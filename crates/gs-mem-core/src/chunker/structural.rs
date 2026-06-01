//! Structural markdown chunker — splits a page body into [`Chunk`]s on
//! heading boundaries.
//!
//! Strategy: iterate the body line-by-line. Whenever a line matches the
//! markdown ATX heading pattern (`^#+\s+`), close the current chunk (if
//! non-empty) and start a new one whose first line is the heading. The
//! result is a sequence of `Chunk`s that respect markdown structure
//! without requiring a full AST parser. Each chunk gets a stable
//! `content_fp` and a `simhash` derived from the chunk body so downstream
//! indexers (Tantivy + vector) can dedupe and skip-gate cleanly.
//!
//! This module stays dependency-light by using the existing `simhash` module
//! and `Fingerprint` primitive in this crate.

use uuid::Uuid;

use crate::chunker::simhash::simhash64;
use crate::error::Result;
use crate::fingerprint::Fingerprint;
use crate::types::{Chunk, Slug};

/// Walk `body` and emit one [`Chunk`] per heading-bounded section.
///
/// The page id used for each chunk's `page_id` is derived deterministically
/// from `slug` so callers that don't yet have a database-assigned page id
/// still get stable identifiers (the indexer overwrites `page_id` after
/// page persistence).
pub fn chunk_markdown(slug: &Slug, body: &str) -> Result<Vec<Chunk>> {
    let page_id = page_id_from_slug(slug);
    let sections = split_by_headings(body);

    let mut out = Vec::with_capacity(sections.len().max(1));
    for (ord, body) in sections.into_iter().enumerate() {
        let trimmed = body.trim();
        if trimmed.is_empty() {
            continue;
        }
        let body_owned = trimmed.to_string();
        let content_fp = Fingerprint::of(&body_owned.as_str());
        let simhash = simhash64(&body_owned);
        out.push(Chunk {
            id: Uuid::new_v4(),
            page_id,
            ord: ord as u32,
            body: body_owned,
            content_fp,
            simhash,
            embedding: None,
        });
    }

    if out.is_empty() && !body.trim().is_empty() {
        // Whole-body fallback: caller passed a body with no recognized
        // structure. Emit a single chunk so the page is still indexable.
        let body_owned = body.trim().to_string();
        let content_fp = Fingerprint::of(&body_owned.as_str());
        let simhash = simhash64(&body_owned);
        out.push(Chunk {
            id: Uuid::new_v4(),
            page_id,
            ord: 0,
            body: body_owned,
            content_fp,
            simhash,
            embedding: None,
        });
    }

    Ok(out)
}

/// Derive a deterministic UUID from a slug for chunks created before the
/// storage layer assigns a database page id. Stable across runs so dedup gates
/// work against the same chunked-without-page representation.
fn page_id_from_slug(slug: &Slug) -> Uuid {
    let fp = Fingerprint::of(&slug.as_str());
    let bytes = fp.as_bytes();
    // Take the first 16 bytes of the BLAKE2 fingerprint and stamp them as
    // a UUID. Not RFC-4122-compliant in version bits, but uniqueness is
    // what matters and BLAKE2 collision-resistance covers it.
    #[allow(
        clippy::expect_used,
        reason = "Fingerprint::as_bytes always returns at least 32 bytes (BLAKE2b-256)"
    )]
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&bytes[..16]);
    Uuid::from_bytes(buf)
}

/// Split `body` into sections where each section starts with an ATX
/// heading line (`#`, `##`, …). Text before the first heading is its own
/// section (the "preamble").
fn split_by_headings(body: &str) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in body.lines() {
        if is_atx_heading(line) && !current.trim().is_empty() {
            sections.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        sections.push(current);
    }
    sections
}

/// True for lines like `# Title`, `## Subtitle`, up to `###### h6`. Allows
/// the standard `#` followed by at least one whitespace character.
fn is_atx_heading(line: &str) -> bool {
    let trimmed = line.trim_start();
    let mut chars = trimmed.chars();
    let mut hashes = 0;
    for c in chars.by_ref() {
        if c == '#' {
            hashes += 1;
            if hashes > 6 {
                return false;
            }
        } else {
            return hashes >= 1 && c.is_whitespace();
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slug(s: &str) -> Slug {
        Slug::new(s).expect("valid test slug")
    }

    #[test]
    fn chunks_split_on_headings() {
        let body = "\
Preamble paragraph.

# First Section
First section content.

## Subsection
Subsection content.

# Second Section
Second section content.
";
        let chunks = chunk_markdown(&slug("test/page"), body).unwrap();
        assert_eq!(chunks.len(), 4); // preamble + first + sub + second
                                     // First chunk is the preamble, no heading.
        assert!(chunks[0].body.contains("Preamble paragraph"));
        assert!(chunks[1].body.starts_with("# First Section"));
        assert!(chunks[2].body.starts_with("## Subsection"));
        assert!(chunks[3].body.starts_with("# Second Section"));
    }

    #[test]
    fn chunks_have_unique_content_fingerprints() {
        let body = "# A\nfoo\n# B\nbar\n";
        let chunks = chunk_markdown(&slug("test/page"), body).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_ne!(chunks[0].content_fp, chunks[1].content_fp);
        assert_ne!(chunks[0].simhash, chunks[1].simhash);
    }

    #[test]
    fn chunks_have_sequential_ord() {
        let body = "# A\na\n# B\nb\n# C\nc\n";
        let chunks = chunk_markdown(&slug("test/page"), body).unwrap();
        assert_eq!(chunks[0].ord, 0);
        assert_eq!(chunks[1].ord, 1);
        assert_eq!(chunks[2].ord, 2);
    }

    #[test]
    fn empty_body_returns_no_chunks() {
        let chunks = chunk_markdown(&slug("test/page"), "").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn body_without_headings_returns_single_chunk() {
        let body = "Just one paragraph, no headings.";
        let chunks = chunk_markdown(&slug("test/page"), body).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].ord, 0);
        assert!(chunks[0].body.contains("Just one paragraph"));
    }

    #[test]
    fn page_id_deterministic_for_same_slug() {
        let body = "# A\na\n";
        let a = chunk_markdown(&slug("test/page"), body).unwrap();
        let b = chunk_markdown(&slug("test/page"), body).unwrap();
        assert_eq!(a[0].page_id, b[0].page_id);
        // Different slug → different page_id.
        let c = chunk_markdown(&slug("other/page"), body).unwrap();
        assert_ne!(a[0].page_id, c[0].page_id);
    }

    #[test]
    fn chunk_body_preserves_subsection_text() {
        let body = "# Section\nLine 1\nLine 2\n## Sub\nNested\n";
        let chunks = chunk_markdown(&slug("p"), body).unwrap();
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].body.contains("Line 1"));
        assert!(chunks[0].body.contains("Line 2"));
        assert!(chunks[1].body.contains("Nested"));
    }

    #[test]
    fn h7_or_more_hashes_not_treated_as_heading() {
        // Markdown spec: only h1-h6. Seven hashes → not a heading.
        let body = "####### Not a heading\nbody\n";
        let chunks = chunk_markdown(&slug("p"), body).unwrap();
        assert_eq!(chunks.len(), 1); // single fallback chunk
    }

    #[test]
    fn hash_without_trailing_space_not_a_heading() {
        // "#word" is not a heading per ATX spec — needs whitespace after the run.
        let body = "#tag-not-heading\nbody\n";
        let chunks = chunk_markdown(&slug("p"), body).unwrap();
        assert_eq!(chunks.len(), 1);
    }
}
