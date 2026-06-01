//! Dialect compression — optional text pre-processing that shrinks
//! indexed content while preserving semantic signals. Ported from
//! Inmortal-Cogn8ty-memory (mempalace) with adaptations for our schema.
//!
//! Not wired into the default indexing path. Callers opt in via
//! `DialectCompressor::new().compress(text)`.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Mapping of content words → short emotion codes.
const EMOTION_SIGNALS: &[(&str, &str)] = &[
    ("decided", "determ"),
    ("prefer", "convict"),
    ("worried", "anx"),
    ("excited", "excite"),
    ("frustrated", "frust"),
    ("confused", "confuse"),
    ("love", "love"),
    ("hate", "rage"),
    ("hope", "hope"),
    ("fear", "fear"),
    ("trust", "trust"),
    ("happy", "joy"),
    ("sad", "grief"),
];

/// Mapping of content words → semantic flag tags (propagated as tokens).
const FLAG_SIGNALS: &[(&str, &str)] = &[
    ("decided", "DECISION"),
    ("chose", "DECISION"),
    ("switched", "DECISION"),
    ("migrated", "DECISION"),
    ("created", "ORIGIN"),
    ("started", "ORIGIN"),
    ("core", "CORE"),
    ("fundamental", "CORE"),
    ("breakthrough", "PIVOT"),
    ("api", "TECHNICAL"),
    ("database", "TECHNICAL"),
    ("architecture", "TECHNICAL"),
];

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "for", "on", "with",
    "at", "by", "from", "as", "and", "but", "or", "if", "this", "that", "it", "its", "i", "we",
    "you", "they", "my", "our", "their", "have", "has", "had", "will", "would", "could", "should",
    "because", "like", "about", "into",
];

#[allow(clippy::expect_used)]
static TOKEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Za-z][A-Za-z_-]{2,}").expect("valid regex"));

fn stopword_set() -> &'static HashSet<&'static str> {
    static S: Lazy<HashSet<&'static str>> = Lazy::new(|| STOPWORDS.iter().copied().collect());
    &S
}

fn emotion_map() -> &'static HashMap<&'static str, &'static str> {
    static M: Lazy<HashMap<&'static str, &'static str>> =
        Lazy::new(|| EMOTION_SIGNALS.iter().copied().collect());
    &M
}

fn flag_map() -> &'static HashMap<&'static str, &'static str> {
    static M: Lazy<HashMap<&'static str, &'static str>> =
        Lazy::new(|| FLAG_SIGNALS.iter().copied().collect());
    &M
}

/// Compressor state. Entity codes supplied at construction time (personal brain context).
#[derive(Debug, Clone, Default)]
pub struct DialectCompressor {
    entity_codes: HashMap<String, String>,
}

impl DialectCompressor {
    /// Construct a new, empty compressor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert/overwrite an entity → short-code mapping. Builder-style.
    #[must_use]
    pub fn with_entity(mut self, name: impl Into<String>, code: impl Into<String>) -> Self {
        self.entity_codes
            .insert(name.into().to_lowercase(), code.into());
        self
    }

    /// Compress `text` by (1) entity-code substitution, (2) emotion signal emission,
    /// (3) flag signal emission, (4) stopword removal. Output is space-joined tokens.
    #[must_use]
    pub fn compress(&self, text: &str) -> String {
        let stop = stopword_set();
        let emo = emotion_map();
        let flag = flag_map();

        let mut out: Vec<String> = Vec::new();
        for mat in TOKEN_RE.find_iter(text) {
            let raw = mat.as_str();
            let lower = raw.to_lowercase();

            if let Some(code) = self.entity_codes.get(&lower) {
                out.push(code.clone());
                continue;
            }

            if stop.contains(lower.as_str()) {
                continue;
            }

            out.push(lower.clone());

            if let Some(e) = emo.get(lower.as_str()) {
                out.push(format!("emotion_{e}"));
            }
            if let Some(f) = flag.get(lower.as_str()) {
                out.push((*f).to_string());
            }
        }

        out.join(" ")
    }

    /// Compression ratio = `output_len / input_len` (lower = more compression).
    #[must_use]
    pub fn ratio(&self, original: &str) -> f32 {
        let compressed = self.compress(original);
        #[allow(clippy::cast_precision_loss)]
        let orig_len = original.len() as f32;
        if orig_len == 0.0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let comp_len = compressed.len() as f32;
        comp_len / orig_len
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn strips_stopwords() {
        let d = DialectCompressor::new();
        let out = d.compress("the architecture of the system");
        assert!(out.contains("architecture"));
        assert!(out.contains("TECHNICAL"));
        assert!(!out.split_whitespace().any(|t| t == "the"));
        assert!(!out.split_whitespace().any(|t| t == "of"));
    }

    #[test]
    fn emits_emotion_and_flag_signals() {
        let d = DialectCompressor::new();
        let out = d.compress("I decided to migrate");
        assert!(out.contains("decided"));
        assert!(out.contains("emotion_determ"));
        assert!(out.contains("DECISION"));
    }

    #[test]
    fn substitutes_entities() {
        let d = DialectCompressor::new().with_entity("GalaxyCorp", "gc");
        let out = d.compress("The GalaxyCorp database is large");
        assert!(out.split_whitespace().any(|t| t == "gc"));
        assert!(!out.contains("galaxycorp"));
        assert!(out.contains("database"));
        assert!(out.contains("TECHNICAL"));
    }

    #[test]
    fn compression_ratio_is_below_one_for_stopword_heavy() {
        let d = DialectCompressor::new();
        let long = "the the the the the the the the the the architecture";
        let ratio = d.ratio(long);
        assert!(ratio < 0.5, "expected heavy compression, got ratio {ratio}");
    }

    #[test]
    fn empty_input_is_empty_output() {
        let d = DialectCompressor::new();
        assert_eq!(d.compress(""), "");
        assert!((d.ratio("") - 1.0).abs() < f32::EPSILON);
    }
}
