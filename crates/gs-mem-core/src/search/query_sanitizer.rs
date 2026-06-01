use once_cell::sync::Lazy;
use regex::Regex;

// Compile-time-verified regex literals; `expect` here is sound and cannot panic
// at runtime. Justified allow rather than fallible init over a `Lazy`.
#[allow(clippy::expect_used)]
static CLEAN: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^A-Za-z0-9_'\- ]+").expect("valid regex"));
#[allow(clippy::expect_used)]
static WS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));

/// Sanitize a user query for safe FTS5 / tantivy use. Strips punctuation that
/// could trigger parse errors or operator injection; collapses whitespace.
pub fn sanitize(raw: &str) -> String {
    let no_punct = CLEAN.replace_all(raw, " ");
    WS.replace_all(no_punct.as_ref(), " ").trim().to_string()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn strips_fts_operators() {
        assert_eq!(
            sanitize(r#"hello AND "world" NEAR x"#),
            "hello AND world NEAR x"
        );
        assert_eq!(sanitize("foo(bar)"), "foo bar");
        assert_eq!(sanitize("  multi   space  "), "multi space");
    }

    #[test]
    fn preserves_content_words() {
        assert_eq!(sanitize("rust-lang's serde"), "rust-lang's serde");
    }
}
