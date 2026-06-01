use once_cell::sync::Lazy;
use regex::Regex;

#[allow(clippy::expect_used)]
static WIKI: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\[\[([A-Za-z0-9_/.\-]+)\]\]").expect("valid regex"));
#[allow(clippy::expect_used)]
static MD_LINK: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\[[^\]]*\]\(\.?/?([A-Za-z0-9_/\-]+?)(?:\.md)?\)").expect("valid regex")
});

/// Extract referenced slugs from markdown body. Deduped, excluding the source slug.
#[must_use]
pub fn extract_links(body: &str, source_slug: &str) -> Vec<String> {
    use std::collections::BTreeSet;

    let mut set = BTreeSet::new();
    for cap in WIKI.captures_iter(body) {
        if let Some(matched) = cap.get(1) {
            let slug = matched.as_str().to_string();
            if slug != source_slug {
                set.insert(slug);
            }
        }
    }

    for cap in MD_LINK.captures_iter(body) {
        if let Some(matched) = cap.get(1) {
            let slug = matched.as_str().to_string();
            if !slug.starts_with("http") && slug != source_slug {
                set.insert(slug);
            }
        }
    }

    set.into_iter().collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn extracts_wiki_links() {
        let body = "This references [[concept-x]] and [[projects/foo]].";
        let links = extract_links(body, "self");
        assert_eq!(
            links,
            vec!["concept-x".to_string(), "projects/foo".to_string()]
        );
    }

    #[test]
    fn extracts_markdown_links() {
        let body = "See [the intro](./intro.md) and [the guide](guide.md) and [README](readme).";
        let links = extract_links(body, "self");
        assert!(links.contains(&"intro".to_string()));
        assert!(links.contains(&"guide".to_string()));
        assert!(links.contains(&"readme".to_string()));
    }

    #[test]
    fn deduplicates_and_excludes_self() {
        let body = "[[self]] and [[other]] and [[other]] again";
        let links = extract_links(body, "self");
        assert_eq!(links, vec!["other".to_string()]);
    }

    #[test]
    fn ignores_urls() {
        let body = "See [google](https://google.com)";
        let links = extract_links(body, "self");
        assert!(links.is_empty());
    }
}
