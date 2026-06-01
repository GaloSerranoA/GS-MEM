//! Pin test: every public enum in immortal-gmem carries `#[non_exhaustive]`.

use std::fs;
use std::path::{Path, PathBuf};

fn assert_enum_non_exhaustive(file: &Path, enum_name: &str) {
    let src = fs::read_to_string(file).unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
    let needle = format!("pub enum {enum_name}");
    let pos = src
        .find(&needle)
        .unwrap_or_else(|| panic!("`pub enum {enum_name}` not found in {}", file.display()));
    let preceding = &src[..pos];
    let nearest = preceding.rfind("#[non_exhaustive]");
    assert!(
        nearest.is_some_and(|start| preceding[start..].lines().count() <= 5),
        "{enum_name} in {} must carry #[non_exhaustive] within 5 lines of its `pub enum` declaration.",
        file.display()
    );
}

fn src(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join(rel)
}

#[test]
fn gmem_error_is_non_exhaustive() {
    assert_enum_non_exhaustive(&src("error.rs"), "GmemError");
}
#[test]
fn page_scope_is_non_exhaustive() {
    assert_enum_non_exhaustive(&src("types.rs"), "PageScope");
}
#[test]
fn query_intent_is_non_exhaustive() {
    assert_enum_non_exhaustive(&src("search/intent.rs"), "QueryIntent");
}
#[test]
fn search_strategy_is_non_exhaustive() {
    assert_enum_non_exhaustive(&src("search/strategy.rs"), "SearchStrategy");
}
