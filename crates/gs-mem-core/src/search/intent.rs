use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryIntent {
    Tag,
    Backlink,
    Temporal,
    Semantic,
    Relational,
}

pub fn classify_intent(q: &str) -> QueryIntent {
    let q = q.to_ascii_lowercase();
    if q.contains("tag:") || q.starts_with('#') {
        QueryIntent::Tag
    } else if q.contains("backlink") || q.contains("links to") {
        QueryIntent::Backlink
    } else if q.contains("today") || q.contains("yesterday") || q.contains("timeline") {
        QueryIntent::Temporal
    } else if q.contains("related") || q.contains("depends on") || q.contains("connects") {
        QueryIntent::Relational
    } else {
        QueryIntent::Semantic
    }
}
