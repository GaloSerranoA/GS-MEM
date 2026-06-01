use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpoTriple {
    pub id: Option<i64>,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid_from: OffsetDateTime,
    pub valid_to: Option<OffsetDateTime>,
    pub confidence: f32,
    pub source_slug: Option<String>,
    pub source_file: Option<String>,
    pub extracted_at: OffsetDateTime,
}

impl SpoTriple {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        let now = OffsetDateTime::now_utc();
        Self {
            id: None,
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            valid_from: now,
            valid_to: None,
            confidence: 1.0,
            source_slug: None,
            source_file: None,
            extracted_at: now,
        }
    }

    pub fn is_current(&self) -> bool {
        self.valid_to.is_none()
    }
}
