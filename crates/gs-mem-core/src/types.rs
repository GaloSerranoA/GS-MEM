//! Core domain types for the brain: `Page`, `Slug`, `PageScope`, `Chunk`, `Tag`, `Link`.
//!
//! Absorbed verbatim from `gmem/src/types.rs` on 2026-04-18 (phase 1/15 of the
//! master absorption plan). No semantic changes — only module namespace moves.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use time::OffsetDateTime;
use uuid::Uuid;

use crate::error::{GmemError, Result};
use crate::fingerprint::Fingerprint;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct Slug(String);

impl Slug {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if is_valid_slug(&value) {
            Ok(Self(value))
        } else {
            Err(GmemError::InvalidSlug(value))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Slug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for Slug {
    type Err = GmemError;

    fn from_str(s: &str) -> Result<Self> {
        Self::new(s)
    }
}

fn is_valid_slug(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('/')
        && !value.ends_with('/')
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '/' | '.'))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum PageScope {
    Project,
    Concept,
    Reference,
    Person,
    Plan,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Page {
    pub id: Uuid,
    pub slug: Slug,
    pub scope: PageScope,
    pub title: Option<String>,
    pub compiled_truth: String,
    pub timeline: Vec<String>,
    pub frontmatter: Option<serde_json::Value>,
    pub content_fp: Fingerprint,
    pub created_at: OffsetDateTime,
    pub updated_at: OffsetDateTime,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub page_id: Uuid,
    pub ord: u32,
    pub body: String,
    pub content_fp: Fingerprint,
    pub simhash: u64,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tag {
    pub page_id: Uuid,
    pub tag: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Link {
    pub id: Option<i64>,
    pub from_slug: Slug,
    pub to_slug: Slug,
    pub edge_type: String,
    pub valid_at: OffsetDateTime,
    pub invalid_at: Option<OffsetDateTime>,
}
