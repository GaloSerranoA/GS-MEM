//! Typed `GmemError` for the brain primitives.
//!
//! Variant set grows as each phase lands. Phases 1-4 in.
//! `#[non_exhaustive]` keeps downstream matches future-safe.

use thiserror::Error;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum GmemError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("sqlite error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    #[error("page not found: {slug}")]
    NotFound { slug: String },
    #[error("invalid slug: {0}")]
    InvalidSlug(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("embedding error: {0}")]
    Embedding(String),
    #[error("search error: {0}")]
    Search(String),
    #[error("mcp error: {0}")]
    Mcp(String),
    #[error("other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, GmemError>;
