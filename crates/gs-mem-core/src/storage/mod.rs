use std::path::Path;

use crate::error::Result;
use crate::fingerprint::Fingerprint;
use crate::graph::triples::SpoTriple;
use crate::indexer::source_version::SourceVersion;
use crate::types::{Link, Page, Slug};

pub mod schema;
pub mod sqlite;

pub use sqlite::SqliteStorage;

pub trait Storage: Sized {
    fn open(path: impl AsRef<Path>) -> Result<Self>;
    fn init_schema(&self) -> Result<()>;
    fn get_page(&self, slug: &Slug) -> Result<Page>;
    fn put_page(&self, page: &Page) -> Result<()>;
    fn list_pages(&self, tag: Option<&str>, limit: usize) -> Result<Vec<Page>>;
    fn search_keyword(&self, query: &str, limit: usize) -> Result<Vec<(Slug, String)>>;
    fn get_source_version(&self, slug: &Slug) -> Result<Option<SourceVersion>>;
    fn put_source_version(
        &self,
        slug: &Slug,
        mtime_us: i64,
        content_fp: &Fingerprint,
        logic_fp: &Fingerprint,
    ) -> Result<()>;
    fn upsert_link(&self, from: &Slug, to: &Slug, edge_type: &str) -> Result<()>;
    fn close_links_from(&self, from: &Slug) -> Result<()>;
    fn backlinks_to(&self, to: &Slug, only_current: bool) -> Result<Vec<Link>>;
    fn add_triple(&self, triple: &SpoTriple) -> Result<i64>;
    fn close_triple(&self, id: i64) -> Result<()>;
    fn triples_by_subject(&self, subject: &str, only_current: bool) -> Result<Vec<SpoTriple>>;
    fn triples_by_object(&self, object: &str, only_current: bool) -> Result<Vec<SpoTriple>>;
    /// BFS over current (invalid_at IS NULL) backlinks starting at `anchor`.
    /// Returns `(slug, hop_distance)` pairs for every slug reachable within
    /// `max_depth` hops (the anchor itself is excluded).
    fn bfs_backlinks(&self, anchor: &Slug, max_depth: u8) -> Result<Vec<(Slug, u8)>>;
}
