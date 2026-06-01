use crate::error::Result;
use crate::types::{Link, Slug};

pub mod bitemporal;
pub mod link_detector;
pub mod triples;

pub trait GraphStore {
    fn open_edge(&self, from: &Slug, to: &Slug, edge_type: &str, valid_at_unix: i64) -> Result<()>;
    fn close_edge(
        &self,
        from: &Slug,
        to: &Slug,
        edge_type: &str,
        invalid_at_unix: i64,
    ) -> Result<()>;
    fn backlinks(&self, slug: &Slug) -> Result<Vec<Link>>;
}
