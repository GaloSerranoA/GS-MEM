use crate::error::Result;
use crate::types::{Chunk, Slug};

pub mod dialect;
pub mod simhash;
pub mod structural;

#[derive(Debug, Default, Clone)]
pub struct StructuralChunker;

pub trait Chunker {
    fn chunk_page(&self, slug: &Slug, body: &str) -> Result<Vec<Chunk>>;
}

impl Chunker for StructuralChunker {
    fn chunk_page(&self, slug: &Slug, body: &str) -> Result<Vec<Chunk>> {
        structural::chunk_markdown(slug, body)
    }
}
