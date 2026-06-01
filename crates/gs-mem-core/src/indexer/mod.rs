use crate::chunker::Chunker;
use crate::embedding::EmbeddingProvider;
use crate::error::Result;
use crate::fingerprint::Fingerprint;
use crate::types::{Chunk, Slug};

pub mod memo;
pub mod source_version;

#[derive(Debug)]
pub struct Indexer<C, E> {
    pub chunker: C,
    pub embedder: E,
    pub logic_fp: Fingerprint,
}

impl<C, E> Indexer<C, E>
where
    C: Chunker,
    E: EmbeddingProvider,
{
    pub fn new(chunker: C, embedder: E) -> Self {
        let logic_fp = Fingerprint::of(&"gmem-indexer-v0");
        Self {
            chunker,
            embedder,
            logic_fp,
        }
    }

    /// Run the full indexing pipeline on a page body:
    ///
    ///   1. Chunk the body via the configured [`Chunker`] (one chunk per
    ///      heading-bounded section by default).
    ///   2. For each chunk, compute an embedding via the configured
    ///      [`EmbeddingProvider`] and attach it to the chunk in-place.
    ///   3. Return the complete [`Vec<Chunk>`] ready for persistence.
    ///
    /// Caller is responsible for: writing the chunks to the chunks table
    /// (`storage::sqlite::insert_chunks`) and routing each chunk to the
    /// vector index (`search::vector_index::VectorIndex::insert`).
    pub fn index_markdown(&self, slug: &Slug, body: &str) -> Result<Vec<Chunk>> {
        let mut chunks = self.chunker.chunk_page(slug, body)?;
        for chunk in &mut chunks {
            let embedding = self.embedder.embed(&chunk.body)?;
            chunk.embedding = Some(embedding);
        }
        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::StructuralChunker;
    use crate::embedding::FakeRandomEmbedder;

    fn slug(s: &str) -> Slug {
        Slug::new(s).expect("valid test slug")
    }

    #[test]
    fn index_markdown_produces_chunks_with_embeddings() {
        let indexer = Indexer::new(StructuralChunker, FakeRandomEmbedder);
        let body = "# A\nfirst section\n# B\nsecond section\n";
        let chunks = indexer.index_markdown(&slug("test/p"), body).unwrap();
        assert_eq!(chunks.len(), 2);
        for c in &chunks {
            assert!(c.embedding.is_some(), "chunk must have embedding");
            #[allow(clippy::unwrap_used, reason = "checked Some on the line above")]
            let dim = c.embedding.as_ref().unwrap().len();
            assert_eq!(dim, FakeRandomEmbedder.dim());
        }
    }

    #[test]
    fn index_markdown_empty_body_returns_no_chunks() {
        let indexer = Indexer::new(StructuralChunker, FakeRandomEmbedder);
        let chunks = indexer.index_markdown(&slug("p"), "").unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn index_markdown_preserves_chunk_order() {
        let indexer = Indexer::new(StructuralChunker, FakeRandomEmbedder);
        let body = "# A\na\n# B\nb\n# C\nc\n";
        let chunks = indexer.index_markdown(&slug("p"), body).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].ord, 0);
        assert_eq!(chunks[1].ord, 1);
        assert_eq!(chunks[2].ord, 2);
    }
}
