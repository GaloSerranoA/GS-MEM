use crate::error::Result;
use crate::fingerprint::Fingerprint;

pub mod sovereign;

pub use sovereign::SovereignEmbedder;

pub trait EmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    fn dim(&self) -> usize;
}

#[derive(Debug, Default, Clone)]
pub struct FakeRandomEmbedder;

impl FakeRandomEmbedder {
    pub const VERSION: &'static str = "v0";
}

impl EmbeddingProvider for FakeRandomEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let fp = Fingerprint::of(&text);
        let mut embedding = Vec::with_capacity(fp.as_bytes().len());

        for (idx, byte) in fp.as_bytes().iter().enumerate() {
            let centered = (*byte as f32 / 255.0) - 0.5;
            embedding.push(centered + idx as f32 * 0.001);
        }

        Ok(embedding)
    }

    fn dim(&self) -> usize {
        16
    }
}
