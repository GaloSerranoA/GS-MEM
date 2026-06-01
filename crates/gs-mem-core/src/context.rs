use std::sync::{Arc, Mutex};

use crate::config::Config;
use crate::embedding::{EmbeddingProvider, SovereignEmbedder};
use crate::error::Result;
use crate::search::vector_index::VectorIndex;
use crate::search::SearchEngine;
use crate::storage::SqliteStorage;

pub struct Context {
    storage: Arc<Mutex<SqliteStorage>>,
    engine: SearchEngine,
    embedder: Option<Arc<dyn EmbeddingProvider + Send + Sync>>,
    vectors: Option<Arc<VectorIndex>>,
}

impl Context {
    pub fn open(config: &Config) -> Result<Self> {
        let storage = SqliteStorage::open(&config.db_path)?;
        storage.init_schema()?;
        let storage = Arc::new(Mutex::new(storage));

        let (embedder, vectors) = match SovereignEmbedder::default_paths() {
            Ok(embedder) => {
                let vectors = Arc::new(VectorIndex::new(&config.index_root, embedder.dim())?);
                (
                    Some(Arc::new(embedder) as Arc<dyn EmbeddingProvider + Send + Sync>),
                    Some(vectors),
                )
            }
            Err(_) => (None, None),
        };

        Ok(Self {
            storage,
            engine: SearchEngine::default(),
            embedder,
            vectors,
        })
    }

    pub fn storage_only(storage: Arc<Mutex<SqliteStorage>>) -> Self {
        Self {
            storage,
            engine: SearchEngine::default(),
            embedder: None,
            vectors: None,
        }
    }

    pub fn with_embedder(
        storage: Arc<Mutex<SqliteStorage>>,
        embedder: Arc<dyn EmbeddingProvider + Send + Sync>,
        vectors: Arc<VectorIndex>,
    ) -> Self {
        Self {
            storage,
            engine: SearchEngine::default(),
            embedder: Some(embedder),
            vectors: Some(vectors),
        }
    }

    pub fn storage(&self) -> &Arc<Mutex<SqliteStorage>> {
        &self.storage
    }

    pub(crate) fn engine(&self) -> &SearchEngine {
        &self.engine
    }

    pub(crate) fn embedder(&self) -> Option<&Arc<dyn EmbeddingProvider + Send + Sync>> {
        self.embedder.as_ref()
    }

    pub(crate) fn vectors(&self) -> Option<&Arc<VectorIndex>> {
        self.vectors.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn context_is_send_sync() {
        assert_send_sync::<Context>();
        assert_send_sync::<SovereignEmbedder>();
    }
}
