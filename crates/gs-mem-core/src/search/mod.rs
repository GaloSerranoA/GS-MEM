//! Search orchestration over BM25 (Tantivy) + vector + backlink + tag
//! sub-indexes, fused via Reciprocal Rank Fusion.
//!
//! The orchestrator is parameterized over `SearchBackend` so callers can plug
//! in production indexes (`TantivyIndex`, `VectorIndex`) or test doubles. RRF
//! fusion remains the single hybrid strategy; intent classification is
//! preserved as a hook for future per-intent strategy selection.

use crate::error::Result;

pub mod backlink_hop;
pub mod hnsw;
pub mod intent;
pub mod query_sanitizer;
pub mod rrf;
pub mod strategy;
pub mod tantivy_index;
pub mod vector_index;

use intent::{classify_intent, QueryIntent};
use rrf::reciprocal_rank_fusion;
use strategy::SearchStrategy;

/// Pluggable backend that exposes the per-strategy search primitives. Real
/// implementations wire `TantivyIndex` and `VectorIndex`; tests can supply
/// in-memory doubles.
pub trait SearchBackend {
    /// BM25 / full-text search. Returns ranked slugs.
    fn full_text(&self, query: &str, limit: usize) -> Result<Vec<String>>;
    /// Vector / embedding search. Returns ranked slugs.
    fn vector(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<String>>;
    /// Tag intersection search. Empty default — implementations may override.
    fn tag_intersection(&self, _tags: &[&str], _limit: usize) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
    /// Backlink-hop search starting from `slug`. Empty default.
    fn backlink_hop(&self, _slug: &str, _max_depth: u8, _limit: usize) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone)]
pub struct SearchEngine {
    pub strategies: Vec<SearchStrategy>,
}

impl Default for SearchEngine {
    fn default() -> Self {
        Self {
            strategies: vec![SearchStrategy::Hybrid { rrf_k: 60 }],
        }
    }
}

impl SearchEngine {
    /// Run the configured strategies via `backend`, fuse the resulting ranked
    /// lists with Reciprocal Rank Fusion, and return the top `limit` slugs.
    ///
    /// `query_embedding` is required when any active strategy needs vector
    /// search (`Vector` or `Hybrid`). When the embedding is empty and a
    /// vector strategy is requested, that sub-strategy is skipped (the
    /// remaining sub-strategies still contribute to the fused ranking).
    pub fn search<B: SearchBackend>(
        &self,
        backend: &B,
        query: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<String>> {
        if limit == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }

        // Intent classification reserved for future strategy selection.
        // Phase 2 will route per-intent (e.g., Reference → FullText only,
        // Concept → Hybrid). Computed here for the side effect of validating
        // the classifier path; per gmem's silent-by-design Gate 6 we do not
        // emit a tracing event.
        let _intent: QueryIntent = classify_intent(query);

        let mut ranked_lists: Vec<Vec<String>> = Vec::new();
        let mut rrf_k: u32 = 60;

        for strategy in &self.strategies {
            match strategy {
                SearchStrategy::FullText => {
                    ranked_lists.push(backend.full_text(query, limit)?);
                }
                SearchStrategy::Vector => {
                    if !query_embedding.is_empty() {
                        ranked_lists.push(backend.vector(query_embedding, limit)?);
                    }
                }
                SearchStrategy::Hybrid { rrf_k: k } => {
                    rrf_k = *k;
                    ranked_lists.push(backend.full_text(query, limit)?);
                    if !query_embedding.is_empty() {
                        ranked_lists.push(backend.vector(query_embedding, limit)?);
                    }
                }
                SearchStrategy::TagIntersection => {
                    // Tag extraction from a free-text query is a separate concern.
                    // Callers using this strategy should tokenize and pass tags via
                    // a dedicated method; here we leave the slot empty so RRF
                    // ignores it.
                }
                SearchStrategy::BacklinkHop { max_depth: _ } => {
                    // Backlink-hop needs a starting slug, not a free-text query.
                    // Callers using this strategy should drive backlink_hop
                    // directly.
                }
            }
        }

        let fused = reciprocal_rank_fusion(&ranked_lists, rrf_k);
        Ok(fused
            .into_iter()
            .take(limit)
            .map(|(slug, _score)| slug)
            .collect())
    }

    /// Stable identifier for registry lookup per the Algorithm
    /// Integration Contract v1.0 §6.2.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        "immortal-gmem::search::SearchEngine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only backend: returns canned ranked lists per call type.
    struct FakeBackend {
        full_text_results: Vec<String>,
        vector_results: Vec<String>,
    }

    impl SearchBackend for FakeBackend {
        fn full_text(&self, _query: &str, _limit: usize) -> Result<Vec<String>> {
            Ok(self.full_text_results.clone())
        }
        fn vector(&self, _q: &[f32], _limit: usize) -> Result<Vec<String>> {
            Ok(self.vector_results.clone())
        }
    }

    #[test]
    fn empty_query_returns_empty() {
        let engine = SearchEngine::default();
        let backend = FakeBackend {
            full_text_results: vec!["a".into(), "b".into()],
            vector_results: vec!["c".into()],
        };
        let r = engine.search(&backend, "", &[1.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn limit_zero_returns_empty() {
        let engine = SearchEngine::default();
        let backend = FakeBackend {
            full_text_results: vec!["a".into()],
            vector_results: vec!["b".into()],
        };
        let r = engine.search(&backend, "anything", &[1.0], 0).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn hybrid_fuses_full_text_and_vector() {
        let engine = SearchEngine::default(); // Hybrid { rrf_k: 60 }
        let backend = FakeBackend {
            full_text_results: vec!["a".into(), "b".into(), "c".into()],
            vector_results: vec!["b".into(), "c".into(), "d".into()],
        };
        let r = engine.search(&backend, "anything", &[1.0, 2.0], 4).unwrap();
        // "b" appears at rank 1 in vector and rank 2 in full_text → highest fused score.
        // "a" at rank 1 in full_text only.
        // "c" at rank 2 in full_text and rank 2 in vector.
        // RRF score = sum(1 / (k + rank + 1))
        assert_eq!(r[0], "b", "b should win RRF; got: {r:?}");
    }

    #[test]
    fn full_text_only_strategy_skips_vector() {
        let engine = SearchEngine {
            strategies: vec![SearchStrategy::FullText],
        };
        let backend = FakeBackend {
            full_text_results: vec!["a".into(), "b".into()],
            vector_results: vec!["should-not-appear".into()],
        };
        let r = engine.search(&backend, "q", &[1.0], 5).unwrap();
        assert_eq!(r, vec!["a", "b"]);
    }

    #[test]
    fn vector_only_strategy_skips_full_text() {
        let engine = SearchEngine {
            strategies: vec![SearchStrategy::Vector],
        };
        let backend = FakeBackend {
            full_text_results: vec!["should-not-appear".into()],
            vector_results: vec!["v1".into(), "v2".into()],
        };
        let r = engine.search(&backend, "q", &[1.0], 5).unwrap();
        assert_eq!(r, vec!["v1", "v2"]);
    }

    #[test]
    fn empty_embedding_skips_vector_substrategy_in_hybrid() {
        let engine = SearchEngine::default(); // Hybrid
        let backend = FakeBackend {
            full_text_results: vec!["a".into()],
            vector_results: vec!["should-not-appear".into()],
        };
        let r = engine.search(&backend, "q", &[], 5).unwrap();
        assert_eq!(r, vec!["a"]);
    }

    #[test]
    fn tag_intersection_default_returns_empty() {
        let backend = FakeBackend {
            full_text_results: vec![],
            vector_results: vec![],
        };
        let r = backend.tag_intersection(&["tag1"], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn backlink_hop_default_returns_empty() {
        let backend = FakeBackend {
            full_text_results: vec![],
            vector_results: vec![],
        };
        let r = backend.backlink_hop("some-slug", 2, 5).unwrap();
        assert!(r.is_empty());
    }
}
