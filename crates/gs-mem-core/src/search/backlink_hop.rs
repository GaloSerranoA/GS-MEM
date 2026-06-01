//! Graph-traversal search: BFS over backlinks from an anchor slug, plus a
//! graph-distance rerank utility that boosts BM25/vector candidates which
//! sit close to an anchor in the link graph.

use std::collections::HashMap;

use crate::error::Result;
use crate::storage::{SqliteStorage, Storage};
use crate::types::Slug;

/// Pure graph search: return slugs reachable from `anchor` within `max_depth`
/// hops, ordered by distance (closest first), capped at `limit`.
pub fn search(
    storage: &SqliteStorage,
    anchor: &Slug,
    max_depth: u8,
    limit: usize,
) -> Result<Vec<(Slug, u8)>> {
    let mut out = storage.bfs_backlinks(anchor, max_depth)?;
    out.truncate(limit);
    Ok(out)
}

/// Given `(slug, base_score)` candidates from some other retriever, boost
/// scores for slugs within `max_depth` hops of `anchor`:
/// `score *= 1.0 + distance_boost / (hop + 1)`. Candidates not reachable
/// are unchanged. Result is re-sorted by score descending.
pub fn rerank_by_graph_distance(
    storage: &SqliteStorage,
    anchor: &Slug,
    candidates: Vec<(String, f32)>,
    max_depth: u8,
    distance_boost: f32,
) -> Result<Vec<(String, f32)>> {
    let nearby = storage.bfs_backlinks(anchor, max_depth)?;
    let distance_map: HashMap<String, u8> = nearby
        .into_iter()
        .map(|(s, d)| (s.as_str().to_string(), d))
        .collect();

    let mut out: Vec<(String, f32)> = candidates
        .into_iter()
        .map(|(slug, score)| {
            let boost = match distance_map.get(&slug) {
                Some(hop) => distance_boost / f32::from(*hop + 1),
                None => 0.0,
            };
            (slug, score * (1.0 + boost))
        })
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(out)
}
