use std::collections::HashMap;
use std::hash::Hash;

pub fn reciprocal_rank_fusion<T>(ranked_lists: &[Vec<T>], k: u32) -> Vec<(T, f32)>
where
    T: Clone + Eq + Hash,
{
    let mut scores: HashMap<T, f32> = HashMap::new();

    for ranked in ranked_lists {
        for (rank, item) in ranked.iter().enumerate() {
            let denom = k as f32 + rank as f32 + 1.0;
            let score = 1.0 / denom;
            let entry = scores.entry(item.clone()).or_insert(0.0);
            *entry += score;
        }
    }

    let mut fused: Vec<(T, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused
}
