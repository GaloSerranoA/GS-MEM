//! Property-based tests for the Phase 2M HNSW vector index.
//!
//! Phase 2O (2026-04-19): closes the audit-identified "no proptest"
//! gap for HNSW. The 12 example tests in `src/search/hnsw.rs` cover
//! distance properties + recall on a fixed 500-vector synthetic
//! corpus. This file fuzzes random corpora to verify HNSW invariants
//! hold across the input distribution.
//!
//! ## Properties verified
//!
//! 1. `insert` never panics on dim-correct embeddings.
//! 2. `search(q, k, ef)` always returns ≤ k results.
//! 3. All returned IDs were previously inserted (no ghost IDs).
//! 4. Self-query: searching for an inserted vector returns it in top-k.
//! 5. Recall@k ≥ 0.85 vs brute-force ground truth on random
//!    100-vector 32-dim corpora (TEST-INTEGRITY contract).
//! 6. Returned similarities are in [-1.0, 1.0] (cosine similarity range,
//!    plus small Hamming-distance correction overhead → tolerance to 1.05).
//! 7. Search results are ordered by descending similarity.
//!
//! These property tests are slower than the unit tests because each
//! case constructs a fresh HnswIndex with up to 100 inserts. Configured
//! to 32 cases per property for that reason.

use std::collections::{HashMap, HashSet};

use gsmem::search::hnsw::HnswIndex;
use proptest::prelude::*;

// ─────────────────────────────────────────────────────────────────────────
// Strategies
// ─────────────────────────────────────────────────────────────────────────

/// Generate a random unit vector of `dim` dimensions from a u64 seed.
fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut raw = Vec::with_capacity(dim);
    for _ in 0..dim {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map upper 32 bits to [-1, 1).
        let bits = (s >> 32) as u32;
        let v = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
        raw.push(v);
    }
    let n: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        raw.into_iter().map(|x| x / n).collect()
    } else {
        // Fallback: place all mass on first dim.
        let mut v = vec![0.0_f32; dim];
        v[0] = 1.0;
        v
    }
}

/// Build a random index with `n` vectors.
fn build_random_index(dim: usize, n: usize, base_seed: u64) -> HnswIndex {
    let idx = HnswIndex::with_dim(dim);
    for i in 0..n {
        let v = random_unit_vec(dim, base_seed.wrapping_add(i as u64));
        idx.insert(&format!("v{i}"), &v);
    }
    idx
}

// ─────────────────────────────────────────────────────────────────────────
// Property 1-3: basic invariants
// ─────────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn insert_never_panics_on_random_input(
        dim in 4usize..=32,
        n in 0usize..=20,
        seed in any::<u64>(),
    ) {
        let _ = build_random_index(dim, n, seed);
    }

    #[test]
    fn search_returns_at_most_k_results(
        dim in 4usize..=32,
        n in 1usize..=30,
        k in 1usize..=10,
        ef in 10usize..=50,
        seed in any::<u64>(),
        query_seed in any::<u64>(),
    ) {
        let idx = build_random_index(dim, n, seed);
        let q = random_unit_vec(dim, query_seed);
        let results = idx.search(&q, k, ef);
        prop_assert!(
            results.len() <= k,
            "search returned {} results, expected ≤ k = {k}",
            results.len()
        );
        prop_assert!(
            results.len() <= n,
            "search returned {} results, but only {n} indexed",
            results.len()
        );
    }

    #[test]
    fn search_returns_only_inserted_ids(
        dim in 4usize..=32,
        n in 1usize..=30,
        k in 1usize..=10,
        ef in 10usize..=50,
        seed in any::<u64>(),
        query_seed in any::<u64>(),
    ) {
        let idx = build_random_index(dim, n, seed);
        let q = random_unit_vec(dim, query_seed);
        let results = idx.search(&q, k, ef);
        let valid_ids: HashSet<String> = (0..n).map(|i| format!("v{i}")).collect();
        for (id, _) in &results {
            prop_assert!(
                valid_ids.contains(id),
                "search returned ghost id '{id}' that was never inserted"
            );
        }
        // No duplicate IDs in a single result set.
        let unique: HashSet<&String> = results.iter().map(|(id, _)| id).collect();
        prop_assert_eq!(
            unique.len(),
            results.len(),
            "search returned duplicate IDs"
        );
    }

    // ── Property 6: similarities are in valid cosine range ──────────────

    #[test]
    fn similarities_in_valid_cosine_range(
        dim in 4usize..=32,
        n in 1usize..=30,
        k in 1usize..=10,
        ef in 10usize..=50,
        seed in any::<u64>(),
        query_seed in any::<u64>(),
    ) {
        let idx = build_random_index(dim, n, seed);
        let q = random_unit_vec(dim, query_seed);
        let results = idx.search(&q, k, ef);
        for (id, sim) in &results {
            prop_assert!(
                (-1.05..=1.05).contains(sim),
                "similarity {sim} for id {id} outside cosine range [-1.05, 1.05]"
            );
        }
    }

    // ── Property 7: results are ordered by descending similarity ────────

    #[test]
    fn results_ordered_by_descending_similarity(
        dim in 4usize..=32,
        n in 2usize..=30,
        k in 2usize..=10,
        ef in 10usize..=50,
        seed in any::<u64>(),
        query_seed in any::<u64>(),
    ) {
        let idx = build_random_index(dim, n, seed);
        let q = random_unit_vec(dim, query_seed);
        let results = idx.search(&q, k, ef);
        for window in results.windows(2) {
            let (_, s_a) = &window[0];
            let (_, s_b) = &window[1];
            prop_assert!(
                s_a >= s_b,
                "results not in descending order: {s_a} < {s_b}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Property 4: self-query
// ─────────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn self_query_returns_self_in_top_k(
        dim in 8usize..=32,
        n in 5usize..=50,
        k in 5usize..=20,
        ef in 50usize..=100,
        seed in any::<u64>(),
        target_idx in any::<u64>(),
    ) {
        let idx = HnswIndex::with_dim(dim);
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
        for i in 0..n {
            let v = random_unit_vec(dim, seed.wrapping_add(i as u64));
            let id = format!("v{i}");
            idx.insert(&id, &v);
            vectors.insert(id, v);
        }
        let target_id = format!("v{}", (target_idx as usize) % n);
        let target_vec = &vectors[&target_id];
        let results = idx.search(target_vec, k.min(n), ef);
        let returned_ids: HashSet<String> =
            results.iter().map(|(id, _)| id.clone()).collect();
        prop_assert!(
            returned_ids.contains(&target_id),
            "self-query for {target_id} did not return itself in top-{k}; got {returned_ids:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Property 5: recall vs brute-force ground truth
// ─────────────────────────────────────────────────────────────────────────

/// Brute-force k-NN cosine over `vectors` against `query`. Returns
/// the k nearest IDs by cosine similarity (highest first).
fn brute_force_knn(query: &[f32], vectors: &[(String, Vec<f32>)], k: usize) -> Vec<String> {
    let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut scored: Vec<(f32, &str)> = vectors
        .iter()
        .map(|(id, v)| {
            let v_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let sim = if q_norm > 0.0 && v_norm > 0.0 {
                dot / (q_norm * v_norm)
            } else {
                0.0
            };
            (sim, id.as_str())
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(_, id)| id.to_string())
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn recall_at_k_meets_target_on_random_corpus(
        seed in any::<u64>(),
        query_seed in any::<u64>(),
    ) {
        // Phase 2O recall contract — 16 random corpora × random queries.
        // Smaller than the 500-vector unit test (proptest cases run more
        // often), but each case rebuilds a fresh index so the validation
        // signal is independent.
        //
        // TEST-INTEGRITY: this is the SPEC contract for HNSW utility.
        // Do NOT lower below 0.85 — tune ef_search / M instead.
        let dim = 32;
        let n = 100;
        let k = 10;
        let ef = 80;
        let idx = HnswIndex::with_dim(dim);
        let mut vectors: Vec<(String, Vec<f32>)> = Vec::with_capacity(n);
        for i in 0..n {
            let v = random_unit_vec(dim, seed.wrapping_add(i as u64));
            let id = format!("v{i}");
            idx.insert(&id, &v);
            vectors.push((id, v));
        }
        let q = random_unit_vec(dim, query_seed);
        let truth: HashSet<String> =
            brute_force_knn(&q, &vectors, k).into_iter().collect();
        let approx: HashSet<String> = idx
            .search(&q, k, ef)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let hits = truth.intersection(&approx).count();
        let recall = hits as f64 / k as f64;
        prop_assert!(
            recall >= 0.85,
            "HNSW recall@{k} = {recall:.3} (target ≥ 0.85) for n={n} dim={dim} ef={ef}. \
             Tune ef_search or M; do not lower the threshold."
        );
    }
}
