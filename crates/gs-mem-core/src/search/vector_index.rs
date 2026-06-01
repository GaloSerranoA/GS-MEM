//! Sovereign vector index for the gmem brain.
//!
//! Phase 2A (2026-04-19): rusqlite-backed disk persistence on top of the
//! existing in-memory cosine plain-scan. Embeddings are persisted to
//! `index_dir/vectors.db` (SQLite) on every insert and lazily loaded into
//! the in-memory cache on construction so search latency stays in-memory
//! for hot reads. Crash-safe: every insert is a synchronous COMMIT.
//!
//! Phase 2M (2026-04-19): sub-O(N) approximate search via the new
//! [`super::hnsw::HnswIndex`]. Every insert now mirrors into both the
//! linear cache (for persistence + delete) and the HNSW graph (for
//! search). `search()` queries HNSW; the linear cache is the
//! crash-safety + iteration backstop. Recall on the validation corpus
//! is 1.000 vs brute-force ground truth at default parameters; see
//! `hnsw::tests::recall_at_10_meets_target_on_500_random_64dim`.

use std::path::{Path, PathBuf};
use std::sync::RwLock;

use rusqlite::{params, Connection};

use super::hnsw::HnswIndex;
use crate::error::{GmemError, Result};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS vectors (
  id        TEXT PRIMARY KEY,
  dim       INTEGER NOT NULL,
  embedding BLOB NOT NULL
);
"#;

/// On-disk + in-memory vector store with sub-O(N) cosine-similarity
/// nearest-neighbor search via HNSW (Phase 2M).
///
/// Persistence model: every `insert` is a synchronous SQLite write to
/// `index_dir/vectors.db`. The in-memory linear cache is the
/// crash-safety + iteration backstop, and the HNSW graph is the
/// search index. On construction, existing rows are loaded into both
/// the linear cache AND the HNSW graph so the first `search()` call
/// is instant.
#[derive(Debug)]
pub struct VectorIndex {
    pub dimensions: usize,
    pub index_dir: PathBuf,
    /// `(id, embedding)` pairs — write-through cache of the SQLite contents.
    /// Used for delete + iteration; not the search path.
    items: RwLock<Vec<(String, Vec<f32>)>>,
    /// HNSW proximity index — sub-O(N) approximate-nearest-neighbor
    /// search at default ef_construction=200, M=16. See
    /// [`super::hnsw::HnswIndex`].
    hnsw: HnswIndex,
    /// SQLite handle. Wrapped in a Mutex via the rusqlite `Connection`'s
    /// internal locking — only one writer at a time, which is fine for a
    /// gmem brain (single-process embedded index).
    conn: std::sync::Mutex<Connection>,
}

impl VectorIndex {
    /// Open or create an index at `index_dir`. Loads any existing rows
    /// from `index_dir/vectors.db` into the in-memory cache.
    pub fn new(index_dir: impl AsRef<Path>, dimensions: usize) -> Result<Self> {
        let dir = index_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(GmemError::Io)?;
        let db_path = dir.join("vectors.db");
        let conn = Connection::open(&db_path)?;
        // Crash-safety: WAL mode + synchronous=NORMAL gives durable commits
        // without the fsync cost of synchronous=FULL.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.execute(SCHEMA, [])?;

        // Load existing rows into the in-memory cache AND the HNSW graph.
        let mut items = Vec::new();
        let hnsw = HnswIndex::with_dim(dimensions);
        {
            let mut stmt = conn.prepare("SELECT id, dim, embedding FROM vectors")?;
            let rows = stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let dim: i64 = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                Ok((id, dim as usize, blob))
            })?;
            for row in rows {
                let (id, dim, blob) = row?;
                if dim != dimensions {
                    return Err(GmemError::Embedding(format!(
                        "stored vector for id {id} has dim {dim}, expected {dimensions}"
                    )));
                }
                let embedding = decode_embedding(&blob, dimensions)?;
                hnsw.insert(&id, &embedding);
                items.push((id, embedding));
            }
        }

        Ok(Self {
            dimensions,
            index_dir: dir,
            items: RwLock::new(items),
            hnsw,
            conn: std::sync::Mutex::new(conn),
        })
    }

    /// Number of vectors currently held in the index (in-memory cache).
    pub fn len(&self) -> usize {
        self.items.read().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert (or overwrite) the embedding for `id`. Write-through to disk.
    pub fn insert(&self, id: &str, embedding: &[f32]) -> Result<()> {
        if embedding.len() != self.dimensions {
            return Err(GmemError::Embedding(format!(
                "dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }
        let blob = encode_embedding(embedding);
        // Disk first — if it fails, the in-memory state is unchanged.
        {
            let conn = self
                .conn
                .lock()
                .map_err(|_| GmemError::Other("vectors.db lock poisoned".to_string()))?;
            conn.execute(
                "INSERT OR REPLACE INTO vectors (id, dim, embedding) VALUES (?1, ?2, ?3)",
                params![id, self.dimensions as i64, &blob],
            )?;
        }
        // Then in-memory cache.
        let mut g = self
            .items
            .write()
            .map_err(|_| GmemError::Other("vector index lock poisoned".to_string()))?;
        if let Some(existing) = g.iter_mut().find(|(eid, _)| eid == id) {
            existing.1 = embedding.to_vec();
        } else {
            g.push((id.to_string(), embedding.to_vec()));
        }
        // Phase 2M: mirror into HNSW for sub-O(N) search.
        self.hnsw.insert(id, embedding);
        Ok(())
    }

    /// Return the IDs of the `limit` nearest neighbors to `query`, ranked by
    /// cosine similarity (highest first).
    ///
    /// Phase 2M: search uses the HNSW index for sub-O(N) approximate
    /// nearest-neighbor lookup. `ef_search` is auto-set to
    /// `max(limit * 4, 50)` for default-quality recall (≥ 0.90 vs
    /// brute-force on the validation corpus).
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<String>> {
        if query.len() != self.dimensions {
            return Err(GmemError::Embedding(format!(
                "dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            )));
        }
        if limit == 0 {
            return Ok(Vec::new());
        }
        if self.hnsw.is_empty() {
            return Ok(Vec::new());
        }
        let ef_search = (limit * 4).max(50);
        // Over-fetch so we can filter out deleted items (the HNSW
        // graph is append-only; deletes tombstone in the linear cache,
        // and we filter the HNSW result against the live set).
        let hits = self.hnsw.search(query, limit * 2, ef_search);
        let g = self
            .items
            .read()
            .map_err(|_| GmemError::Other("vector index lock poisoned".to_string()))?;
        let alive: std::collections::HashSet<&str> = g.iter().map(|(id, _)| id.as_str()).collect();
        Ok(hits
            .into_iter()
            .filter(|(id, _)| alive.contains(id.as_str()))
            .map(|(id, _sim)| id)
            .take(limit)
            .collect())
    }

    /// Delete the vector with the given `id`. Idempotent: returns `Ok(())`
    /// even if no such id exists (matches DELETE semantics).
    pub fn delete(&self, id: &str) -> Result<()> {
        {
            let conn = self
                .conn
                .lock()
                .map_err(|_| GmemError::Other("vectors.db lock poisoned".to_string()))?;
            conn.execute("DELETE FROM vectors WHERE id = ?1", params![id])?;
        }
        let mut g = self
            .items
            .write()
            .map_err(|_| GmemError::Other("vector index lock poisoned".to_string()))?;
        g.retain(|(eid, _)| eid != id);
        Ok(())
    }
}

/// Encode a `Vec<f32>` as little-endian bytes for SQLite BLOB storage.
fn encode_embedding(emb: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(emb.len() * 4);
    for v in emb {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decode a SQLite BLOB into `Vec<f32>`. Errors if `bytes.len()` does not
/// equal `expected_dim * 4`.
fn decode_embedding(bytes: &[u8], expected_dim: usize) -> Result<Vec<f32>> {
    let expected_bytes = expected_dim * 4;
    if bytes.len() != expected_bytes {
        return Err(GmemError::Embedding(format!(
            "stored embedding has {} bytes, expected {} ({}*4)",
            bytes.len(),
            expected_bytes,
            expected_dim
        )));
    }
    let mut out = Vec::with_capacity(expected_dim);
    for chunk in bytes.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

// Phase 2M: cosine similarity helpers retired — search is now driven by
// the HNSW graph in `super::hnsw`, which holds its own distance fn.

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn idx_in(tmp: &TempDir, dim: usize) -> VectorIndex {
        VectorIndex::new(tmp.path(), dim).expect("open vector index")
    }

    #[test]
    fn empty_index_returns_empty_results() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 3);
        assert!(v.is_empty());
        let r = v.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn insert_then_search_retrieves_nearest() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 3);
        v.insert("a", &[1.0, 0.0, 0.0]).unwrap();
        v.insert("b", &[0.0, 1.0, 0.0]).unwrap();
        v.insert("c", &[0.0, 0.0, 1.0]).unwrap();
        let nearest = v.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(nearest, vec!["a"]);
        let top2 = v.search(&[0.99, 0.1, 0.0], 2).unwrap();
        assert_eq!(top2[0], "a");
        assert_eq!(top2[1], "b");
    }

    #[test]
    fn dimension_mismatch_on_insert_errors() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 3);
        let err = v.insert("x", &[1.0, 0.0]).unwrap_err();
        assert!(matches!(err, GmemError::Embedding(_)));
    }

    #[test]
    fn dimension_mismatch_on_search_errors() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 3);
        v.insert("a", &[1.0, 0.0, 0.0]).unwrap();
        let err = v.search(&[1.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, GmemError::Embedding(_)));
    }

    #[test]
    fn insert_overwrites_existing_id() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 3);
        v.insert("x", &[1.0, 0.0, 0.0]).unwrap();
        v.insert("x", &[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(v.len(), 1);
        let r = v.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(r, vec!["x"]);
    }

    #[test]
    fn search_with_limit_zero_returns_empty() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 3);
        v.insert("a", &[1.0, 0.0, 0.0]).unwrap();
        assert!(v.search(&[1.0, 0.0, 0.0], 0).unwrap().is_empty());
    }

    #[test]
    fn search_handles_zero_query_gracefully() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 3);
        v.insert("a", &[1.0, 0.0, 0.0]).unwrap();
        let r = v.search(&[0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(r, vec!["a"]);
    }

    #[test]
    fn search_ranks_by_cosine_not_dot_product() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 3);
        v.insert("small", &[1.0, 0.0, 0.0]).unwrap();
        v.insert("large", &[10.0, 0.0, 0.0]).unwrap();
        let r = v.search(&[5.0, 0.0, 0.0], 2).unwrap();
        assert!(r.contains(&"small".to_string()));
        assert!(r.contains(&"large".to_string()));
    }

    // ── Phase 2A: persistence tests ──────────────────────────────────────

    #[test]
    fn persistence_roundtrip_across_open_close() {
        let tmp = TempDir::new().unwrap();
        // Open, insert, drop.
        {
            let v = idx_in(&tmp, 4);
            v.insert("alpha", &[1.0, 0.0, 0.0, 0.0]).unwrap();
            v.insert("beta", &[0.0, 1.0, 0.0, 0.0]).unwrap();
            v.insert("gamma", &[0.0, 0.0, 1.0, 0.0]).unwrap();
            assert_eq!(v.len(), 3);
        }
        // Reopen — should see the same vectors.
        let v2 = idx_in(&tmp, 4);
        assert_eq!(v2.len(), 3);
        let nearest = v2.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(nearest, vec!["alpha"]);
        let nearest2 = v2.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(nearest2, vec!["beta"]);
    }

    #[test]
    fn persistence_overwrite_persists_across_reopen() {
        let tmp = TempDir::new().unwrap();
        {
            let v = idx_in(&tmp, 2);
            v.insert("k", &[1.0, 0.0]).unwrap();
            v.insert("k", &[0.0, 1.0]).unwrap(); // overwrite
        }
        let v2 = idx_in(&tmp, 2);
        assert_eq!(v2.len(), 1);
        let r = v2.search(&[0.0, 1.0], 1).unwrap();
        assert_eq!(r, vec!["k"]);
    }

    #[test]
    fn persistence_dim_mismatch_on_reopen_errors() {
        let tmp = TempDir::new().unwrap();
        {
            let v = idx_in(&tmp, 4);
            v.insert("x", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        }
        // Reopen with wrong dimension — should refuse to load.
        let err = VectorIndex::new(tmp.path(), 3).unwrap_err();
        assert!(matches!(err, GmemError::Embedding(_)));
    }

    #[test]
    fn delete_removes_from_disk_and_memory() {
        let tmp = TempDir::new().unwrap();
        {
            let v = idx_in(&tmp, 3);
            v.insert("a", &[1.0, 0.0, 0.0]).unwrap();
            v.insert("b", &[0.0, 1.0, 0.0]).unwrap();
            v.delete("a").unwrap();
            assert_eq!(v.len(), 1);
        }
        let v2 = idx_in(&tmp, 3);
        assert_eq!(v2.len(), 1);
        let r = v2.search(&[1.0, 0.0, 0.0], 1).unwrap();
        // "a" is gone; "b" wins by default (only candidate).
        assert_eq!(r, vec!["b"]);
    }

    #[test]
    fn delete_nonexistent_id_is_noop() {
        let tmp = TempDir::new().unwrap();
        let v = idx_in(&tmp, 2);
        v.insert("a", &[1.0, 0.0]).unwrap();
        v.delete("nonexistent").unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let original = vec![1.5_f32, -2.7, 0.0, f32::INFINITY, f32::NEG_INFINITY];
        let bytes = encode_embedding(&original);
        let decoded = decode_embedding(&bytes, original.len()).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn decode_rejects_wrong_byte_count() {
        let bytes = vec![0u8; 13]; // not a multiple of 4
        let err = decode_embedding(&bytes, 4).unwrap_err();
        assert!(matches!(err, GmemError::Embedding(_)));
    }
}
