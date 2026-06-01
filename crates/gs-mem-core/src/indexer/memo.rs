//! Evaluation-memo cache backed by the `eval_memo` SQLite table.
//!
//! Used by the indexer to short-circuit re-embedding of unchanged content:
//! `put` stores `(content_fp, kind, payload)` after a costly computation;
//! `get` returns the cached payload when the same content_fp is seen
//! again. This is the cocoindex-style skip-gate referenced by the indexer.
//!
//! Schema (from `crate::storage::schema::EVAL_MEMO_TABLE`):
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS eval_memo (
//!   content_fp BLOB PRIMARY KEY,
//!   kind       TEXT NOT NULL,
//!   payload    BLOB NOT NULL,
//!   created_at INTEGER NOT NULL
//! );
//! ```

use rusqlite::{params, Connection, OptionalExtension};

use crate::error::Result;
use crate::fingerprint::Fingerprint;

pub trait EvaluationMemory {
    fn get(&self, content_fp: &Fingerprint) -> Result<Option<Vec<u8>>>;
    fn put(&self, content_fp: &Fingerprint, kind: &str, payload: &[u8]) -> Result<()>;
}

#[derive(Debug)]
pub struct SqliteMemo<'a> {
    conn: &'a Connection,
}

impl<'a> SqliteMemo<'a> {
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }
}

impl EvaluationMemory for SqliteMemo<'_> {
    fn get(&self, content_fp: &Fingerprint) -> Result<Option<Vec<u8>>> {
        let row = self
            .conn
            .query_row(
                "SELECT payload FROM eval_memo WHERE content_fp = ?1",
                params![content_fp.as_bytes()],
                |r| r.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        Ok(row)
    }

    fn put(&self, content_fp: &Fingerprint, kind: &str, payload: &[u8]) -> Result<()> {
        let now_secs: i64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        // INSERT OR REPLACE so re-puts with the same fp are idempotent.
        self.conn.execute(
            "INSERT OR REPLACE INTO eval_memo (content_fp, kind, payload, created_at) \
                 VALUES (?1, ?2, ?3, ?4)",
            params![content_fp.as_bytes(), kind, payload, now_secs],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::schema::EVAL_MEMO_TABLE;

    fn fresh_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("open in-memory");
        conn.execute(EVAL_MEMO_TABLE, []).expect("create eval_memo");
        conn
    }

    #[test]
    fn put_then_get_roundtrip() {
        let conn = fresh_conn();
        let memo = SqliteMemo::new(&conn);
        let fp = Fingerprint::of(&"hello world");
        let payload = b"cached-embedding-bytes";
        memo.put(&fp, "embedding-v1", payload).unwrap();
        let got = memo.get(&fp).unwrap();
        assert_eq!(got.as_deref(), Some(payload.as_slice()));
    }

    #[test]
    fn get_on_miss_returns_none() {
        let conn = fresh_conn();
        let memo = SqliteMemo::new(&conn);
        let fp = Fingerprint::of(&"never-stored");
        let got = memo.get(&fp).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn put_is_idempotent_on_same_fingerprint() {
        let conn = fresh_conn();
        let memo = SqliteMemo::new(&conn);
        let fp = Fingerprint::of(&"key");
        memo.put(&fp, "v1", b"first").unwrap();
        // Replacement updates the payload.
        memo.put(&fp, "v1", b"second").unwrap();
        let got = memo.get(&fp).unwrap();
        assert_eq!(got.as_deref(), Some(b"second".as_slice()));
        // And only one row remains.
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM eval_memo", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn different_fingerprints_yield_different_rows() {
        let conn = fresh_conn();
        let memo = SqliteMemo::new(&conn);
        let fp_a = Fingerprint::of(&"alpha");
        let fp_b = Fingerprint::of(&"beta");
        memo.put(&fp_a, "v", b"A").unwrap();
        memo.put(&fp_b, "v", b"B").unwrap();
        assert_eq!(memo.get(&fp_a).unwrap().as_deref(), Some(b"A".as_slice()));
        assert_eq!(memo.get(&fp_b).unwrap().as_deref(), Some(b"B".as_slice()));
    }
}
