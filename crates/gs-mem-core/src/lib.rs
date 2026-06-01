//! # immortal-gmem
//!
//! Sovereign brain primitives for NANTAR — absorbed from g-memory-rs on
//! 2026-04-18. Full master plan:
//! `plans/2026-04-18-gmem-self-contained-absorption-into-nantar` in gmem memory.
//!
//! ## Sovereignty invariant
//!
//! No external `g-memory-rs` dependency. The original gmem repo at
//! `D:\OneDrive\AI\G-Memory-rs\` continues to exist as a standalone personal
//! MCP tool for Claude Desktop / Claude Code / Codex. NANTAR owns its own
//! copy from absorption day forward.
//!
//! ## Absorption progress
//!
//! - [x] **Phase 1 — kernel** (types / error / fingerprint / config)
//! - [x] **Phase 2 — index** (chunker / search / embedding / graph)
//! - [x] **Phase 3 — storage** (storage / indexer / sync + search::backlink_hop)
//! - [x] **Phase 4 — mcp** (JSON-RPC 2.0 stdio server, feature-gated `mcp`)
//!
//! ## Public surface
//!
//! Kernel (Phase 1):
//! - [`Slug`], [`Page`], [`PageScope`], [`Chunk`], [`Tag`], [`Link`]
//! - [`Fingerprint`] — Blake2b-128 content fingerprint
//! - [`Config`] — paths (env: `IMMORTAL_GMEM_ROOT`, subdir: `%LOCALAPPDATA%\immortal-gmem\`)
//! - [`GmemError`], [`Result`] — typed error taxonomy
//!
//! Index (Phase 2):
//! - [`chunker`] — `StructuralChunker` (heading-aware markdown chunking),
//!   `simhash64`, `DialectCompressor`
//! - [`search`] — `SearchEngine` facade, `TantivyIndex` (BM25, Lucene-quality),
//!   `VectorIndex` (SQLite-backed HNSW cosine search), `QueryIntent` classifier,
//!   `reciprocal_rank_fusion`, `SearchStrategy` enum, `query_sanitizer::sanitize`,
//!   `backlink_hop` (BFS graph traversal + graph-distance rerank)
//! - [`embedding`] — `EmbeddingProvider` trait, `SovereignEmbedder`
//!   (immortal-nn + MiniLM L6 v2, 384-dim), `FakeRandomEmbedder` test double
//! - [`graph`] — `GraphStore` trait, `SpoTriple`, bi-temporal edges, link detector
//!
//! Storage (Phase 3):
//! - [`storage`] — `Storage` trait, `SqliteStorage` (rusqlite-backed brain.db),
//!   `schema` SQL DDL constants
//! - [`indexer`] — `Indexer<C,E>` orchestrator (chunker + embedder + logic_fp),
//!   `SourceVersion` (mtime + content_fp + logic_fp), `EvaluationMemory` trait +
//!   `SqliteMemo` (eval_memo SQLite cache — cocoindex skip-gate pattern)
//! - [`sync`] — `SyncScanner` for mtime-driven markdown corpus scanning
//!
//! MCP (Phase 4, feature-gated `mcp`):
//! - `mcp::serve_stdio(config)` — blocking JSON-RPC 2.0 stdio server,
//!   protocol 2024-11-05, 7 tools
//!   (gmem.get_page/put_page/list_pages/search/backlinks/triples_add/triples_find)

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod chunker;
pub mod config;
pub mod context;
pub mod embedding;
pub mod error;
pub mod fingerprint;
pub mod graph;
pub mod indexer;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod search;
pub mod storage;
pub mod sync;
pub mod tools;
pub mod types;

pub use config::Config;
pub use context::Context;
pub use error::{GmemError, Result};
pub use fingerprint::Fingerprint;
pub use types::{Chunk, Link, Page, PageScope, Slug, Tag};

pub use chunker::{Chunker, StructuralChunker};
pub use embedding::{EmbeddingProvider, FakeRandomEmbedder, SovereignEmbedder};
pub use graph::GraphStore;
pub use search::{strategy::SearchStrategy, SearchEngine};

pub use indexer::{
    memo::{EvaluationMemory, SqliteMemo},
    source_version::SourceVersion,
    Indexer,
};
pub use storage::{SqliteStorage, Storage};
pub use sync::SyncScanner;
