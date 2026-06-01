# Changelog

All notable changes to GS-MEM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-31

Initial standalone release — extracted from the NANTAR framework's
`immortal-gmem` crate into an independent AGPL-3.0 project.

### Added

- **`gs-mem-core`** (lib `gsmem`): SQLite-backed page store, bi-temporal SPO
  knowledge graph + backlinks, keyword search; BM25 (Tantivy) + HNSW vector
  search and a sovereign MiniLM-L6-v2 embedder in-crate; optional `mcp` feature.
- **`gs-mem-mcp`**: JSON-RPC 2.0 stdio MCP server exposing the 7 memory tools.
- **`gs-mem-server`**: axum HTTP REST API over the same tools, serving an
  embedded React web UI (pages, search, page editor, knowledge-graph view).
- Vendored `immortal-nn` + `immortal-tokenizer` (no external NANTAR dependency).
- Multi-stage Docker build + `docker compose` deployment.
- Model weights distributed via Git LFS.
