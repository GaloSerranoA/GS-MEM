# GS-MEM

**Sovereign, self-hostable memory engine.** A SQLite-backed page store with a
bi-temporal knowledge graph, full-text search, an MCP server for AI agents, an
HTTP REST API, and a web UI — all in pure Rust, offline, with no external API.

> Extracted from the [NANTAR](https://github.com/serragi) framework's
> `immortal-gmem` crate into a standalone project.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**Created by Galo Serrano Abad — NANTAR AI ROBOTICS.**

## What it is

GS-MEM stores durable **pages** (compiled markdown knowledge) and **SPO triples**
(a bi-temporal knowledge graph), and exposes them three ways:

- **MCP server** (`gs-mem-mcp`) — JSON-RPC 2.0 over stdio, 7 tools. A drop-in
  memory for Claude Desktop / Claude Code / Codex.
- **HTTP API** (`gs-mem-server`) — an axum REST API over the same tools.
- **Web UI** — a React SPA (pages, search, page editor, knowledge-graph view),
  embedded directly in the server binary.

"Sovereign" means: pure-Rust inference (bundled MiniLM-L6-v2 weights for the
on-device embedder), bundled SQLite (no system dependency), no calls home.

## Features

- **7 memory tools:** `get_page`, `put_page`, `list_pages`, `search`,
  `backlinks`, `triples_add`, `triples_find`.
- **Bi-temporal SPO knowledge graph** + wiki-style `[[backlinks]]`.
- **Search:** keyword search (SQLite) is exposed via the API/MCP today; the core
  library additionally ships BM25 (Tantivy) + HNSW vector search and a sovereign
  neural embedder for advanced retrieval (not yet wired into the 7 tools).
- **One self-contained server binary** (UI embedded) plus a Docker image.

## Architecture (Cargo workspace)

| Crate | Role |
|---|---|
| `crates/gs-mem-core` (lib `gsmem`) | storage, search, graph, embedder, tool logic, optional `mcp` feature |
| `crates/gs-mem-mcp` | stdio MCP server binary |
| `crates/gs-mem-server` | axum HTTP API + embedded web UI |
| `crates/immortal-nn`, `crates/immortal-tokenizer` | vendored sovereign NN + WordPiece tokenizer |
| `frontend/` | React + Vite web UI |

## Quickstart

### Docker

```sh
docker compose -f docker/docker-compose.yml up --build
# open http://localhost:8088
```

### From source

Prerequisites: Rust 1.75+, Node 20+.

```sh
# 1. build the web UI (embedded into the server at compile time)
npm ci --prefix frontend && npm run build --prefix frontend
# 2. run the HTTP server + UI
cargo run --release -p gs-mem-server      # http://localhost:8088
```

### As an MCP server (Claude Desktop / Claude Code / Codex)

```sh
cargo build --release -p gs-mem-mcp
```

Then point your MCP client at the built binary:

```json
{
  "mcpServers": {
    "gs-mem": {
      "command": "/abs/path/to/target/release/gs-mem-mcp",
      "env": { "IMMORTAL_GMEM_DATA_DIR": "/abs/path/to/gsmem-data" }
    }
  }
}
```

## HTTP API

| Method & path | Tool |
|---|---|
| `GET /api/health` | liveness → `{"status":"ok"}` |
| `GET /api/pages?tag=&limit=` | `list_pages` |
| `GET /api/pages/{slug}` | `get_page` |
| `PUT /api/pages/{slug}` | `put_page` (body: `{ body, title?, scope?, frontmatter? }`) |
| `GET /api/search?q=&limit=` | `search` |
| `GET /api/pages/{slug}/backlinks` | `backlinks` |
| `GET /api/triples?subject=` \| `?object=` | `triples_find` |
| `POST /api/triples` | `triples_add` (body: `{ subject, predicate, object, confidence? }`) |

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `IMMORTAL_GMEM_DATA_DIR` | OS data dir `/immortal-gmem` | location of `brain.db` + the search index |
| `IMMORTAL_GMEM_ROOT` | `$OneDrive` or CWD | markdown corpus root (for ingestion) |
| `IMMORTAL_GMEM_MODEL_DIR` | path search | sovereign embedder model directory |
| `GS_MEM_PORT` | `8088` | HTTP server port |
| `RUST_LOG` | `gs_mem_server=info` | `tracing` filter |

## Development

```sh
cargo nextest run --workspace        # Rust tests
npm run dev --prefix frontend        # Vite dev server (proxies /api → :8088)
```

The core crate enforces an "enterprise doctrine" test suite (no `anyhow`/`log`,
`#[non_exhaustive]` public enums, no `unsafe`, no public `Result<_, String>`).
Please keep new core code compliant.

## License

GS-MEM is **dual-licensed** — see [LICENSING.md](LICENSING.md):

- **[AGPL-3.0-only](LICENSE)** — free & open source. Self-host, modify, and
  redistribute under the AGPL's copyleft + network-source-disclosure terms.
- **Commercial license** — for closed-source/proprietary or SaaS use without AGPL
  obligations. Contact **hello@serragi.com**.

Model weights and vendored components are attributed in [NOTICE](NOTICE).
