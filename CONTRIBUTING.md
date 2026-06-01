# Contributing to GS-MEM

Thanks for your interest! GS-MEM is a Rust Cargo workspace plus a React/Vite
frontend.

## Prerequisites

- Rust 1.75+ (with `cargo-nextest` recommended: `cargo install cargo-nextest`)
- Node 20+
- Git LFS (the MiniLM model weights under `models/` are LFS objects)

## Build & test

```sh
# Rust
cargo check --workspace
cargo nextest run --workspace
cargo fmt --all --check
cargo clippy -p gs-mem-core -p gs-mem-mcp -p gs-mem-server --all-targets -- -D warnings

# Frontend
npm ci --prefix frontend
npm run build --prefix frontend     # produces frontend/dist (embedded by the server)
npm run dev --prefix frontend       # dev server, proxies /api → :8088
```

## Conventions

- **Errors:** use `thiserror`-based typed errors; never `anyhow`, `log`, or
  `env_logger`. Use `tracing` for logging.
- **`gs-mem-core` doctrine:** the `tests/enterprise_doctrine.rs` gates enforce
  `#![forbid(unsafe_code)]`, the clippy lint floor, `#[non_exhaustive]` public
  enums, and no public function returning `Result<_, String>`. Keep new core
  code compliant.
- **Tests are hermetic:** never touch the default data dir / a real `brain.db`.
  Set `IMMORTAL_GMEM_DATA_DIR` to a temp dir in tests.
- Run `cargo fmt` before committing.

## Pull requests

1. Branch from `main`.
2. Keep changes focused; add tests for new behavior.
3. Ensure `cargo nextest run --workspace` and the frontend build pass.
4. Describe the change and link any relevant issue.

## Contributor licensing

GS-MEM is dual-licensed (AGPL-3.0 **and** a commercial license — see
[LICENSING.md](LICENSING.md)). To keep that sustainable, by submitting a
contribution you agree that:

1. your contribution is provided under **AGPL-3.0-only**; **and**
2. you grant the project's copyright holder (Galo Serrano Abad) a perpetual,
   irrevocable, worldwide, royalty-free right to **also** license your contribution
   under other terms, including the **commercial license**.

You retain copyright to your contribution. If you cannot grant (2), please say so in
your pull request so it can be handled separately.
