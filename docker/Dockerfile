# syntax=docker/dockerfile:1
#
# GS-MEM — multi-stage build.
#   1. node   : build the React UI -> frontend/dist
#   2. rust   : build gs-mem-server (rust-embed embeds frontend/dist AT COMPILE TIME)
#   3. runtime: slim Debian with just the static-ish binary
#
# Build from the repo root:  docker build -f docker/Dockerfile -t gs-mem .
# Or via compose:            docker compose -f docker/docker-compose.yml up --build

# --- Stage 1: build the React UI ---
FROM node:22-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: build the Rust server (embeds the built UI) ---
FROM rust:1-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
# rust-embed reads ../../frontend/dist relative to the server crate at COMPILE time,
# so the built UI must be present before `cargo build`.
COPY --from=frontend /app/frontend/dist ./frontend/dist
RUN cargo build --release -p gs-mem-server

# --- Stage 3: slim runtime ---
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/gs-mem-server /usr/local/bin/gs-mem-server

# Container-friendly defaults. /data holds brain.db + the tantivy index (mount a volume).
# /models/sovereign holds the MiniLM weights IF you enable the sovereign embedder
# (the current 7 tools use SQL keyword search and do not require the model).
ENV IMMORTAL_GMEM_DATA_DIR=/data \
    IMMORTAL_GMEM_MODEL_DIR=/models/sovereign \
    GS_MEM_PORT=8088 \
    RUST_LOG=gs_mem_server=info
EXPOSE 8088
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8088/api/health || exit 1

CMD ["gs-mem-server"]
