//! GS-MEM stdio MCP server binary.
//!
//! Boots a blocking JSON-RPC 2.0 server on stdin/stdout (MCP protocol
//! `2024-11-05`) exposing the 7 GS-MEM tools: `get_page`, `put_page`,
//! `list_pages`, `search`, `backlinks`, `triples_add`, `triples_find`.
//!
//! Point an MCP client (Claude Desktop / Claude Code / Codex) at this binary as
//! the server `command`. Data locations are resolved by [`gsmem::Config::load`]
//! (see the `IMMORTAL_GMEM_*` environment variables).

fn main() -> gsmem::Result<()> {
    let config = gsmem::Config::load()?;
    gsmem::mcp::serve_stdio(config)
}
