//! Minimal MCP (Model Context Protocol) server over stdio.
//!
//! Wire protocol: JSON-RPC 2.0, newline-delimited (one JSON message per line).
//! Protocol version: 2024-11-05.
//!
//! Hand-rolled rather than depending on `rmcp` because rmcp 1.5's macro-driven
//! API has a significant dependency surface (`schemars`, proc-macro boilerplate).
//! The MCP wire protocol is small enough that a direct impl is simpler and
//! more auditable.

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::Config;
use crate::context::Context;
use crate::error::{GmemError, Result};

pub mod tools;

const PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Deserialize)]
struct RpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i32,
    message: String,
}

/// Run an MCP server on stdin/stdout until EOF. Blocking.
pub fn serve_stdio(config: Config) -> Result<()> {
    let ctx = Context::open(&config)?;

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();

    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(());
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: RpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(err) => {
                write_response(
                    &mut writer,
                    &RpcResponse {
                        jsonrpc: "2.0",
                        id: Value::Null,
                        result: None,
                        error: Some(RpcError {
                            code: -32700,
                            message: format!("parse error: {err}"),
                        }),
                    },
                )?;
                continue;
            }
        };

        let is_notification = req.id.is_none();
        let id = req.id.clone().unwrap_or(Value::Null);
        let result = dispatch(&ctx, &req.method, &req.params);

        if is_notification {
            continue;
        }

        let resp = match result {
            Ok(value) => RpcResponse {
                jsonrpc: "2.0",
                id,
                result: Some(value),
                error: None,
            },
            Err(err) => RpcResponse {
                jsonrpc: "2.0",
                id,
                result: None,
                error: Some(RpcError {
                    code: error_code(&err),
                    message: err.to_string(),
                }),
            },
        };
        write_response(&mut writer, &resp)?;
    }
}

fn write_response(writer: &mut impl Write, resp: &RpcResponse) -> Result<()> {
    let s = serde_json::to_string(resp)?;
    writer.write_all(s.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

fn error_code(err: &GmemError) -> i32 {
    match err {
        GmemError::NotFound { .. } => -32001,
        GmemError::InvalidSlug(_) => -32602,
        _ => -32603,
    }
}

fn dispatch(ctx: &Context, method: &str, params: &Value) -> Result<Value> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "immortal-gmem",
                "version": env!("CARGO_PKG_VERSION")
            }
        })),
        "initialized" | "notifications/initialized" => Ok(Value::Null),
        "tools/list" => Ok(json!({ "tools": tools::list() })),
        "tools/call" => tools::call(ctx, params),
        "ping" => Ok(json!({})),
        _ => Err(GmemError::Mcp(format!("unknown method: {method}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_code_mapping() {
        assert_eq!(
            error_code(&GmemError::NotFound { slug: "x".into() }),
            -32001
        );
        assert_eq!(error_code(&GmemError::InvalidSlug("x".into())), -32602);
        assert_eq!(error_code(&GmemError::Other("x".into())), -32603);
    }
}
