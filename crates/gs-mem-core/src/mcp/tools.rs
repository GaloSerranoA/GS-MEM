//! MCP tool handlers for immortal-gmem.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::error::{GmemError, Result};
use crate::storage::SqliteStorage;

/// Return the list of MCP tool descriptors (for `tools/list`).
pub fn list() -> Value {
    json!([
        {
            "name": "gmem.get_page",
            "description": "Fetch a page by slug. Returns compiled_truth + timeline + metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "slug": {
                        "type": "string",
                        "description": "Page slug (e.g. 'projects/foo/bar')"
                    }
                },
                "required": ["slug"]
            }
        },
        {
            "name": "gmem.put_page",
            "description": "Upsert a page. Body is the compiled-truth content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "slug": { "type": "string" },
                    "body": { "type": "string" },
                    "title": { "type": "string" },
                    "scope": {
                        "type": "string",
                        "enum": ["project", "concept", "reference", "person", "plan"]
                    },
                    "frontmatter": { "type": "object" }
                },
                "required": ["slug", "body"]
            }
        },
        {
            "name": "gmem.list_pages",
            "description": "List pages, optionally filtered by tag.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tag": { "type": "string" },
                    "limit": { "type": "integer", "default": 20 }
                }
            }
        },
        {
            "name": "gmem.search",
            "description": "Keyword search (SQL LIKE) over title + compiled_truth.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "default": 10 }
                },
                "required": ["query"]
            }
        },
        {
            "name": "gmem.backlinks",
            "description": "List current backlinks to a slug.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "slug": { "type": "string" }
                },
                "required": ["slug"]
            }
        },
        {
            "name": "gmem.triples_add",
            "description": "Add a new SPO triple.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string" },
                    "predicate": { "type": "string" },
                    "object": { "type": "string" },
                    "confidence": { "type": "number" }
                },
                "required": ["subject", "predicate", "object"]
            }
        },
        {
            "name": "gmem.triples_find",
            "description": "Find triples by exact subject or exact object.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": { "type": "string" },
                    "object": { "type": "string" }
                }
            }
        }
    ])
}

/// Dispatch a `tools/call` request.
pub fn call(storage: &Arc<Mutex<SqliteStorage>>, params: &Value) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'name' in tools/call params".into()))?;
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));

    let raw = match name {
        "gmem.get_page" => crate::tools::dispatch_tool(storage, "get_page", &args)?,
        "gmem.put_page" => crate::tools::dispatch_tool(storage, "put_page", &args)?,
        "gmem.list_pages" => crate::tools::dispatch_tool(storage, "list_pages", &args)?,
        "gmem.search" => crate::tools::dispatch_tool(storage, "search", &args)?,
        "gmem.backlinks" => crate::tools::dispatch_tool(storage, "backlinks", &args)?,
        "gmem.triples_add" => crate::tools::dispatch_tool(storage, "triples_add", &args)?,
        "gmem.triples_find" => crate::tools::dispatch_tool(storage, "triples_find", &args)?,
        _ => return Err(GmemError::Mcp(format!("unknown tool: {name}"))),
    };
    let text = serde_json::to_string(&raw)?;

    Ok(json!({
        "content": [{ "type": "text", "text": text }]
    }))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn tools_list_returns_seven() {
        let list = list();
        let arr = list.as_array().expect("tools must be array");
        assert_eq!(arr.len(), 7);
        let names: Vec<_> = arr
            .iter()
            .map(|t| t["name"].as_str().unwrap_or(""))
            .collect();
        assert!(names.contains(&"gmem.get_page"));
        assert!(names.contains(&"gmem.put_page"));
        assert!(names.contains(&"gmem.list_pages"));
        assert!(names.contains(&"gmem.search"));
        assert!(names.contains(&"gmem.backlinks"));
        assert!(names.contains(&"gmem.triples_add"));
        assert!(names.contains(&"gmem.triples_find"));
    }

    #[test]
    fn triples_and_backlinks_tools_roundtrip() {
        let temp = tempfile::tempdir().expect("tempdir");
        let storage = SqliteStorage::open(temp.path()).expect("open storage");
        storage.init_schema().expect("init schema");
        let storage = Arc::new(Mutex::new(storage));

        let put_resp = call(
            &storage,
            &json!({
                "name": "gmem.put_page",
                "arguments": {
                    "slug": "alpha",
                    "body": "See [[beta]]"
                }
            }),
        )
        .expect("put page");
        assert_eq!(put_resp["content"][0]["type"], "text");

        let backlinks_resp = call(
            &storage,
            &json!({
                "name": "gmem.backlinks",
                "arguments": {
                    "slug": "beta"
                }
            }),
        )
        .expect("backlinks");
        let backlinks_text = backlinks_resp["content"][0]["text"]
            .as_str()
            .expect("backlinks text");
        let backlinks_json: serde_json::Value =
            serde_json::from_str(backlinks_text).expect("parse backlinks");
        assert_eq!(backlinks_json.as_array().expect("array").len(), 1);
        assert_eq!(backlinks_json[0]["from_slug"], "alpha");

        let add_resp = call(
            &storage,
            &json!({
                "name": "gmem.triples_add",
                "arguments": {
                    "subject": "alice",
                    "predicate": "knows",
                    "object": "bob",
                    "confidence": 0.7
                }
            }),
        )
        .expect("triples add");
        let add_text = add_resp["content"][0]["text"].as_str().expect("add text");
        let add_json: serde_json::Value = serde_json::from_str(add_text).expect("parse add");
        assert!(add_json["id"].as_i64().unwrap_or_default() > 0);

        let find_resp = call(
            &storage,
            &json!({
                "name": "gmem.triples_find",
                "arguments": {
                    "subject": "alice"
                }
            }),
        )
        .expect("triples find");
        let find_text = find_resp["content"][0]["text"].as_str().expect("find text");
        let find_json: serde_json::Value = serde_json::from_str(find_text).expect("parse find");
        assert_eq!(find_json.as_array().expect("array").len(), 1);
        assert_eq!(find_json[0]["predicate"], "knows");
    }
}
