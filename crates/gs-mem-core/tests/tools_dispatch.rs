use std::sync::{Arc, Mutex};

use serde_json::json;

use gsmem::storage::SqliteStorage;
use gsmem::Context;

#[test]
fn dispatch_tool_returns_raw_page_json_without_mcp_envelope() {
    let temp = tempfile::tempdir().expect("tempdir");
    let storage = SqliteStorage::open(temp.path()).expect("open storage");
    storage.init_schema().expect("init schema");
    let storage = Arc::new(Mutex::new(storage));
    let ctx = Context::storage_only(storage);

    let put = gsmem::tools::dispatch_tool(
        &ctx,
        "put_page",
        &json!({
            "slug": "alpha",
            "body": "See [[beta]]",
            "title": "Alpha"
        }),
    )
    .expect("put page");
    assert_eq!(put, json!({ "ok": true, "slug": "alpha" }));

    let page = gsmem::tools::dispatch_tool(&ctx, "get_page", &json!({ "slug": "alpha" }))
        .expect("get page");
    assert_eq!(page["slug"], "alpha");
    assert_eq!(page["title"], "Alpha");
    assert_eq!(page["compiled_truth"], "See [[beta]]");
    assert!(page.get("content").is_none());
}
