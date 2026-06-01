use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use axum::http::{header, Method, Request, StatusCode};
use serde_json::{json, Value};
use tower::ServiceExt;

use gs_mem_server::app;
use gsmem::{Config, SqliteStorage};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn test_state() -> (tempfile::TempDir, Arc<Mutex<SqliteStorage>>) {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let original = std::env::var_os("IMMORTAL_GMEM_DATA_DIR");
    let temp = tempfile::tempdir().expect("tempdir");

    std::env::set_var("IMMORTAL_GMEM_DATA_DIR", temp.path());
    let config = Config::load().expect("config");
    config.ensure_local_dirs().expect("local dirs");
    let storage = SqliteStorage::open(&config.db_path).expect("open storage");
    storage.init_schema().expect("init schema");

    match original {
        Some(value) => std::env::set_var("IMMORTAL_GMEM_DATA_DIR", value),
        None => std::env::remove_var("IMMORTAL_GMEM_DATA_DIR"),
    }

    (temp, Arc::new(Mutex::new(storage)))
}

async fn json_body(response: axum::response::Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    serde_json::from_slice(&bytes).expect("json body")
}

fn json_request(method: Method, uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request")
}

#[tokio::test]
async fn health_returns_ok() {
    let (_temp, state) = test_state();
    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(json_body(response).await, json!({ "status": "ok" }));
}

#[tokio::test]
async fn ui_root_returns_html() {
    let (_temp, state) = test_state();
    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .expect("content type");
    assert!(content_type.starts_with("text/html"));

    let bytes = to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body bytes");
    let body = String::from_utf8(bytes.to_vec()).expect("html body");
    assert!(body.contains(r#"<div id="root"></div>"#));
}

#[tokio::test]
async fn put_page_then_get_page_returns_same_slug_and_body() {
    let (_temp, state) = test_state();
    let app = app(state);

    let put = app
        .clone()
        .oneshot(json_request(
            Method::PUT,
            "/api/pages/alpha",
            json!({
                "body": "Alpha body",
                "title": "Alpha"
            }),
        ))
        .await
        .expect("put response");
    assert_eq!(put.status(), StatusCode::OK);

    let get = app
        .oneshot(
            Request::builder()
                .uri("/api/pages/alpha")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("get response");
    assert_eq!(get.status(), StatusCode::OK);
    let body = json_body(get).await;
    assert_eq!(body["slug"], "alpha");
    assert_eq!(body["compiled_truth"], "Alpha body");
}

#[tokio::test]
async fn post_triple_then_find_by_subject_returns_it() {
    let (_temp, state) = test_state();
    let app = app(state);

    let post = app
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/triples",
            json!({
                "subject": "alice",
                "predicate": "knows",
                "object": "bob",
                "confidence": 0.7
            }),
        ))
        .await
        .expect("post response");
    assert_eq!(post.status(), StatusCode::OK);

    let get = app
        .oneshot(
            Request::builder()
                .uri("/api/triples?subject=alice")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("get response");
    assert_eq!(get.status(), StatusCode::OK);
    let triples = json_body(get).await;
    let triples = triples.as_array().expect("triples array");
    assert_eq!(triples.len(), 1);
    assert_eq!(triples[0]["subject"], "alice");
    assert_eq!(triples[0]["predicate"], "knows");
    assert_eq!(triples[0]["object"], "bob");
}
