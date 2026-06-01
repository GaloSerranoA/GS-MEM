use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Path, Query, State};
use axum::http::{header, Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use rust_embed::RustEmbed;
use serde_json::{json, Map, Value};
use thiserror::Error;
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use gsmem::{Config, GmemError, SqliteStorage};

pub type AppState = Arc<Mutex<SqliteStorage>>;

const UI_NOT_BUILT_HTML: &str = r#"<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>GS-MEM</title></head>
<body><main><h1>GS-MEM UI not built</h1><p>GS-MEM UI not built &mdash; run <code>npm run build</code> in <code>frontend/</code>.</p></main></body>
</html>"#;

#[derive(RustEmbed)]
// Path is resolved relative to this crate's manifest dir (crates/gs-mem-server),
// so this points at <repo>/frontend/dist. The folder must exist at compile time
// (a committed .gitkeep guarantees that); run `npm run build` in frontend/ to
// embed the real UI.
#[folder = "../../frontend/dist"]
struct UiAssets;

#[derive(Debug, Error)]
pub enum AppError {
    #[error(transparent)]
    Gmem(#[from] GmemError),
    #[error("bad input: {0}")]
    BadInput(String),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::Gmem(GmemError::NotFound { .. }) => StatusCode::NOT_FOUND,
            Self::Gmem(GmemError::InvalidSlug(_)) | Self::Gmem(GmemError::Mcp(_)) => {
                StatusCode::BAD_REQUEST
            }
            Self::BadInput(_) => StatusCode::BAD_REQUEST,
            Self::Gmem(_) | Self::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(json!({ "error": self.to_string() }))).into_response()
    }
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/pages", get(list_pages))
        .route("/api/pages/{slug}", get(get_page).put(put_page))
        .route("/api/search", get(search))
        .route("/api/pages/{slug}/backlinks", get(backlinks))
        .route("/api/triples", get(triples_find).post(triples_add))
        .fallback(ui_fallback)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

pub fn build_state(config: &Config) -> gsmem::Result<AppState> {
    config.ensure_local_dirs()?;
    let storage = SqliteStorage::open(&config.db_path)?;
    storage.init_schema()?;
    Ok(Arc::new(Mutex::new(storage)))
}

pub async fn run() -> Result<(), AppError> {
    init_tracing();

    let config = Config::load()?;
    let state = build_state(&config)?;
    let port = port_from_env()?;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = TcpListener::bind(addr).await?;

    tracing::info!(%addr, "serving gs-mem HTTP API");
    axum::serve(listener, app(state)).await?;
    Ok(())
}

fn init_tracing() {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("gs_mem_server=info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

fn port_from_env() -> Result<u16, AppError> {
    match std::env::var("GS_MEM_PORT") {
        Ok(value) => value
            .parse::<u16>()
            .map_err(|err| AppError::BadInput(format!("invalid GS_MEM_PORT: {err}"))),
        Err(std::env::VarError::NotPresent) => Ok(8088),
        Err(err) => Err(AppError::BadInput(format!("invalid GS_MEM_PORT: {err}"))),
    }
}

async fn health() -> Result<Json<Value>, AppError> {
    Ok(Json(json!({ "status": "ok" })))
}

async fn list_pages(
    State(state): State<AppState>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Json<Value>, AppError> {
    let mut args = Map::new();
    if let Some(tag) = query.get("tag") {
        args.insert("tag".to_string(), Value::String(tag.clone()));
    }
    insert_limit(&mut args, &query)?;

    dispatch(&state, "list_pages", Value::Object(args))
}

async fn get_page(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<Json<Value>, AppError> {
    dispatch(&state, "get_page", json!({ "slug": slug }))
}

async fn put_page(
    State(state): State<AppState>,
    Path(slug): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    let mut args = object_body(body)?;
    args.insert("slug".to_string(), Value::String(slug));
    dispatch(&state, "put_page", Value::Object(args))
}

async fn search(
    State(state): State<AppState>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Json<Value>, AppError> {
    let q = query
        .get("q")
        .ok_or_else(|| AppError::BadInput("missing 'q'".to_string()))?;
    let mut args = Map::new();
    args.insert("query".to_string(), Value::String(q.clone()));
    insert_limit(&mut args, &query)?;

    dispatch(&state, "search", Value::Object(args))
}

async fn backlinks(
    State(state): State<AppState>,
    Path(slug): Path<String>,
) -> Result<Json<Value>, AppError> {
    dispatch(&state, "backlinks", json!({ "slug": slug }))
}

async fn triples_find(
    State(state): State<AppState>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Json<Value>, AppError> {
    let subject = query.get("subject");
    let object = query.get("object");
    match (subject, object) {
        (Some(subject), None) => dispatch(&state, "triples_find", json!({ "subject": subject })),
        (None, Some(object)) => dispatch(&state, "triples_find", json!({ "object": object })),
        (Some(_), Some(_)) | (None, None) => Err(AppError::BadInput(
            "provide exactly one of 'subject' or 'object'".to_string(),
        )),
    }
}

async fn triples_add(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, AppError> {
    dispatch(&state, "triples_add", body)
}

async fn ui_fallback(request: Request<Body>) -> Response {
    if request.method() != Method::GET {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }

    let path = request.uri().path().trim_start_matches('/');
    if path == "api" || path.starts_with("api/") {
        return StatusCode::NOT_FOUND.into_response();
    }

    let asset_path = if path.is_empty() { "index.html" } else { path };
    if let Some(asset) = UiAssets::get(asset_path) {
        return embedded_asset_response(asset_path, asset.data.into_owned());
    }

    if let Some(index) = UiAssets::get("index.html") {
        return embedded_asset_response("index.html", index.data.into_owned());
    }

    html_response(UI_NOT_BUILT_HTML)
}

fn embedded_asset_response(path: &str, bytes: Vec<u8>) -> Response {
    let content_type = mime_guess::from_path(path)
        .first_or_octet_stream()
        .to_string();
    ([(header::CONTENT_TYPE, content_type)], bytes).into_response()
}

fn html_response(html: &'static str) -> Response {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], html).into_response()
}

fn dispatch(state: &AppState, name: &str, args: Value) -> Result<Json<Value>, AppError> {
    Ok(Json(gsmem::tools::dispatch_tool(state, name, &args)?))
}

fn object_body(body: Value) -> Result<Map<String, Value>, AppError> {
    match body {
        Value::Object(map) => Ok(map),
        _ => Err(AppError::BadInput(
            "request body must be a JSON object".to_string(),
        )),
    }
}

fn insert_limit(
    args: &mut Map<String, Value>,
    query: &HashMap<String, String>,
) -> Result<(), AppError> {
    if let Some(limit) = query.get("limit") {
        let limit = limit
            .parse::<u64>()
            .map_err(|err| AppError::BadInput(format!("invalid limit: {err}")))?;
        args.insert("limit".to_string(), Value::from(limit));
    }
    Ok(())
}
