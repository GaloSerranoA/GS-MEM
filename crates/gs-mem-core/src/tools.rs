//! Always-compiled GS-MEM tool business logic.
//!
//! The optional MCP server owns JSON-RPC framing and the MCP content envelope.
//! This module returns the raw JSON payloads so other frontends can reuse the
//! same storage-backed operations without enabling the `mcp` feature.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::context::Context;
use crate::error::{GmemError, Result};
use crate::fingerprint::Fingerprint;
use crate::graph::link_detector::extract_links;
use crate::graph::triples::SpoTriple;
use crate::search::vector_index::VectorIndex;
use crate::search::SearchBackend;
use crate::storage::{SqliteStorage, Storage};
use crate::types::{Page, PageScope, Slug};

/// Dispatch one raw GS-MEM tool by its bare name.
pub fn dispatch_tool(ctx: &Context, name: &str, args: &Value) -> Result<Value> {
    match name {
        "get_page" => get_page(ctx, args),
        "put_page" => put_page(ctx, args),
        "list_pages" => list_pages(ctx, args),
        "search" => search(ctx, args),
        "backlinks" => backlinks(ctx, args),
        "triples_add" => triples_add(ctx, args),
        "triples_find" => triples_find(ctx, args),
        _ => Err(GmemError::Mcp(format!("unknown tool: {name}"))),
    }
}

pub fn get_page(ctx: &Context, args: &Value) -> Result<Value> {
    let slug = args
        .get("slug")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'slug'".into()))?;
    let slug = Slug::new(slug)?;
    let guard = lock_storage(ctx.storage())?;
    let page = guard.get_page(&slug)?;
    Ok(json!({
        "slug": page.slug.as_str(),
        "scope": scope_to_str(&page.scope),
        "title": page.title,
        "compiled_truth": page.compiled_truth,
        "timeline": page.timeline,
        "frontmatter": page.frontmatter,
    }))
}

pub fn put_page(ctx: &Context, args: &Value) -> Result<Value> {
    let slug = args
        .get("slug")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'slug'".into()))?;
    let body = args
        .get("body")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'body'".into()))?;
    let title = args.get("title").and_then(Value::as_str).map(str::to_owned);
    let scope_str = args
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or("concept");
    let scope = match scope_str {
        "project" => PageScope::Project,
        "concept" => PageScope::Concept,
        "reference" => PageScope::Reference,
        "person" => PageScope::Person,
        "plan" => PageScope::Plan,
        other => return Err(GmemError::Mcp(format!("invalid scope: {other}"))),
    };
    let frontmatter = args.get("frontmatter").cloned().filter(|v| !v.is_null());

    let now = OffsetDateTime::now_utc();
    let body_owned = body.to_string();
    let page = Page {
        id: Uuid::new_v4(),
        slug: Slug::new(slug)?,
        scope,
        title,
        compiled_truth: body_owned.clone(),
        timeline: vec![],
        frontmatter,
        content_fp: Fingerprint::of(&body_owned),
        created_at: now,
        updated_at: now,
    };

    {
        let guard = lock_storage(ctx.storage())?;
        guard.put_page(&page)?;
        guard.close_links_from(&page.slug)?;
        for target in extract_links(&page.compiled_truth, page.slug.as_str()) {
            if let Ok(to_slug) = Slug::new(&target) {
                guard.upsert_link(&page.slug, &to_slug, "references")?;
            }
        }
    }

    if let (Some(embedder), Some(vectors)) = (ctx.embedder(), ctx.vectors()) {
        let emb = embedder.embed(&page.compiled_truth)?;
        vectors.insert(page.slug.as_str(), &emb)?;
    }

    Ok(json!({ "ok": true, "slug": slug }))
}

pub fn list_pages(ctx: &Context, args: &Value) -> Result<Value> {
    let tag = args.get("tag").and_then(Value::as_str);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .min(usize::MAX as u64) as usize;
    let guard = lock_storage(ctx.storage())?;
    let pages = guard.list_pages(tag, limit)?;
    let out: Vec<_> = pages
        .iter()
        .map(|p| {
            json!({
                "slug": p.slug.as_str(),
                "scope": scope_to_str(&p.scope),
                "title": p.title,
            })
        })
        .collect();
    Ok(json!(out))
}

pub fn search(ctx: &Context, args: &Value) -> Result<Value> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'query'".into()))?;
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(10)
        .min(usize::MAX as u64) as usize;
    if limit == 0 || query.trim().is_empty() {
        return Ok(json!([]));
    }

    let query_embedding = match ctx.embedder() {
        Some(embedder) => embedder.embed(query)?,
        None => Vec::new(),
    };
    let backend = ContextBackend {
        storage: ctx.storage(),
        vectors: ctx.vectors().map(|vectors| vectors.as_ref()),
    };
    let slugs = ctx
        .engine()
        .search(&backend, query, &query_embedding, limit)?;

    let guard = lock_storage(ctx.storage())?;
    let mut out = Vec::with_capacity(slugs.len());
    for slug in slugs {
        let slug = Slug::new(&slug)?;
        let page = guard.get_page(&slug)?;
        out.push(json!({
            "slug": page.slug.as_str(),
            "snippet": page_snippet(&page.compiled_truth),
        }));
    }

    Ok(json!(out))
}

pub fn backlinks(ctx: &Context, args: &Value) -> Result<Value> {
    let slug = args
        .get("slug")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'slug'".into()))?;
    let slug = Slug::new(slug)?;
    let guard = lock_storage(ctx.storage())?;
    let links = guard.backlinks_to(&slug, true)?;
    let out: Vec<_> = links
        .iter()
        .map(|link| {
            json!({
                "from_slug": link.from_slug.as_str(),
                "edge_type": link.edge_type,
            })
        })
        .collect();
    Ok(json!(out))
}

pub fn triples_add(ctx: &Context, args: &Value) -> Result<Value> {
    let subject = args
        .get("subject")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'subject'".into()))?;
    let predicate = args
        .get("predicate")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'predicate'".into()))?;
    let object = args
        .get("object")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'object'".into()))?;

    let mut triple = SpoTriple::new(subject, predicate, object);
    if let Some(confidence) = args.get("confidence").and_then(Value::as_f64) {
        triple.confidence = confidence as f32;
    }

    let guard = lock_storage(ctx.storage())?;
    let id = guard.add_triple(&triple)?;
    Ok(json!({ "id": id }))
}

pub fn triples_find(ctx: &Context, args: &Value) -> Result<Value> {
    let subject = args.get("subject").and_then(Value::as_str);
    let object = args.get("object").and_then(Value::as_str);
    let triples = match (subject, object) {
        (Some(subject), None) => {
            let guard = lock_storage(ctx.storage())?;
            guard.triples_by_subject(subject, true)?
        }
        (None, Some(object)) => {
            let guard = lock_storage(ctx.storage())?;
            guard.triples_by_object(object, true)?
        }
        (Some(_), Some(_)) | (None, None) => {
            return Err(GmemError::Mcp(
                "provide exactly one of 'subject' or 'object'".into(),
            ))
        }
    };

    let out: Vec<_> = triples
        .iter()
        .map(|triple| {
            json!({
                "id": triple.id,
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "confidence": triple.confidence,
                "is_current": triple.is_current(),
            })
        })
        .collect();
    Ok(json!(out))
}

struct ContextBackend<'a> {
    storage: &'a Arc<Mutex<SqliteStorage>>,
    vectors: Option<&'a VectorIndex>,
}

impl SearchBackend for ContextBackend<'_> {
    fn full_text(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let guard = lock_storage(self.storage)?;
        let hits = guard.search_keyword(query, limit)?;
        Ok(hits
            .into_iter()
            .map(|(slug, _snippet)| slug.as_str().to_string())
            .collect())
    }

    fn vector(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<String>> {
        match self.vectors {
            Some(vectors) => vectors.search(query_embedding, limit),
            None => Ok(Vec::new()),
        }
    }
}

fn lock_storage(
    storage: &Arc<Mutex<SqliteStorage>>,
) -> Result<std::sync::MutexGuard<'_, SqliteStorage>> {
    storage
        .lock()
        .map_err(|_| GmemError::Mcp("storage mutex poisoned".into()))
}

fn scope_to_str(scope: &PageScope) -> &'static str {
    match scope {
        PageScope::Project => "project",
        PageScope::Concept => "concept",
        PageScope::Reference => "reference",
        PageScope::Person => "person",
        PageScope::Plan => "plan",
    }
}

fn page_snippet(compiled_truth: &str) -> String {
    compiled_truth
        .chars()
        .take(200)
        .collect::<String>()
        .trim()
        .to_string()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use crate::context::Context;
    use crate::embedding::{EmbeddingProvider, FakeRandomEmbedder};
    use crate::search::vector_index::VectorIndex;

    fn temp_storage() -> (tempfile::TempDir, Arc<Mutex<SqliteStorage>>) {
        let temp = tempfile::tempdir().expect("tempdir");
        let storage = SqliteStorage::open(temp.path()).expect("open storage");
        storage.init_schema().expect("init schema");
        (temp, Arc::new(Mutex::new(storage)))
    }

    #[test]
    fn put_page_indexes_embedding_and_search_can_return_vector_hit() {
        let (_storage_temp, storage) = temp_storage();
        let index_temp = tempfile::tempdir().expect("vector tempdir");
        let vectors = Arc::new(VectorIndex::new(index_temp.path(), 16).expect("vector index"));
        let embedder: Arc<dyn EmbeddingProvider + Send + Sync> = Arc::new(FakeRandomEmbedder);
        let ctx = Context::with_embedder(storage, embedder, Arc::clone(&vectors));

        put_page(
            &ctx,
            &json!({
                "slug": "vector-only",
                "body": "vector::only::alpha"
            }),
        )
        .expect("put page");
        assert_eq!(vectors.len(), 1);

        let hits = search(
            &ctx,
            &json!({
                "query": "vector::only::alpha",
                "limit": 1
            }),
        )
        .expect("search");

        let hits = hits.as_array().expect("hits array");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["slug"], "vector-only");
        assert_eq!(hits[0]["snippet"], "vector::only::alpha");
    }

    #[test]
    fn search_without_embedder_degrades_to_keyword_hits() {
        let (_storage_temp, storage) = temp_storage();
        let ctx = Context::storage_only(storage);

        put_page(
            &ctx,
            &json!({
                "slug": "keyword-only",
                "body": "keyword degradation still works"
            }),
        )
        .expect("put page");

        let hits = search(
            &ctx,
            &json!({
                "query": "degradation",
                "limit": 5
            }),
        )
        .expect("search");

        let hits = hits.as_array().expect("hits array");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0]["slug"], "keyword-only");
        assert_eq!(hits[0]["snippet"], "keyword degradation still works");
    }
}
