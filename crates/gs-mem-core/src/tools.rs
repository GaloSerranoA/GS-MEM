//! Always-compiled GS-MEM tool business logic.
//!
//! The optional MCP server owns JSON-RPC framing and the MCP content envelope.
//! This module returns the raw JSON payloads so other frontends can reuse the
//! same storage-backed operations without enabling the `mcp` feature.

use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::error::{GmemError, Result};
use crate::fingerprint::Fingerprint;
use crate::graph::link_detector::extract_links;
use crate::graph::triples::SpoTriple;
use crate::storage::{SqliteStorage, Storage};
use crate::types::{Page, PageScope, Slug};

/// Dispatch one raw GS-MEM tool by its bare name.
pub fn dispatch_tool(
    storage: &Arc<Mutex<SqliteStorage>>,
    name: &str,
    args: &Value,
) -> Result<Value> {
    match name {
        "get_page" => get_page(storage, args),
        "put_page" => put_page(storage, args),
        "list_pages" => list_pages(storage, args),
        "search" => search(storage, args),
        "backlinks" => backlinks(storage, args),
        "triples_add" => triples_add(storage, args),
        "triples_find" => triples_find(storage, args),
        _ => Err(GmemError::Mcp(format!("unknown tool: {name}"))),
    }
}

pub fn get_page(storage: &Arc<Mutex<SqliteStorage>>, args: &Value) -> Result<Value> {
    let slug = args
        .get("slug")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'slug'".into()))?;
    let slug = Slug::new(slug)?;
    let guard = lock_storage(storage)?;
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

pub fn put_page(storage: &Arc<Mutex<SqliteStorage>>, args: &Value) -> Result<Value> {
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

    let guard = lock_storage(storage)?;
    guard.put_page(&page)?;
    guard.close_links_from(&page.slug)?;
    for target in extract_links(&page.compiled_truth, page.slug.as_str()) {
        if let Ok(to_slug) = Slug::new(&target) {
            guard.upsert_link(&page.slug, &to_slug, "references")?;
        }
    }
    Ok(json!({ "ok": true, "slug": slug }))
}

pub fn list_pages(storage: &Arc<Mutex<SqliteStorage>>, args: &Value) -> Result<Value> {
    let tag = args.get("tag").and_then(Value::as_str);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .min(usize::MAX as u64) as usize;
    let guard = lock_storage(storage)?;
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

pub fn search(storage: &Arc<Mutex<SqliteStorage>>, args: &Value) -> Result<Value> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'query'".into()))?;
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(10)
        .min(usize::MAX as u64) as usize;
    let guard = lock_storage(storage)?;
    let hits = guard.search_keyword(query, limit)?;
    let out: Vec<_> = hits
        .iter()
        .map(|(slug, snippet)| {
            json!({
                "slug": slug.as_str(),
                "snippet": snippet,
            })
        })
        .collect();
    Ok(json!(out))
}

pub fn backlinks(storage: &Arc<Mutex<SqliteStorage>>, args: &Value) -> Result<Value> {
    let slug = args
        .get("slug")
        .and_then(Value::as_str)
        .ok_or_else(|| GmemError::Mcp("missing 'slug'".into()))?;
    let slug = Slug::new(slug)?;
    let guard = lock_storage(storage)?;
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

pub fn triples_add(storage: &Arc<Mutex<SqliteStorage>>, args: &Value) -> Result<Value> {
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

    let guard = lock_storage(storage)?;
    let id = guard.add_triple(&triple)?;
    Ok(json!({ "id": id }))
}

pub fn triples_find(storage: &Arc<Mutex<SqliteStorage>>, args: &Value) -> Result<Value> {
    let subject = args.get("subject").and_then(Value::as_str);
    let object = args.get("object").and_then(Value::as_str);
    let triples = match (subject, object) {
        (Some(subject), None) => {
            let guard = lock_storage(storage)?;
            guard.triples_by_subject(subject, true)?
        }
        (None, Some(object)) => {
            let guard = lock_storage(storage)?;
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
