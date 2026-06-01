use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};

use rusqlite::{params, types::Type, Connection, OptionalExtension, Row};

use crate::error::{GmemError, Result};
use crate::fingerprint::Fingerprint;
use crate::graph::triples::SpoTriple;
use crate::indexer::source_version::SourceVersion;
use crate::storage::schema::ALL_SCHEMA;
use crate::storage::Storage;
use crate::types::{Link, Page, PageScope, Slug};
use time::OffsetDateTime;
use uuid::Uuid;

#[derive(Debug)]
pub struct SqliteStorage {
    conn: Connection,
    db_path: PathBuf,
}

impl SqliteStorage {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        <Self as Storage>::open(path)
    }

    pub fn init_schema(&self) -> Result<()> {
        <Self as Storage>::init_schema(self)
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}

impl Storage for SqliteStorage {
    fn open(path: impl AsRef<Path>) -> Result<Self> {
        let db_path = normalize_db_path(path.as_ref());
        if let Some(parent) = db_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;

        Ok(Self { conn, db_path })
    }

    fn init_schema(&self) -> Result<()> {
        for statement in ALL_SCHEMA {
            self.conn.execute(statement, [])?;
        }
        Ok(())
    }

    fn get_page(&self, _slug: &Slug) -> Result<Page> {
        let mut stmt = self.conn.prepare(
            "SELECT id, slug, scope, title, compiled_truth, timeline, frontmatter, content_fp, created_at, updated_at
             FROM pages
             WHERE slug = ?1",
        )?;

        let page = stmt
            .query_row(params![_slug.as_str()], page_from_row)
            .optional()?;

        page.ok_or_else(|| GmemError::NotFound {
            slug: _slug.as_str().into(),
        })
    }

    fn put_page(&self, page: &Page) -> Result<()> {
        let timeline = serde_json::to_string(&page.timeline)?;
        let frontmatter = serialize_frontmatter(page.frontmatter.as_ref())?;

        self.conn.execute(
            "INSERT OR REPLACE INTO pages
             (id, slug, scope, title, compiled_truth, timeline, frontmatter, content_fp, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                page.id.to_string(),
                page.slug.as_str(),
                scope_as_str(&page.scope),
                page.title.as_deref(),
                &page.compiled_truth,
                timeline,
                frontmatter,
                page.content_fp.as_bytes().as_slice(),
                timestamp_to_micros(page.created_at),
                timestamp_to_micros(page.updated_at),
            ],
        )?;

        Ok(())
    }

    fn list_pages(&self, tag: Option<&str>, limit: usize) -> Result<Vec<Page>> {
        let limit = i64::try_from(limit)
            .map_err(|err| GmemError::Other(format!("invalid limit: {err}")))?;

        let sql = "SELECT p.id, p.slug, p.scope, p.title, p.compiled_truth, p.timeline, p.frontmatter, p.content_fp, p.created_at, p.updated_at
                   FROM pages p
                   JOIN tags t ON t.page_id = p.id
                   WHERE t.tag = ?1
                   ORDER BY p.updated_at DESC
                   LIMIT ?2";
        let sql_no_tag = "SELECT id, slug, scope, title, compiled_truth, timeline, frontmatter, content_fp, created_at, updated_at
                          FROM pages
                          ORDER BY updated_at DESC
                          LIMIT ?1";

        let mut stmt = self
            .conn
            .prepare(if tag.is_some() { sql } else { sql_no_tag })?;

        let rows = match tag {
            Some(tag) => stmt.query_map(params![tag, limit], page_from_row)?,
            None => stmt.query_map(params![limit], page_from_row)?,
        };

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    fn search_keyword(&self, query: &str, limit: usize) -> Result<Vec<(Slug, String)>> {
        let limit = i64::try_from(limit)
            .map_err(|err| GmemError::Other(format!("invalid limit: {err}")))?;
        let sanitized = crate::search::query_sanitizer::sanitize(query);
        let mut stmt = self.conn.prepare(
            "SELECT slug, substr(compiled_truth, 1, 200) AS snippet
             FROM pages
             WHERE compiled_truth LIKE '%' || ?1 || '%'
                OR title LIKE '%' || ?1 || '%'
             ORDER BY updated_at DESC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![sanitized, limit], |row| {
            Ok((
                parse_slug(row.get::<_, String>(0)?, 0)?,
                row.get::<_, String>(1)?,
            ))
        })?;

        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    fn get_source_version(&self, slug: &Slug) -> Result<Option<SourceVersion>> {
        let mut stmt = self.conn.prepare(
            "SELECT mtime_us, content_fp, logic_fp
             FROM source_versions
             WHERE slug = ?1",
        )?;

        stmt.query_row(params![slug.as_str()], |row| {
            Ok(SourceVersion {
                mtime_us: row.get(0)?,
                content_fp: parse_fingerprint(row.get::<_, Vec<u8>>(1)?, 1)?,
                logic_fp: parse_fingerprint(row.get::<_, Vec<u8>>(2)?, 2)?,
            })
        })
        .optional()
        .map_err(Into::into)
    }

    fn put_source_version(
        &self,
        slug: &Slug,
        mtime_us: i64,
        content_fp: &Fingerprint,
        logic_fp: &Fingerprint,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO source_versions (slug, mtime_us, content_fp, logic_fp)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                slug.as_str(),
                mtime_us,
                content_fp.as_bytes().as_slice(),
                logic_fp.as_bytes().as_slice(),
            ],
        )?;

        Ok(())
    }

    fn upsert_link(&self, from: &Slug, to: &Slug, edge_type: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO links (from_slug, to_slug, edge_type, valid_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                from.as_str(),
                to.as_str(),
                edge_type,
                timestamp_to_micros(OffsetDateTime::now_utc()),
            ],
        )?;

        Ok(())
    }

    fn close_links_from(&self, from: &Slug) -> Result<()> {
        self.conn.execute(
            "UPDATE links
             SET invalid_at = ?1
             WHERE from_slug = ?2 AND invalid_at IS NULL",
            params![
                timestamp_to_micros(OffsetDateTime::now_utc()),
                from.as_str()
            ],
        )?;

        Ok(())
    }

    fn backlinks_to(&self, to: &Slug, only_current: bool) -> Result<Vec<Link>> {
        let sql = if only_current {
            "SELECT id, from_slug, to_slug, edge_type, valid_at, invalid_at
             FROM links
             WHERE to_slug = ?1 AND invalid_at IS NULL
             ORDER BY id ASC"
        } else {
            "SELECT id, from_slug, to_slug, edge_type, valid_at, invalid_at
             FROM links
             WHERE to_slug = ?1
             ORDER BY id ASC"
        };

        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(params![to.as_str()], link_from_row)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    fn add_triple(&self, triple: &SpoTriple) -> Result<i64> {
        let id = self.conn.query_row(
            "INSERT INTO triples
             (subject, predicate, object, valid_from, valid_to, confidence, source_slug, source_file, extracted_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             RETURNING id",
            params![
                &triple.subject,
                &triple.predicate,
                &triple.object,
                timestamp_to_micros(triple.valid_from),
                triple.valid_to.map(timestamp_to_micros),
                triple.confidence,
                triple.source_slug.as_deref(),
                triple.source_file.as_deref(),
                timestamp_to_micros(triple.extracted_at),
            ],
            |row| row.get(0),
        )?;

        Ok(id)
    }

    fn close_triple(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE triples
             SET valid_to = ?1
             WHERE id = ?2 AND valid_to IS NULL",
            params![timestamp_to_micros(OffsetDateTime::now_utc()), id],
        )?;

        Ok(())
    }

    fn triples_by_subject(&self, subject: &str, only_current: bool) -> Result<Vec<SpoTriple>> {
        let sql = if only_current {
            "SELECT id, subject, predicate, object, valid_from, valid_to, confidence, source_slug, source_file, extracted_at
             FROM triples
             WHERE subject = ?1 AND valid_to IS NULL
             ORDER BY id ASC"
        } else {
            "SELECT id, subject, predicate, object, valid_from, valid_to, confidence, source_slug, source_file, extracted_at
             FROM triples
             WHERE subject = ?1
             ORDER BY id ASC"
        };

        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(params![subject], triple_from_row)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    fn triples_by_object(&self, object: &str, only_current: bool) -> Result<Vec<SpoTriple>> {
        let sql = if only_current {
            "SELECT id, subject, predicate, object, valid_from, valid_to, confidence, source_slug, source_file, extracted_at
             FROM triples
             WHERE object = ?1 AND valid_to IS NULL
             ORDER BY id ASC"
        } else {
            "SELECT id, subject, predicate, object, valid_from, valid_to, confidence, source_slug, source_file, extracted_at
             FROM triples
             WHERE object = ?1
             ORDER BY id ASC"
        };

        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map(params![object], triple_from_row)?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    fn bfs_backlinks(&self, anchor: &Slug, max_depth: u8) -> Result<Vec<(Slug, u8)>> {
        use std::collections::{HashMap, VecDeque};
        let mut visited: HashMap<String, u8> = HashMap::new();
        visited.insert(anchor.as_str().to_string(), 0);
        let mut queue: VecDeque<(String, u8)> = VecDeque::new();
        queue.push_back((anchor.as_str().to_string(), 0));

        let mut stmt = self
            .conn
            .prepare("SELECT from_slug FROM links WHERE to_slug = ?1 AND invalid_at IS NULL")?;

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            let neighbors: Vec<String> = stmt
                .query_map(params![current], |row| row.get::<_, String>(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            for from_slug in neighbors {
                if visited.contains_key(&from_slug) {
                    continue;
                }
                visited.insert(from_slug.clone(), depth + 1);
                queue.push_back((from_slug, depth + 1));
            }
        }

        let mut out = Vec::with_capacity(visited.len());
        for (slug_str, dist) in visited {
            if dist == 0 {
                continue;
            }
            if let Ok(s) = Slug::new(&slug_str) {
                out.push((s, dist));
            }
        }
        out.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));
        Ok(out)
    }
}

fn normalize_db_path(path: &Path) -> PathBuf {
    match path.extension() {
        Some(_) => path.to_path_buf(),
        None => path.join("brain.db"),
    }
}

fn page_from_row(row: &Row<'_>) -> rusqlite::Result<Page> {
    let id = parse_uuid(row.get::<_, String>(0)?, 0)?;
    let slug = parse_slug(row.get::<_, String>(1)?, 1)?;
    let scope = parse_scope(row.get::<_, String>(2)?, 2)?;
    let title = row.get(3)?;
    let compiled_truth = row.get(4)?;
    let timeline = parse_timeline(row.get::<_, String>(5)?, 5)?;
    let frontmatter = parse_frontmatter(row.get::<_, Option<String>>(6)?, 6)?;
    let content_fp = parse_fingerprint(row.get::<_, Vec<u8>>(7)?, 7)?;
    let created_at = parse_timestamp(row.get::<_, i64>(8)?, 8)?;
    let updated_at = parse_timestamp(row.get::<_, i64>(9)?, 9)?;

    Ok(Page {
        id,
        slug,
        scope,
        title,
        compiled_truth,
        timeline,
        frontmatter,
        content_fp,
        created_at,
        updated_at,
    })
}

fn triple_from_row(row: &Row<'_>) -> rusqlite::Result<SpoTriple> {
    let valid_to = row
        .get::<_, Option<i64>>(5)?
        .map(|value| parse_timestamp(value, 5))
        .transpose()?;

    Ok(SpoTriple {
        id: row.get(0)?,
        subject: row.get(1)?,
        predicate: row.get(2)?,
        object: row.get(3)?,
        valid_from: parse_timestamp(row.get::<_, i64>(4)?, 4)?,
        valid_to,
        confidence: row.get(6)?,
        source_slug: row.get(7)?,
        source_file: row.get(8)?,
        extracted_at: parse_timestamp(row.get::<_, i64>(9)?, 9)?,
    })
}

fn link_from_row(row: &Row<'_>) -> rusqlite::Result<Link> {
    let invalid_at = row
        .get::<_, Option<i64>>(5)?
        .map(|value| parse_timestamp(value, 5))
        .transpose()?;

    Ok(Link {
        id: row.get(0)?,
        from_slug: parse_slug(row.get::<_, String>(1)?, 1)?,
        to_slug: parse_slug(row.get::<_, String>(2)?, 2)?,
        edge_type: row.get(3)?,
        valid_at: parse_timestamp(row.get::<_, i64>(4)?, 4)?,
        invalid_at,
    })
}

fn serialize_frontmatter(frontmatter: Option<&serde_json::Value>) -> Result<String> {
    serde_json::to_string(frontmatter.unwrap_or(&serde_json::Value::Null)).map_err(Into::into)
}

fn parse_uuid(value: String, col: usize) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(&value).map_err(|err| conversion_err(col, Type::Text, err))
}

fn parse_slug(value: String, col: usize) -> rusqlite::Result<Slug> {
    Slug::new(value).map_err(|err| conversion_err(col, Type::Text, err))
}

fn parse_scope(value: String, col: usize) -> rusqlite::Result<PageScope> {
    let scope = match value.as_str() {
        "project" => PageScope::Project,
        "concept" => PageScope::Concept,
        "reference" => PageScope::Reference,
        "person" => PageScope::Person,
        "plan" => PageScope::Plan,
        _ => {
            return Err(conversion_err(
                col,
                Type::Text,
                IoError::new(
                    ErrorKind::InvalidData,
                    format!("invalid page scope: {value}"),
                ),
            ))
        }
    };

    Ok(scope)
}

fn parse_timeline(value: String, col: usize) -> rusqlite::Result<Vec<String>> {
    if value.is_empty() {
        return Ok(Vec::new());
    }

    serde_json::from_str(&value).map_err(|err| conversion_err(col, Type::Text, err))
}

fn parse_frontmatter(
    value: Option<String>,
    col: usize,
) -> rusqlite::Result<Option<serde_json::Value>> {
    match value {
        Some(value) => {
            let parsed = serde_json::from_str::<serde_json::Value>(&value)
                .map_err(|err| conversion_err(col, Type::Text, err))?;
            if parsed.is_null() {
                Ok(None)
            } else {
                Ok(Some(parsed))
            }
        }
        None => Ok(None),
    }
}

fn parse_fingerprint(
    value: Vec<u8>,
    col: usize,
) -> rusqlite::Result<crate::fingerprint::Fingerprint> {
    let bytes: [u8; 16] = value.try_into().map_err(|bytes: Vec<u8>| {
        conversion_err(
            col,
            Type::Blob,
            IoError::new(
                ErrorKind::InvalidData,
                format!("invalid fingerprint length: {}", bytes.len()),
            ),
        )
    })?;

    Ok(bytes.into())
}

fn parse_timestamp(value: i64, col: usize) -> rusqlite::Result<OffsetDateTime> {
    OffsetDateTime::from_unix_timestamp_nanos(i128::from(value) * 1_000)
        .map_err(|err| conversion_err(col, Type::Integer, err))
}

fn timestamp_to_micros(value: OffsetDateTime) -> i64 {
    (value.unix_timestamp_nanos() / 1_000) as i64
}

fn scope_as_str(scope: &PageScope) -> &'static str {
    match scope {
        PageScope::Project => "project",
        PageScope::Concept => "concept",
        PageScope::Reference => "reference",
        PageScope::Person => "person",
        PageScope::Plan => "plan",
    }
}

fn conversion_err(
    col: usize,
    ty: Type,
    err: impl std::error::Error + Send + Sync + 'static,
) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(col, ty, Box::new(err))
}
