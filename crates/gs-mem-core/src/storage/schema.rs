pub const PAGES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS pages (
  id           TEXT PRIMARY KEY,
  slug         TEXT NOT NULL UNIQUE,
  scope        TEXT NOT NULL,
  title        TEXT,
  compiled_truth TEXT NOT NULL DEFAULT '',
  timeline     TEXT NOT NULL DEFAULT '',
  frontmatter  TEXT,
  content_fp   BLOB NOT NULL,
  created_at   INTEGER NOT NULL,
  updated_at   INTEGER NOT NULL
);
"#;

pub const CHUNKS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS chunks (
  id           TEXT PRIMARY KEY,
  page_id      TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  ord          INTEGER NOT NULL,
  body         TEXT NOT NULL,
  content_fp   BLOB NOT NULL,
  simhash      INTEGER NOT NULL,
  embedding    BLOB
);
"#;

pub const CHUNKS_PAGE_IDX: &str = r#"
CREATE INDEX IF NOT EXISTS chunks_page_idx ON chunks(page_id);
"#;

pub const CHUNKS_SIMHASH_IDX: &str = r#"
CREATE INDEX IF NOT EXISTS chunks_simhash_idx ON chunks(simhash);
"#;

pub const TAGS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS tags (
  page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  tag     TEXT NOT NULL,
  PRIMARY KEY (page_id, tag)
);
"#;

pub const TAGS_TAG_IDX: &str = r#"
CREATE INDEX IF NOT EXISTS tags_tag_idx ON tags(tag);
"#;

pub const LINKS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS links (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  from_slug   TEXT NOT NULL,
  to_slug     TEXT NOT NULL,
  edge_type   TEXT NOT NULL,
  valid_at    INTEGER NOT NULL,
  invalid_at  INTEGER NULL
);
"#;

pub const LINKS_FROM_IDX: &str = r#"
CREATE INDEX IF NOT EXISTS links_from_idx ON links(from_slug) WHERE invalid_at IS NULL;
"#;

pub const LINKS_TO_IDX: &str = r#"
CREATE INDEX IF NOT EXISTS links_to_idx   ON links(to_slug)   WHERE invalid_at IS NULL;
"#;

pub const SOURCE_VERSIONS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS source_versions (
  slug       TEXT PRIMARY KEY,
  mtime_us   INTEGER NOT NULL,
  content_fp BLOB NOT NULL,
  logic_fp   BLOB NOT NULL
);
"#;

pub const EVAL_MEMO_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS eval_memo (
  content_fp BLOB PRIMARY KEY,
  kind       TEXT NOT NULL,
  payload    BLOB NOT NULL,
  created_at INTEGER NOT NULL
);
"#;

pub const TRIPLES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS triples (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  subject        TEXT NOT NULL,
  predicate      TEXT NOT NULL,
  object         TEXT NOT NULL,
  valid_from     INTEGER NOT NULL,
  valid_to       INTEGER,
  confidence     REAL NOT NULL DEFAULT 1.0,
  source_slug    TEXT,
  source_file    TEXT,
  extracted_at   INTEGER NOT NULL
);
"#;

pub const TRIPLES_SUBJECT_IDX: &str =
    "CREATE INDEX IF NOT EXISTS triples_subject_idx ON triples(subject) WHERE valid_to IS NULL;";
pub const TRIPLES_OBJECT_IDX: &str =
    "CREATE INDEX IF NOT EXISTS triples_object_idx ON triples(object) WHERE valid_to IS NULL;";
pub const TRIPLES_PREDICATE_IDX: &str =
    "CREATE INDEX IF NOT EXISTS triples_predicate_idx ON triples(predicate) WHERE valid_to IS NULL;";

pub const ALL_SCHEMA: &[&str] = &[
    PAGES_TABLE,
    CHUNKS_TABLE,
    CHUNKS_PAGE_IDX,
    CHUNKS_SIMHASH_IDX,
    TAGS_TABLE,
    TAGS_TAG_IDX,
    LINKS_TABLE,
    LINKS_FROM_IDX,
    LINKS_TO_IDX,
    SOURCE_VERSIONS_TABLE,
    EVAL_MEMO_TABLE,
    TRIPLES_TABLE,
    TRIPLES_SUBJECT_IDX,
    TRIPLES_OBJECT_IDX,
    TRIPLES_PREDICATE_IDX,
];
