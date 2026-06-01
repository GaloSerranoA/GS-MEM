use std::path::{Path, PathBuf};

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, SchemaBuilder, Value, STORED, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

use crate::error::{GmemError, Result};

const WRITER_HEAP_BYTES: usize = 50_000_000;

pub struct TantivyIndex {
    index_dir: PathBuf,
    schema: Schema,
    field_slug: Field,
    field_title: Field,
    field_body: Field,
    field_tags: Field,
    index: Index,
    reader: IndexReader,
}

impl TantivyIndex {
    fn schema_with_fields() -> (Schema, Field, Field, Field, Field) {
        let mut builder = SchemaBuilder::default();
        let field_slug = builder.add_text_field("slug", TEXT | STORED);
        let field_title = builder.add_text_field("title", TEXT | STORED);
        let field_body = builder.add_text_field("body", TEXT | STORED);
        let field_tags = builder.add_text_field("tags", TEXT | STORED);
        (
            builder.build(),
            field_slug,
            field_title,
            field_body,
            field_tags,
        )
    }

    /// Open existing or create new tantivy index at `index_dir`.
    pub fn open_or_create(index_dir: impl AsRef<Path>) -> Result<Self> {
        let index_dir = index_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&index_dir)?;

        let (schema, field_slug, field_title, field_body, field_tags) = Self::schema_with_fields();
        let index = Index::open_in_dir(&index_dir)
            .or_else(|_| Index::create_in_dir(&index_dir, schema.clone()))
            .map_err(GmemError::Tantivy)?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(GmemError::Tantivy)?;

        Ok(Self {
            index_dir,
            schema,
            field_slug,
            field_title,
            field_body,
            field_tags,
            index,
            reader,
        })
    }

    /// Upsert a document. Does NOT commit — caller commits in batch.
    pub fn upsert(
        &self,
        writer: &mut IndexWriter,
        slug: &str,
        title: Option<&str>,
        body: &str,
        tags: &[String],
    ) -> Result<()> {
        let term = tantivy::Term::from_field_text(self.field_slug, slug);
        writer.delete_term(term);

        let title_str = title.unwrap_or("");
        let tags_joined = tags.join(" ");

        writer
            .add_document(doc!(
                self.field_slug => slug,
                self.field_title => title_str,
                self.field_body => body,
                self.field_tags => tags_joined,
            ))
            .map_err(GmemError::Tantivy)?;
        Ok(())
    }

    /// Open a writer. Caller drops it after commit to release the lock.
    pub fn writer(&self) -> Result<IndexWriter> {
        self.index
            .writer(WRITER_HEAP_BYTES)
            .map_err(GmemError::Tantivy)
    }

    /// BM25 search. Returns (slug, score, snippet) tuples.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32, String)>> {
        use crate::search::query_sanitizer::sanitize;

        let sanitized = sanitize(query);
        if sanitized.trim().is_empty() {
            return Ok(Vec::new());
        }

        self.reader.reload().map_err(GmemError::Tantivy)?;
        let searcher = self.reader.searcher();
        let qp = QueryParser::for_index(
            &self.index,
            vec![self.field_title, self.field_body, self.field_tags],
        );
        let parsed = qp
            .parse_query(&sanitized)
            .map_err(|err| GmemError::Search(format!("tantivy query parse: {err}")))?;

        let hits = searcher
            .search(&parsed, &TopDocs::with_limit(limit))
            .map_err(GmemError::Tantivy)?;

        let mut out = Vec::with_capacity(hits.len());
        for (score, addr) in hits {
            let retrieved: TantivyDocument = searcher.doc(addr).map_err(GmemError::Tantivy)?;
            let slug = retrieved
                .get_first(self.field_slug)
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string();
            let body = retrieved
                .get_first(self.field_body)
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string();
            let snippet_end = body
                .char_indices()
                .nth(200)
                .map(|(idx, _)| idx)
                .unwrap_or(body.len());
            let snippet = body[..snippet_end].to_string();
            out.push((slug, score, snippet));
        }
        Ok(out)
    }

    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}
