//! Filesystem scanner — walks the markdown corpus under `Config::markdown_root`
//! and lists every `.md`/`.markdown` file. Used by the indexer to discover
//! pages and by `Config::ensure_local_dirs` workflows.
//!
//! Absorbed from `gmem/src/sync/mod.rs` on 2026-04-18. Flattened from module
//! directory to single file since there is only one scanner implementation.

use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::error::{GmemError, Result};

#[derive(Debug, Clone)]
pub struct SyncScanner {
    root: PathBuf,
}

impl SyncScanner {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn scan(&self) -> Result<Vec<PathBuf>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }

        let mut files = Vec::new();
        for entry in WalkDir::new(&self.root) {
            let entry = entry.map_err(|err| GmemError::Other(err.to_string()))?;
            if entry.file_type().is_file() && is_markdown(entry.path()) {
                files.push(entry.path().to_path_buf());
            }
        }

        Ok(files)
    }
}

fn is_markdown(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("md") | Some("markdown")
    )
}
