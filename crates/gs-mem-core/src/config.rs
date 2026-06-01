//! Brain paths for NANTAR — markdown corpus root, SQLite brain.db, tantivy index root.
//!
//! Adapted from `gmem/src/config.rs` on 2026-04-18. Two NANTAR-specific
//! renames prevent collision with the external `D:\OneDrive\AI\G-Memory-rs\`
//! personal MCP tool:
//!
//! 1. `GMEM_ROOT` env var → `IMMORTAL_GMEM_ROOT` (NANTAR's markdown root, opt-in override)
//! 2. `%LOCALAPPDATA%\gmem\` → `%LOCALAPPDATA%\immortal-gmem\`
//!    (NANTAR's brain.db + tantivy index live under `immortal-gmem/` —
//!    external gmem's personal brain at `gmem/` is untouched)

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{GmemError, Result};

#[derive(Clone, Debug)]
pub struct Config {
    pub markdown_root: PathBuf,
    pub db_path: PathBuf,
    pub index_root: PathBuf,
}

impl Config {
    pub fn load() -> Result<Self> {
        let markdown_root = match env::var_os("IMMORTAL_GMEM_ROOT") {
            Some(root) => PathBuf::from(root),
            None => match env::var_os("OneDrive") {
                Some(one_drive) => PathBuf::from(one_drive),
                None => env::current_dir()?,
            },
        };

        let db_root = match env::var_os("IMMORTAL_GMEM_DATA_DIR") {
            Some(root) => PathBuf::from(root),
            None => {
                let local_data = dirs::data_local_dir()
                    .ok_or_else(|| GmemError::Other("LOCALAPPDATA is unavailable".to_string()))?;
                local_data.join("immortal-gmem")
            }
        };

        Ok(Self {
            markdown_root,
            db_path: db_root.join("brain.db"),
            index_root: db_root.join("index"),
        })
    }

    pub fn with_markdown_root(markdown_root: impl Into<PathBuf>) -> Result<Self> {
        let mut config = Self::load()?;
        config.markdown_root = markdown_root.into();
        Ok(config)
    }

    pub fn ensure_local_dirs(&self) -> Result<()> {
        if let Some(parent) = self.db_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir_all(&self.index_root)?;
        Ok(())
    }

    pub fn markdown_root(&self) -> &Path {
        &self.markdown_root
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn data_dir_override_uses_brain_and_index_under_env_dir() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let original = env::var_os("IMMORTAL_GMEM_DATA_DIR");
        let temp = tempfile::tempdir().expect("tempdir");

        env::set_var("IMMORTAL_GMEM_DATA_DIR", temp.path());
        let config = Config::load().expect("config loads with data dir override");

        assert_eq!(config.db_path, temp.path().join("brain.db"));
        assert_eq!(config.index_root, temp.path().join("index"));

        match original {
            Some(value) => env::set_var("IMMORTAL_GMEM_DATA_DIR", value),
            None => env::remove_var("IMMORTAL_GMEM_DATA_DIR"),
        }
    }
}
