//! Enterprise doctrine gate for `immortal-gmem`.
//!
//! Structural regression gates for the sovereign brain primitives library —
//! 4-phase architecture (kernel/index/storage/mcp feature-gated), absorbed
//! from g-memory-rs on 2026-04-18. Zero downstream consumers at landing.
//! Silent-by-design: no `tracing` dependency (JSON-RPC stdio server requires
//! no logging infrastructure). `GmemError` typed error with 11 variants.
//!
//! Gates:
//!
//! 1. Unsafe-code discipline: `forbid(unsafe_code)` OR `deny(unsafe_code)` at `lib.rs`.
//! 2. Production lint floor (`warn(clippy::unwrap_used, expect_used, panic)`) gated on `cfg_attr(not(test))`.
//! 3. Local error type `GmemError` is `#[non_exhaustive]`.
//! 4. No `Result<_, String>` on public API.
//! 5. No `anyhow` dependency.
//! 6. No `log` / `env_logger` dependency (silent crate — tracing NOT required).

use std::fs;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn src_dir() -> PathBuf {
    crate_root().join("src")
}

fn lib_rs() -> PathBuf {
    src_dir().join("lib.rs")
}

fn cargo_toml() -> PathBuf {
    crate_root().join("Cargo.toml")
}

#[allow(dead_code)]
fn read_all_src_rs() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, acc: &mut Vec<(PathBuf, String)>) {
        for entry in fs::read_dir(dir).expect("read_dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                walk(&path, acc);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                let content = fs::read_to_string(&path).expect("read_to_string");
                acc.push((path, content));
            }
        }
    }
    let mut out = Vec::new();
    walk(&src_dir(), &mut out);
    out
}

#[allow(dead_code)]
fn assert_error_non_exhaustive(file: &Path, enum_name: &str) {
    let src = fs::read_to_string(file).unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
    let needle = format!("pub enum {enum_name}");
    let pos = src
        .find(&needle)
        .unwrap_or_else(|| panic!("`pub enum {enum_name}` not found in {}", file.display()));
    let preceding = &src[..pos];
    let nearest = preceding.rfind("#[non_exhaustive]");
    assert!(
        nearest.is_some_and(|start| preceding[start..].lines().count() <= 4),
        "{enum_name} in {} must be marked #[non_exhaustive] within 4 \
         lines of its `pub enum` declaration.",
        file.display()
    );
}

#[allow(dead_code)]
fn outer_result_error_type(line: &str) -> Option<&str> {
    let start = line.find("-> Result<")?;
    let after = &line[start + "-> Result<".len()..];
    let mut depth: i32 = 1;
    let mut comma_idx: Option<usize> = None;
    let mut close_idx: Option<usize> = None;
    for (i, ch) in after.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    close_idx = Some(i);
                    break;
                }
            }
            ',' if depth == 1 && comma_idx.is_none() => comma_idx = Some(i),
            _ => {}
        }
    }
    let comma = comma_idx?;
    let close = close_idx?;
    Some(after[comma + 1..close].trim())
}

#[test]
fn gate_1_unsafe_code_disciplined() {
    let src = fs::read_to_string(lib_rs())
        .expect("read lib.rs")
        .replace("\r\n", "\n");
    assert!(
        src.contains("#![forbid(unsafe_code)]") || src.contains("#![deny(unsafe_code)]"),
        "lib.rs must declare `#![forbid(unsafe_code)]` OR `#![deny(unsafe_code)]`. \
         immortal-gmem has zero unsafe sites → prefer `forbid`."
    );
}

#[test]
fn gate_2_production_lint_floor_declared() {
    let src = fs::read_to_string(lib_rs())
        .expect("read lib.rs")
        .replace("\r\n", "\n");
    for required in &[
        "clippy::unwrap_used",
        "clippy::expect_used",
        "clippy::panic",
    ] {
        assert!(
            src.contains(required),
            "lib.rs is missing `{required}`. Enterprise contract: non-test code must warn on unwrap/expect/panic."
        );
    }
    assert!(
        src.contains("cfg_attr(\n    not(test)") || src.contains("cfg_attr(not(test)"),
        "lib.rs lint floor must be gated on not(test) via cfg_attr."
    );
}

#[test]
fn gate_3_local_errors_are_non_exhaustive() {
    assert_error_non_exhaustive(&src_dir().join("error.rs"), "GmemError");
}

#[test]
fn gate_4_no_public_result_string() {
    let src = read_all_src_rs();
    let mut offenders: Vec<String> = Vec::new();
    for (path, content) in &src {
        for (i, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            if !(trimmed.starts_with("pub fn")
                || trimmed.starts_with("pub async fn")
                || trimmed.starts_with("pub(crate) fn"))
            {
                continue;
            }
            if let Some(err_type) = outer_result_error_type(line) {
                if err_type == "String" || err_type == "&str" {
                    offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "public API must use a typed error — never Result<_, String>:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn gate_5_no_anyhow_dependency() {
    let toml = fs::read_to_string(cargo_toml()).expect("read Cargo.toml");
    let deps_section = toml
        .split("[dependencies]")
        .nth(1)
        .expect("Cargo.toml is missing [dependencies]");
    let body = deps_section
        .split_once("\n[")
        .map_or(deps_section, |(body, _rest)| body);
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((name, _)) = trimmed.split_once('=') else {
            continue;
        };
        if name.trim() == "anyhow" {
            panic!("`anyhow` crate found in [dependencies] — use typed errors. Line: {trimmed}");
        }
    }
}

#[test]
fn gate_6_no_log_env_logger_dep() {
    // immortal-gmem is silent by design — JSON-RPC stdio server needs no logging.
    // Gate 6 (relaxed): forbid `log` and `env_logger`; tracing is NOT required.
    let toml = fs::read_to_string(cargo_toml()).expect("read Cargo.toml");
    let deps_section = toml
        .split("[dependencies]")
        .nth(1)
        .expect("Cargo.toml is missing [dependencies]");
    let body = deps_section
        .split_once("\n[")
        .map_or(deps_section, |(body, _rest)| body);
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("log ") || trimmed.starts_with("log=") {
            panic!("`log` crate found in [dependencies] — immortal-gmem is silent by design. Line: {trimmed}");
        }
        if trimmed.starts_with("env_logger") {
            panic!(
                "`env_logger` crate found in [dependencies] — immortal-gmem is silent by design. Line: {trimmed}"
            );
        }
    }
}
