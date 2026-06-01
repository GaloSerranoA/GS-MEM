//! Enterprise doctrine gate for `immortal-nn`.
//!
//! These tests enforce the structural invariants that make this crate a
//! sovereign neural inference engine. Each gate has a narrative: a
//! regression it prevents. Bumping any gate requires attesting the whole
//! set in the commit message.
//!
//! Gates (as of v1.0 of this doctrine):
//!
//! 1. `#![forbid(unsafe_code)]` declared at crate root — this crate is
//!    100% safe Rust.
//! 2. `#![warn(missing_docs)]` declared at crate root — every public
//!    item carries documentation.
//! 3. `Cargo.toml` declares no external inference runtime dependencies.
//!    The only runtime dependency allowed by Plan 44 is CPU scheduling
//!    (`rayon`), with no GPU backend and no model runtime.
//! 4. `NnError` is `#[non_exhaustive]` — new variants can be added
//!    without `SemVer` breakage.
//! 5. No `Result<_, String>` on the public API — every fallible entry
//!    point returns `NnResult<T>`.
//! 6. Clippy `pedantic` enabled in `Cargo.toml` `[lints.clippy]`.

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

#[test]
fn gate_1_forbid_unsafe_declared_at_crate_root() {
    let lib = fs::read_to_string(lib_rs()).expect("read lib.rs");
    assert!(
        lib.contains("#![forbid(unsafe_code)]"),
        "lib.rs must declare #![forbid(unsafe_code)] at the crate root. \
         Enterprise contract: this crate is 100% safe Rust."
    );
}

#[test]
fn gate_2_warn_missing_docs_declared_at_crate_root() {
    let lib = fs::read_to_string(lib_rs()).expect("read lib.rs");
    assert!(
        lib.contains("#![warn(missing_docs)]"),
        "lib.rs must declare #![warn(missing_docs)] at the crate root. \
         Enterprise contract: every public item carries documentation."
    );
}

#[test]
fn gate_3_only_cpu_scheduling_runtime_dependency() {
    let toml = fs::read_to_string(cargo_toml()).expect("read Cargo.toml");

    // Grab the body between `[dependencies]` and the next section header.
    let deps_section = toml
        .split("[dependencies]")
        .nth(1)
        .expect("Cargo.toml is missing a [dependencies] section (it must exist even when empty)");

    let body: &str = deps_section
        .split_once("\n[")
        .map_or(deps_section, |(body, _rest)| body);

    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("rayon ") || trimmed.starts_with("rayon=") {
            continue;
        }
        panic!(
            "immortal-nn must declare no external inference runtime dependencies. \
             Found a non-comment line in [dependencies]: `{trimmed}`. \
             Plan 44 allows only CPU scheduling via rayon; conversion tooling \
             lives in immortal-nn-convert and no GPU/model runtime may leak in.",
        );
    }
}

#[test]
fn gate_4_nn_error_is_non_exhaustive() {
    let lib = fs::read_to_string(lib_rs()).expect("read lib.rs");
    // Look for `#[non_exhaustive]` on a line immediately preceding `pub enum NnError`.
    let marker = "#[non_exhaustive]";
    let enum_line = "pub enum NnError";
    let pos = lib
        .find(enum_line)
        .expect("`pub enum NnError` must exist in lib.rs");
    let preceding = &lib[..pos];
    let nearest = preceding.rfind(marker);
    assert!(
        nearest.is_some_and(|start| preceding[start..].lines().count() <= 4),
        "NnError must be marked #[non_exhaustive] so new variants can be added \
         without SemVer breakage. Expected `#[non_exhaustive]` within 4 lines of \
         `pub enum NnError` in lib.rs."
    );
}

#[test]
fn gate_5_no_public_result_string() {
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
            if line.contains("Result<") && (line.contains(", String>") || line.contains(", &str>"))
            {
                offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "public API must use NnResult<T> (never Result<_, String>):\n{}",
        offenders.join("\n")
    );
}

#[test]
fn gate_conv_is_pure_inference() {
    // The conv module must stay inference-only and sovereign — no RNG
    // (training concern), no external crates. Scan every file in
    // src/conv/ for banned items in non-comment code.
    let conv_dir = src_dir().join("conv");
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&conv_dir).expect("read conv/") {
        let entry = entry.expect("entry");
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            files.push(p);
        }
    }
    assert!(
        !files.is_empty(),
        "expected src/conv/*.rs but found no files"
    );
    let banned_rng = ["thread_rng", "rand::random", "from_entropy", "ChaCha20Rng"];
    let banned_crates = ["use ndarray", "use serde", "use rand"];
    for path in &files {
        let body = fs::read_to_string(path).expect("read conv file");
        for (i, line) in body.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") || t.starts_with("///") || t.starts_with("//!") {
                continue;
            }
            for b in &banned_rng {
                assert!(
                    !line.contains(b),
                    "{}:{}: conv module must not use `{b}` — inference engine is pure.",
                    path.display(),
                    i + 1
                );
            }
            for b in &banned_crates {
                assert!(
                    !line.contains(b),
                    "{}:{}: conv module must not depend on `{b}` — sovereign contract.",
                    path.display(),
                    i + 1
                );
            }
        }
    }
}

#[test]
fn gate_rnn_is_pure_inference() {
    // The rnn module is inference-only. Same contract as conv/: no RNG,
    // no external crates leaking in.
    let rnn_dir = src_dir().join("rnn");
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&rnn_dir).expect("read rnn/") {
        let entry = entry.expect("entry");
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            files.push(p);
        }
    }
    assert!(
        !files.is_empty(),
        "expected src/rnn/*.rs but found no files"
    );
    let banned_rng = ["thread_rng", "rand::random", "from_entropy", "ChaCha20Rng"];
    let banned_crates = ["use ndarray", "use serde", "use rand"];
    for path in &files {
        let body = fs::read_to_string(path).expect("read rnn file");
        for (i, line) in body.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") || t.starts_with("///") || t.starts_with("//!") {
                continue;
            }
            for b in &banned_rng {
                assert!(
                    !line.contains(b),
                    "{}:{}: rnn module must not use `{b}` — inference engine is pure.",
                    path.display(),
                    i + 1
                );
            }
            for b in &banned_crates {
                assert!(
                    !line.contains(b),
                    "{}:{}: rnn module must not depend on `{b}` — sovereign contract.",
                    path.display(),
                    i + 1
                );
            }
        }
    }
}

#[test]
fn gate_transformer_is_pure_inference() {
    // The transformer module is inference-only. Same contract as conv/
    // and rnn/: no RNG, no external crates.
    let tx_dir = src_dir().join("transformer");
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&tx_dir).expect("read transformer/") {
        let entry = entry.expect("entry");
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            files.push(p);
        }
    }
    assert!(
        !files.is_empty(),
        "expected src/transformer/*.rs but found no files"
    );
    let banned_rng = ["thread_rng", "rand::random", "from_entropy", "ChaCha20Rng"];
    let banned_crates = ["use ndarray", "use serde", "use rand"];
    for path in &files {
        let body = fs::read_to_string(path).expect("read transformer file");
        for (i, line) in body.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") || t.starts_with("///") || t.starts_with("//!") {
                continue;
            }
            for b in &banned_rng {
                assert!(
                    !line.contains(b),
                    "{}:{}: transformer module must not use `{b}` — inference engine is pure.",
                    path.display(),
                    i + 1
                );
            }
            for b in &banned_crates {
                assert!(
                    !line.contains(b),
                    "{}:{}: transformer module must not depend on `{b}` — sovereign contract.",
                    path.display(),
                    i + 1
                );
            }
        }
    }
}

#[test]
fn gate_transformer_no_hashmap() {
    // Determinism: transformer public API must not expose HashMap/HashSet
    // in non-comment lines. Weight lookups go through the deterministic
    // SovereignModel map.
    let tx_dir = src_dir().join("transformer");
    for entry in fs::read_dir(&tx_dir).expect("read transformer/") {
        let entry = entry.expect("entry");
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let body = fs::read_to_string(&p).expect("read transformer file");
        for (i, line) in body.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") || t.starts_with("///") || t.starts_with("//!") {
                continue;
            }
            if t.starts_with("use ") {
                continue;
            }
            assert!(
                !line.contains("HashMap"),
                "{}:{}: transformer must not reference HashMap — determinism contract",
                p.display(),
                i + 1
            );
            assert!(
                !line.contains("HashSet"),
                "{}:{}: transformer must not reference HashSet — determinism contract",
                p.display(),
                i + 1
            );
        }
    }
}

#[test]
fn gate_6_clippy_pedantic_declared() {
    let toml = fs::read_to_string(cargo_toml()).expect("read Cargo.toml");
    let lints_section = toml
        .split("[lints.clippy]")
        .nth(1)
        .expect("Cargo.toml must declare [lints.clippy]");
    let body = lints_section
        .split_once("\n[")
        .map_or(lints_section, |(body, _rest)| body);
    assert!(
        body.contains("pedantic") && body.contains("warn"),
        "Cargo.toml [lints.clippy] must declare `pedantic = \"warn\"`. \
         Enterprise contract: pedantic lints catch the long tail of \
         ergonomics + correctness smells."
    );
}
