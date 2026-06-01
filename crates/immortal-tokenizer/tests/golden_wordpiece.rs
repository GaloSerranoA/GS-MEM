//! Golden corpus byte-identical acceptance test.
//!
//! TEST-INTEGRITY CONTRACT: If this test fails, the sovereign implementation
//! is wrong. The fixture MUST NOT be edited to match observed output. To
//! regenerate (only when upstream HF MiniLM vocab legitimately changes),
//! re-run `gen_golden` and commit the new fixture as an explicit human
//! decision.

use std::fs::File;
use std::io::{BufRead, BufReader};

use immortal_tokenizer::SovereignTokenizer;
use serde::Deserialize;

const TOK_FIXTURE: &str = "tests/fixtures/minilm-l6-v2.sovereign-tokenizer";
const GOLDEN_JSONL: &str = "tests/fixtures/golden_wordpiece.jsonl";

#[derive(Debug, Deserialize)]
struct GoldenLine {
    text: String,
    expected_ids: Vec<u32>,
}

#[test]
fn golden_wordpiece_byte_identical() {
    let tok = SovereignTokenizer::load(TOK_FIXTURE).expect("load sovereign fixture");
    let reader = BufReader::new(File::open(GOLDEN_JSONL).expect("open golden jsonl"));

    let mut total = 0usize;
    let mut diffs: Vec<String> = Vec::new();

    for (lineno, line) in reader.lines().enumerate() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        let g: GoldenLine = serde_json::from_str(&line).expect("parse golden line");
        total += 1;

        let actual = tok.encode(&g.text, true).expect("encode").ids;
        if actual != g.expected_ids {
            let first_diff = actual
                .iter()
                .zip(g.expected_ids.iter())
                .position(|(a, e)| a != e)
                .unwrap_or_else(|| actual.len().min(g.expected_ids.len()));
            if diffs.len() < 3 {
                diffs.push(format!(
                    "line {lineno}: text={:?}\n  expected[..15]={:?}\n  actual[..15]={:?}\n  first_diff_idx={first_diff}",
                    g.text,
                    &g.expected_ids.iter().take(15).collect::<Vec<_>>(),
                    &actual.iter().take(15).collect::<Vec<_>>(),
                ));
            } else {
                // just count
            }
        }
    }

    let diff_count = diffs.len();
    if !diffs.is_empty() {
        panic!(
            "golden corpus drift: diverged on at least {diff_count} of {total} lines:\n{}",
            diffs.join("\n")
        );
    }
    assert_eq!(total, 1000, "expected exactly 1,000 golden lines");
}
