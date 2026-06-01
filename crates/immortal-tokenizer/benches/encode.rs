//! Criterion benchmarks for SovereignTokenizer::encode.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use immortal_tokenizer::SovereignTokenizer;

const FIXTURE: &str = "tests/fixtures/minilm-l6-v2.sovereign-tokenizer";

fn load_tok() -> SovereignTokenizer {
    SovereignTokenizer::load(FIXTURE).expect("load")
}

fn bench_short(c: &mut Criterion) {
    let tok = load_tok();
    let s = "Hello, world!";
    c.bench_function("encode/short", |b| {
        b.iter(|| {
            let _ = tok.encode(black_box(s), true).expect("encode");
        })
    });
}

fn bench_medium(c: &mut Criterion) {
    let tok = load_tok();
    let s = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    c.bench_function("encode/medium_500c", |b| {
        b.iter(|| {
            let _ = tok.encode(black_box(&s), true).expect("encode");
        })
    });
}

fn bench_long(c: &mut Criterion) {
    let tok = load_tok();
    let s = "The quick brown fox jumps over the lazy dog. ".repeat(80);
    c.bench_function("encode/long_2KB_truncated", |b| {
        b.iter(|| {
            let _ = tok.encode(black_box(&s), true).expect("encode");
        })
    });
}

criterion_group!(benches, bench_short, bench_medium, bench_long);
criterion_main!(benches);
