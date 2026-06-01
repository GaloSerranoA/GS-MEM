//! # immortal-tokenizer
//!
//! Sovereign, inference-only WordPiece tokenizer.
//!
//! ## Doctrine gates (Phase 1)
//! 1. Zero production panics — `unwrap`/`expect`/`panic!` forbidden in `src/**`
//!    outside `#[cfg(test)]`. Enforced by `cargo clippy --lib -- -D warnings`
//!    in combination with the crate-level lint configuration below.
//! 2. Typed error boundary — all `pub fn` return `Result<_, TokenizerError>`.
//! 3. `#[non_exhaustive]` on all `pub enum` and boundary `pub struct`.
//! 4. No external regex / C FFI in runtime dep tree.
//! 5. Golden corpus test green.
//! 6. Format boundary isolation — only `format.rs` references `bincode`/`IMTK`.

#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented,
        clippy::unreachable,
    )
)]

pub mod error;
pub mod format;
pub mod normalizer;
pub mod pretokenizer;
pub mod tokenizer;
pub mod truncation;
pub mod wordpiece;

pub use error::{Result, TokenizerError};
pub use tokenizer::{Encoding, SovereignTokenizer, TokenId};
