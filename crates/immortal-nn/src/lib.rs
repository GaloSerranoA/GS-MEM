//! # immortal-nn
//!
//! Sovereign pure-Rust neural inference engine for the IMMORTAL framework.
//!
//! ## Design principles
//! - **Zero external runtime dependencies.** Not even `serde` or `ndarray`.
//! - **Inference only.** No autograd, no tape, no backward pass.
//! - **Nd-native.** 2-D / 3-D / 4-D tensors, row-major contiguous.
//! - **Functional ops.** `&Tensor -> Tensor`, no mutation.
//! - **Auditable.** Every op validates shapes and fails fast with a typed
//!   [`NnError`] — there is no `Result<_, String>` on the public API.
//!
//! ## Crate layout
//! - [`tensor`] — `Shape`, `Tensor`, indexing
//! - [`ops`]    — pure functional ops (matmul, attention, CRF Viterbi, layer norm, …)
//! - [`blocks`] — architecture blocks (Linear, Embedding, MultiHeadAttention, BiLstm, …)
//! - [`heads`]  — task heads (TokenClassifier, SequenceClassifier, BiaffineParser)
//! - [`mod@format`] — `.sovereign` binary format primitives (CRC32-checked)
//!
//! ## Enterprise contract
//! - `#![forbid(unsafe_code)]`
//! - `#![warn(missing_docs)]`
//! - Clippy `pedantic` enabled at crate level
//! - Typed [`NnError`] (`#[non_exhaustive]`) + [`NnResult`] alias
//! - `tests/enterprise_doctrine.rs` structural-regression gate

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(
    not(test),
    warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
// Domain-specific allows for a pure-Rust neural inference library:
#![allow(
    clippy::missing_errors_doc,       // NnResult is self-documenting; errors are well-typed
    clippy::missing_panics_doc,       // shape invariants panic intentionally; documented in type contract
    clippy::doc_markdown,             // math/tensor notation in doc comments
    clippy::many_single_char_names,   // standard ML variable names (i, j, k, r, c, n, m, d)
    clippy::similar_names,            // LSTM weight names (w_ih/w_hh, b_ih/b_hh) are the standard
    clippy::cast_possible_truncation, // NN tensor shapes fit in u32; validated by the model loader
    clippy::cast_precision_loss,      // f32 index arithmetic acceptable in neural inference
    clippy::needless_range_loop,      // multi-dimensional tensor math uses index arithmetic by design
    clippy::must_use_candidate,       // builder/factory methods; callers are expected to use results
    clippy::should_implement_trait,   // Activation::from_str follows the existing model-loading API
)]

pub mod blocks;
pub mod causal_lm;
pub mod conv;
pub mod deberta;
pub mod format;
pub mod heads;
pub mod ops;
pub mod rnn;
pub mod tensor;
pub mod transformer;
pub mod xlm_roberta;

pub use format::{SovereignModel, SovereignWriter};
pub use tensor::{Shape, Tensor};
pub use xlm_roberta::{XlmRobertaEncoder, XlmRobertaSequenceClassifier};

// ---------------------------------------------------------------------------
// Architecture identifiers — single source of truth shared with exporters.
// ---------------------------------------------------------------------------

/// Token classifier with BiLSTM encoder + CRF decoder. Used for NER, POS, SRL.
pub const ARCH_TOKEN_CLASSIFIER_BILSTM_CRF: &str = "token_classifier_bilstm_crf";

/// Token classifier with Transformer encoder + CRF decoder.
pub const ARCH_TOKEN_CLASSIFIER_TRANSFORMER_CRF: &str = "token_classifier_transformer_crf";

/// Sequence classifier with Transformer encoder. Used for sentiment, WSD, intent.
pub const ARCH_SEQUENCE_CLASSIFIER_TRANSFORMER: &str = "sequence_classifier_transformer";

/// DeBERTa-v2/v3 sequence classifier with disentangled attention (tasksource NLI).
pub const ARCH_DEBERTA_V2_CLASSIFIER: &str = "deberta_v2_classifier";

/// Biaffine dependency parser. Used for dependency parsing and coreference.
pub const ARCH_BIAFFINE_PARSER: &str = "biaffine_parser";

/// Causal language model — GPT-2 family (GPT-2, DistilGPT-2, GPT-Neo).
pub const ARCH_CAUSAL_LM_GPT2: &str = "causal_lm_gpt2";

/// Causal language model — LLaMA family (LLaMA, LLaMA-2, CodeLlama).
pub const ARCH_CAUSAL_LM_LLAMA: &str = "causal_lm_llama";

/// Causal language model — Mistral family (Mistral, Mixtral).
pub const ARCH_CAUSAL_LM_MISTRAL: &str = "causal_lm_mistral";

/// XLM-R encoder for BGE-style dense embedding inference.
pub const ARCH_XLM_ROBERTA_ENCODER: &str = "xlm_roberta_encoder";

/// XLM-R sequence classifier for BGE-style cross-encoder reranking.
pub const ARCH_XLM_ROBERTA_SEQUENCE_CLASSIFIER: &str = "xlm_roberta_sequence_classifier";

/// All known architecture identifiers (for loader validation).
pub const KNOWN_ARCHITECTURES: &[&str] = &[
    ARCH_TOKEN_CLASSIFIER_BILSTM_CRF,
    ARCH_TOKEN_CLASSIFIER_TRANSFORMER_CRF,
    ARCH_SEQUENCE_CLASSIFIER_TRANSFORMER,
    ARCH_BIAFFINE_PARSER,
    ARCH_CAUSAL_LM_GPT2,
    ARCH_CAUSAL_LM_LLAMA,
    ARCH_CAUSAL_LM_MISTRAL,
    ARCH_XLM_ROBERTA_ENCODER,
    ARCH_XLM_ROBERTA_SEQUENCE_CLASSIFIER,
    ARCH_DEBERTA_V2_CLASSIFIER,
];

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// All errors produced by `immortal-nn`.
///
/// `#[non_exhaustive]` — callers must always use a wildcard match arm so
/// that new variants can be added without breaking downstream code.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NnError {
    /// Operand shapes did not match the operation's contract.
    ShapeMismatch {
        /// What the op expected.
        expected: String,
        /// What it actually got.
        got: String,
    },
    /// Axis index is out of range for the tensor's rank.
    InvalidAxis {
        /// Requested axis.
        axis: usize,
        /// Tensor rank.
        ndim: u8,
    },
    /// Operation requires a contiguous layout but the input is not contiguous.
    NonContiguous,
    /// Tensor rank is not supported (must be 2, 3, or 4).
    UnsupportedRank {
        /// The offending rank.
        rank: u8,
    },
    /// Element count does not match the requested shape.
    ElementCountMismatch {
        /// Number of elements provided.
        provided: usize,
        /// Number of elements the shape requires.
        required: usize,
    },
    /// Divisibility constraint failed (e.g. `embed_dim % num_heads != 0`).
    DivisibilityError {
        /// Human-readable name of the failing constraint.
        what: String,
        /// Numerator.
        numerator: usize,
        /// Denominator.
        denominator: usize,
    },
    /// Architecture identifier in metadata is unknown.
    UnknownArchitecture(String),
    /// CRC32 mismatch when validating a `.sovereign` tensor blob.
    ChecksumMismatch {
        /// Tensor name.
        name: String,
        /// Expected CRC32.
        expected: u32,
        /// Computed CRC32.
        got: u32,
    },
    /// I/O error while reading a `.sovereign` file.
    Io(String),
    /// Generic format error (corrupt file, bad magic, unknown version, ...).
    Format(String),
}

impl core::fmt::Display for NnError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            Self::InvalidAxis { axis, ndim } => {
                write!(f, "invalid axis {axis} for tensor of rank {ndim}")
            }
            Self::NonContiguous => write!(f, "tensor is not contiguous"),
            Self::UnsupportedRank { rank } => {
                write!(f, "unsupported tensor rank {rank} (expected 2..=4)")
            }
            Self::ElementCountMismatch { provided, required } => write!(
                f,
                "element count mismatch: provided {provided}, required {required}"
            ),
            Self::DivisibilityError {
                what,
                numerator,
                denominator,
            } => write!(f, "{what}: {numerator} not divisible by {denominator}"),
            Self::UnknownArchitecture(s) => write!(f, "unknown architecture: {s}"),
            Self::ChecksumMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "CRC32 mismatch for tensor {name:?}: expected 0x{expected:08x}, got 0x{got:08x}"
            ),
            Self::Io(s) => write!(f, "io error: {s}"),
            Self::Format(s) => write!(f, "format error: {s}"),
        }
    }
}

impl std::error::Error for NnError {}

/// Standard result type for `immortal-nn`.
pub type NnResult<T> = Result<T, NnError>;
