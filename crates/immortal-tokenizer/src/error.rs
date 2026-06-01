//! Typed error boundary for immortal-tokenizer.

use crate::truncation::TruncationStrategy;

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TokenizerError {
    #[error("io error reading tokenizer file: {0}")]
    Io(#[from] std::io::Error),

    #[error("tokenizer header corrupt: {reason}")]
    HeaderCorrupt { reason: String },

    #[error("tokenizer payload corrupt: {reason}")]
    PayloadCorrupt { reason: String },

    #[error("unsupported tokenizer format version: found major={found_major}, this build supports major={supported_major}")]
    UnsupportedFormatVersion {
        found_major: u16,
        supported_major: u16,
    },

    #[error("bincode decode failed: {0}")]
    BincodeDecode(#[from] bincode::error::DecodeError),

    #[error("bincode encode failed: {0}")]
    BincodeEncode(#[from] bincode::error::EncodeError),

    #[error("vocab validation failed: {reason}")]
    VocabInvalid { reason: String },

    #[error("required special token missing from vocab: {token}")]
    SpecialTokenMissing { token: String },

    #[error("input exceeds max_input_chars_per_word ({word_len} > {limit}); returned [UNK]")]
    WordTooLong { word_len: usize, limit: usize },

    #[error("truncation strategy {strategy:?} cannot satisfy max_length={max_length} with {special_tokens} required special tokens")]
    TruncationImpossible {
        strategy: TruncationStrategy,
        max_length: usize,
        special_tokens: usize,
    },
}

impl TokenizerError {
    /// Stable short identifier per variant — used by downstream error-code-routing code.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Io(_) => "io",
            Self::HeaderCorrupt { .. } => "header_corrupt",
            Self::PayloadCorrupt { .. } => "payload_corrupt",
            Self::UnsupportedFormatVersion { .. } => "unsupported_format_version",
            Self::BincodeDecode(_) => "bincode_decode",
            Self::BincodeEncode(_) => "bincode_encode",
            Self::VocabInvalid { .. } => "vocab_invalid",
            Self::SpecialTokenMissing { .. } => "special_token_missing",
            Self::WordTooLong { .. } => "word_too_long",
            Self::TruncationImpossible { .. } => "truncation_impossible",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_codes_are_unique_and_stable() {
        let samples = [
            TokenizerError::HeaderCorrupt { reason: "x".into() },
            TokenizerError::PayloadCorrupt { reason: "x".into() },
            TokenizerError::UnsupportedFormatVersion {
                found_major: 2,
                supported_major: 1,
            },
            TokenizerError::VocabInvalid { reason: "x".into() },
            TokenizerError::SpecialTokenMissing {
                token: "[CLS]".into(),
            },
            TokenizerError::WordTooLong {
                word_len: 200,
                limit: 100,
            },
        ];
        let codes: Vec<&str> = samples.iter().map(|e| e.error_code()).collect();
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len(), "error codes must be unique");
    }

    #[test]
    fn display_includes_variant_context() {
        let err = TokenizerError::SpecialTokenMissing {
            token: "[CLS]".into(),
        };
        assert!(format!("{err}").contains("[CLS]"));
    }
}
