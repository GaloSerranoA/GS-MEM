//! WordPiece greedy longest-match subword tokenization.

use std::collections::HashMap;
use std::sync::Arc;

use crate::tokenizer::TokenId;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WordPieceConfig {
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

#[derive(Debug, Clone)]
pub struct WordPiece {
    cfg: WordPieceConfig,
    vocab: Arc<HashMap<String, TokenId>>,
    unk_id: TokenId,
}

impl WordPiece {
    pub fn new(
        cfg: WordPieceConfig,
        vocab: Arc<HashMap<String, TokenId>>,
    ) -> crate::error::Result<Self> {
        let unk_id = vocab.get(&cfg.unk_token).copied().ok_or_else(|| {
            crate::error::TokenizerError::SpecialTokenMissing {
                token: cfg.unk_token.clone(),
            }
        })?;
        Ok(Self { cfg, vocab, unk_id })
    }

    /// Tokenize one pre-tokenized word into subword IDs.
    pub fn tokenize(&self, word: &str) -> Vec<TokenId> {
        let char_count = word.chars().count();
        if char_count > self.cfg.max_input_chars_per_word {
            return vec![self.unk_id];
        }
        if word.is_empty() {
            return Vec::new();
        }

        let chars: Vec<(usize, char)> = word.char_indices().collect();
        let mut output: Vec<TokenId> = Vec::new();
        let mut start_char_idx = 0usize;

        while start_char_idx < chars.len() {
            let start_byte = chars[start_char_idx].0;
            let mut found: Option<(usize, TokenId)> = None;

            let mut end_char_idx = chars.len();
            while end_char_idx > start_char_idx {
                let end_byte = if end_char_idx < chars.len() {
                    chars[end_char_idx].0
                } else {
                    word.len()
                };
                let sub: &str = &word[start_byte..end_byte];
                let candidate: String = if start_char_idx > 0 {
                    format!("{}{}", self.cfg.continuing_subword_prefix, sub)
                } else {
                    sub.to_string()
                };
                if let Some(&id) = self.vocab.get(&candidate) {
                    found = Some((end_char_idx, id));
                    break;
                }
                end_char_idx -= 1;
            }

            match found {
                Some((end, id)) => {
                    output.push(id);
                    start_char_idx = end;
                }
                None => {
                    return vec![self.unk_id];
                }
            }
        }

        output
    }

    pub fn unk_id(&self) -> TokenId {
        self.unk_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(vocab: &[(&str, TokenId)]) -> WordPiece {
        let cfg = WordPieceConfig {
            unk_token: "[UNK]".into(),
            continuing_subword_prefix: "##".into(),
            max_input_chars_per_word: 100,
        };
        let map: HashMap<String, TokenId> =
            vocab.iter().map(|(k, v)| ((*k).to_string(), *v)).collect();
        WordPiece::new(cfg, Arc::new(map)).expect("unk present")
    }

    #[test]
    fn single_exact_match() {
        let wp = make(&[("[UNK]", 0), ("hello", 1)]);
        assert_eq!(wp.tokenize("hello"), vec![1]);
    }

    #[test]
    fn greedy_longest_with_prefix() {
        let wp = make(&[("[UNK]", 0), ("un", 1), ("##afford", 2), ("##able", 3)]);
        assert_eq!(wp.tokenize("unaffordable"), vec![1, 2, 3]);
    }

    #[test]
    fn whole_word_unk_on_unknown_start() {
        let wp = make(&[("[UNK]", 0), ("ok", 1)]);
        assert_eq!(wp.tokenize("xyz"), vec![0]);
    }

    #[test]
    fn over_long_word_returns_unk() {
        let wp = make(&[("[UNK]", 0), ("a", 1)]);
        let long: String = "a".repeat(101);
        assert_eq!(wp.tokenize(&long), vec![0]);
    }

    #[test]
    fn empty_word_empty_output() {
        let wp = make(&[("[UNK]", 0)]);
        assert!(wp.tokenize("").is_empty());
    }

    #[test]
    fn missing_unk_token_is_error() {
        let cfg = WordPieceConfig {
            unk_token: "[UNK]".into(),
            continuing_subword_prefix: "##".into(),
            max_input_chars_per_word: 100,
        };
        let vocab: HashMap<String, TokenId> = [("hello".to_string(), 1u32)].into();
        let err = WordPiece::new(cfg, Arc::new(vocab)).expect_err("should fail");
        assert!(matches!(
            err,
            crate::error::TokenizerError::SpecialTokenMissing { .. }
        ));
    }

    #[test]
    fn middle_subword_failure_falls_back_to_unk() {
        let wp = make(&[("[UNK]", 0), ("ab", 1), ("##cd", 2)]);
        assert_eq!(wp.tokenize("abqz"), vec![0]);
    }
}
