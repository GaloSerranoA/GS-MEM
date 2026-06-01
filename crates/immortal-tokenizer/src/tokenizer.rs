//! `SovereignTokenizer` facade — the only type most consumers touch.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Result, TokenizerError};
use crate::format::{
    self, PaddingDirectionDisk, PaddingStrategyDisk, SovereignTokenizerData, TruncationStrategyDisk,
};
use crate::normalizer::{Normalizer, NormalizerConfig};
use crate::pretokenizer::{PreTokConfig, PreTokenizer};
use crate::truncation::{Truncation, TruncationConfig, TruncationStrategy};
use crate::wordpiece::{WordPiece, WordPieceConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaddingStrategy {
    Disabled,
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaddingDirection {
    Right,
    Left,
}

#[derive(Debug, Clone)]
struct PaddingConfig {
    strategy: PaddingStrategy,
    direction: PaddingDirection,
    pad_id: TokenId,
    pad_type_id: u32,
}

pub type TokenId = u32;

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Encoding {
    pub ids: Vec<TokenId>,
    pub type_ids: Vec<u32>,
    pub offsets: Vec<(usize, usize)>,
    /// 1 for real tokens, 0 for padding. Populated by `encode` after padding.
    pub attention_mask: Vec<u32>,
}

impl Encoding {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct SovereignTokenizer {
    normalizer: Normalizer,
    pretokenizer: PreTokenizer,
    wordpiece: WordPiece,
    truncation: TruncationConfig,
    padding: PaddingConfig,
    vocab: Arc<HashMap<String, TokenId>>,
    reverse_vocab: Arc<Vec<String>>,
    cls_id: TokenId,
    sep_id: TokenId,
    pad_id: TokenId,
    unk_id: TokenId,
    mask_id: TokenId,
    continuing_prefix: String,
}

impl SovereignTokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let data = format::load(path)?;
        Self::from_data(data)
    }

    pub fn from_data(data: SovereignTokenizerData) -> Result<Self> {
        if data.vocab.is_empty() {
            return Err(TokenizerError::VocabInvalid {
                reason: "vocab is empty".into(),
            });
        }
        let mut vocab: HashMap<String, TokenId> = HashMap::with_capacity(data.vocab.len());
        for (i, tok) in data.vocab.iter().enumerate() {
            let id = u32::try_from(i).map_err(|_| TokenizerError::VocabInvalid {
                reason: format!("vocab too large: {}", data.vocab.len()),
            })?;
            if vocab.insert(tok.clone(), id).is_some() {
                return Err(TokenizerError::VocabInvalid {
                    reason: format!("duplicate vocab entry: {tok}"),
                });
            }
        }
        let vocab = Arc::new(vocab);
        let reverse_vocab = Arc::new(data.vocab.clone());

        let specials = &data.special_tokens;
        let lookup = |name: &str, s: &str| -> Result<TokenId> {
            vocab
                .get(s)
                .copied()
                .ok_or_else(|| TokenizerError::SpecialTokenMissing {
                    token: format!("{name} ({s})"),
                })
        };
        let cls_id = lookup("cls", &specials.cls)?;
        let sep_id = lookup("sep", &specials.sep)?;
        let pad_id = lookup("pad", &specials.pad)?;
        let unk_id = lookup("unk", &specials.unk)?;
        let mask_id = lookup("mask", &specials.mask)?;

        if data.padding.enabled && data.padding.pad_id != pad_id {
            return Err(TokenizerError::VocabInvalid {
                reason: format!(
                    "padding pad_id {} does not match special pad token id {pad_id}",
                    data.padding.pad_id
                ),
            });
        }

        let normalizer = Normalizer::new(NormalizerConfig {
            lowercase: data.normalizer.lowercase,
            strip_accents: data.normalizer.strip_accents,
            handle_chinese_chars: data.normalizer.handle_chinese_chars,
            clean_text: data.normalizer.clean_text,
        });
        let pretokenizer = PreTokenizer::new(PreTokConfig {
            do_basic_split: data.pretokenizer.do_basic_split,
            never_split: data.pretokenizer.never_split.clone(),
        });
        let wordpiece = WordPiece::new(
            WordPieceConfig {
                unk_token: data.wordpiece.unk_token.clone(),
                continuing_subword_prefix: data.wordpiece.continuing_subword_prefix.clone(),
                max_input_chars_per_word: data.wordpiece.max_input_chars_per_word,
            },
            Arc::clone(&vocab),
        )?;
        let truncation = TruncationConfig {
            max_length: data.truncation.max_length,
            strategy: match data.truncation.strategy {
                TruncationStrategyDisk::LongestFirst => TruncationStrategy::LongestFirst,
                TruncationStrategyDisk::OnlyFirst => TruncationStrategy::OnlyFirst,
                TruncationStrategyDisk::OnlySecond => TruncationStrategy::OnlySecond,
            },
            stride: data.truncation.stride,
        };
        let padding = PaddingConfig {
            strategy: if data.padding.enabled {
                match data.padding.strategy {
                    PaddingStrategyDisk::Disabled => PaddingStrategy::Disabled,
                    PaddingStrategyDisk::Fixed(n) => PaddingStrategy::Fixed(n),
                }
            } else {
                PaddingStrategy::Disabled
            },
            direction: match data.padding.direction {
                PaddingDirectionDisk::Right => PaddingDirection::Right,
                PaddingDirectionDisk::Left => PaddingDirection::Left,
            },
            pad_id: data.padding.pad_id,
            pad_type_id: data.padding.pad_type_id,
        };

        Ok(Self {
            normalizer,
            pretokenizer,
            wordpiece,
            truncation,
            padding,
            vocab,
            reverse_vocab,
            cls_id,
            sep_id,
            pad_id,
            unk_id,
            mask_id,
            continuing_prefix: data.wordpiece.continuing_subword_prefix,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        let mut enc = self.encode_raw(text)?;
        if add_special_tokens {
            // Reserve 2 slots (CLS + SEP) in the budget.
            let budget = self.truncation.max_length.saturating_sub(2);
            if enc.ids.len() > budget {
                enc.ids.truncate(budget);
                enc.type_ids.truncate(budget);
                enc.offsets.truncate(budget);
                enc.attention_mask.truncate(budget);
            }
            // Prepend CLS, append SEP.
            enc.ids.insert(0, self.cls_id);
            enc.type_ids.insert(0, 0);
            enc.offsets.insert(0, (0, 0));
            enc.attention_mask.insert(0, 1);
            enc.ids.push(self.sep_id);
            enc.type_ids.push(0);
            enc.offsets.push((0, 0));
            enc.attention_mask.push(1);
        } else {
            Truncation::apply_single(&mut enc, &self.truncation, 0)?;
        }
        self.apply_padding(&mut enc);
        Ok(enc)
    }

    /// Override the configured max_length at runtime.
    ///
    /// Used by the `tokenizers-compat` shim to honor `Tokenizer::with_truncation`
    /// calls from consumers mid-migration. Not part of the stable native API —
    /// callers building fresh tokenizers should pass the right max_length
    /// via the `.sovereign-tokenizer` file instead.
    pub fn set_max_length(&mut self, max_length: usize) {
        self.truncation.max_length = max_length;
    }

    fn apply_padding(&self, enc: &mut Encoding) {
        let target = match self.padding.strategy {
            PaddingStrategy::Disabled => return,
            PaddingStrategy::Fixed(n) => n,
        };
        if enc.ids.len() >= target {
            return;
        }
        let pad_count = target - enc.ids.len();
        match self.padding.direction {
            PaddingDirection::Right => {
                for _ in 0..pad_count {
                    enc.ids.push(self.padding.pad_id);
                    enc.type_ids.push(self.padding.pad_type_id);
                    enc.offsets.push((0, 0));
                    enc.attention_mask.push(0);
                }
            }
            PaddingDirection::Left => {
                let mut new_ids = Vec::with_capacity(target);
                let mut new_type_ids = Vec::with_capacity(target);
                let mut new_offsets = Vec::with_capacity(target);
                let mut new_attention = Vec::with_capacity(target);
                for _ in 0..pad_count {
                    new_ids.push(self.padding.pad_id);
                    new_type_ids.push(self.padding.pad_type_id);
                    new_offsets.push((0, 0));
                    new_attention.push(0);
                }
                new_ids.extend_from_slice(&enc.ids);
                new_type_ids.extend_from_slice(&enc.type_ids);
                new_offsets.extend_from_slice(&enc.offsets);
                new_attention.extend_from_slice(&enc.attention_mask);
                enc.ids = new_ids;
                enc.type_ids = new_type_ids;
                enc.offsets = new_offsets;
                enc.attention_mask = new_attention;
            }
        }
    }

    pub fn encode_pair(
        &self,
        text_a: &str,
        text_b: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut a = self.encode_raw(text_a)?;
        let mut b = self.encode_raw(text_b)?;
        let reserved = if add_special_tokens { 3 } else { 0 };
        Truncation::apply_pair(&mut a, &mut b, &self.truncation, reserved)?;
        let mut out = if add_special_tokens {
            let cap = a.ids.len() + b.ids.len() + 3;
            let mut out = Encoding {
                ids: Vec::with_capacity(cap),
                type_ids: Vec::with_capacity(cap),
                offsets: Vec::with_capacity(cap),
                attention_mask: Vec::with_capacity(cap),
            };
            out.ids.push(self.cls_id);
            out.type_ids.push(0);
            out.offsets.push((0, 0));
            out.attention_mask.push(1);
            for (i, id) in a.ids.iter().enumerate() {
                out.ids.push(*id);
                out.type_ids.push(0);
                out.offsets.push(a.offsets[i]);
                out.attention_mask.push(1);
            }
            out.ids.push(self.sep_id);
            out.type_ids.push(0);
            out.offsets.push((0, 0));
            out.attention_mask.push(1);
            for (i, id) in b.ids.iter().enumerate() {
                out.ids.push(*id);
                out.type_ids.push(1);
                out.offsets.push(b.offsets[i]);
                out.attention_mask.push(1);
            }
            out.ids.push(self.sep_id);
            out.type_ids.push(1);
            out.offsets.push((0, 0));
            out.attention_mask.push(1);
            out
        } else {
            let mut out = a;
            for (i, id) in b.ids.iter().enumerate() {
                out.ids.push(*id);
                out.type_ids.push(1);
                out.offsets.push(b.offsets[i]);
                out.attention_mask.push(1);
            }
            out
        };
        self.apply_padding(&mut out);
        Ok(out)
    }

    fn encode_raw(&self, text: &str) -> Result<Encoding> {
        let (normalized, alignment) = self.normalizer.normalize(text);
        let pre_tokens = self.pretokenizer.pre_tokenize(&normalized, &alignment);

        let mut ids: Vec<TokenId> = Vec::new();
        let mut offsets: Vec<(usize, usize)> = Vec::new();

        for pt in pre_tokens {
            let sub_ids = self.wordpiece.tokenize(pt.text);
            for id in sub_ids {
                ids.push(id);
                offsets.push(pt.offsets);
            }
        }
        let type_ids = vec![0u32; ids.len()];
        let attention_mask = vec![1u32; ids.len()];
        Ok(Encoding {
            ids,
            type_ids,
            offsets,
            attention_mask,
        })
    }

    pub fn decode(&self, ids: &[TokenId]) -> Result<String> {
        let mut out = String::new();
        let mut first = true;
        for &id in ids {
            let idx = id as usize;
            if idx >= self.reverse_vocab.len() {
                return Err(TokenizerError::VocabInvalid {
                    reason: format!(
                        "id {id} out of range (vocab size {})",
                        self.reverse_vocab.len()
                    ),
                });
            }
            let tok = &self.reverse_vocab[idx];
            if id == self.cls_id || id == self.sep_id || id == self.pad_id {
                continue;
            }
            if let Some(rest) = tok.strip_prefix(&self.continuing_prefix) {
                out.push_str(rest);
            } else {
                if !first {
                    out.push(' ');
                }
                out.push_str(tok);
            }
            first = false;
        }
        Ok(out)
    }

    pub fn vocab_size(&self) -> usize {
        self.reverse_vocab.len()
    }
    pub fn token_id(&self, s: &str) -> Option<TokenId> {
        self.vocab.get(s).copied()
    }
    pub fn id_to_token(&self, id: TokenId) -> Option<&str> {
        self.reverse_vocab.get(id as usize).map(|s| s.as_str())
    }
    pub fn cls_id(&self) -> TokenId {
        self.cls_id
    }
    pub fn sep_id(&self) -> TokenId {
        self.sep_id
    }
    pub fn pad_id(&self) -> TokenId {
        self.pad_id
    }
    pub fn unk_id(&self) -> TokenId {
        self.unk_id
    }
    pub fn mask_id(&self) -> TokenId {
        self.mask_id
    }
    pub fn max_length(&self) -> usize {
        self.truncation.max_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        NormalizerDisk, PaddingDirectionDisk, PaddingDisk, PaddingStrategyDisk, PreTokenizerDisk,
        SpecialTokensDisk, TruncationDisk, WordPieceDisk,
    };

    fn tiny_tokenizer() -> SovereignTokenizer {
        tiny_tokenizer_with_padding(PaddingDisk::default())
    }

    fn tiny_tokenizer_with_padding(padding: PaddingDisk) -> SovereignTokenizer {
        let data = SovereignTokenizerData {
            vocab: vec![
                "[PAD]".into(),
                "[UNK]".into(),
                "[CLS]".into(),
                "[SEP]".into(),
                "[MASK]".into(),
                "hello".into(),
                "world".into(),
                "un".into(),
                "##afford".into(),
                "##able".into(),
                ",".into(),
                "!".into(),
            ],
            normalizer: NormalizerDisk {
                lowercase: true,
                strip_accents: true,
                handle_chinese_chars: true,
                clean_text: true,
            },
            pretokenizer: PreTokenizerDisk {
                do_basic_split: true,
                never_split: vec![
                    "[CLS]".into(),
                    "[SEP]".into(),
                    "[PAD]".into(),
                    "[UNK]".into(),
                    "[MASK]".into(),
                ],
            },
            wordpiece: WordPieceDisk {
                unk_token: "[UNK]".into(),
                continuing_subword_prefix: "##".into(),
                max_input_chars_per_word: 100,
            },
            truncation: TruncationDisk {
                max_length: 16,
                strategy: TruncationStrategyDisk::LongestFirst,
                stride: 0,
            },
            special_tokens: SpecialTokensDisk {
                cls: "[CLS]".into(),
                sep: "[SEP]".into(),
                pad: "[PAD]".into(),
                unk: "[UNK]".into(),
                mask: "[MASK]".into(),
            },
            padding,
        };
        SovereignTokenizer::from_data(data).expect("construct")
    }

    #[test]
    fn send_sync_clone() {
        fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
        assert_send_sync_clone::<SovereignTokenizer>();
    }

    #[test]
    fn encode_with_specials_wraps_in_cls_sep() {
        let tok = tiny_tokenizer();
        let enc = tok.encode("hello world", true).expect("encode");
        assert_eq!(enc.ids.first().copied(), Some(tok.cls_id()));
        assert_eq!(enc.ids.last().copied(), Some(tok.sep_id()));
    }

    #[test]
    fn encode_unaffordable_uses_wordpiece() {
        let tok = tiny_tokenizer();
        let enc = tok.encode("unaffordable", false).expect("encode");
        assert_eq!(
            enc.ids,
            vec![
                tok.token_id("un").unwrap(),
                tok.token_id("##afford").unwrap(),
                tok.token_id("##able").unwrap(),
            ]
        );
    }

    #[test]
    fn vocab_round_trip() {
        let tok = tiny_tokenizer();
        for s in [
            "hello", "world", "un", "##afford", "##able", "[CLS]", "[SEP]", "[UNK]",
        ] {
            let id = tok
                .token_id(s)
                .unwrap_or_else(|| panic!("missing token: {s}"));
            assert_eq!(tok.id_to_token(id).unwrap(), s);
        }
    }

    #[test]
    fn decode_drops_specials_and_strips_prefix() {
        let tok = tiny_tokenizer();
        let enc = tok.encode("unaffordable", true).expect("encode");
        let out = tok.decode(&enc.ids).expect("decode");
        assert_eq!(out, "unaffordable");
    }

    #[test]
    fn decode_whitespace_shape_matches_hf_lossy_collapse() {
        let tok = tiny_tokenizer();

        let enc = tok.encode("Hello   world", false).expect("encode");
        let out = tok.decode(&enc.ids).expect("decode");
        assert_eq!(out, "hello world");

        let enc = tok.encode("\tHello\nworld", false).expect("encode");
        let out = tok.decode(&enc.ids).expect("decode");
        assert_eq!(out, "hello world");

        let enc = tok.encode("  hello  world", false).expect("encode");
        let out = tok.decode(&enc.ids).expect("decode");
        assert_eq!(out, "hello world");
    }

    #[test]
    fn encode_pair_has_two_seps() {
        let tok = tiny_tokenizer();
        let enc = tok
            .encode_pair("hello", "world", true)
            .expect("encode_pair");
        let seps: Vec<usize> = enc
            .ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == tok.sep_id())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(seps.len(), 2);
    }

    #[test]
    fn encode_pair_applies_fixed_padding() {
        let tok = tiny_tokenizer_with_padding(PaddingDisk {
            enabled: true,
            strategy: PaddingStrategyDisk::Fixed(8),
            direction: PaddingDirectionDisk::Right,
            pad_id: 0,
            pad_type_id: 0,
        });
        let enc = tok
            .encode_pair("hello", "world", true)
            .expect("encode_pair");
        assert_eq!(enc.ids.len(), 8);
        assert_eq!(enc.type_ids.len(), 8);
        assert_eq!(enc.offsets.len(), 8);
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 1, 1, 0, 0, 0]);
        assert_eq!(&enc.ids[5..], &[tok.pad_id(), tok.pad_id(), tok.pad_id()]);
    }

    #[test]
    fn padding_pad_id_must_match_special_pad() {
        let mut data = SovereignTokenizerData {
            vocab: vec![
                "[PAD]".into(),
                "[UNK]".into(),
                "[CLS]".into(),
                "[SEP]".into(),
                "[MASK]".into(),
                "hello".into(),
            ],
            normalizer: NormalizerDisk {
                lowercase: true,
                strip_accents: true,
                handle_chinese_chars: true,
                clean_text: true,
            },
            pretokenizer: PreTokenizerDisk {
                do_basic_split: true,
                never_split: vec![],
            },
            wordpiece: WordPieceDisk {
                unk_token: "[UNK]".into(),
                continuing_subword_prefix: "##".into(),
                max_input_chars_per_word: 100,
            },
            truncation: TruncationDisk {
                max_length: 16,
                strategy: TruncationStrategyDisk::LongestFirst,
                stride: 0,
            },
            special_tokens: SpecialTokensDisk {
                cls: "[CLS]".into(),
                sep: "[SEP]".into(),
                pad: "[PAD]".into(),
                unk: "[UNK]".into(),
                mask: "[MASK]".into(),
            },
            padding: PaddingDisk::default(),
        };
        data.padding.enabled = true;
        data.padding.strategy = PaddingStrategyDisk::Fixed(8);
        data.padding.pad_id = 99;

        let err = SovereignTokenizer::from_data(data).expect_err("bad pad id");
        assert!(matches!(err, TokenizerError::VocabInvalid { .. }));
    }

    #[test]
    fn missing_special_in_vocab_rejected() {
        let data = SovereignTokenizerData {
            vocab: vec!["[PAD]".into(), "[UNK]".into(), "hello".into()],
            normalizer: NormalizerDisk {
                lowercase: true,
                strip_accents: true,
                handle_chinese_chars: true,
                clean_text: true,
            },
            pretokenizer: PreTokenizerDisk {
                do_basic_split: true,
                never_split: vec![],
            },
            wordpiece: WordPieceDisk {
                unk_token: "[UNK]".into(),
                continuing_subword_prefix: "##".into(),
                max_input_chars_per_word: 100,
            },
            truncation: TruncationDisk {
                max_length: 16,
                strategy: TruncationStrategyDisk::LongestFirst,
                stride: 0,
            },
            special_tokens: SpecialTokensDisk {
                cls: "[CLS]".into(),
                sep: "[SEP]".into(),
                pad: "[PAD]".into(),
                unk: "[UNK]".into(),
                mask: "[MASK]".into(),
            },
            padding: PaddingDisk::default(),
        };
        let err = SovereignTokenizer::from_data(data).expect_err("missing CLS");
        assert!(matches!(err, TokenizerError::SpecialTokenMissing { .. }));
    }
}
