//! Truncation strategies operating on `Encoding`s.

use crate::error::{Result, TokenizerError};
use crate::tokenizer::Encoding;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TruncationConfig {
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

impl TruncationConfig {
    pub const fn default_512() -> Self {
        Self {
            max_length: 512,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        }
    }
}

pub struct Truncation;

impl Truncation {
    pub fn apply_single(enc: &mut Encoding, cfg: &TruncationConfig, reserved: usize) -> Result<()> {
        if cfg.max_length <= reserved {
            return Err(TokenizerError::TruncationImpossible {
                strategy: cfg.strategy,
                max_length: cfg.max_length,
                special_tokens: reserved,
            });
        }
        let budget = cfg.max_length - reserved;
        if enc.ids.len() > budget {
            enc.ids.truncate(budget);
            enc.type_ids.truncate(budget);
            enc.offsets.truncate(budget);
            enc.attention_mask.truncate(budget);
        }
        Ok(())
    }

    pub fn apply_pair(
        a: &mut Encoding,
        b: &mut Encoding,
        cfg: &TruncationConfig,
        reserved: usize,
    ) -> Result<()> {
        if cfg.max_length <= reserved {
            return Err(TokenizerError::TruncationImpossible {
                strategy: cfg.strategy,
                max_length: cfg.max_length,
                special_tokens: reserved,
            });
        }
        let budget = cfg.max_length - reserved;
        if a.ids.len() + b.ids.len() <= budget {
            return Ok(());
        }

        match cfg.strategy {
            TruncationStrategy::LongestFirst => {
                while a.ids.len() + b.ids.len() > budget {
                    if a.ids.is_empty() && b.ids.is_empty() {
                        break;
                    }
                    if a.ids.len() >= b.ids.len() {
                        pop(a);
                    } else {
                        pop(b);
                    }
                }
            }
            TruncationStrategy::OnlyFirst => {
                if b.ids.len() >= budget {
                    return Err(TokenizerError::TruncationImpossible {
                        strategy: cfg.strategy,
                        max_length: cfg.max_length,
                        special_tokens: reserved,
                    });
                }
                let keep = budget - b.ids.len();
                a.ids.truncate(keep);
                a.type_ids.truncate(keep);
                a.offsets.truncate(keep);
                a.attention_mask.truncate(keep);
            }
            TruncationStrategy::OnlySecond => {
                if a.ids.len() >= budget {
                    return Err(TokenizerError::TruncationImpossible {
                        strategy: cfg.strategy,
                        max_length: cfg.max_length,
                        special_tokens: reserved,
                    });
                }
                let keep = budget - a.ids.len();
                b.ids.truncate(keep);
                b.type_ids.truncate(keep);
                b.offsets.truncate(keep);
                b.attention_mask.truncate(keep);
            }
        }
        Ok(())
    }
}

fn pop(e: &mut Encoding) {
    e.ids.pop();
    e.type_ids.pop();
    e.offsets.pop();
    e.attention_mask.pop();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(ids: &[u32]) -> Encoding {
        Encoding {
            ids: ids.to_vec(),
            type_ids: vec![0; ids.len()],
            offsets: (0..ids.len()).map(|i| (i, i + 1)).collect(),
            attention_mask: vec![1; ids.len()],
        }
    }

    #[test]
    fn single_respects_reserved() {
        let mut e = enc(&[10, 20, 30, 40, 50, 60]);
        let cfg = TruncationConfig {
            max_length: 5,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        };
        Truncation::apply_single(&mut e, &cfg, 2).expect("ok");
        assert_eq!(e.ids, vec![10, 20, 30]);
    }

    #[test]
    fn single_impossible_when_budget_negative() {
        let mut e = enc(&[10]);
        let cfg = TruncationConfig {
            max_length: 2,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        };
        let err = Truncation::apply_single(&mut e, &cfg, 2).expect_err("impossible");
        assert!(matches!(err, TokenizerError::TruncationImpossible { .. }));
    }

    #[test]
    fn longest_first_pair_alternates() {
        let mut a = enc(&[1, 2, 3, 4]);
        let mut b = enc(&[10, 20]);
        let cfg = TruncationConfig {
            max_length: 5,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        };
        Truncation::apply_pair(&mut a, &mut b, &cfg, 3).expect("ok");
        assert_eq!(a.ids.len() + b.ids.len(), 2);
    }

    #[test]
    fn only_first_truncates_first() {
        let mut a = enc(&[1, 2, 3, 4]);
        let mut b = enc(&[10]);
        let cfg = TruncationConfig {
            max_length: 3,
            strategy: TruncationStrategy::OnlyFirst,
            stride: 0,
        };
        Truncation::apply_pair(&mut a, &mut b, &cfg, 0).expect("ok");
        assert_eq!(a.ids, vec![1, 2]);
        assert_eq!(b.ids, vec![10]);
    }

    #[test]
    fn only_first_errors_if_second_already_over_budget() {
        let mut a = enc(&[1]);
        let mut b = enc(&[10, 20, 30]);
        let cfg = TruncationConfig {
            max_length: 3,
            strategy: TruncationStrategy::OnlyFirst,
            stride: 0,
        };
        let err = Truncation::apply_pair(&mut a, &mut b, &cfg, 0).expect_err("impossible");
        assert!(matches!(err, TokenizerError::TruncationImpossible { .. }));
    }
}
