//! BertPreTokenizer — whitespace + punctuation + CJK-codepoint splitter, no regex.

use crate::normalizer::Alignment;

/// A single pre-token with offsets into the ORIGINAL input (via alignment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreToken<'b> {
    pub text: &'b str,
    pub offsets: (usize, usize),
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreTokConfig {
    pub do_basic_split: bool,
    pub never_split: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PreTokenizer {
    cfg: PreTokConfig,
}

impl PreTokenizer {
    pub fn new(cfg: PreTokConfig) -> Self {
        Self { cfg }
    }

    pub fn pre_tokenize<'n>(
        &self,
        normalized: &'n str,
        alignment: &Alignment,
    ) -> Vec<PreToken<'n>> {
        if !self.cfg.do_basic_split {
            if normalized.is_empty() {
                return Vec::new();
            }
            return vec![PreToken {
                text: normalized,
                offsets: (0, alignment.to_original(normalized.len())),
            }];
        }

        let mut tokens: Vec<PreToken<'n>> = Vec::new();
        let mut i = 0usize;

        while i < normalized.len() {
            if let Some(nsp) = self.match_never_split(normalized, i) {
                let end = i + nsp.len();
                tokens.push(PreToken {
                    text: &normalized[i..end],
                    offsets: (alignment.to_original(i), alignment.to_original(end)),
                });
                i = end;
                continue;
            }
            let ch = match normalized[i..].chars().next() {
                Some(c) => c,
                None => break,
            };
            if ch.is_whitespace() {
                i += ch.len_utf8();
                continue;
            }
            if is_punct(ch) {
                let end = i + ch.len_utf8();
                tokens.push(PreToken {
                    text: &normalized[i..end],
                    offsets: (alignment.to_original(i), alignment.to_original(end)),
                });
                i = end;
                continue;
            }
            let start = i;
            while i < normalized.len() {
                let c = match normalized[i..].chars().next() {
                    Some(c) => c,
                    None => break,
                };
                if c.is_whitespace()
                    || is_punct(c)
                    || self.match_never_split(normalized, i).is_some()
                {
                    break;
                }
                i += c.len_utf8();
            }
            if start != i {
                tokens.push(PreToken {
                    text: &normalized[start..i],
                    offsets: (alignment.to_original(start), alignment.to_original(i)),
                });
            }
        }

        tokens
    }

    fn match_never_split<'a>(&'a self, s: &str, at: usize) -> Option<&'a String> {
        self.cfg
            .never_split
            .iter()
            .filter(|ns| !ns.is_empty())
            .find(|ns| s[at..].starts_with(ns.as_str()))
    }
}

/// Matches HF BertPreTokenizer's `is_bert_punc`: ASCII punct OR Unicode
/// category P* (Pc/Pd/Pe/Pf/Pi/Po/Ps).
/// Authoritative: vendor/tokenizers/.../pre_tokenizers/bert.rs:5-7.
fn is_punct(ch: char) -> bool {
    use unicode_categories::UnicodeCategories;
    ch.is_ascii_punctuation() || ch.is_punctuation()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id_align(s: &str) -> Alignment {
        Alignment::Identity { len: s.len() }
    }

    #[test]
    fn whitespace_splits() {
        let pt = PreTokenizer::new(PreTokConfig {
            do_basic_split: true,
            never_split: vec![],
        });
        let norm = "hello world";
        let out = pt.pre_tokenize(norm, &id_align(norm));
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "hello");
        assert_eq!(out[0].offsets, (0, 5));
        assert_eq!(out[1].text, "world");
        assert_eq!(out[1].offsets, (6, 11));
    }

    #[test]
    fn punct_separated() {
        let pt = PreTokenizer::new(PreTokConfig {
            do_basic_split: true,
            never_split: vec![],
        });
        let norm = "hello, world!";
        let out = pt.pre_tokenize(norm, &id_align(norm));
        let texts: Vec<&str> = out.iter().map(|t| t.text).collect();
        assert_eq!(texts, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn never_split_preserved() {
        let pt = PreTokenizer::new(PreTokConfig {
            do_basic_split: true,
            never_split: vec!["[CLS]".into(), "[SEP]".into()],
        });
        let norm = "[CLS] hello [SEP]";
        let out = pt.pre_tokenize(norm, &id_align(norm));
        let texts: Vec<&str> = out.iter().map(|t| t.text).collect();
        assert_eq!(texts, vec!["[CLS]", "hello", "[SEP]"]);
    }

    #[test]
    fn empty_never_split_entry_is_ignored() {
        let pt = PreTokenizer::new(PreTokConfig {
            do_basic_split: true,
            never_split: vec!["".into()],
        });
        let norm = "hello";
        let out = pt.pre_tokenize(norm, &id_align(norm));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "hello");
    }

    #[test]
    fn offsets_respect_alignment() {
        let norm = "cafe";
        let align = Alignment::Explicit(vec![0, 1, 2, 3, 5]);
        let pt = PreTokenizer::new(PreTokConfig {
            do_basic_split: true,
            never_split: vec![],
        });
        let out = pt.pre_tokenize(norm, &align);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].offsets, (0, 5));
    }

    #[test]
    fn empty_input() {
        let pt = PreTokenizer::new(PreTokConfig {
            do_basic_split: true,
            never_split: vec![],
        });
        let out = pt.pre_tokenize("", &id_align(""));
        assert!(out.is_empty());
    }

    #[test]
    fn do_basic_split_off_returns_whole_input() {
        let pt = PreTokenizer::new(PreTokConfig {
            do_basic_split: false,
            never_split: vec![],
        });
        let norm = "hello world!";
        let out = pt.pre_tokenize(norm, &id_align(norm));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "hello world!");
    }
}
