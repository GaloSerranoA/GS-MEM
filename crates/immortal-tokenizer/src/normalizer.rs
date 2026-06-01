//! BertNormalizer + Alignment (offset bookkeeping across normalization).

use std::borrow::Cow;

/// Maps byte offsets in the normalized string back to byte offsets in the original input.
///
/// Invariants:
/// - For `Identity { len }`, `to_original(n) == n.min(len)` for all `n`.
/// - For `Explicit(map)`, `map.len() == normalized.len() + 1` (one-past-end sentinel).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Alignment {
    Identity { len: usize },
    Explicit(Vec<usize>),
}

impl Alignment {
    pub fn to_original(&self, norm_byte: usize) -> usize {
        match self {
            Self::Identity { len } => norm_byte.min(*len),
            Self::Explicit(map) => {
                if map.is_empty() {
                    0
                } else {
                    map[norm_byte.min(map.len() - 1)]
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Identity { len } => *len,
            Self::Explicit(map) => map.len().saturating_sub(1),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct NormalizerConfig {
    pub lowercase: bool,
    pub strip_accents: bool,
    pub handle_chinese_chars: bool,
    pub clean_text: bool,
}

impl NormalizerConfig {
    pub const fn identity() -> Self {
        Self {
            lowercase: false,
            strip_accents: false,
            handle_chinese_chars: false,
            clean_text: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Normalizer {
    cfg: NormalizerConfig,
}

impl Normalizer {
    pub fn new(cfg: NormalizerConfig) -> Self {
        Self { cfg }
    }

    /// Normalize input, returning `(normalized, alignment_map_into_original)`.
    ///
    /// Zero-copy happy path: if no transformations apply, returns
    /// `Cow::Borrowed` + `Alignment::Identity` with no allocation.
    pub fn normalize<'a>(&self, input: &'a str) -> (Cow<'a, str>, Alignment) {
        // Fast path 1: no flags → pure zero-copy.
        if !self.cfg.lowercase
            && !self.cfg.strip_accents
            && !self.cfg.handle_chinese_chars
            && !self.cfg.clean_text
        {
            return (
                Cow::Borrowed(input),
                Alignment::Identity { len: input.len() },
            );
        }
        // Fast path 2: only lowercase + pure ASCII + already lowercase.
        if self.cfg.lowercase
            && !self.cfg.strip_accents
            && !self.cfg.handle_chinese_chars
            && !self.cfg.clean_text
            && input.is_ascii()
            && !input.bytes().any(|b| b.is_ascii_uppercase())
        {
            return (
                Cow::Borrowed(input),
                Alignment::Identity { len: input.len() },
            );
        }

        self.slow_normalize(input)
    }

    fn slow_normalize<'a>(&self, input: &'a str) -> (Cow<'a, str>, Alignment) {
        use unicode_normalization::UnicodeNormalization;

        let mut out = String::with_capacity(input.len());
        let mut map: Vec<usize> = Vec::with_capacity(input.len() + 1);

        let mut orig_byte = 0usize;
        for ch in input.chars() {
            let orig_ch_len = ch.len_utf8();

            // 1. clean_text: drop invalid / BOM / control chars (not tab/newline/CR).
            if self.cfg.clean_text && should_clean(ch) {
                orig_byte += orig_ch_len;
                continue;
            }

            // 2. clean_text: map any whitespace to ' ' (HF MAPS, does NOT collapse).
            //    Authoritative: vendor/tokenizers/tokenizers/src/normalizers/bert.rs:92-96.
            let ch = if self.cfg.clean_text && is_hf_whitespace(ch) {
                ' '
            } else {
                ch
            };

            // 3. CJK char: unconditionally wrap in spaces (HF emits
            //    ` c ` per CJK char; vendor/tokenizers/.../bert.rs:98-108).
            if self.cfg.handle_chinese_chars && is_chinese_char(ch) {
                map.push(orig_byte);
                out.push(' ');
                for _ in 0..ch.len_utf8() {
                    map.push(orig_byte);
                }
                out.push(ch);
                map.push(orig_byte + orig_ch_len);
                out.push(' ');
                orig_byte += orig_ch_len;
                continue;
            }

            // 4. NFD + optional strip_accents / lowercase.
            if self.cfg.strip_accents {
                for nfd_ch in ch.nfd() {
                    if unicode_normalization::char::is_combining_mark(nfd_ch) {
                        continue;
                    }
                    if self.cfg.lowercase {
                        push_lowercase_mapped(&mut out, &mut map, nfd_ch, orig_byte);
                    } else {
                        push_mapped_char(&mut out, &mut map, nfd_ch, orig_byte);
                    }
                }
            } else {
                if self.cfg.lowercase {
                    push_lowercase_mapped(&mut out, &mut map, ch, orig_byte);
                } else {
                    push_mapped_char(&mut out, &mut map, ch, orig_byte);
                }
            }
            orig_byte += orig_ch_len;
        }

        // HF does NOT trim trailing whitespace; leave as-is.
        map.push(orig_byte);
        (Cow::Owned(out), Alignment::Explicit(map))
    }
}

/// Matches HF `bert::is_control`: true for Unicode "Other" category chars
/// (Cc/Cf/Cn/Co) with the hard-coded exceptions `\t`/`\n`/`\r`.
/// Authoritative: vendor/tokenizers/.../bert.rs:15-25.
fn should_clean(ch: char) -> bool {
    if ch as u32 == 0 || ch == '\u{FFFD}' {
        return true;
    }
    if matches!(ch, '\t' | '\n' | '\r') {
        return false;
    }
    use unicode_categories::UnicodeCategories;
    ch.is_other()
}

/// HF whitespace: \t \n \r plus anything `char::is_whitespace` recognizes.
fn is_hf_whitespace(ch: char) -> bool {
    matches!(ch, '\t' | '\n' | '\r') || ch.is_whitespace()
}

fn is_chinese_char(ch: char) -> bool {
    let c = ch as u32;
    matches!(
        c,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

fn push_lowercase_mapped(out: &mut String, map: &mut Vec<usize>, ch: char, orig_byte: usize) {
    for lower in ch.to_lowercase() {
        push_mapped_char(out, map, lower, orig_byte);
    }
}

fn push_mapped_char(out: &mut String, map: &mut Vec<usize>, ch: char, orig_byte: usize) {
    for _ in 0..ch.len_utf8() {
        map.push(orig_byte);
    }
    out.push(ch);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_alignment_maps_byte_for_byte() {
        let a = Alignment::Identity { len: 5 };
        for i in 0..=5 {
            assert_eq!(a.to_original(i), i);
        }
        assert_eq!(a.to_original(99), 5);
    }

    #[test]
    fn explicit_alignment_respects_sentinel() {
        let a = Alignment::Explicit(vec![0, 2, 3]);
        assert_eq!(a.to_original(0), 0);
        assert_eq!(a.to_original(1), 2);
        assert_eq!(a.to_original(2), 3);
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn alignment_empty() {
        assert!(Alignment::Identity { len: 0 }.is_empty());
        assert!(Alignment::Explicit(vec![0]).is_empty());
    }

    #[test]
    fn identity_config_is_noop() {
        let n = Normalizer::new(NormalizerConfig::identity());
        let (out, align) = n.normalize("hello");
        assert!(matches!(out, Cow::Borrowed(_)));
        assert_eq!(&*out, "hello");
        assert!(matches!(align, Alignment::Identity { len: 5 }));
    }

    #[test]
    fn lowercase_ascii() {
        let n = Normalizer::new(NormalizerConfig {
            lowercase: true,
            strip_accents: false,
            handle_chinese_chars: false,
            clean_text: false,
        });
        let (out, _) = n.normalize("Hello WORLD");
        assert_eq!(&*out, "hello world");
    }

    #[test]
    fn lowercase_preserves_multi_char_expansion_and_alignment() {
        let n = Normalizer::new(NormalizerConfig {
            lowercase: true,
            strip_accents: false,
            handle_chinese_chars: false,
            clean_text: false,
        });
        let (out, align) = n.normalize("İ");
        assert_eq!(&*out, "i\u{307}");
        assert_eq!(align.len(), out.len());
        assert_eq!(align.to_original(0), 0);
        assert_eq!(align.to_original(1), 0);
        assert_eq!(align.to_original(2), 0);
        assert_eq!(align.to_original(out.len()), "İ".len());
    }

    #[test]
    fn strip_accents_nfd() {
        let n = Normalizer::new(NormalizerConfig {
            lowercase: false,
            strip_accents: true,
            handle_chinese_chars: false,
            clean_text: false,
        });
        let (out, _) = n.normalize("café");
        assert_eq!(&*out, "cafe");
    }

    #[test]
    fn cjk_wrap() {
        let n = Normalizer::new(NormalizerConfig {
            lowercase: false,
            strip_accents: false,
            handle_chinese_chars: true,
            clean_text: false,
        });
        let (out, _) = n.normalize("hello世界");
        assert_eq!(&*out, "hello 世  界 ");
    }

    #[test]
    fn clean_text_maps_whitespace_but_does_not_collapse() {
        // HF BertNormalizer::do_clean_text filters control chars and MAPS any
        // whitespace to ' '. It does NOT collapse multiple whitespace.
        // Authoritative: vendor/tokenizers/.../bert.rs:92-96.
        let n = Normalizer::new(NormalizerConfig {
            lowercase: false,
            strip_accents: false,
            handle_chinese_chars: false,
            clean_text: true,
        });
        let (out, _) = n.normalize("hello   world\t\t");
        assert_eq!(&*out, "hello   world  ");
    }

    #[test]
    fn alignment_maps_accented_chars_to_original_bytes() {
        let n = Normalizer::new(NormalizerConfig {
            lowercase: false,
            strip_accents: true,
            handle_chinese_chars: false,
            clean_text: false,
        });
        let (out, align) = n.normalize("café");
        assert_eq!(&*out, "cafe");
        assert_eq!(align.to_original(0), 0);
        assert_eq!(align.to_original(3), 3);
    }

    #[test]
    fn bert_full_pipeline_matches_hf_semantics() {
        // Walks the full HF BertNormalizer pipeline for a multi-feature input.
        // Order per vendor/tokenizers/.../bert.rs:119-136:
        //   clean_text → handle_chinese_chars → strip_accents → lowercase.
        // Expected output derived by hand-tracing each stage against the HF source.
        let n = Normalizer::new(NormalizerConfig {
            lowercase: true,
            strip_accents: true,
            handle_chinese_chars: true,
            clean_text: true,
        });
        let (out, _) = n.normalize("Héllo\u{0}WORLD 世界");
        // 1. clean_text drops \u{0}, maps ' '→' ' → "HélloWORLD 世界"
        // 2. handle_chinese_chars wraps 世,界 unconditionally → "HélloWORLD  世  界 "
        // 3. strip_accents NFD(é)='e'+mark, drop mark → "HelloWORLD  世  界 "
        // 4. lowercase → "helloworld  世  界 "
        assert_eq!(&*out, "helloworld  世  界 ");
    }
}
