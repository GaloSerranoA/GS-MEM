//! On-disk `.sovereign-tokenizer` codec — the ONLY module aware of serialization layout.
//!
//! File layout (little-endian):
//!   bytes 0..5    : magic "IMTK\0"
//!   bytes 5..7    : major version (u16 LE)
//!   bytes 7..9    : minor version (u16 LE)
//!   bytes 9..13   : payload length (u32 LE)
//!   bytes 13..    : bincode-encoded SovereignTokenizerData

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Result, TokenizerError};

pub const FORMAT_MAGIC: &[u8; 5] = b"IMTK\0";
pub const FORMAT_VERSION_MAJOR: u16 = 1;
pub const FORMAT_VERSION_MINOR: u16 = 0;
const HEADER_LEN: usize = 5 + 2 + 2 + 4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SovereignTokenizerData {
    pub vocab: Vec<String>,
    pub normalizer: NormalizerDisk,
    pub pretokenizer: PreTokenizerDisk,
    pub wordpiece: WordPieceDisk,
    pub truncation: TruncationDisk,
    pub special_tokens: SpecialTokensDisk,
    #[serde(default)]
    pub padding: PaddingDisk,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PaddingDisk {
    /// If false, padding is skipped entirely (backwards-compat with files
    /// produced before this field existed — bincode default fills `false`).
    pub enabled: bool,
    pub strategy: PaddingStrategyDisk,
    pub direction: PaddingDirectionDisk,
    pub pad_id: u32,
    pub pad_type_id: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum PaddingStrategyDisk {
    #[default]
    Disabled,
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum PaddingDirectionDisk {
    #[default]
    Right,
    Left,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NormalizerDisk {
    pub lowercase: bool,
    pub strip_accents: bool,
    pub handle_chinese_chars: bool,
    pub clean_text: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreTokenizerDisk {
    pub do_basic_split: bool,
    pub never_split: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WordPieceDisk {
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TruncationStrategyDisk {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TruncationDisk {
    pub max_length: usize,
    pub strategy: TruncationStrategyDisk,
    pub stride: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpecialTokensDisk {
    pub cls: String,
    pub sep: String,
    pub pad: String,
    pub unk: String,
    pub mask: String,
}

pub fn load(path: impl AsRef<Path>) -> Result<SovereignTokenizerData> {
    let bytes = std::fs::read(path.as_ref())?;
    load_from_bytes(&bytes)
}

pub fn load_from_bytes(bytes: &[u8]) -> Result<SovereignTokenizerData> {
    if bytes.len() < HEADER_LEN {
        return Err(TokenizerError::HeaderCorrupt {
            reason: format!(
                "file too small: {} bytes < {} header bytes",
                bytes.len(),
                HEADER_LEN
            ),
        });
    }
    if &bytes[0..5] != FORMAT_MAGIC {
        return Err(TokenizerError::HeaderCorrupt {
            reason: format!(
                "bad magic: expected {:?}, got {:?}",
                FORMAT_MAGIC,
                &bytes[0..5]
            ),
        });
    }
    let major = u16::from_le_bytes([bytes[5], bytes[6]]);
    let minor = u16::from_le_bytes([bytes[7], bytes[8]]);
    if major != FORMAT_VERSION_MAJOR {
        return Err(TokenizerError::UnsupportedFormatVersion {
            found_major: major,
            supported_major: FORMAT_VERSION_MAJOR,
        });
    }
    if minor > FORMAT_VERSION_MINOR {
        tracing::warn!(
            "loading tokenizer with newer minor version {minor} > {FORMAT_VERSION_MINOR}; additive fields may be ignored"
        );
    }
    let payload_len = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]) as usize;
    let payload_end =
        HEADER_LEN
            .checked_add(payload_len)
            .ok_or_else(|| TokenizerError::HeaderCorrupt {
                reason: format!("declared payload length {payload_len} overflows header size"),
            })?;
    if payload_end > bytes.len() {
        return Err(TokenizerError::HeaderCorrupt {
            reason: format!(
                "declared payload length {payload_len} exceeds available bytes {}",
                bytes.len() - HEADER_LEN
            ),
        });
    }
    let payload = &bytes[HEADER_LEN..payload_end];
    let (data, _): (SovereignTokenizerData, _) =
        bincode::serde::decode_from_slice(payload, bincode::config::standard())?;
    Ok(data)
}

pub fn save(data: &SovereignTokenizerData, path: impl AsRef<Path>) -> Result<()> {
    let bytes = save_to_bytes(data)?;
    std::fs::write(path.as_ref(), bytes)?;
    Ok(())
}

pub fn save_to_bytes(data: &SovereignTokenizerData) -> Result<Vec<u8>> {
    let payload = bincode::serde::encode_to_vec(data, bincode::config::standard())?;
    let payload_len = u32::try_from(payload.len()).map_err(|_| TokenizerError::PayloadCorrupt {
        reason: format!("payload length {} exceeds u32::MAX", payload.len()),
    })?;

    let mut buf = Vec::with_capacity(HEADER_LEN + payload.len());
    buf.write_all(FORMAT_MAGIC)?;
    buf.write_all(&FORMAT_VERSION_MAJOR.to_le_bytes())?;
    buf.write_all(&FORMAT_VERSION_MINOR.to_le_bytes())?;
    buf.write_all(&payload_len.to_le_bytes())?;
    buf.write_all(&payload)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> SovereignTokenizerData {
        SovereignTokenizerData {
            vocab: vec![
                "[PAD]".into(),
                "[UNK]".into(),
                "[CLS]".into(),
                "[SEP]".into(),
                "[MASK]".into(),
                "hello".into(),
                "##o".into(),
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
                    "[UNK]".into(),
                    "[CLS]".into(),
                    "[SEP]".into(),
                    "[PAD]".into(),
                    "[MASK]".into(),
                ],
            },
            wordpiece: WordPieceDisk {
                unk_token: "[UNK]".into(),
                continuing_subword_prefix: "##".into(),
                max_input_chars_per_word: 100,
            },
            truncation: TruncationDisk {
                max_length: 512,
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
        }
    }

    #[test]
    fn round_trip_preserves_data() {
        let data = sample_data();
        let bytes = save_to_bytes(&data).expect("save");
        let loaded = load_from_bytes(&bytes).expect("load");
        assert_eq!(data, loaded);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = save_to_bytes(&sample_data()).expect("save");
        bytes[0] = b'X';
        let err = load_from_bytes(&bytes).expect_err("should fail");
        assert!(matches!(err, TokenizerError::HeaderCorrupt { .. }));
    }

    #[test]
    fn wrong_major_rejected() {
        let mut bytes = save_to_bytes(&sample_data()).expect("save");
        bytes[5] = 99;
        let err = load_from_bytes(&bytes).expect_err("should fail");
        match err {
            TokenizerError::UnsupportedFormatVersion {
                found_major,
                supported_major,
            } => {
                assert_eq!(found_major, 99);
                assert_eq!(supported_major, FORMAT_VERSION_MAJOR);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn higher_minor_warns_but_loads() {
        let mut bytes = save_to_bytes(&sample_data()).expect("save");
        bytes[7] = 99;
        let loaded = load_from_bytes(&bytes).expect("should still load");
        assert_eq!(loaded.vocab.len(), 7);
    }

    #[test]
    fn truncated_header_rejected() {
        let bytes = b"IM";
        let err = load_from_bytes(bytes).expect_err("too small");
        assert!(matches!(err, TokenizerError::HeaderCorrupt { .. }));
    }

    #[test]
    fn payload_length_overflow_rejected() {
        let mut bytes = save_to_bytes(&sample_data()).expect("save");
        bytes[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = load_from_bytes(&bytes).expect_err("should fail");
        assert!(matches!(err, TokenizerError::HeaderCorrupt { .. }));
    }

    #[test]
    fn little_endian_explicit() {
        let bytes = save_to_bytes(&sample_data()).expect("save");
        assert_eq!(&bytes[5..7], &FORMAT_VERSION_MAJOR.to_le_bytes());
        assert_eq!(&bytes[7..9], &FORMAT_VERSION_MINOR.to_le_bytes());
    }
}
