//! Blake2b-128 content fingerprint for cocoindex-style skip-gate dedup.
//!
//! Absorbed verbatim from `gmem/src/fingerprint.rs` on 2026-04-18.
//! `Fingerprint::of(&T)` canonicalizes `T` via `serde_json`, then hashes the
//! bytes with Blake2b producing a 16-byte fingerprint. Two equal `T`s always
//! produce equal `Fingerprint`s; any field change produces a different one.

use std::fmt;

use blake2::digest::consts::U16;
use blake2::{Blake2b, Digest};
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize, Default)]
#[serde(transparent)]
pub struct Fingerprint([u8; 16]);

impl Fingerprint {
    pub fn of<T: Serialize>(value: &T) -> Fingerprint {
        let bytes = serde_json::to_vec(value).unwrap_or_else(|err| err.to_string().into_bytes());
        let digest = Blake2b::<U16>::digest(bytes);
        let mut out = [0_u8; 16];
        out.copy_from_slice(&digest);
        Self(out)
    }

    pub const fn zero() -> Self {
        Self([0; 16])
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl From<[u8; 16]> for Fingerprint {
    fn from(value: [u8; 16]) -> Self {
        Self(value)
    }
}
