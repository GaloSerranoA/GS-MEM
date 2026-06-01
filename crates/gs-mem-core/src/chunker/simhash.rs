use sha2::{Digest, Sha256};

pub fn simhash64(text: &str) -> u64 {
    let mut weights = [0_i32; 64];

    for token in text.split_whitespace() {
        let hash = token_hash(token);
        for (idx, slot) in weights.iter_mut().enumerate() {
            if (hash >> idx) & 1 == 1 {
                *slot += 1;
            } else {
                *slot -= 1;
            }
        }
    }

    let mut out = 0_u64;
    for (idx, slot) in weights.iter().enumerate() {
        if *slot >= 0 {
            out |= 1_u64 << idx;
        }
    }
    out
}

fn token_hash(token: &str) -> u64 {
    let digest = Sha256::digest(token.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}
