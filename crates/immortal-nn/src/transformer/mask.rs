//! Attention masks.
//!
//! Mask convention (matches [`crate::ops::scaled_dot_product_attention`]):
//! `1.0` = **allowed**, `0.0` = **masked out**.

use crate::tensor::{Shape, Tensor};

/// Build a `[seq_len, seq_len]` causal mask. Row `i` allows columns
/// `j <= i` and masks `j > i`, producing a lower-triangular "1" pattern
/// which lets each decoder position attend only to itself and earlier
/// positions.
#[must_use]
pub fn causal_mask(seq_len: usize) -> Tensor {
    let mut mask = Tensor::zeros(Shape::d2(seq_len, seq_len));
    let data = mask.data_mut();
    for i in 0..seq_len {
        for j in 0..=i {
            data[i * seq_len + j] = 1.0;
        }
    }
    mask
}
