//! Sinusoidal positional encoding (Vaswani et al., 2017).
//!
//! For each position `pos` and dimension `i`:
//! ```text
//! PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
//! PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
//! ```
//!
//! The encoding is a pure function of `(pos, d_model)` — no RNG, fully
//! deterministic, no learnable parameters. Add it to the input
//! embeddings before the first transformer block.

use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

/// Cached sinusoidal positional encoding.
///
/// Pre-computes the `[max_len, d_model]` table at construction time and
/// exposes `forward(x)` which adds the relevant prefix of the table to
/// each sequence in a batch.
#[derive(Debug, Clone)]
pub struct PositionalEncoding {
    table: Tensor,
    max_len: usize,
    d_model: usize,
}

impl PositionalEncoding {
    /// Construct a new encoding table for sequences up to `max_len` and
    /// model width `d_model`.
    ///
    /// # Errors
    /// Returns [`NnError::Format`] if either argument is zero.
    pub fn new(max_len: usize, d_model: usize) -> NnResult<Self> {
        if max_len == 0 || d_model == 0 {
            return Err(NnError::Format(format!(
                "PositionalEncoding: max_len and d_model must be >= 1; got {max_len}, {d_model}"
            )));
        }
        let mut data = vec![0.0_f32; max_len * d_model];
        // `2 * (i / 2)` snaps both (2i) and (2i+1) to the same exponent
        // pair — the standard sinusoidal scheme.
        for pos in 0..max_len {
            for i in 0..d_model {
                let pair = (i / 2) * 2;
                let exponent = pair as f32 / d_model as f32;
                let denom = 10000.0_f32.powf(exponent);
                let angle = pos as f32 / denom;
                data[pos * d_model + i] = if i.is_multiple_of(2) {
                    angle.sin()
                } else {
                    angle.cos()
                };
            }
        }
        Ok(Self {
            table: Tensor::from_vec(data, Shape::d2(max_len, d_model))?,
            max_len,
            d_model,
        })
    }

    /// Max sequence length this encoding supports.
    #[must_use]
    pub const fn max_len(&self) -> usize {
        self.max_len
    }

    /// Model width this encoding was built for.
    #[must_use]
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    /// Return the raw `[max_len, d_model]` table.
    #[must_use]
    pub const fn table(&self) -> &Tensor {
        &self.table
    }

    /// Add the positional encoding to the input.
    ///
    /// `x` must have rank 3 `[B, S, d_model]`. Returns a new tensor of
    /// the same shape with the `[S, d_model]` prefix of the table added
    /// to each batch.
    ///
    /// # Errors
    /// - [`NnError::ShapeMismatch`] on wrong rank or feature count.
    /// - [`NnError::Format`] if `S > max_len`.
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        if x.shape().ndim() != 3 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-3 [B, S, d_model]".into(),
                got: format!("rank-{}", x.shape().ndim()),
            });
        }
        let d = x.shape().dims();
        let (b, s, e) = (d[0], d[1], d[2]);
        if e != self.d_model {
            return Err(NnError::ShapeMismatch {
                expected: format!("d_model {}", self.d_model),
                got: format!("{e}"),
            });
        }
        if s > self.max_len {
            return Err(NnError::Format(format!(
                "PositionalEncoding: sequence len {s} exceeds max_len {}",
                self.max_len
            )));
        }

        let mut out = x.clone();
        let out_data = out.data_mut();
        let pe = self.table.data();
        for bi in 0..b {
            for si in 0..s {
                let dst_off = (bi * s + si) * e;
                let src_off = si * self.d_model;
                for di in 0..e {
                    out_data[dst_off + di] += pe[src_off + di];
                }
            }
        }
        Ok(out)
    }
}
