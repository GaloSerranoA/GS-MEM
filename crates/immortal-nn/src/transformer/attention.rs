//! Cross-attention + multi-head cross-attention for encoder-decoder setups.
//!
//! Unlike [`crate::ops::scaled_dot_product_attention`] which requires
//! `Q.shape == K.shape == V.shape`, cross-attention handles the case
//! where the query sequence (from the decoder) and the key/value
//! sequence (from the encoder memory) have different lengths.

use crate::blocks::Linear;
use crate::ops::matmul;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

/// Scaled dot-product attention with differing query / key-value sequence lengths.
///
/// - `q`: `[B, H, Sq, D]`
/// - `k`: `[B, H, Skv, D]`
/// - `v`: `[B, H, Skv, D]`
/// - `mask`: optional `[Sq, Skv]` (row = query position, col = key position;
///   `1.0` = allowed, `0.0` = masked). Broadcast across `(B, H)`.
///
/// Returns `[B, H, Sq, D]`.
///
/// # Errors
/// [`NnError::ShapeMismatch`] on any rank or dimension mismatch.
pub fn cross_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> NnResult<Tensor> {
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.shape().ndim() != 4 {
            return Err(NnError::ShapeMismatch {
                expected: format!("{name}: rank-4 [B, H, S, D]"),
                got: format!("rank-{}", t.shape().ndim()),
            });
        }
    }
    let qd = q.shape().dims();
    let kd = k.shape().dims();
    let vd = v.shape().dims();
    let (b, h, s_q, d) = (qd[0], qd[1], qd[2], qd[3]);
    if kd[0] != b || kd[1] != h || kd[3] != d {
        return Err(NnError::ShapeMismatch {
            expected: format!("K: [{b}, {h}, *, {d}]"),
            got: format!("{kd:?}"),
        });
    }
    if vd[0] != b || vd[1] != h || vd[2] != kd[2] || vd[3] != d {
        return Err(NnError::ShapeMismatch {
            expected: format!("V: [{b}, {h}, {}, {d}]", kd[2]),
            got: format!("{vd:?}"),
        });
    }
    let s_kv = kd[2];

    let scale = 1.0 / (d as f32).sqrt();
    let mut out_data = vec![0.0_f32; b * h * s_q * d];

    // Work per (B, H). We slice each head into `[Sq, D]` and `[Skv, D]`
    // 2-D tensors, transpose K, matmul, softmax, matmul V.
    let q_src = q.data();
    let k_src = k.data();
    let v_src = v.data();

    let mut scores = vec![0.0_f32; s_q * s_kv];
    for bi in 0..b {
        for hi in 0..h {
            // Q slice: [Sq, D]
            let q_off = ((bi * h + hi) * s_q) * d;
            let q2 = Tensor::from_vec(q_src[q_off..q_off + s_q * d].to_vec(), Shape::d2(s_q, d))?;
            // K slice: [Skv, D] → transpose to [D, Skv]
            let k_off = ((bi * h + hi) * s_kv) * d;
            let mut kt = vec![0.0_f32; d * s_kv];
            for kk in 0..s_kv {
                for dd in 0..d {
                    kt[dd * s_kv + kk] = k_src[k_off + kk * d + dd];
                }
            }
            let kt2 = Tensor::from_vec(kt, Shape::d2(d, s_kv))?;
            // V slice: [Skv, D]
            let v_off = ((bi * h + hi) * s_kv) * d;
            let v2 = Tensor::from_vec(v_src[v_off..v_off + s_kv * d].to_vec(), Shape::d2(s_kv, d))?;

            // scores = Q @ K^T / sqrt(D)
            let raw_scores = matmul(&q2, &kt2)?;
            for (out_s, raw) in scores.iter_mut().zip(raw_scores.data()) {
                *out_s = raw * scale;
            }

            // Mask before softmax.
            if let Some(m) = mask {
                if m.shape().ndim() != 2
                    || m.shape().dims()[0] != s_q
                    || m.shape().dims()[1] != s_kv
                {
                    return Err(NnError::ShapeMismatch {
                        expected: format!("mask [{s_q}, {s_kv}]"),
                        got: format!("{}", m.shape()),
                    });
                }
                let md = m.data();
                for (idx, s) in scores.iter_mut().enumerate() {
                    if md[idx] == 0.0 {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }

            // Row-wise softmax.
            let mut attn = vec![0.0_f32; s_q * s_kv];
            for row in 0..s_q {
                let row_off = row * s_kv;
                let row_slice = &scores[row_off..row_off + s_kv];
                let max = row_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0_f32;
                for col in 0..s_kv {
                    let e = if max == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (row_slice[col] - max).exp()
                    };
                    attn[row_off + col] = e;
                    denom += e;
                }
                if denom > 0.0 {
                    for col in 0..s_kv {
                        attn[row_off + col] /= denom;
                    }
                }
            }
            let attn_t = Tensor::from_vec(attn, Shape::d2(s_q, s_kv))?;

            // out = attn @ V → [Sq, D]
            let out_head = matmul(&attn_t, &v2)?;
            let out_off = ((bi * h + hi) * s_q) * d;
            out_data[out_off..out_off + s_q * d].copy_from_slice(out_head.data());
        }
    }

    Tensor::from_vec(out_data, Shape::d4(b, h, s_q, d))
}

/// Multi-head cross-attention. Separate input for query vs key/value.
#[derive(Debug)]
pub struct MultiHeadCrossAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadCrossAttention {
    /// Construct from existing projection layers. All four linear layers
    /// must have the same `in_features` (the model dimension `d_model`).
    ///
    /// # Errors
    /// [`NnError::DivisibilityError`] if `d_model % num_heads != 0`.
    pub fn from_parts(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
    ) -> NnResult<Self> {
        let d_model = q_proj.in_features();
        if !d_model.is_multiple_of(num_heads) {
            return Err(NnError::DivisibilityError {
                what: "d_model % num_heads".into(),
                numerator: d_model,
                denominator: num_heads,
            });
        }
        let head_dim = d_model / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    /// Forward. `query`: `[B, Sq, E]`, `key_value`: `[B, Skv, E]`, optional
    /// `mask`: `[Sq, Skv]`. Returns `[B, Sq, E]`.
    ///
    /// # Errors
    /// [`NnError::ShapeMismatch`] on wrong rank or mismatched batch.
    pub fn forward(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        mask: Option<&Tensor>,
    ) -> NnResult<Tensor> {
        if query.shape().ndim() != 3 || key_value.shape().ndim() != 3 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-3 [B, S, E] for both query and key_value".into(),
                got: format!(
                    "ranks {}/{}",
                    query.shape().ndim(),
                    key_value.shape().ndim()
                ),
            });
        }
        let qd = query.shape().dims();
        let kd = key_value.shape().dims();
        if qd[0] != kd[0] || qd[2] != kd[2] {
            return Err(NnError::ShapeMismatch {
                expected: format!("query {qd:?} vs kv {kd:?}: batch and E must match"),
                got: format!("{qd:?} vs {kd:?}"),
            });
        }
        let (b, s_q) = (qd[0], qd[1]);
        let s_kv = kd[1];

        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key_value)?;
        let v = self.v_proj.forward(key_value)?;

        let q4 = split_heads(&q, b, s_q, self.num_heads, self.head_dim)?;
        let k4 = split_heads(&k, b, s_kv, self.num_heads, self.head_dim)?;
        let v4 = split_heads(&v, b, s_kv, self.num_heads, self.head_dim)?;

        let attn_out = cross_attention(&q4, &k4, &v4, mask)?;
        let merged = merge_heads(&attn_out, b, s_q, self.num_heads, self.head_dim)?;
        self.o_proj.forward(&merged)
    }

    /// Number of attention heads.
    #[must_use]
    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Per-head dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// `[B, S, H*D]` → `[B, H, S, D]`.
fn split_heads(t: &Tensor, b: usize, s: usize, h: usize, d: usize) -> NnResult<Tensor> {
    let bshd = t.reshape(Shape::d4(b, s, h, d))?;
    let mut out = Tensor::zeros(Shape::d4(b, h, s, d));
    let src = bshd.data();
    let dst = out.data_mut();
    for bi in 0..b {
        for si in 0..s {
            for hi in 0..h {
                for di in 0..d {
                    dst[((bi * h + hi) * s + si) * d + di] = src[((bi * s + si) * h + hi) * d + di];
                }
            }
        }
    }
    Ok(out)
}

/// `[B, H, Sq, D]` → `[B, Sq, H*D]`.
fn merge_heads(t: &Tensor, b: usize, s: usize, h: usize, d: usize) -> NnResult<Tensor> {
    let mut transposed = Tensor::zeros(Shape::d4(b, s, h, d));
    let src = t.data();
    let dst = transposed.data_mut();
    for bi in 0..b {
        for hi in 0..h {
            for si in 0..s {
                for di in 0..d {
                    dst[((bi * s + si) * h + hi) * d + di] = src[((bi * h + hi) * s + si) * d + di];
                }
            }
        }
    }
    transposed.reshape(Shape::d3(b, s, h * d))
}
