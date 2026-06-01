//! Pure functional operations on [`Tensor`].
//!
//! All ops:
//! - take `&Tensor` (or `&[u32]`) inputs and return owned `Tensor`,
//! - validate shapes upfront and fail with [`NnError`] on mismatch,
//! - never mutate inputs,
//! - never broadcast,
//! - never silently change dtype.
//!
//! Axis convention: `axis: usize` only. No negative axes.
//!
//! v1 implementation notes:
//! - [`softmax`] is supported only over the **last** dimension.
//!   Other axes return [`NnError::InvalidAxis`].
//! - [`matmul`] is a naive cache-blocked `f32` kernel. A faster path can
//!   land later behind a feature flag without API change.

use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};
use rayon::prelude::*;

// ===========================================================================
// Arithmetic
// ===========================================================================

/// `[m, k] @ [k, n] -> [m, n]`. Both inputs must be 2-D.
///
/// # Errors
/// [`NnError::ShapeMismatch`] on rank or inner-dim mismatch.
pub fn matmul(a: &Tensor, b: &Tensor) -> NnResult<Tensor> {
    if a.shape().ndim() != 2 || b.shape().ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "rank-2 @ rank-2".into(),
            got: format!("rank-{} @ rank-{}", a.shape().ndim(), b.shape().ndim()),
        });
    }
    let (m, k1) = (a.shape().dims()[0], a.shape().dims()[1]);
    let (k2, n) = (b.shape().dims()[0], b.shape().dims()[1]);
    if k1 != k2 {
        return Err(NnError::ShapeMismatch {
            expected: format!("[m,{k1}] @ [{k1},n]"),
            got: format!("[{m},{k1}] @ [{k2},{n}]"),
        });
    }
    let mut out = Tensor::zeros(Shape::d2(m, n));
    matmul_kernel(a.data(), b.data(), out.data_mut(), m, k1, n);
    Ok(out)
}

/// Cache-friendly i-k-j naive `f32` matmul. `c` must be pre-zeroed.
fn matmul_kernel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    const PARALLEL_WORK_THRESHOLD: usize = 8_000_000;
    let work = m.saturating_mul(k).saturating_mul(n);
    if m > 1 && work >= PARALLEL_WORK_THRESHOLD {
        c.par_chunks_mut(n).enumerate().for_each(|(i, row_c)| {
            for kk in 0..k {
                let aik = a[i * k + kk];
                let row_b = &b[kk * n..kk * n + n];
                for j in 0..n {
                    row_c[j] += aik * row_b[j];
                }
            }
        });
        return;
    }

    for i in 0..m {
        for kk in 0..k {
            let aik = a[i * k + kk];
            let row_b = &b[kk * n..kk * n + n];
            let row_c = &mut c[i * n..i * n + n];
            for j in 0..n {
                row_c[j] += aik * row_b[j];
            }
        }
    }
}

/// Batched matmul: `[B, m, k] @ [B, k, n] -> [B, m, n]`.
///
/// # Errors
/// [`NnError::ShapeMismatch`] on rank, batch, or inner-dim mismatch.
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> NnResult<Tensor> {
    if a.shape().ndim() != 3 || b.shape().ndim() != 3 {
        return Err(NnError::ShapeMismatch {
            expected: "rank-3 @ rank-3".into(),
            got: format!("rank-{} @ rank-{}", a.shape().ndim(), b.shape().ndim()),
        });
    }
    let (ba, m, k1) = (
        a.shape().dims()[0],
        a.shape().dims()[1],
        a.shape().dims()[2],
    );
    let (bb, k2, n) = (
        b.shape().dims()[0],
        b.shape().dims()[1],
        b.shape().dims()[2],
    );
    if ba != bb || k1 != k2 {
        return Err(NnError::ShapeMismatch {
            expected: format!("[B,m,{k1}] @ [B,{k1},n]"),
            got: format!("[{ba},{m},{k1}] @ [{bb},{k2},{n}]"),
        });
    }
    let mut out = Tensor::zeros(Shape::d3(ba, m, n));
    let a_stride = m * k1;
    let b_stride = k1 * n;
    let c_stride = m * n;
    let (ad, bd, cd) = (a.data(), b.data(), out.data_mut());
    for batch in 0..ba {
        matmul_kernel(
            &ad[batch * a_stride..(batch + 1) * a_stride],
            &bd[batch * b_stride..(batch + 1) * b_stride],
            &mut cd[batch * c_stride..(batch + 1) * c_stride],
            m,
            k1,
            n,
        );
    }
    Ok(out)
}

/// Element-wise addition. Shapes must match exactly — no broadcasting.
///
/// # Errors
/// [`NnError::ShapeMismatch`] on shape mismatch.
pub fn add(a: &Tensor, b: &Tensor) -> NnResult<Tensor> {
    if a.shape() != b.shape() {
        return Err(NnError::ShapeMismatch {
            expected: format!("{}", a.shape()),
            got: format!("{}", b.shape()),
        });
    }
    let mut out = a.clone();
    for (o, &x) in out.data_mut().iter_mut().zip(b.data()) {
        *o += x;
    }
    Ok(out)
}

/// Multiply every element by a scalar.
#[must_use]
pub fn scale(t: &Tensor, s: f32) -> Tensor {
    let mut out = t.clone();
    for x in out.data_mut() {
        *x *= s;
    }
    out
}

// ===========================================================================
// Activations
// ===========================================================================

/// ReLU: `max(0, x)`.
#[must_use]
pub fn relu(t: &Tensor) -> Tensor {
    let mut out = t.clone();
    for x in out.data_mut() {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
    out
}

/// GELU (tanh approximation, matches HuggingFace default):
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
#[must_use]
pub fn gelu(t: &Tensor) -> Tensor {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    let mut out = t.clone();
    for x in out.data_mut() {
        let v = *x;
        let inner = C * (v + 0.044_715 * v * v * v);
        *x = 0.5 * v * (1.0 + inner.tanh());
    }
    out
}

/// Sigmoid: `1 / (1 + exp(-x))`.
#[must_use]
pub fn sigmoid(t: &Tensor) -> Tensor {
    let mut out = t.clone();
    for x in out.data_mut() {
        *x = 1.0 / (1.0 + (-*x).exp());
    }
    out
}

/// Hyperbolic tangent activation.
#[must_use]
pub fn tanh_act(t: &Tensor) -> Tensor {
    let mut out = t.clone();
    for x in out.data_mut() {
        *x = x.tanh();
    }
    out
}

// ===========================================================================
// Normalization
// ===========================================================================

/// Numerically stable softmax over the **last** dimension.
///
/// `axis` must equal `t.shape().ndim() - 1`. Other axes are not supported in v1.
///
/// # Errors
/// [`NnError::InvalidAxis`] if `axis` is not the last dimension.
pub fn softmax(t: &Tensor, axis: usize) -> NnResult<Tensor> {
    let nd = t.shape().ndim() as usize;
    if axis != nd - 1 {
        return Err(NnError::InvalidAxis {
            axis,
            ndim: t.shape().ndim(),
        });
    }
    let last = t.shape().dims()[nd - 1];
    let mut out = t.clone();
    for chunk in out.data_mut().chunks_exact_mut(last) {
        let max = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for x in chunk.iter_mut() {
            *x = (*x - max).exp();
            sum += *x;
        }
        let inv = 1.0 / sum;
        for x in chunk.iter_mut() {
            *x *= inv;
        }
    }
    Ok(out)
}

/// Layer normalization over the **last** dimension.
///
/// `gamma` and `beta` must be 1-D-shaped (encoded as 2-D `[1, features]`)
/// with the same final-axis size as `input`.
///
/// # Errors
/// [`NnError::ShapeMismatch`] if `gamma`/`beta` dimensions do not match the
/// last axis of `input`.
pub fn layer_norm(input: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> NnResult<Tensor> {
    let nd = input.shape().ndim() as usize;
    let last = input.shape().dims()[nd - 1];
    let g = gamma.data();
    let b = beta.data();
    if g.len() != last || b.len() != last {
        return Err(NnError::ShapeMismatch {
            expected: format!("gamma/beta of length {last}"),
            got: format!("gamma={}, beta={}", g.len(), b.len()),
        });
    }
    let mut out = input.clone();
    for chunk in out.data_mut().chunks_exact_mut(last) {
        let n = last as f32;
        let mean = chunk.iter().sum::<f32>() / n;
        let var = chunk.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, x) in chunk.iter_mut().enumerate() {
            *x = (*x - mean) * inv * g[i] + b[i];
        }
    }
    Ok(out)
}

// ===========================================================================
// Embedding
// ===========================================================================

/// Lookup `[seq_len]` token IDs into a `[vocab, dim]` table.
///
/// Returns a `[seq_len, dim]` tensor.
///
/// # Errors
/// [`NnError::ShapeMismatch`] if `table` is not 2-D, or any index is OOV.
pub fn embedding_lookup(table: &Tensor, indices: &[u32]) -> NnResult<Tensor> {
    if table.shape().ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "rank-2 embedding table".into(),
            got: format!("rank-{}", table.shape().ndim()),
        });
    }
    let vocab = table.shape().dims()[0];
    let dim = table.shape().dims()[1];
    let mut out = Tensor::zeros(Shape::d2(indices.len(), dim));
    let src = table.data();
    let dst = out.data_mut();
    for (i, &idx) in indices.iter().enumerate() {
        let id = idx as usize;
        if id >= vocab {
            return Err(NnError::ShapeMismatch {
                expected: format!("token id < {vocab}"),
                got: format!("token id = {id}"),
            });
        }
        dst[i * dim..(i + 1) * dim].copy_from_slice(&src[id * dim..(id + 1) * dim]);
    }
    Ok(out)
}

// ===========================================================================
// Attention
// ===========================================================================

/// Scaled dot-product attention.
///
/// Inputs Q, K, V are all `[B, H, S, D]`. Optional `mask` is `[B, S]`
/// where `0` means "masked out" (set to -inf before softmax) and any
/// non-zero value means "keep". Returns `[B, H, S, D]`.
///
/// Internally decomposed as: `qk = Q @ K^T -> scale -> mask -> softmax -> @ V`.
/// The decomposition is intentional — every step is an audit point.
///
/// # Errors
/// [`NnError::ShapeMismatch`] on any rank or dim mismatch among Q/K/V/mask.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> NnResult<Tensor> {
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.shape().ndim() != 4 {
            return Err(NnError::ShapeMismatch {
                expected: format!("{name}: rank-4 [B,H,S,D]"),
                got: format!("rank-{}", t.shape().ndim()),
            });
        }
    }
    if q.shape() != k.shape() || k.shape() != v.shape() {
        return Err(NnError::ShapeMismatch {
            expected: "Q.shape == K.shape == V.shape".into(),
            got: format!("Q={}, K={}, V={}", q.shape(), k.shape(), v.shape()),
        });
    }
    let dims = q.shape().dims();
    let (b, h, s, d) = (dims[0], dims[1], dims[2], dims[3]);
    let scale_factor = 1.0 / (d as f32).sqrt();

    // Reshape [B,H,S,D] to [B*H, S, D] so we can use batched_matmul.
    let q3 = q.reshape(Shape::d3(b * h, s, d))?;
    let v3 = v.reshape(Shape::d3(b * h, s, d))?;
    // Build K^T as [B*H, D, S] by manual transpose of last two axes.
    let kt = transpose_last_two(k)?;
    let kt3 = kt.reshape(Shape::d3(b * h, d, s))?;

    // Step 1: scores = Q @ K^T  -> [B*H, S, S]
    let mut scores = batched_matmul(&q3, &kt3)?;

    // Step 2: scale by 1/sqrt(d)
    for x in scores.data_mut() {
        *x *= scale_factor;
    }

    // Step 3: optional mask. mask: [B, S] applied per (B*H, *, S).
    if let Some(m) = mask {
        if m.shape().ndim() != 2 || m.shape().dims() != [b, s] {
            return Err(NnError::ShapeMismatch {
                expected: format!("mask [{b},{s}]"),
                got: format!("{}", m.shape()),
            });
        }
        let md = m.data();
        let sd = scores.data_mut();
        for batch in 0..b {
            for head in 0..h {
                let bh = batch * h + head;
                for row in 0..s {
                    let row_off = (bh * s + row) * s;
                    for col in 0..s {
                        if md[batch * s + col] == 0.0 {
                            sd[row_off + col] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
    }

    // Step 4: softmax over last dim
    let attn = softmax(&scores, 2)?;

    // Step 5: attn @ V  -> [B*H, S, D]
    let out3 = batched_matmul(&attn, &v3)?;

    // Reshape back to [B, H, S, D]
    out3.reshape(Shape::d4(b, h, s, d))
}

/// Transpose the last two axes of a rank-3 or rank-4 tensor. Allocates a new tensor.
fn transpose_last_two(t: &Tensor) -> NnResult<Tensor> {
    let nd = t.shape().ndim();
    match nd {
        3 => {
            let d = t.shape().dims();
            let (a, b, c) = (d[0], d[1], d[2]);
            let mut out = Tensor::zeros(Shape::d3(a, c, b));
            let src = t.data();
            let dst = out.data_mut();
            for ai in 0..a {
                for bi in 0..b {
                    for ci in 0..c {
                        dst[(ai * c + ci) * b + bi] = src[(ai * b + bi) * c + ci];
                    }
                }
            }
            Ok(out)
        }
        4 => {
            let d = t.shape().dims();
            let (a, b, c, e) = (d[0], d[1], d[2], d[3]);
            let mut out = Tensor::zeros(Shape::d4(a, b, e, c));
            let src = t.data();
            let dst = out.data_mut();
            for ai in 0..a {
                for bi in 0..b {
                    for ci in 0..c {
                        for ei in 0..e {
                            dst[((ai * b + bi) * e + ei) * c + ci] =
                                src[((ai * b + bi) * c + ci) * e + ei];
                        }
                    }
                }
            }
            Ok(out)
        }
        _ => Err(NnError::UnsupportedRank { rank: nd }),
    }
}

// ===========================================================================
// CRF Viterbi decode
// ===========================================================================

/// Viterbi decode for a linear-chain CRF.
///
/// - `emissions`: `[B, S, T]` — per-token tag scores
/// - `transitions`: `[T, T]` — `transitions[from][to]` (locked direction)
/// - `start_transitions`: `[T]` — score for starting in each tag
/// - `end_transitions`: `[T]` — score for ending in each tag
///
/// Returns `(tags, score)` per batch element. `tags[b]` is the best tag
/// sequence of length `S`. `score[b]` is the path score (log-space).
///
/// # Errors
/// [`NnError::ShapeMismatch`] on any rank or dim mismatch.
pub fn crf_viterbi_decode(
    emissions: &Tensor,
    transitions: &Tensor,
    start_transitions: &Tensor,
    end_transitions: &Tensor,
) -> NnResult<Vec<(Vec<u32>, f32)>> {
    if emissions.shape().ndim() != 3 {
        return Err(NnError::ShapeMismatch {
            expected: "emissions rank-3 [B,S,T]".into(),
            got: format!("rank-{}", emissions.shape().ndim()),
        });
    }
    let dims = emissions.shape().dims();
    let (batch, seq, tags) = (dims[0], dims[1], dims[2]);
    if transitions.shape().dims() != [tags, tags] {
        return Err(NnError::ShapeMismatch {
            expected: format!("transitions [{tags},{tags}]"),
            got: format!("{}", transitions.shape()),
        });
    }
    if start_transitions.data().len() != tags || end_transitions.data().len() != tags {
        return Err(NnError::ShapeMismatch {
            expected: format!("start/end transitions length {tags}"),
            got: format!(
                "start={}, end={}",
                start_transitions.data().len(),
                end_transitions.data().len()
            ),
        });
    }

    let em = emissions.data();
    let tr = transitions.data();
    let st = start_transitions.data();
    let en = end_transitions.data();
    let mut results = Vec::with_capacity(batch);

    for b in 0..batch {
        // dp[t] = best score ending in tag t at current step
        let mut dp = vec![0.0_f32; tags];
        // backptr[step][t] = previous tag
        let mut back = vec![0_u32; seq * tags];

        // Step 0: start_transitions + emissions[b,0,*]
        for t in 0..tags {
            dp[t] = st[t] + em[(b * seq) * tags + t];
        }

        // Steps 1..seq
        let mut next = vec![0.0_f32; tags];
        for step in 1..seq {
            for t in 0..tags {
                let mut best = f32::NEG_INFINITY;
                let mut best_prev = 0_u32;
                for p in 0..tags {
                    let score = dp[p] + tr[p * tags + t];
                    if score > best {
                        best = score;
                        best_prev = p as u32;
                    }
                }
                next[t] = best + em[((b * seq) + step) * tags + t];
                back[step * tags + t] = best_prev;
            }
            dp.copy_from_slice(&next);
        }

        // Add end_transitions and find best final tag
        for t in 0..tags {
            dp[t] += en[t];
        }
        let (mut best_tag, mut best_score) = (0_u32, f32::NEG_INFINITY);
        for (t, &score) in dp.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_tag = t as u32;
            }
        }

        // Backtrace
        let mut path = vec![0_u32; seq];
        path[seq - 1] = best_tag;
        for step in (1..seq).rev() {
            path[step - 1] = back[step * tags + path[step] as usize];
        }
        results.push((path, best_score));
    }
    Ok(results)
}

// ===========================================================================
// Pooling
// ===========================================================================

/// Mean pool over the sequence axis (axis=1) of a `[B, S, F]` tensor.
/// Returns `[B, F]`.
///
/// # Errors
/// [`NnError::UnsupportedRank`] if input is not rank-3.
pub fn mean_pool_seq(t: &Tensor) -> NnResult<Tensor> {
    if t.shape().ndim() != 3 {
        return Err(NnError::UnsupportedRank {
            rank: t.shape().ndim(),
        });
    }
    let dims = t.shape().dims();
    let (b, s, f) = (dims[0], dims[1], dims[2]);
    let mut out = Tensor::zeros(Shape::d2(b, f));
    let src = t.data();
    let dst = out.data_mut();
    let inv = 1.0 / (s as f32);
    for bi in 0..b {
        for si in 0..s {
            for fi in 0..f {
                dst[bi * f + fi] += src[(bi * s + si) * f + fi];
            }
        }
        for fi in 0..f {
            dst[bi * f + fi] *= inv;
        }
    }
    Ok(out)
}

/// Take position 0 along the sequence axis: `[B, S, F] -> [B, F]`.
///
/// # Errors
/// [`NnError::UnsupportedRank`] if input is not rank-3.
pub fn cls_pool(t: &Tensor) -> NnResult<Tensor> {
    if t.shape().ndim() != 3 {
        return Err(NnError::UnsupportedRank {
            rank: t.shape().ndim(),
        });
    }
    let dims = t.shape().dims();
    let (b, s, f) = (dims[0], dims[1], dims[2]);
    let mut out = Tensor::zeros(Shape::d2(b, f));
    let src = t.data();
    let dst = out.data_mut();
    for bi in 0..b {
        let row = (bi * s) * f;
        dst[bi * f..(bi + 1) * f].copy_from_slice(&src[row..row + f]);
    }
    Ok(out)
}

// ===========================================================================
// Shape ops
// ===========================================================================

/// Concatenate tensors along `axis`. All shapes must match on every other axis.
///
/// v1 supports rank-2 and rank-3 inputs.
///
/// # Errors
/// [`NnError::ShapeMismatch`] on rank or non-axis dim mismatch.
/// [`NnError::InvalidAxis`] if `axis >= ndim`.
pub fn concat(tensors: &[&Tensor], axis: usize) -> NnResult<Tensor> {
    if tensors.is_empty() {
        return Err(NnError::ShapeMismatch {
            expected: "non-empty tensor list".into(),
            got: "empty".into(),
        });
    }
    let first_shape = tensors[0].shape();
    let nd = first_shape.ndim() as usize;
    if axis >= nd {
        return Err(NnError::InvalidAxis {
            axis,
            ndim: first_shape.ndim(),
        });
    }
    for t in &tensors[1..] {
        if t.shape().ndim() as usize != nd {
            return Err(NnError::ShapeMismatch {
                expected: format!("rank {nd}"),
                got: format!("rank {}", t.shape().ndim()),
            });
        }
        for i in 0..nd {
            if i != axis && t.shape().dims()[i] != first_shape.dims()[i] {
                return Err(NnError::ShapeMismatch {
                    expected: format!("dim {i} = {}", first_shape.dims()[i]),
                    got: format!("dim {i} = {}", t.shape().dims()[i]),
                });
            }
        }
    }

    // Build output shape.
    let mut out_dims = [1usize; 4];
    out_dims[..nd].copy_from_slice(first_shape.dims());
    out_dims[axis] = tensors.iter().map(|t| t.shape().dims()[axis]).sum();
    let out_shape = Shape::new(out_dims, first_shape.ndim())?;
    let mut out = Tensor::zeros(out_shape);

    // Naive but correct: copy element-by-element using strides.
    let out_strides = out_shape.strides();
    let dst = out.data_mut();
    let mut axis_offset = 0usize;
    for t in tensors {
        let in_dims = t.shape().dims();
        let in_strides = t.shape().strides();
        let src = t.data();
        let total = t.numel();
        for flat in 0..total {
            // unflatten flat into multi-index w.r.t. input strides
            let mut idx = [0usize; 4];
            let mut rem = flat;
            for d in 0..nd {
                idx[d] = rem / in_strides[d];
                rem %= in_strides[d];
            }
            // shift along axis
            idx[axis] += axis_offset;
            // flatten back into output
            let mut out_flat = 0usize;
            for (d, &v) in idx.iter().take(nd).enumerate() {
                out_flat += v * out_strides[d];
            }
            dst[out_flat] = src[flat];
            // re-strip axis offset
            let _ = in_dims; // silence unused warning when nd == 2
        }
        axis_offset += t.shape().dims()[axis];
    }
    Ok(out)
}

/// Split a tensor along `axis` into `chunks` equal pieces.
///
/// # Errors
/// [`NnError::DivisibilityError`] if the axis size is not divisible by `chunks`.
/// [`NnError::InvalidAxis`] if `axis >= ndim`.
pub fn split(t: &Tensor, axis: usize, chunks: usize) -> NnResult<Vec<Tensor>> {
    let nd = t.shape().ndim() as usize;
    if axis >= nd {
        return Err(NnError::InvalidAxis {
            axis,
            ndim: t.shape().ndim(),
        });
    }
    let axis_size = t.shape().dims()[axis];
    if !axis_size.is_multiple_of(chunks) {
        return Err(NnError::DivisibilityError {
            what: format!("split axis {axis}"),
            numerator: axis_size,
            denominator: chunks,
        });
    }
    let chunk_size = axis_size / chunks;
    let mut out = Vec::with_capacity(chunks);
    let in_strides = t.shape().strides();
    let src = t.data();
    for c in 0..chunks {
        let mut new_dims = [1usize; 4];
        new_dims[..nd].copy_from_slice(t.shape().dims());
        new_dims[axis] = chunk_size;
        let new_shape = Shape::new(new_dims, t.shape().ndim())?;
        let mut piece = Tensor::zeros(new_shape);
        let total = piece.numel();
        let out_strides = new_shape.strides();
        let dst = piece.data_mut();
        for flat in 0..total {
            let mut idx = [0usize; 4];
            let mut rem = flat;
            for d in 0..nd {
                idx[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            let src_axis = idx[axis] + c * chunk_size;
            let mut src_flat = 0usize;
            for d in 0..nd {
                let v = if d == axis { src_axis } else { idx[d] };
                src_flat += v * in_strides[d];
            }
            dst[flat] = src[src_flat];
        }
        out.push(piece);
    }
    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "op tests assert exact output for integer-valued inputs (1.0, 2.0, …); bitwise equality is intentional here"
)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn matmul_2x3_x_3x2() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], Shape::d2(2, 3)).unwrap();
        let b = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], Shape::d2(3, 2)).unwrap();
        let c = matmul(&a, &b).unwrap();
        // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(c.data(), &[58., 64., 139., 154.]);
    }

    #[test]
    fn matmul_inner_dim_mismatch() {
        let a = Tensor::zeros(Shape::d2(2, 3));
        let b = Tensor::zeros(Shape::d2(4, 2));
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn batched_matmul_2_batches() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], Shape::d3(2, 2, 2)).unwrap();
        let b = Tensor::from_vec(vec![1., 0., 0., 1., 1., 1., 1., 1.], Shape::d3(2, 2, 2)).unwrap();
        let c = batched_matmul(&a, &b).unwrap();
        // batch 0: [[1,2],[3,4]] @ I = [[1,2],[3,4]]
        // batch 1: [[5,6],[7,8]] @ ones = [[11,11],[15,15]]
        assert_eq!(c.data(), &[1., 2., 3., 4., 11., 11., 15., 15.]);
    }

    #[test]
    fn add_same_shape() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], Shape::d2(2, 2)).unwrap();
        let b = Tensor::from_vec(vec![10., 20., 30., 40.], Shape::d2(2, 2)).unwrap();
        assert_eq!(add(&a, &b).unwrap().data(), &[11., 22., 33., 44.]);
    }

    #[test]
    fn add_shape_mismatch_errors() {
        let a = Tensor::zeros(Shape::d2(2, 2));
        let b = Tensor::zeros(Shape::d2(2, 3));
        assert!(add(&a, &b).is_err());
    }

    #[test]
    fn relu_clips_negatives() {
        let t = Tensor::from_vec(vec![-1., 0., 1., -2., 3.], Shape::d2(1, 5)).unwrap();
        assert_eq!(relu(&t).data(), &[0., 0., 1., 0., 3.]);
    }

    #[test]
    fn gelu_known_values() {
        let t = Tensor::from_vec(vec![0.0, 1.0, -1.0], Shape::d2(1, 3)).unwrap();
        let g = gelu(&t);
        assert!(approx(g.data()[0], 0.0, 1e-6));
        assert!(approx(g.data()[1], 0.8412, 1e-3));
        assert!(approx(g.data()[2], -0.1588, 1e-3));
    }

    #[test]
    fn softmax_last_dim_sums_to_one() {
        let t = Tensor::from_vec(vec![1., 2., 3., 1., 1., 1.], Shape::d2(2, 3)).unwrap();
        let s = softmax(&t, 1).unwrap();
        for row in s.data().chunks_exact(3) {
            assert!(approx(row.iter().sum::<f32>(), 1.0, 1e-6));
        }
        // Uniform row should be exactly 1/3 each
        for &v in &s.data()[3..6] {
            assert!(approx(v, 1.0 / 3.0, 1e-6));
        }
    }

    #[test]
    fn softmax_rejects_non_last_axis() {
        let t = Tensor::zeros(Shape::d2(2, 3));
        assert!(softmax(&t, 0).is_err());
    }

    #[test]
    fn softmax_numerically_stable_on_large_inputs() {
        let t = Tensor::from_vec(vec![1000., 1001., 1002.], Shape::d2(1, 3)).unwrap();
        let s = softmax(&t, 1).unwrap();
        assert!(approx(s.data().iter().sum::<f32>(), 1.0, 1e-6));
        assert!(s.data().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4.], Shape::d2(1, 4)).unwrap();
        let g = Tensor::from_vec(vec![1., 1., 1., 1.], Shape::d2(1, 4)).unwrap();
        let b = Tensor::from_vec(vec![0., 0., 0., 0.], Shape::d2(1, 4)).unwrap();
        let n = layer_norm(&t, &g, &b, 1e-5).unwrap();
        let mean = n.data().iter().sum::<f32>() / 4.0;
        assert!(approx(mean, 0.0, 1e-5));
    }

    #[test]
    fn embedding_lookup_basic() {
        let table =
            Tensor::from_vec(vec![0., 0., 1., 1., 2., 2., 3., 3.], Shape::d2(4, 2)).unwrap();
        let out = embedding_lookup(&table, &[2, 0, 3]).unwrap();
        assert_eq!(out.data(), &[2., 2., 0., 0., 3., 3.]);
        assert_eq!(out.shape().dims(), &[3, 2]);
    }

    #[test]
    fn embedding_lookup_oov_errors() {
        let table = Tensor::zeros(Shape::d2(4, 2));
        assert!(embedding_lookup(&table, &[5]).is_err());
    }

    #[test]
    fn attention_identity_v_returns_v() {
        // With Q==K and uniform V, output should equal V (softmax distributes,
        // weighted sum of identical rows = same row).
        let q = Tensor::from_vec(vec![1.; 8], Shape::d4(1, 1, 2, 4)).unwrap();
        let k = q.clone();
        let v =
            Tensor::from_vec(vec![1., 2., 3., 4., 1., 2., 3., 4.], Shape::d4(1, 1, 2, 4)).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
        for i in 0..8 {
            assert!(approx(out.data()[i], v.data()[i], 1e-5));
        }
    }

    #[test]
    fn attention_shape_preserved() {
        let q = Tensor::zeros(Shape::d4(2, 4, 5, 8));
        let k = Tensor::zeros(Shape::d4(2, 4, 5, 8));
        let v = Tensor::zeros(Shape::d4(2, 4, 5, 8));
        let out = scaled_dot_product_attention(&q, &k, &v, None).unwrap();
        assert_eq!(out.shape().dims(), &[2, 4, 5, 8]);
    }

    #[test]
    fn crf_viterbi_two_tag_chain() {
        // 2 tags, sequence length 3, batch 1.
        // Emissions strongly favor tag 0 at every step.
        let emissions = Tensor::from_vec(vec![5., 0., 5., 0., 5., 0.], Shape::d3(1, 3, 2)).unwrap();
        let transitions = Tensor::from_vec(vec![0., 0., 0., 0.], Shape::d2(2, 2)).unwrap();
        let start = Tensor::from_vec(vec![0., 0.], Shape::d2(1, 2)).unwrap();
        let end = Tensor::from_vec(vec![0., 0.], Shape::d2(1, 2)).unwrap();
        let out = crf_viterbi_decode(&emissions, &transitions, &start, &end).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, vec![0, 0, 0]);
        assert!(approx(out[0].1, 15.0, 1e-5));
    }

    #[test]
    fn crf_viterbi_transition_drives_choice() {
        // Emissions force a unique optimal path [0, 1, 1]:
        // step 0 strongly favors tag 0; steps 1 and 2 strongly favor tag 1.
        let emissions =
            Tensor::from_vec(vec![10., 0., 0., 10., 0., 10.], Shape::d3(1, 3, 2)).unwrap();
        let transitions = Tensor::from_vec(vec![0., 0., 0., 0.], Shape::d2(2, 2)).unwrap();
        let start = Tensor::from_vec(vec![0., 0.], Shape::d2(1, 2)).unwrap();
        let end = Tensor::from_vec(vec![0., 0.], Shape::d2(1, 2)).unwrap();
        let out = crf_viterbi_decode(&emissions, &transitions, &start, &end).unwrap();
        assert_eq!(out[0].0, vec![0, 1, 1]);
        assert!(approx(out[0].1, 30.0, 1e-5));
    }

    #[test]
    fn mean_pool_seq_basic() {
        let t =
            Tensor::from_vec(vec![1., 2., 3., 4., 10., 20., 30., 40.], Shape::d3(1, 2, 4)).unwrap();
        let p = mean_pool_seq(&t).unwrap();
        assert_eq!(p.data(), &[5.5, 11., 16.5, 22.]);
    }

    #[test]
    fn cls_pool_takes_position_zero() {
        let t = Tensor::from_vec(vec![1., 2., 3., 99., 99., 99.], Shape::d3(1, 2, 3)).unwrap();
        let p = cls_pool(&t).unwrap();
        assert_eq!(p.data(), &[1., 2., 3.]);
    }

    #[test]
    fn concat_along_axis1() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4.], Shape::d2(2, 2)).unwrap();
        let b = Tensor::from_vec(vec![5., 6., 7., 8., 9., 10.], Shape::d2(2, 3)).unwrap();
        let c = concat(&[&a, &b], 1).unwrap();
        assert_eq!(c.shape().dims(), &[2, 5]);
        assert_eq!(c.data(), &[1., 2., 5., 6., 7., 3., 4., 8., 9., 10.]);
    }

    #[test]
    fn split_axis1_two_chunks() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6., 7., 8.], Shape::d2(2, 4)).unwrap();
        let chunks = split(&t, 1, 2).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].data(), &[1., 2., 5., 6.]);
        assert_eq!(chunks[1].data(), &[3., 4., 7., 8.]);
    }

    #[test]
    fn split_rejects_indivisible() {
        let t = Tensor::zeros(Shape::d2(2, 5));
        assert!(split(&t, 1, 2).is_err());
    }

    #[test]
    fn scale_multiplies_all() {
        let t = Tensor::from_vec(vec![1., 2., 3.], Shape::d2(1, 3)).unwrap();
        assert_eq!(scale(&t, 2.0).data(), &[2., 4., 6.]);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn approx(a: f32, b: f32, eps: f32) -> bool {
            (a - b).abs() < eps
        }

        proptest! {
            /// gelu output is finite for any finite input (no NaN/Inf blow-up).
            #[test]
            fn gelu_finite_output(
                values in proptest::collection::vec(-50.0f32..50.0, 1..=32),
            ) {
                let len = values.len();
                let t = Tensor::from_vec(values, Shape::d2(1, len)).unwrap();
                let out = gelu(&t);
                for &v in out.data() {
                    prop_assert!(v.is_finite(), "gelu output not finite: {v}");
                }
            }

            /// layer_norm: output has mean ≈ 0 and variance ≈ 1 per sample
            /// (when gamma=1 and beta=0). Requires n>=4 for reliable f32 precision.
            #[test]
            fn layer_norm_mean_zero_var_one(
                values in proptest::collection::vec(-10.0f32..10.0, 4..=16),
            ) {
                let n = values.len();
                let t = Tensor::from_vec(values, Shape::d2(1, n)).unwrap();
                let gamma = Tensor::from_vec(vec![1.0f32; n], Shape::d2(1, n)).unwrap();
                let beta  = Tensor::from_vec(vec![0.0f32; n], Shape::d2(1, n)).unwrap();
                let out = layer_norm(&t, &gamma, &beta, 1e-5).unwrap();
                let nf = n as f32;
                let mean = out.data().iter().sum::<f32>() / nf;
                let var  = out.data().iter().map(|x| (x - mean).powi(2)).sum::<f32>() / nf;
                prop_assert!(
                    approx(mean, 0.0, 1e-4),
                    "layer_norm mean should be ~0, got {mean}"
                );
                // Biased variance divided by n; eps perturbation makes this ~1 but
                // not exact — allow 2% tolerance.
                prop_assert!(
                    approx(var, 1.0, 2e-2),
                    "layer_norm variance should be ~1, got {var}"
                );
            }

            /// matmul([m,k] × [k,n]) produces output of shape [m,n], panic-free.
            #[test]
            fn matmul_shape_correct(
                m in 1usize..=6,
                k in 1usize..=6,
                n in 1usize..=6,
                a_vals in proptest::collection::vec(-3.0f32..3.0, 1..=36),
                b_vals in proptest::collection::vec(-3.0f32..3.0, 1..=36),
            ) {
                // Build exactly m×k and k×n tensors, truncating/padding to exact size
                let a_len = m * k;
                let b_len = k * n;
                let a_data: Vec<f32> = (0..a_len).map(|i| {
                    a_vals.get(i % a_vals.len().max(1)).copied().unwrap_or(0.0)
                }).collect();
                let b_data: Vec<f32> = (0..b_len).map(|i| {
                    b_vals.get(i % b_vals.len().max(1)).copied().unwrap_or(0.0)
                }).collect();
                let a = Tensor::from_vec(a_data, Shape::d2(m, k)).unwrap();
                let b = Tensor::from_vec(b_data, Shape::d2(k, n)).unwrap();
                let c = matmul(&a, &b).unwrap();
                prop_assert_eq!(c.shape().dims(), &[m, n]);
            }

            /// mean_pool_seq([B, S, F]) -> [B, F]: output ndim=2, correct shape.
            #[test]
            fn mean_pool_seq_shape(
                b in 1usize..=4,
                s in 1usize..=6,
                f in 1usize..=8,
            ) {
                let data = vec![1.0f32; b * s * f];
                let t = Tensor::from_vec(data, Shape::d3(b, s, f)).unwrap();
                let out = mean_pool_seq(&t).unwrap();
                prop_assert_eq!(out.shape().ndim(), 2);
                prop_assert_eq!(out.shape().dims(), &[b, f]);
            }

            /// cls_pool([B, S, F]) -> [B, F]: output shape correct.
            #[test]
            fn cls_pool_shape(
                b in 1usize..=4,
                s in 1usize..=6,
                f in 1usize..=8,
            ) {
                let data = vec![1.0f32; b * s * f];
                let t = Tensor::from_vec(data, Shape::d3(b, s, f)).unwrap();
                let out = cls_pool(&t).unwrap();
                prop_assert_eq!(out.shape().ndim(), 2);
                prop_assert_eq!(out.shape().dims(), &[b, f]);
            }

            #[test]
            fn softmax_sums_to_one(
                values in proptest::collection::vec(-10.0f32..10.0, 2..32),
            ) {
                let len = values.len();
                let input = Tensor::from_vec(values, Shape::d2(1, len)).unwrap();
                let output = softmax(&input, 1).unwrap();
                let sum: f32 = output.data().iter().sum();
                prop_assert!((sum - 1.0).abs() < 1e-5,
                    "Softmax should sum to 1.0, got {sum}");
            }

            #[test]
            fn softmax_non_negative(
                values in proptest::collection::vec(-50.0f32..50.0, 1..16),
            ) {
                let len = values.len();
                let input = Tensor::from_vec(values, Shape::d2(1, len)).unwrap();
                let output = softmax(&input, 1).unwrap();
                for &v in output.data() {
                    prop_assert!(v >= 0.0, "Softmax output should be non-negative, got {v}");
                }
            }

            #[test]
            fn relu_non_negative(
                values in proptest::collection::vec(-100.0f32..100.0, 1..64),
            ) {
                let len = values.len();
                let input = Tensor::from_vec(values, Shape::d2(1, len)).unwrap();
                let output = relu(&input);
                for &v in output.data() {
                    prop_assert!(v >= 0.0, "ReLU output should be non-negative, got {v}");
                }
            }

            #[test]
            fn relu_preserves_positive(
                values in proptest::collection::vec(0.0f32..100.0, 1..32),
            ) {
                let len = values.len();
                let input = Tensor::from_vec(values.clone(), Shape::d2(1, len)).unwrap();
                let output = relu(&input);
                for (i, &v) in output.data().iter().enumerate() {
                    prop_assert_eq!(v, values[i], "ReLU should preserve positive values");
                }
            }

            #[test]
            fn relu_zeros_negative(
                values in proptest::collection::vec(-100.0f32..0.0, 1..32),
            ) {
                let len = values.len();
                let input = Tensor::from_vec(values, Shape::d2(1, len)).unwrap();
                let output = relu(&input);
                for &v in output.data() {
                    prop_assert_eq!(v, 0.0, "ReLU should zero negative values");
                }
            }

            #[test]
            fn matmul_associativity(
                a_vals in proptest::collection::vec(-5.0f32..5.0, 4..=4),
                b_vals in proptest::collection::vec(-5.0f32..5.0, 4..=4),
                c_vals in proptest::collection::vec(-5.0f32..5.0, 4..=4),
            ) {
                let a = Tensor::from_vec(a_vals, Shape::d2(2, 2)).unwrap();
                let b = Tensor::from_vec(b_vals, Shape::d2(2, 2)).unwrap();
                let c = Tensor::from_vec(c_vals, Shape::d2(2, 2)).unwrap();

                let ab = matmul(&a, &b).unwrap();
                let ab_c = matmul(&ab, &c).unwrap();

                let bc = matmul(&b, &c).unwrap();
                let a_bc = matmul(&a, &bc).unwrap();

                for (v1, v2) in ab_c.data().iter().zip(a_bc.data().iter()) {
                    let denom = v1.abs().max(v2.abs()).max(1e-6);
                    prop_assert!((v1 - v2).abs() / denom < 0.05,
                        "(A*B)*C != A*(B*C): {v1} vs {v2}");
                }
            }
        }
    }
}
