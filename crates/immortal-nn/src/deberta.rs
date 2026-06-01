//! DeBERTa-v2/v3 sequence classifier with disentangled attention.
//!
//! Implements the DeBERTa-v2 architecture (Microsoft) as used by the
//! `tasksource` NLI models: word embedding + LayerNorm (no absolute position,
//! no token-type for v3), N layers of *disentangled* self-attention
//! (content-to-content + content-to-position + position-to-content with
//! bucketed relative-position embeddings, `share_att_key`), a `ContextPooler`
//! (GELU(dense(first-token))), and a linear classifier.
//!
//! Mirrors HF `transformers.models.deberta_v2.modeling_deberta_v2` exactly
//! (verified logit-for-logit). For query `i`, key `j`, head `h`:
//! ```text
//! idx          = clamp(bucket(i - j) + att_span, 0, 2*att_span - 1)
//! score[i][j]  = (q[i]·k[j] + q[i]·pos_k[idx] + k[j]·pos_q[idx]) / sqrt(hd * 3)
//! ```
//! where `pos_q = query_proj(rel_emb)`, `pos_k = key_proj(rel_emb)`
//! (share_att_key), and `bucket` is identity for short sequences.

use crate::blocks::{Embedding, LayerNorm, Linear};
use crate::format::SovereignModel;
use crate::ops;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult, ARCH_DEBERTA_V2_CLASSIFIER};
use std::path::Path;

use crate::heads::ClassPrediction;

/// Error function (Abramowitz & Stegun 7.1.26, max abs error ~1.5e-7).
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

/// Exact GELU (erf form), matching PyTorch `nn.functional.gelu` default.
fn gelu_exact(t: &Tensor) -> Tensor {
    let mut out = t.clone();
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    for v in out.data_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * inv_sqrt2));
    }
    out
}

/// Log-bucket a relative position (DeBERTa-v2 `make_log_bucket_position`).
/// Identity for `|rel| < bucket_size/2`; the COPA sequences never trigger the
/// log branch, but the full form keeps it correct for long inputs.
fn log_bucket(rel: i64, bucket_size: i64, max_position: i64) -> i64 {
    let mid = bucket_size / 2;
    let sign = if rel < 0 { -1 } else { 1 };
    let abs_pos = if rel < mid && rel > -mid {
        mid - 1
    } else {
        rel.abs()
    };
    if abs_pos <= mid {
        rel
    } else {
        let lp = (((abs_pos as f64 / mid as f64).ln()
            / ((max_position - 1) as f64 / mid as f64).ln())
            * (mid - 1) as f64)
            .ceil() as i64
            + mid;
        lp * sign
    }
}

/// `idx` into the relative-position embedding for `delta = i - j`.
fn rel_idx(delta: i64, bucket_size: i64, max_position: i64, att_span: i64) -> usize {
    let b = log_bucket(delta, bucket_size, max_position);
    (b + att_span).clamp(0, att_span * 2 - 1) as usize
}

#[inline]
fn dot(a: &[f32], ai: usize, b: &[f32], bi: usize, hd: usize) -> f32 {
    let mut s = 0.0f32;
    for d in 0..hd {
        s += a[ai + d] * b[bi + d];
    }
    s
}

/// One DeBERTa-v2 layer: disentangled self-attention + FFN (both post-norm).
struct DebertaLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    attn_out: Linear,
    attn_norm: LayerNorm,
    ffn_up: Linear,
    ffn_down: Linear,
    ffn_norm: LayerNorm,
}

impl DebertaLayer {
    fn load(model: &SovereignModel, prefix: &str) -> NnResult<Self> {
        Ok(Self {
            q_proj: Linear::load(model, &format!("{prefix}.q_proj"))?,
            k_proj: Linear::load(model, &format!("{prefix}.k_proj"))?,
            v_proj: Linear::load(model, &format!("{prefix}.v_proj"))?,
            attn_out: Linear::load(model, &format!("{prefix}.attn_out"))?,
            attn_norm: LayerNorm::load(model, &format!("{prefix}.attn_norm"))?,
            ffn_up: Linear::load(model, &format!("{prefix}.ffn_up"))?,
            ffn_down: Linear::load(model, &format!("{prefix}.ffn_down"))?,
            ffn_norm: LayerNorm::load(model, &format!("{prefix}.ffn_norm"))?,
        })
    }

    /// `x: [S, H]`; `pos_q`/`pos_k`: cached `[2*att_span, H]` position
    /// projections (query/key proj of the LayerNorm'd rel-embeddings, constant
    /// across forwards under share_att_key).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        pos_q: &Tensor,
        pos_k: &Tensor,
        num_heads: usize,
        hd: usize,
        att_span: i64,
        bucket_size: i64,
        max_position: i64,
    ) -> NnResult<Tensor> {
        let s = x.shape().dims()[0];
        let h = num_heads * hd;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let (qd, kd, vd, pqd, pkd) = (q.data(), k.data(), v.data(), pos_q.data(), pos_k.data());

        let scale = ((hd * 3) as f32).sqrt();
        // Precompute idx for every (i - j) delta in [-(S-1), S-1].
        let mut ctx = vec![0.0f32; s * h];
        for head in 0..num_heads {
            let off = head * hd;
            for i in 0..s {
                let qi = i * h + off;
                let mut scores = vec![0.0f32; s];
                for j in 0..s {
                    let kj = j * h + off;
                    let idx = rel_idx(i as i64 - j as i64, bucket_size, max_position, att_span);
                    let pos = idx * h + off;
                    let content = dot(qd, qi, kd, kj, hd);
                    let c2p = dot(qd, qi, pkd, pos, hd);
                    let p2c = dot(kd, kj, pqd, pos, hd);
                    scores[j] = (content + c2p + p2c) / scale;
                }
                // softmax over keys
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    sum += *sc;
                }
                let inv = 1.0 / sum;
                // context = sum_j prob_j * v_j
                for (j, &sj) in scores.iter().enumerate() {
                    let p = sj * inv;
                    let vj = j * h + off;
                    let ci = i * h + off;
                    for d in 0..hd {
                        ctx[ci + d] += p * vd[vj + d];
                    }
                }
            }
        }

        let ctx_t = Tensor::from_vec(ctx, Shape::d2(s, h))?;
        // attention output: dense -> residual -> LayerNorm (post-norm)
        let attn = self.attn_out.forward(&ctx_t)?;
        let x1 = self.attn_norm.forward(&ops::add(x, &attn)?)?;
        // FFN: dense -> gelu -> dense -> residual -> LayerNorm
        let up = self.ffn_up.forward(&x1)?;
        let act = gelu_exact(&up);
        let down = self.ffn_down.forward(&act)?;
        self.ffn_norm.forward(&ops::add(&x1, &down)?)
    }
}

/// DeBERTa-v2/v3 sequence classifier (e.g. tasksource NLI head).
pub struct DebertaClassifier {
    embedding: Embedding,
    embed_norm: LayerNorm,
    /// Per-layer cached position projections `(pos_q, pos_k)` = query/key proj of
    /// the LayerNorm'd relative-position embeddings (constant across forwards
    /// under share_att_key; precomputed once at load).
    rel_proj: Vec<(Tensor, Tensor)>,
    layers: Vec<DebertaLayer>,
    pooler_dense: Linear,
    classifier: Linear,
    num_heads: usize,
    att_span: i64,
    max_position: i64,
    label_map: Vec<String>,
}

impl DebertaClassifier {
    /// Load a DeBERTa-v2/v3 classifier from a `.sovereign` file.
    pub fn load(path: &Path) -> NnResult<Self> {
        let model = SovereignModel::load(path)?;
        let arch = model.architecture();
        if arch != ARCH_DEBERTA_V2_CLASSIFIER {
            return Err(NnError::UnknownArchitecture(arch.to_string()));
        }

        let embedding = Embedding::load(&model, "embedding")?;
        let embed_norm = LayerNorm::load(&model, "embed_norm")?;
        let rel_embeddings = model.tensor("rel_embeddings.weight")?.clone();
        let rel_norm = LayerNorm::load(&model, "rel_norm")?;
        let pooler_dense = Linear::load(&model, "pooler_dense")?;
        let classifier = Linear::load(&model, "classifier")?;

        let num_layers: usize = model
            .metadata("num_layers")
            .and_then(|s| s.parse().ok())
            .unwrap_or(12);
        let num_heads: usize = model
            .metadata("num_heads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(12);
        // att_span = pos_ebd_size = rel_embeddings rows / 2 (= position_buckets).
        let att_span = (rel_embeddings.shape().dims()[0] / 2) as i64;
        let max_position: i64 = model
            .metadata("max_seq_len")
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(DebertaLayer::load(&model, &format!("encoder.layer.{i}"))?);
        }

        // Precompute per-layer position projections ONCE (q/k proj of the
        // LayerNorm'd relative-position embeddings). Constant across forwards
        // under share_att_key -> turns a large per-forward cost into a one-time
        // load cost (bit-identical to recomputing each forward).
        let rel_emb = rel_norm.forward(&rel_embeddings)?;
        let mut rel_proj = Vec::with_capacity(layers.len());
        for layer in &layers {
            rel_proj.push((
                layer.q_proj.forward(&rel_emb)?,
                layer.k_proj.forward(&rel_emb)?,
            ));
        }

        let label_map = crate::heads::parse_label_map(&model)?;

        Ok(Self {
            embedding,
            embed_norm,
            rel_proj,
            layers,
            pooler_dense,
            classifier,
            num_heads,
            att_span,
            max_position,
            label_map,
        })
    }

    /// Raw classifier logits `[num_labels]` (pre-softmax).
    pub fn logits(&self, token_ids: &[u32]) -> NnResult<Vec<f32>> {
        let hd = self.embedding.embed_dim() / self.num_heads;
        let bucket_size = self.att_span * 2; // position_buckets

        // Embeddings: word -> LayerNorm (no position / token-type for v3).
        let emb = self
            .embed_norm
            .forward(&self.embedding.forward(token_ids)?)?;
        let mut x = emb;
        for (i, layer) in self.layers.iter().enumerate() {
            let (pos_q, pos_k) = &self.rel_proj[i];
            x = layer.forward(
                &x,
                pos_q,
                pos_k,
                self.num_heads,
                hd,
                self.att_span,
                bucket_size,
                self.max_position,
            )?;
        }

        // ContextPooler: first token -> dense -> GELU.
        let h = self.embedding.embed_dim();
        let first = Tensor::from_vec(x.data()[0..h].to_vec(), Shape::d2(1, h))?;
        let pooled = gelu_exact(&self.pooler_dense.forward(&first)?);
        let logits = self.classifier.forward(&pooled)?;
        Ok(logits.data().to_vec())
    }

    /// Predict the best label (softmax argmax).
    pub fn predict(&self, token_ids: &[u32]) -> NnResult<ClassPrediction> {
        let raw = self.logits(token_ids)?;
        let logits = Tensor::from_vec(raw.clone(), Shape::d2(1, raw.len()))?;
        let probs = ops::softmax(&logits, 1)?;
        let scores = probs.data();
        let (best_id, &best) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let label = self
            .label_map
            .get(best_id)
            .cloned()
            .unwrap_or_else(|| format!("CLASS_{best_id}"));
        Ok(ClassPrediction {
            label,
            label_id: best_id,
            score: best,
            all_scores: scores.to_vec(),
        })
    }
}
