//! Llama-family causal language model + greedy generation.
//!
//! Implements the LLaMA decoder (Meta) as used by Llama-2/3 and Llama-derived
//! fine-tunes: token embedding, N pre-norm decoder layers (RMSNorm + RoPE +
//! grouped-query causal self-attention + SwiGLU MLP), a final RMSNorm, and an
//! `lm_head` (tied to the embedding when absent). Mirrors HF
//! `transformers.models.llama.modeling_llama` exactly:
//! ```text
//! h = h + attn(rmsnorm1(h));   h = h + swiglu(rmsnorm2(h))
//! attn: q,k,v = proj(h); rope(q,k); repeat_kv(k,v); softmax(qk^T/sqrt(hd) + causal)·v
//! rmsnorm(x) = x * rsqrt(mean(x^2) + eps) * weight   (no mean-subtraction/bias)
//! ```
//! Greedy generation (deterministic) with a per-layer KV cache (O(n)/step),
//! verified bit-identical to the full-recompute path.

use crate::blocks::{Embedding, Linear};
use crate::format::SovereignModel;
use crate::ops;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult, ARCH_CAUSAL_LM_LLAMA};
use std::path::Path;

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RMSNorm over the last dim: `x * rsqrt(mean(x^2) + eps) * weight`.
struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    fn load(model: &SovereignModel, name: &str, eps: f32) -> NnResult<Self> {
        Ok(Self {
            weight: model.tensor(name)?.data().to_vec(),
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        let dims = x.shape().dims();
        let (s, h) = (dims[0], dims[1]);
        let d = x.data();
        let mut out = vec![0.0f32; s * h];
        for i in 0..s {
            let row = &d[i * h..(i + 1) * h];
            let ss: f32 = row.iter().map(|v| v * v).sum::<f32>() / h as f32;
            let inv = 1.0 / (ss + self.eps).sqrt();
            for j in 0..h {
                out[i * h + j] = row[j] * inv * self.weight[j];
            }
        }
        Tensor::from_vec(out, Shape::d2(s, h))
    }
}

/// Apply RoPE in place to one head vector `v[..hd]` at absolute position `pos`.
/// Rotates the (d, d+hd/2) pairs by angle `pos * theta^(-2d/hd)` (HF rotate_half).
fn apply_rope(v: &mut [f32], pos: usize, theta: f32) {
    let hd = v.len();
    let half = hd / 2;
    for d in 0..half {
        let inv_freq = theta.powf(-2.0 * d as f32 / hd as f32);
        let ang = pos as f32 * inv_freq;
        let (s, c) = ang.sin_cos();
        let x = v[d];
        let y = v[d + half];
        v[d] = x * c - y * s;
        v[d + half] = y * c + x * s;
    }
}

/// Per-layer KV cache for incremental generation: `k`/`v` hold the projected,
/// RoPE'd keys/values for all positions seen so far, row-major `[len, num_kv*hd]`.
struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
}

impl LayerCache {
    fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
        }
    }
}

/// One LLaMA decoder layer (pre-norm).
struct LlamaLayer {
    norm1: RmsNorm,
    norm2: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ffn_gate: Linear,
    ffn_up: Linear,
    ffn_down: Linear,
}

impl LlamaLayer {
    fn load(model: &SovereignModel, prefix: &str, eps: f32) -> NnResult<Self> {
        Ok(Self {
            norm1: RmsNorm::load(model, &format!("{prefix}.norm1.weight"), eps)?,
            norm2: RmsNorm::load(model, &format!("{prefix}.norm2.weight"), eps)?,
            q_proj: Linear::load(model, &format!("{prefix}.attention.q_proj"))?,
            k_proj: Linear::load(model, &format!("{prefix}.attention.k_proj"))?,
            v_proj: Linear::load(model, &format!("{prefix}.attention.v_proj"))?,
            o_proj: Linear::load(model, &format!("{prefix}.attention.o_proj"))?,
            ffn_gate: Linear::load(model, &format!("{prefix}.ffn_gate"))?,
            ffn_up: Linear::load(model, &format!("{prefix}.ffn_up"))?,
            ffn_down: Linear::load(model, &format!("{prefix}.ffn_down"))?,
        })
    }

    /// `x: [S, H]` -> `[S, H]`. Causal GQA self-attention + SwiGLU, both residual.
    fn forward(
        &self,
        x: &Tensor,
        num_heads: usize,
        num_kv: usize,
        hd: usize,
        theta: f32,
    ) -> NnResult<Tensor> {
        let s = x.shape().dims()[0];
        let h = num_heads * hd;
        let kvh = num_kv * hd;
        let n_rep = num_heads / num_kv;

        // --- attention ---
        let normed = self.norm1.forward(x)?;
        let q = self.q_proj.forward(&normed)?; // [S, num_heads*hd]
        let k = self.k_proj.forward(&normed)?; // [S, num_kv*hd]
        let v = self.v_proj.forward(&normed)?;
        let (mut qd, mut kd) = (q.data().to_vec(), k.data().to_vec());
        let vd = v.data();
        // RoPE on each query/key head at its position.
        for i in 0..s {
            for hh in 0..num_heads {
                apply_rope(&mut qd[i * h + hh * hd..i * h + hh * hd + hd], i, theta);
            }
            for hh in 0..num_kv {
                apply_rope(&mut kd[i * kvh + hh * hd..i * kvh + hh * hd + hd], i, theta);
            }
        }
        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0.0f32; s * h];
        for hh in 0..num_heads {
            let kv = hh / n_rep; // grouped-query: which kv head this query head reads
            let qoff = hh * hd;
            let koff = kv * hd;
            for i in 0..s {
                // causal: attend to keys 0..=i
                let mut scores = vec![0.0f32; i + 1];
                let qi = i * h + qoff;
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kj = j * kvh + koff;
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += qd[qi + d] * kd[kj + d];
                    }
                    *sc = dot * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    sum += *sc;
                }
                let inv = 1.0 / sum;
                let ci = i * h + qoff;
                for (j, &sc) in scores.iter().enumerate() {
                    let p = sc * inv;
                    let vj = j * kvh + koff;
                    for d in 0..hd {
                        ctx[ci + d] += p * vd[vj + d];
                    }
                }
            }
        }
        let ctx_t = Tensor::from_vec(ctx, Shape::d2(s, h))?;
        let attn = self.o_proj.forward(&ctx_t)?;
        let x = ops::add(x, &attn)?;

        // --- SwiGLU MLP ---
        let normed = self.norm2.forward(&x)?;
        let gate = self.ffn_gate.forward(&normed)?;
        let up = self.ffn_up.forward(&normed)?;
        let gd = gate.data();
        let ud = up.data();
        let mut act = vec![0.0f32; gd.len()];
        for n in 0..gd.len() {
            act[n] = silu(gd[n]) * ud[n];
        }
        let inter = gate.shape().dims()[1];
        let act_t = Tensor::from_vec(act, Shape::d2(s, inter))?;
        let down = self.ffn_down.forward(&act_t)?;
        ops::add(&x, &down)
    }

    /// Incremental forward with a KV cache: process `new_x` (the new tokens at
    /// global positions `cache.len..cache.len+n_new`), append their k/v to the
    /// cache, and attend each new query against ALL cached keys (causal).
    /// Bit-identical to `forward` over a full sequence, but O(n) per step.
    #[allow(clippy::too_many_arguments)]
    fn forward_incremental(
        &self,
        new_x: &Tensor,
        cache: &mut LayerCache,
        num_heads: usize,
        num_kv: usize,
        hd: usize,
        theta: f32,
    ) -> NnResult<Tensor> {
        let n_new = new_x.shape().dims()[0];
        let h = num_heads * hd;
        let kvh = num_kv * hd;
        let n_rep = num_heads / num_kv;
        let past = cache.len;

        let normed = self.norm1.forward(new_x)?;
        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;
        let mut qd = q.data().to_vec();
        let mut kd = k.data().to_vec();
        for i in 0..n_new {
            let pos = past + i;
            for hh in 0..num_heads {
                apply_rope(&mut qd[i * h + hh * hd..i * h + hh * hd + hd], pos, theta);
            }
            for hh in 0..num_kv {
                apply_rope(
                    &mut kd[i * kvh + hh * hd..i * kvh + hh * hd + hd],
                    pos,
                    theta,
                );
            }
        }
        cache.k.extend_from_slice(&kd);
        cache.v.extend_from_slice(v.data());
        cache.len = past + n_new;

        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0.0f32; n_new * h];
        for hh in 0..num_heads {
            let kv = hh / n_rep;
            let qoff = hh * hd;
            let koff = kv * hd;
            for i in 0..n_new {
                let gpos = past + i; // attend to cached keys 0..=gpos
                let qi = i * h + qoff;
                let mut scores = vec![0.0f32; gpos + 1];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kj = j * kvh + koff;
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += qd[qi + d] * cache.k[kj + d];
                    }
                    *sc = dot * scale;
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    sum += *sc;
                }
                let inv = 1.0 / sum;
                let ci = i * h + qoff;
                for (j, &sc) in scores.iter().enumerate() {
                    let p = sc * inv;
                    let vj = j * kvh + koff;
                    for d in 0..hd {
                        ctx[ci + d] += p * cache.v[vj + d];
                    }
                }
            }
        }
        let ctx_t = Tensor::from_vec(ctx, Shape::d2(n_new, h))?;
        let attn = self.o_proj.forward(&ctx_t)?;
        let x = ops::add(new_x, &attn)?;
        let normed = self.norm2.forward(&x)?;
        let gate = self.ffn_gate.forward(&normed)?;
        let up = self.ffn_up.forward(&normed)?;
        let gd = gate.data();
        let ud = up.data();
        let mut act = vec![0.0f32; gd.len()];
        for n in 0..gd.len() {
            act[n] = silu(gd[n]) * ud[n];
        }
        let inter = gate.shape().dims()[1];
        let act_t = Tensor::from_vec(act, Shape::d2(n_new, inter))?;
        let down = self.ffn_down.forward(&act_t)?;
        ops::add(&x, &down)
    }
}

/// LLaMA-family causal language model with greedy generation.
pub struct CausalLm {
    embedding: Embedding,
    layers: Vec<LlamaLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    num_heads: usize,
    num_kv: usize,
    head_dim: usize,
    theta: f32,
}

impl CausalLm {
    /// Load a Llama-family causal LM from a `.sovereign` file.
    pub fn load(path: &Path) -> NnResult<Self> {
        let model = SovereignModel::load(path)?;
        if model.architecture() != ARCH_CAUSAL_LM_LLAMA {
            return Err(NnError::UnknownArchitecture(
                model.architecture().to_string(),
            ));
        }
        let meta = |k: &str, dflt: usize| -> usize {
            model
                .metadata(k)
                .and_then(|s| s.parse().ok())
                .unwrap_or(dflt)
        };
        let num_layers = meta("num_layers", 0);
        let num_heads = meta("num_heads", 0);
        let hidden = meta("hidden_dim", 0);
        let num_kv = meta("num_kv_heads", num_heads);
        let head_dim = if num_heads > 0 { hidden / num_heads } else { 0 };
        let theta: f32 = model
            .metadata("rope_theta")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000.0);
        let eps: f32 = model
            .metadata("rms_norm_eps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-5);

        let embedding = Embedding::load(&model, "embedding")?;
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LlamaLayer::load(
                &model,
                &format!("decoder.layer.{i}"),
                eps,
            )?);
        }
        let final_norm = RmsNorm::load(&model, "final_norm.weight", eps)?;
        // lm_head, tied to the embedding table when no separate tensor exists.
        let lm_head = match Linear::load(&model, "lm_head") {
            Ok(l) => l,
            Err(_) => Linear::from_tensors(model.tensor("embedding.weight")?.clone(), None)?,
        };

        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            num_heads,
            num_kv,
            head_dim,
            theta,
        })
    }

    /// Logits over the vocabulary for the LAST position of `token_ids`.
    pub fn last_logits(&self, token_ids: &[u32]) -> NnResult<Vec<f32>> {
        let s = token_ids.len();
        let mut x = self.embedding.forward(token_ids)?; // [S, H]
        for layer in &self.layers {
            x = layer.forward(&x, self.num_heads, self.num_kv, self.head_dim, self.theta)?;
        }
        let x = self.final_norm.forward(&x)?;
        // last row -> lm_head
        let h = self.embedding.embed_dim();
        let last = Tensor::from_vec(x.data()[(s - 1) * h..s * h].to_vec(), Shape::d2(1, h))?;
        Ok(self.lm_head.forward(&last)?.data().to_vec())
    }

    /// Argmax next-token id from a single hidden-state row `[H]`
    /// (`final_norm` -> `lm_head` -> argmax).
    fn argmax_next(&self, last_hidden: &[f32]) -> NnResult<u32> {
        let h = self.embedding.embed_dim();
        let normed = self
            .final_norm
            .forward(&Tensor::from_vec(last_hidden.to_vec(), Shape::d2(1, h))?)?;
        let logits = self.lm_head.forward(&normed)?;
        Ok(logits
            .data()
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0 as u32)
    }

    /// Greedy-decode up to `max_new` tokens after `prompt`, stopping at `eos`,
    /// using a per-layer KV cache (O(n) per step). Returns the new token ids.
    pub fn generate(&self, prompt: &[u32], max_new: usize, eos: u32) -> NnResult<Vec<u32>> {
        if prompt.is_empty() {
            return Ok(Vec::new());
        }
        let h = self.embedding.embed_dim();
        let mut caches: Vec<LayerCache> =
            (0..self.layers.len()).map(|_| LayerCache::new()).collect();
        // Prime the cache with the prompt (one batched incremental step).
        let mut x = self.embedding.forward(prompt)?;
        for (li, layer) in self.layers.iter().enumerate() {
            x = layer.forward_incremental(
                &x,
                &mut caches[li],
                self.num_heads,
                self.num_kv,
                self.head_dim,
                self.theta,
            )?;
        }
        let s = prompt.len();
        let mut next = self.argmax_next(&x.data()[(s - 1) * h..s * h])?;
        let mut out = Vec::new();
        while out.len() < max_new && next != eos {
            out.push(next);
            let mut xn = self.embedding.forward(&[next])?;
            for (li, layer) in self.layers.iter().enumerate() {
                xn = layer.forward_incremental(
                    &xn,
                    &mut caches[li],
                    self.num_heads,
                    self.num_kv,
                    self.head_dim,
                    self.theta,
                )?;
            }
            next = self.argmax_next(&xn.data()[0..h])?;
        }
        Ok(out)
    }
}
