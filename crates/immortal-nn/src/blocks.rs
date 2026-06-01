//! Architecture building blocks (Section 3).
//!
//! Composable blocks that own their weights (loaded from `.sovereign`).
//! Inference-only — no training state, no optimizer, no dropout.
//!
//! Each block has a single `forward()` method.

use crate::format::SovereignModel;
use crate::ops;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

// ---------------------------------------------------------------------------
// Activation enum
// ---------------------------------------------------------------------------

/// Activation function selector, self-describing in `.sovereign` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// GELU (tanh approximation, HuggingFace default).
    GeluTanh,
    /// ReLU.
    Relu,
}

impl Activation {
    /// Parse from the string stored in `.sovereign` metadata.
    pub fn from_str(s: &str) -> NnResult<Self> {
        match s {
            "gelu" | "gelu_tanh" | "gelu_new" => Ok(Self::GeluTanh),
            "relu" => Ok(Self::Relu),
            _ => Err(NnError::Format(format!("unknown activation: {s:?}"))),
        }
    }

    /// Apply the activation element-wise.
    #[must_use]
    pub fn apply(&self, t: &Tensor) -> Tensor {
        match self {
            Self::GeluTanh => ops::gelu(t),
            Self::Relu => ops::relu(t),
        }
    }
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Fully connected layer. `weight: [out_features, in_features]`, optional `bias: [out_features]`.
#[derive(Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    out_features: usize,
    in_features: usize,
}

impl Linear {
    /// Stable identifier for registry lookup per the Algorithm
    /// Integration Contract v1.0 §6.2. Exposed as an inherent method
    /// to preserve the sovereign (zero-external-dep) contract of
    /// `immortal-nn`.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        "immortal-nn::blocks::Linear"
    }

    /// Load from a `SovereignModel` using the given name prefix.
    /// Expects `{prefix}.weight` `[out, in]` and optionally `{prefix}.bias` `[1, out]`.
    pub fn load(model: &SovereignModel, prefix: &str) -> NnResult<Self> {
        let w = model.tensor(&format!("{prefix}.weight"))?;
        if w.shape().ndim() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: format!("{prefix}.weight: rank-2 [out, in]"),
                got: format!("rank-{}", w.shape().ndim()),
            });
        }
        let out_features = w.shape().dims()[0];
        let in_features = w.shape().dims()[1];

        let bias = match model.tensor(&format!("{prefix}.bias")) {
            Ok(b) => Some(b.clone()),
            Err(_) => None,
        };

        Ok(Self {
            weight: w.clone(),
            bias,
            out_features,
            in_features,
        })
    }

    /// Construct directly from weight and optional bias tensors.
    pub fn from_tensors(weight: Tensor, bias: Option<Tensor>) -> NnResult<Self> {
        if weight.shape().ndim() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-2 [out, in]".into(),
                got: format!("rank-{}", weight.shape().ndim()),
            });
        }
        let out_features = weight.shape().dims()[0];
        let in_features = weight.shape().dims()[1];
        Ok(Self {
            weight,
            bias,
            out_features,
            in_features,
        })
    }

    /// `[*, in_features] -> [*, out_features]`.
    ///
    /// Flattens leading dimensions, applies `x @ W^T + bias`, reshapes back.
    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let nd = input.shape().ndim() as usize;
        let last = input.shape().dims()[nd - 1];
        if last != self.in_features {
            return Err(NnError::ShapeMismatch {
                expected: format!("last dim = {}", self.in_features),
                got: format!("last dim = {last}"),
            });
        }

        // Flatten to [batch, in_features]
        let batch: usize = input.data().len() / self.in_features;
        let flat_input = input.reshape(Shape::d2(batch, self.in_features))?;

        // W^T: [in, out]
        let wt = transpose_2d(&self.weight);
        // x @ W^T: [batch, out]
        let mut out = ops::matmul(&flat_input, &wt)?;

        // Add bias
        if let Some(ref b) = self.bias {
            let bd = b.data();
            let od = out.data_mut();
            for row in od.chunks_exact_mut(self.out_features) {
                for (i, v) in row.iter_mut().enumerate() {
                    *v += bd[i];
                }
            }
        }

        // Reshape back: replace last dim with out_features
        let mut out_dims = [1usize; 4];
        let out_nd = input.shape().ndim();
        out_dims[..nd].copy_from_slice(input.shape().dims());
        out_dims[nd - 1] = self.out_features;
        let out_shape = Shape::new(out_dims, out_nd)?;
        out.reshape(out_shape)
    }

    /// Output feature count.
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Input feature count.
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }
}

/// Transpose a 2D tensor `[m, n] -> [n, m]`.
fn transpose_2d(t: &Tensor) -> Tensor {
    let d = t.shape().dims();
    let (m, n) = (d[0], d[1]);
    let mut out = Tensor::zeros(Shape::d2(n, m));
    let src = t.data();
    let dst = out.data_mut();
    for i in 0..m {
        for j in 0..n {
            dst[j * m + i] = src[i * n + j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Embedding lookup table. `table: [vocab_size, embed_dim]`.
#[derive(Debug)]
pub struct Embedding {
    table: Tensor,
    vocab_size: usize,
    embed_dim: usize,
}

impl Embedding {
    /// Load from model. Expects `{prefix}.weight` `[vocab, dim]`.
    pub fn load(model: &SovereignModel, prefix: &str) -> NnResult<Self> {
        let t = model.tensor(&format!("{prefix}.weight"))?;
        if t.shape().ndim() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: format!("{prefix}.weight: rank-2 [vocab, dim]"),
                got: format!("rank-{}", t.shape().ndim()),
            });
        }
        Ok(Self {
            vocab_size: t.shape().dims()[0],
            embed_dim: t.shape().dims()[1],
            table: t.clone(),
        })
    }

    /// Construct directly.
    pub fn from_tensor(table: Tensor) -> NnResult<Self> {
        if table.shape().ndim() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-2 [vocab, dim]".into(),
                got: format!("rank-{}", table.shape().ndim()),
            });
        }
        let vocab_size = table.shape().dims()[0];
        let embed_dim = table.shape().dims()[1];
        Ok(Self {
            table,
            vocab_size,
            embed_dim,
        })
    }

    /// Look up token IDs. Returns `[seq_len, embed_dim]`.
    pub fn forward(&self, token_ids: &[u32]) -> NnResult<Tensor> {
        ops::embedding_lookup(&self.table, token_ids)
    }

    /// Vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embedding dimension.
    #[must_use]
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// Layer normalization over the last dimension.
#[derive(Debug)]
pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f32,
}

impl LayerNorm {
    /// Load from model. Expects `{prefix}.weight` (gamma) and `{prefix}.bias` (beta).
    pub fn load(model: &SovereignModel, prefix: &str) -> NnResult<Self> {
        let gamma = model.tensor(&format!("{prefix}.weight"))?.clone();
        let beta = model.tensor(&format!("{prefix}.bias"))?.clone();
        // Read eps from metadata if available, else default 1e-5
        let eps = model
            .metadata("layer_norm_eps")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1e-5);
        Ok(Self { gamma, beta, eps })
    }

    /// Construct directly.
    #[must_use]
    pub fn from_tensors(gamma: Tensor, beta: Tensor, eps: f32) -> Self {
        Self { gamma, beta, eps }
    }

    /// Normalize over the last dimension.
    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        ops::layer_norm(input, &self.gamma, &self.beta, self.eps)
    }
}

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

/// Multi-head attention with Q/K/V/O projections.
#[derive(Debug)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    /// Load from model. Expects `{prefix}.q_proj`, `k_proj`, `v_proj`, `o_proj`.
    pub fn load(model: &SovereignModel, prefix: &str, num_heads: usize) -> NnResult<Self> {
        let q_proj = Linear::load(model, &format!("{prefix}.q_proj"))?;
        let k_proj = Linear::load(model, &format!("{prefix}.k_proj"))?;
        let v_proj = Linear::load(model, &format!("{prefix}.v_proj"))?;
        let o_proj = Linear::load(model, &format!("{prefix}.o_proj"))?;

        let embed_dim = q_proj.in_features();
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(NnError::DivisibilityError {
                what: "embed_dim % num_heads".into(),
                numerator: embed_dim,
                denominator: num_heads,
            });
        }
        let head_dim = embed_dim / num_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    /// Construct directly from projection layers.
    pub fn from_parts(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: usize,
    ) -> NnResult<Self> {
        let embed_dim = q_proj.in_features();
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(NnError::DivisibilityError {
                what: "embed_dim % num_heads".into(),
                numerator: embed_dim,
                denominator: num_heads,
            });
        }
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    /// `[B, S, E] -> [B, S, E]` with optional mask `[B, S]`.
    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> NnResult<Tensor> {
        let dims = input.shape().dims();
        if input.shape().ndim() != 3 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-3 [B, S, E]".into(),
                got: format!("rank-{}", input.shape().ndim()),
            });
        }
        let (b, s, _e) = (dims[0], dims[1], dims[2]);

        // Project Q, K, V: [B, S, E] -> [B, S, E]
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Reshape to [B, S, H, D] then transpose to [B, H, S, D]
        let q4 = reshape_for_heads(&q, b, s, self.num_heads, self.head_dim)?;
        let k4 = reshape_for_heads(&k, b, s, self.num_heads, self.head_dim)?;
        let v4 = reshape_for_heads(&v, b, s, self.num_heads, self.head_dim)?;

        // Scaled dot-product attention
        let attn_out = ops::scaled_dot_product_attention(&q4, &k4, &v4, mask)?;

        // Reshape back: [B, H, S, D] -> [B, S, H*D] = [B, S, E]
        let merged = merge_heads(&attn_out, b, s, self.num_heads, self.head_dim)?;

        // Output projection
        self.o_proj.forward(&merged)
    }
}

/// `[B, S, H*D]` -> `[B, H, S, D]`
fn reshape_for_heads(t: &Tensor, b: usize, s: usize, h: usize, d: usize) -> NnResult<Tensor> {
    // First reshape to [B, S, H, D]
    let bshd = t.reshape(Shape::d4(b, s, h, d))?;
    // Then transpose axes 1 and 2: [B, S, H, D] -> [B, H, S, D]
    let mut out = Tensor::zeros(Shape::d4(b, h, s, d));
    let src = bshd.data();
    let dst = out.data_mut();
    for bi in 0..b {
        for si in 0..s {
            for hi in 0..h {
                for di in 0..d {
                    let src_idx = ((bi * s + si) * h + hi) * d + di;
                    let dst_idx = ((bi * h + hi) * s + si) * d + di;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    Ok(out)
}

/// `[B, H, S, D]` -> `[B, S, H*D]`
fn merge_heads(t: &Tensor, b: usize, s: usize, h: usize, d: usize) -> NnResult<Tensor> {
    // Transpose [B, H, S, D] -> [B, S, H, D]
    let mut transposed = Tensor::zeros(Shape::d4(b, s, h, d));
    let src = t.data();
    let dst = transposed.data_mut();
    for bi in 0..b {
        for hi in 0..h {
            for si in 0..s {
                for di in 0..d {
                    let src_idx = ((bi * h + hi) * s + si) * d + di;
                    let dst_idx = ((bi * s + si) * h + hi) * d + di;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    // Reshape [B, S, H, D] -> [B, S, H*D]
    transposed.reshape(Shape::d3(b, s, h * d))
}

// ---------------------------------------------------------------------------
// TransformerBlock (pre-norm)
// ---------------------------------------------------------------------------

/// Pre-norm transformer block.
///
/// ```text
/// x + attn(norm1(x))  then  x + ffn(norm2(x))
/// ```
#[derive(Debug)]
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ffn_up: Linear,
    ffn_down: Linear,
    activation: Activation,
}

impl TransformerBlock {
    /// Load from model with a prefix like `"encoder.layer.0"`.
    pub fn load(
        model: &SovereignModel,
        prefix: &str,
        num_heads: usize,
        activation: Activation,
    ) -> NnResult<Self> {
        Ok(Self {
            attention: MultiHeadAttention::load(model, &format!("{prefix}.attention"), num_heads)?,
            norm1: LayerNorm::load(model, &format!("{prefix}.norm1"))?,
            norm2: LayerNorm::load(model, &format!("{prefix}.norm2"))?,
            ffn_up: Linear::load(model, &format!("{prefix}.ffn_up"))?,
            ffn_down: Linear::load(model, &format!("{prefix}.ffn_down"))?,
            activation,
        })
    }

    /// Construct directly.
    #[must_use]
    pub fn from_parts(
        attention: MultiHeadAttention,
        norm1: LayerNorm,
        norm2: LayerNorm,
        ffn_up: Linear,
        ffn_down: Linear,
        activation: Activation,
    ) -> Self {
        Self {
            attention,
            norm1,
            norm2,
            ffn_up,
            ffn_down,
            activation,
        }
    }

    /// `[B, S, E] -> [B, S, E]`.
    pub fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> NnResult<Tensor> {
        // Pre-norm attention with residual
        let normed1 = self.norm1.forward(input)?;
        let attn_out = self.attention.forward(&normed1, mask)?;
        let x = ops::add(input, &attn_out)?;

        // Pre-norm FFN with residual
        let normed2 = self.norm2.forward(&x)?;
        let up = self.ffn_up.forward(&normed2)?;
        let activated = self.activation.apply(&up);
        let down = self.ffn_down.forward(&activated)?;
        ops::add(&x, &down)
    }

    /// Plan 65-NN Phase B (2026-05-04) — post-norm BERT-style forward
    /// pass on the same loaded weights. Used by callers that load
    /// BERT/RoBERTa-family models which were trained with post-norm.
    ///
    /// ```text
    /// y1 = norm1(input + attn(input))
    /// y2 = norm2(y1 + ffn(y1))
    /// ```
    ///
    /// This mirrors `XlmRobertaLayer::forward` (which has been working
    /// correctly for the bge-reranker `.sovereign` runtime) and is the
    /// fix for the SOVEREIGN-EMBEDDER-MINILM-CORRECTNESS bug — the
    /// `crate::blocks::TransformerBlock::forward` (pre-norm) was
    /// being applied to MiniLM-L6-v2 weights that expect post-norm,
    /// producing embeddings uncorrelated with the HF Python reference.
    pub fn forward_post_norm(&self, input: &Tensor, mask: Option<&Tensor>) -> NnResult<Tensor> {
        let attn_out = self.attention.forward(input, mask)?;
        let attn_residual = ops::add(input, &attn_out)?;
        let attn_normed = self.norm1.forward(&attn_residual)?;

        let up = self.ffn_up.forward(&attn_normed)?;
        let activated = self.activation.apply(&up);
        let down = self.ffn_down.forward(&activated)?;
        let ffn_residual = ops::add(&attn_normed, &down)?;
        self.norm2.forward(&ffn_residual)
    }
}

// ---------------------------------------------------------------------------
// LstmCell
// ---------------------------------------------------------------------------

/// Single LSTM cell. Gate order: `i, f, g, o` (locked).
///
/// `w_ih: [4*hidden, input]`, `w_hh: [4*hidden, hidden]`,
/// `b_ih: [1, 4*hidden]`, `b_hh: [1, 4*hidden]`.
#[derive(Debug)]
pub struct LstmCell {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    hidden_size: usize,
}

impl LstmCell {
    /// Load from model.
    pub fn load(model: &SovereignModel, prefix: &str) -> NnResult<Self> {
        let w_ih = model.tensor(&format!("{prefix}.w_ih"))?.clone();
        let w_hh = model.tensor(&format!("{prefix}.w_hh"))?.clone();
        let b_ih = model.tensor(&format!("{prefix}.b_ih"))?.clone();
        let b_hh = model.tensor(&format!("{prefix}.b_hh"))?.clone();
        let hidden_size = w_hh.shape().dims()[1];
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
        })
    }

    /// Construct directly.
    pub fn from_tensors(w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> NnResult<Self> {
        let hidden_size = w_hh.shape().dims()[1];
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
        })
    }

    /// Run one step. `x: [1, input_size]`, `h: [1, hidden]`, `c: [1, hidden]`.
    /// Returns `(h_new, c_new)`.
    fn step(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> NnResult<(Tensor, Tensor)> {
        let hs = self.hidden_size;

        // gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
        let wih_t = transpose_2d(&self.w_ih);
        let whh_t = transpose_2d(&self.w_hh);
        let xw = ops::matmul(x, &wih_t)?;
        let hw = ops::matmul(h, &whh_t)?;

        // Sum all gate contributions
        let xwd = xw.data();
        let hwd = hw.data();
        let bih = self.b_ih.data();
        let bhh = self.b_hh.data();

        let mut gates = vec![0.0_f32; 4 * hs];
        for i in 0..4 * hs {
            gates[i] = xwd[i] + hwd[i] + bih[i] + bhh[i];
        }

        // Split gates: i, f, g, o
        let cd = c.data();
        let mut h_new_data = vec![0.0_f32; hs];
        let mut c_new_data = vec![0.0_f32; hs];

        for j in 0..hs {
            let i_gate = sigmoid_scalar(gates[j]);
            let f_gate = sigmoid_scalar(gates[hs + j]);
            let g_gate = gates[2 * hs + j].tanh();
            let o_gate = sigmoid_scalar(gates[3 * hs + j]);

            c_new_data[j] = f_gate * cd[j] + i_gate * g_gate;
            h_new_data[j] = o_gate * c_new_data[j].tanh();
        }

        let h_new = Tensor::from_vec(h_new_data, Shape::d2(1, hs))?;
        let c_new = Tensor::from_vec(c_new_data, Shape::d2(1, hs))?;
        Ok((h_new, c_new))
    }

    /// Process a full sequence `[S, input_size]`. Returns `[S, hidden]`.
    pub fn forward_seq(&self, input: &Tensor) -> NnResult<Tensor> {
        let seq_len = input.shape().dims()[0];
        let input_size = input.shape().dims()[1];
        let hs = self.hidden_size;

        let mut h = Tensor::zeros(Shape::d2(1, hs));
        let mut c = Tensor::zeros(Shape::d2(1, hs));
        let mut outputs = Vec::with_capacity(seq_len * hs);

        let src = input.data();
        for t in 0..seq_len {
            let x = Tensor::from_vec(
                src[t * input_size..(t + 1) * input_size].to_vec(),
                Shape::d2(1, input_size),
            )?;
            let (h_new, c_new) = self.step(&x, &h, &c)?;
            outputs.extend_from_slice(h_new.data());
            h = h_new;
            c = c_new;
        }

        Tensor::from_vec(outputs, Shape::d2(seq_len, hs))
    }
}

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// BiLstm
// ---------------------------------------------------------------------------

/// Bidirectional LSTM. Concatenates forward and backward outputs (not sums).
/// `[B, S, F] -> [B, S, 2*hidden]`.
#[derive(Debug)]
pub struct BiLstm {
    forward_cell: LstmCell,
    backward_cell: LstmCell,
    hidden_size: usize,
}

impl BiLstm {
    /// Load from model.
    pub fn load(model: &SovereignModel, prefix: &str) -> NnResult<Self> {
        let forward_cell = LstmCell::load(model, &format!("{prefix}.forward"))?;
        let backward_cell = LstmCell::load(model, &format!("{prefix}.backward"))?;
        let hidden_size = forward_cell.hidden_size;
        Ok(Self {
            forward_cell,
            backward_cell,
            hidden_size,
        })
    }

    /// Construct directly.
    pub fn from_cells(forward_cell: LstmCell, backward_cell: LstmCell) -> Self {
        let hidden_size = forward_cell.hidden_size;
        Self {
            forward_cell,
            backward_cell,
            hidden_size,
        }
    }

    /// `[B, S, F] -> [B, S, 2*hidden]`.
    pub fn forward(&self, input: &Tensor) -> NnResult<Tensor> {
        let dims = input.shape().dims();
        if input.shape().ndim() != 3 {
            return Err(NnError::UnsupportedRank {
                rank: input.shape().ndim(),
            });
        }
        let (b, s, f) = (dims[0], dims[1], dims[2]);
        let hs = self.hidden_size;
        let src = input.data();

        let mut result = vec![0.0_f32; b * s * 2 * hs];

        for bi in 0..b {
            // Extract [S, F] for this batch
            let batch_data: Vec<f32> = src[bi * s * f..(bi + 1) * s * f].to_vec();
            let batch_seq = Tensor::from_vec(batch_data, Shape::d2(s, f))?;

            // Forward pass
            let fwd = self.forward_cell.forward_seq(&batch_seq)?;

            // Backward pass: reverse input sequence
            let mut rev_data = vec![0.0_f32; s * f];
            for t in 0..s {
                rev_data[t * f..(t + 1) * f]
                    .copy_from_slice(&batch_seq.data()[(s - 1 - t) * f..(s - t) * f]);
            }
            let rev_seq = Tensor::from_vec(rev_data, Shape::d2(s, f))?;
            let bwd_rev = self.backward_cell.forward_seq(&rev_seq)?;

            // Concatenate: result[b, t, :] = [fwd[t], bwd[S-1-t]]
            let fwd_d = fwd.data();
            let bwd_d = bwd_rev.data();
            for t in 0..s {
                let out_off = (bi * s + t) * 2 * hs;
                result[out_off..out_off + hs].copy_from_slice(&fwd_d[t * hs..(t + 1) * hs]);
                let rev_t = s - 1 - t;
                result[out_off + hs..out_off + 2 * hs]
                    .copy_from_slice(&bwd_d[rev_t * hs..(rev_t + 1) * hs]);
            }
        }

        Tensor::from_vec(result, Shape::d3(b, s, 2 * hs))
    }
}

// ---------------------------------------------------------------------------
// Crf
// ---------------------------------------------------------------------------

/// Linear-chain CRF decoder.
#[derive(Debug)]
pub struct Crf {
    transitions: Tensor,
    start_transitions: Tensor,
    end_transitions: Tensor,
    num_tags: usize,
}

impl Crf {
    /// Load from model.
    pub fn load(model: &SovereignModel, prefix: &str) -> NnResult<Self> {
        let transitions = model.tensor(&format!("{prefix}.transitions"))?.clone();
        let start_transitions = model
            .tensor(&format!("{prefix}.start_transitions"))?
            .clone();
        let end_transitions = model.tensor(&format!("{prefix}.end_transitions"))?.clone();
        let num_tags = transitions.shape().dims()[0];
        Ok(Self {
            transitions,
            start_transitions,
            end_transitions,
            num_tags,
        })
    }

    /// Construct directly.
    pub fn from_tensors(
        transitions: Tensor,
        start_transitions: Tensor,
        end_transitions: Tensor,
    ) -> NnResult<Self> {
        let num_tags = transitions.shape().dims()[0];
        Ok(Self {
            transitions,
            start_transitions,
            end_transitions,
            num_tags,
        })
    }

    /// Decode emissions `[B, S, T]` -> `Vec<(tags, score)>` per batch.
    pub fn decode(&self, emissions: &Tensor) -> NnResult<Vec<(Vec<u32>, f32)>> {
        ops::crf_viterbi_decode(
            emissions,
            &self.transitions,
            &self.start_transitions,
            &self.end_transitions,
        )
    }

    /// Number of tags.
    #[must_use]
    pub const fn num_tags(&self) -> usize {
        self.num_tags
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tensor tests compare exact integer-valued floats (1.0, 2.0, …) where bitwise equality is intentional"
)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn linear_forward_basic() {
        // W = [[1,0],[0,1],[1,1]] (3x2), no bias
        let w = Tensor::from_vec(vec![1., 0., 0., 1., 1., 1.], Shape::d2(3, 2)).unwrap();
        let lin = Linear::from_tensors(w, None).unwrap();
        // x = [2, 3] -> [2*1+3*0, 2*0+3*1, 2*1+3*1] = [2, 3, 5]
        let x = Tensor::from_vec(vec![2., 3.], Shape::d2(1, 2)).unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[1, 3]);
        assert_eq!(y.data(), &[2., 3., 5.]);
    }

    #[test]
    fn linear_forward_with_bias() {
        let w = Tensor::from_vec(vec![1., 0., 0., 1.], Shape::d2(2, 2)).unwrap();
        let b = Tensor::from_vec(vec![10., 20.], Shape::d2(1, 2)).unwrap();
        let lin = Linear::from_tensors(w, Some(b)).unwrap();
        let x = Tensor::from_vec(vec![1., 2.], Shape::d2(1, 2)).unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.data(), &[11., 22.]);
    }

    #[test]
    fn linear_forward_3d() {
        // [B=1, S=2, F=2] through Linear(2->3)
        let w = Tensor::from_vec(vec![1., 0., 0., 1., 1., 1.], Shape::d2(3, 2)).unwrap();
        let lin = Linear::from_tensors(w, None).unwrap();
        let x = Tensor::from_vec(vec![1., 0., 0., 1.], Shape::d3(1, 2, 2)).unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.shape().ndim(), 3);
        assert_eq!(y.shape().dims(), &[1, 2, 3]);
        // [1,0] -> [1,0,1], [0,1] -> [0,1,1]
        assert_eq!(y.data(), &[1., 0., 1., 0., 1., 1.]);
    }

    #[test]
    fn embedding_forward_basic() {
        let table = Tensor::from_vec(vec![0., 0., 1., 1., 2., 2.], Shape::d2(3, 2)).unwrap();
        let emb = Embedding::from_tensor(table).unwrap();
        let y = emb.forward(&[2, 0]).unwrap();
        assert_eq!(y.data(), &[2., 2., 0., 0.]);
    }

    #[test]
    fn layer_norm_basic() {
        let g = Tensor::from_vec(vec![1., 1., 1., 1.], Shape::d2(1, 4)).unwrap();
        let b = Tensor::from_vec(vec![0., 0., 0., 0.], Shape::d2(1, 4)).unwrap();
        let ln = LayerNorm::from_tensors(g, b, 1e-5);
        let x = Tensor::from_vec(vec![1., 2., 3., 4.], Shape::d2(1, 4)).unwrap();
        let y = ln.forward(&x).unwrap();
        let mean: f32 = y.data().iter().sum::<f32>() / 4.0;
        assert!(approx(mean, 0.0, 1e-5));
    }

    #[test]
    fn activation_gelu_relu() {
        let x = Tensor::from_vec(vec![-1., 0., 1.], Shape::d2(1, 3)).unwrap();
        let g = Activation::GeluTanh.apply(&x);
        let r = Activation::Relu.apply(&x);
        assert!(g.data()[0] < 0.0); // GELU(-1) ≈ -0.159
        assert_eq!(r.data()[0], 0.0); // ReLU(-1) = 0
    }

    #[test]
    fn activation_from_str() {
        assert_eq!(Activation::from_str("gelu").unwrap(), Activation::GeluTanh);
        assert_eq!(
            Activation::from_str("gelu_tanh").unwrap(),
            Activation::GeluTanh
        );
        assert_eq!(Activation::from_str("relu").unwrap(), Activation::Relu);
        assert!(Activation::from_str("swish").is_err());
    }

    #[test]
    fn lstm_cell_forward_seq() {
        // Tiny LSTM: input=2, hidden=2
        let hs = 2;
        let input_size = 2;
        // w_ih: [4*hs, input] = [8, 2]
        let w_ih = Tensor::from_vec(
            vec![0.1; 4 * hs * input_size],
            Shape::d2(4 * hs, input_size),
        )
        .unwrap();
        let w_hh = Tensor::from_vec(vec![0.1; 4 * hs * hs], Shape::d2(4 * hs, hs)).unwrap();
        let b_ih = Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap();
        let b_hh = Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap();
        let cell = LstmCell::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();

        let input = Tensor::from_vec(vec![1., 0., 0., 1., 1., 1.], Shape::d2(3, 2)).unwrap();
        let out = cell.forward_seq(&input).unwrap();
        assert_eq!(out.shape().dims(), &[3, 2]);
        // Just verify it produces finite values
        assert!(out.data().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn bilstm_forward_basic() {
        let hs = 2;
        let input_size = 2;
        let make_cell = || {
            let w_ih = Tensor::from_vec(
                vec![0.1; 4 * hs * input_size],
                Shape::d2(4 * hs, input_size),
            )
            .unwrap();
            let w_hh = Tensor::from_vec(vec![0.1; 4 * hs * hs], Shape::d2(4 * hs, hs)).unwrap();
            let b_ih = Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap();
            let b_hh = Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap();
            LstmCell::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap()
        };
        let bilstm = BiLstm::from_cells(make_cell(), make_cell());

        let input = Tensor::from_vec(vec![1., 0., 0., 1., 1., 1.], Shape::d3(1, 3, 2)).unwrap();
        let out = bilstm.forward(&input).unwrap();
        assert_eq!(out.shape().dims(), &[1, 3, 4]); // 2*hidden=4
        assert!(out.data().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn crf_decode_basic() {
        let transitions = Tensor::from_vec(vec![0., 0., 0., 0.], Shape::d2(2, 2)).unwrap();
        let start = Tensor::from_vec(vec![0., 0.], Shape::d2(1, 2)).unwrap();
        let end = Tensor::from_vec(vec![0., 0.], Shape::d2(1, 2)).unwrap();
        let crf = Crf::from_tensors(transitions, start, end).unwrap();

        let emissions = Tensor::from_vec(vec![5., 0., 0., 5., 5., 0.], Shape::d3(1, 3, 2)).unwrap();
        let results = crf.decode(&emissions).unwrap();
        assert_eq!(results[0].0, vec![0, 1, 0]);
    }

    #[test]
    fn transformer_block_forward() {
        // Minimal transformer: E=4, H=2, FFN_DIM=8
        let e = 4;
        let h = 2;
        let ffn = 8;

        let make_linear = |out_f, in_f| {
            let w = Tensor::from_vec(vec![0.01; out_f * in_f], Shape::d2(out_f, in_f)).unwrap();
            Linear::from_tensors(w, None).unwrap()
        };
        let make_ln = |f| {
            let g = Tensor::from_vec(vec![1.0; f], Shape::d2(1, f)).unwrap();
            let b = Tensor::from_vec(vec![0.0; f], Shape::d2(1, f)).unwrap();
            LayerNorm::from_tensors(g, b, 1e-5)
        };

        let mha = MultiHeadAttention::from_parts(
            make_linear(e, e),
            make_linear(e, e),
            make_linear(e, e),
            make_linear(e, e),
            h,
        )
        .unwrap();

        let block = TransformerBlock::from_parts(
            mha,
            make_ln(e),
            make_ln(e),
            make_linear(ffn, e),
            make_linear(e, ffn),
            Activation::GeluTanh,
        );

        let x = Tensor::from_vec(vec![0.1; 3 * e], Shape::d3(1, 3, e)).unwrap();
        let y = block.forward(&x, None).unwrap();
        assert_eq!(y.shape().dims(), &[1, 3, e]);
        assert!(y.data().iter().all(|v| v.is_finite()));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// Helper: construct a Linear layer with identity-ish small weights.
        fn make_linear(out_f: usize, in_f: usize) -> Linear {
            let w = Tensor::from_vec(vec![0.01f32; out_f * in_f], Shape::d2(out_f, in_f)).unwrap();
            Linear::from_tensors(w, None).unwrap()
        }

        fn make_ln(features: usize) -> LayerNorm {
            let g = Tensor::from_vec(vec![1.0f32; features], Shape::d2(1, features)).unwrap();
            let b = Tensor::from_vec(vec![0.0f32; features], Shape::d2(1, features)).unwrap();
            LayerNorm::from_tensors(g, b, 1e-5)
        }

        proptest! {
            /// Linear::forward: output shape is (batch, seq, out_features) for
            /// any conforming (batch, seq, in_features) input.
            #[test]
            fn linear_output_shape(
                batch in 1usize..=4,
                seq   in 1usize..=6,
                in_f  in 1usize..=8,
                out_f in 1usize..=8,
                vals  in proptest::collection::vec(-1.0f32..1.0, 1..=192),
            ) {
                let n = batch * seq * in_f;
                let data: Vec<f32> = (0..n).map(|i| {
                    vals.get(i % vals.len().max(1)).copied().unwrap_or(0.0)
                }).collect();
                let input = Tensor::from_vec(data, Shape::d3(batch, seq, in_f)).unwrap();
                let lin = make_linear(out_f, in_f);
                let out = lin.forward(&input).unwrap();
                prop_assert_eq!(out.shape().dims(), &[batch, seq, out_f]);
                for &v in out.data() {
                    prop_assert!(v.is_finite(), "Linear output contains non-finite value: {v}");
                }
            }

            /// LayerNorm::forward: preserves input shape, output is finite.
            #[test]
            fn layer_norm_shape_preserved(
                batch in 1usize..=4,
                seq   in 1usize..=6,
                feats in 1usize..=8,
                vals  in proptest::collection::vec(-5.0f32..5.0, 1..=192),
            ) {
                let n = batch * seq * feats;
                let data: Vec<f32> = (0..n).map(|i| {
                    vals.get(i % vals.len().max(1)).copied().unwrap_or(0.0)
                }).collect();
                let input = Tensor::from_vec(data, Shape::d3(batch, seq, feats)).unwrap();
                let ln = make_ln(feats);
                let out = ln.forward(&input).unwrap();
                prop_assert_eq!(out.shape().dims(), input.shape().dims());
                for &v in out.data() {
                    prop_assert!(v.is_finite(), "LayerNorm output not finite: {v}");
                }
            }

            /// MultiHeadAttention::forward: output shape matches input shape for
            /// self-attention. embed_dim must be divisible by num_heads (pick
            /// embed_dim = num_heads * head_dim to guarantee this).
            #[test]
            fn mha_output_shape(
                batch     in 1usize..=3,
                seq       in 1usize..=6,
                num_heads in 1usize..=3,
                head_dim  in 1usize..=4,
            ) {
                let embed = num_heads * head_dim;
                let input_data = vec![0.01f32; batch * seq * embed];
                let input = Tensor::from_vec(input_data, Shape::d3(batch, seq, embed)).unwrap();

                let mha = MultiHeadAttention::from_parts(
                    make_linear(embed, embed),
                    make_linear(embed, embed),
                    make_linear(embed, embed),
                    make_linear(embed, embed),
                    num_heads,
                ).unwrap();
                let out = mha.forward(&input, None).unwrap();
                prop_assert_eq!(out.shape().dims(), &[batch, seq, embed]);
            }

            /// TransformerBlock::forward: preserves (batch, seq, d_model) shape.
            #[test]
            fn transformer_block_shape(
                batch     in 1usize..=3,
                seq       in 1usize..=5,
                num_heads in 1usize..=2,
                head_dim  in 1usize..=3,
            ) {
                let e = num_heads * head_dim;   // d_model divisible by num_heads
                let ffn = e * 2;

                let mha = MultiHeadAttention::from_parts(
                    make_linear(e, e),
                    make_linear(e, e),
                    make_linear(e, e),
                    make_linear(e, e),
                    num_heads,
                ).unwrap();
                let block = TransformerBlock::from_parts(
                    mha,
                    make_ln(e),
                    make_ln(e),
                    make_linear(ffn, e),
                    make_linear(e, ffn),
                    Activation::Relu,
                );
                let data = vec![0.01f32; batch * seq * e];
                let x = Tensor::from_vec(data, Shape::d3(batch, seq, e)).unwrap();
                let y = block.forward(&x, None).unwrap();
                prop_assert_eq!(y.shape().dims(), &[batch, seq, e]);
                for &v in y.data() {
                    prop_assert!(v.is_finite(), "TransformerBlock output not finite: {v}");
                }
            }
        }
    }
}
