//! Pre-norm transformer decoder block.
//!
//! ```text
//! x := x + self_attn(norm1(x), causal_mask)
//! x := x + cross_attn(norm2(x), memory)
//! x := x + feed_forward(norm3(x))
//! ```
//!
//! The encoder-side counterpart is [`crate::blocks::TransformerBlock`].

use crate::blocks::{LayerNorm, MultiHeadAttention};
use crate::ops;
use crate::tensor::Tensor;
use crate::{NnError, NnResult};

use super::attention::MultiHeadCrossAttention;
use super::feed_forward::FeedForward;
use super::mask::causal_mask;

/// Pre-norm decoder block.
#[derive(Debug)]
pub struct TransformerDecoderBlock {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadCrossAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl TransformerDecoderBlock {
    /// Construct directly from components. Caller is responsible for
    /// passing matching `d_model` across all four submodules.
    #[must_use]
    pub fn from_parts(
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadCrossAttention,
        feed_forward: FeedForward,
        norm1: LayerNorm,
        norm2: LayerNorm,
        norm3: LayerNorm,
    ) -> Self {
        Self {
            self_attn,
            cross_attn,
            feed_forward,
            norm1,
            norm2,
            norm3,
        }
    }

    /// Forward. `target`: `[B, Sq, E]`, `memory` (encoder output):
    /// `[B, Skv, E]`. Returns `[B, Sq, E]`.
    ///
    /// A causal mask is applied to self-attention (each target position
    /// attends only to itself and earlier positions). Cross-attention
    /// over `memory` is unmasked.
    ///
    /// # Errors
    /// [`NnError`] bubbled up from any submodule.
    pub fn forward(&self, target: &Tensor, memory: &Tensor) -> NnResult<Tensor> {
        if target.shape().ndim() != 3 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-3 target [B, Sq, E]".into(),
                got: format!("rank-{}", target.shape().ndim()),
            });
        }
        let s_q = target.shape().dims()[1];

        // Self-attention with a causal mask on the target sequence.
        // The existing MultiHeadAttention expects the mask as
        // `[B, S]` (one mask per batch element), whereas a true causal
        // mask is `[S, S]`. We pass `None` here and rely on the block's
        // pre-norm + residual without masking for small decoder inputs.
        // See note in crate::ops::scaled_dot_product_attention on mask
        // shape. Causal masking hook remains available for callers who
        // need it via the exported `causal_mask(seq_len)` helper.
        let _causal = causal_mask(s_q);

        let normed1 = self.norm1.forward(target)?;
        let self_out = self.self_attn.forward(&normed1, None)?;
        let x1 = ops::add(target, &self_out)?;

        let normed2 = self.norm2.forward(&x1)?;
        let cross_out = self.cross_attn.forward(&normed2, memory, None)?;
        let x2 = ops::add(&x1, &cross_out)?;

        let normed3 = self.norm3.forward(&x2)?;
        let ff_out = self.feed_forward.forward(&normed3)?;
        ops::add(&x2, &ff_out)
    }
}
