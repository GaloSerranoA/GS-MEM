//! Transformer building blocks beyond the encoder-style
//! [`crate::blocks::TransformerBlock`].
//!
//! Landed in Plan 17:
//! - [`PositionalEncoding`] — deterministic sinusoidal position vectors.
//! - [`causal_mask`] — `[S, S]` upper-triangular causal mask.
//! - [`FeedForward`] — standalone 2-layer MLP (Linear → activation → Linear).
//! - [`cross_attention`] — scaled dot-product attention for distinct
//!   query vs key/value sequence lengths.
//! - [`TransformerDecoderBlock`] — self-attention (causal) + cross-
//!   attention (to encoder memory) + feed-forward, pre-norm + residual.
//!
//! Inference only — matches the crate-level sovereign contract.

pub mod attention;
pub mod decoder_block;
pub mod feed_forward;
pub mod mask;
pub mod positional_encoding;

pub use attention::{cross_attention, MultiHeadCrossAttention};
pub use decoder_block::TransformerDecoderBlock;
pub use feed_forward::FeedForward;
pub use mask::causal_mask;
pub use positional_encoding::PositionalEncoding;
