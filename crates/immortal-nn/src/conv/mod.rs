//! Convolutional inference layers.
//!
//! NCHW layout throughout: input and output tensors are rank-4 with
//! dimensions `[N, C, H, W]`. Weights follow PyTorch's `nn.Conv2d`
//! convention `[OutC, InC, Kh, Kw]`.
//!
//! Inference only — no training, no autograd, no weight initialization.
//! Weights are loaded from the `.sovereign` format or constructed from
//! owned `Tensor`s by the caller. This matches the sovereign inference
//! contract declared at the crate root.

pub mod batch_norm;
pub mod conv2d;
pub mod pooling;

pub use batch_norm::BatchNorm2d;
pub use conv2d::Conv2d;
pub use pooling::{AvgPool2d, MaxPool2d};
