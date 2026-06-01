//! 2-D batch normalization (inference mode).
//!
//! Inference uses the running (population) statistics, not batch statistics.
//! `y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta`.
//!
//! Weights loaded from `.sovereign`:
//! - `{prefix}.gamma` `[1, C]` — scale (γ)
//! - `{prefix}.beta`  `[1, C]` — bias  (β)
//! - `{prefix}.running_mean` `[1, C]`
//! - `{prefix}.running_var`  `[1, C]`

use crate::format::SovereignModel;
use crate::tensor::Tensor;
use crate::{NnError, NnResult};

/// Default PyTorch-compatible epsilon.
pub const DEFAULT_EPS: f32 = 1e-5;

/// 2-D batch normalization, inference only.
#[derive(Debug)]
pub struct BatchNorm2d {
    channels: usize,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f32,
}

impl BatchNorm2d {
    /// Load from a `SovereignModel`. Rejects with [`NnError::ShapeMismatch`]
    /// if any of the four per-channel tensors is missing or the wrong size.
    ///
    /// # Errors
    /// Any missing tensor, numel mismatch, or non-positive `eps` surfaces
    /// as [`NnError`].
    pub fn load(model: &SovereignModel, prefix: &str, channels: usize, eps: f32) -> NnResult<Self> {
        let gamma = model.tensor(&format!("{prefix}.gamma"))?.clone();
        let beta = model.tensor(&format!("{prefix}.beta"))?.clone();
        let running_mean = model.tensor(&format!("{prefix}.running_mean"))?.clone();
        let running_var = model.tensor(&format!("{prefix}.running_var"))?.clone();
        Self::from_tensors(gamma, beta, running_mean, running_var, channels, eps)
    }

    /// Construct directly from owned per-channel tensors.
    ///
    /// # Errors
    /// [`NnError::ShapeMismatch`] if any tensor numel is not `channels`,
    /// or [`NnError::Format`] if `eps <= 0`.
    pub fn from_tensors(
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        channels: usize,
        eps: f32,
    ) -> NnResult<Self> {
        for (name, t) in [
            ("gamma", &gamma),
            ("beta", &beta),
            ("running_mean", &running_mean),
            ("running_var", &running_var),
        ] {
            if t.numel() != channels {
                return Err(NnError::ShapeMismatch {
                    expected: format!("{name} numel {channels}"),
                    got: format!("{name} numel {}", t.numel()),
                });
            }
        }
        if !(eps > 0.0 && eps.is_finite()) {
            return Err(NnError::Format(format!(
                "BatchNorm2d eps must be > 0 and finite, got {eps}"
            )));
        }
        Ok(Self {
            channels,
            gamma,
            beta,
            running_mean,
            running_var,
            eps,
        })
    }

    /// Channel count this layer was built for.
    #[must_use]
    pub const fn channels(&self) -> usize {
        self.channels
    }

    /// Epsilon used in the denominator.
    #[must_use]
    pub const fn eps(&self) -> f32 {
        self.eps
    }

    /// Forward pass. Input `[N, C, H, W]` → output `[N, C, H, W]`.
    ///
    /// # Errors
    /// [`NnError::ShapeMismatch`] on wrong rank or channel count.
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        if x.shape().ndim() != 4 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-4 [N, C, H, W]".into(),
                got: format!("rank-{}", x.shape().ndim()),
            });
        }
        let d = x.shape().dims();
        let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
        if c != self.channels {
            return Err(NnError::ShapeMismatch {
                expected: format!("C={}", self.channels),
                got: format!("C={c}"),
            });
        }

        let mut out = x.clone();
        let out_data = out.data_mut();
        let gamma = self.gamma.data();
        let beta = self.beta.data();
        let mean = self.running_mean.data();
        let var = self.running_var.data();

        let stride_n = c * h * w;
        let stride_c = h * w;

        for ni in 0..n {
            for ci in 0..c {
                let denom = (var[ci] + self.eps).sqrt();
                // Guard against pathological loaded weights (var cannot
                // legitimately be negative, but NaN + eps would leak).
                let scale = if denom > 0.0 { gamma[ci] / denom } else { 0.0 };
                let shift = beta[ci] - scale * mean[ci];
                let base = ni * stride_n + ci * stride_c;
                for hw in 0..stride_c {
                    let idx = base + hw;
                    out_data[idx] = scale * out_data[idx] + shift;
                }
            }
        }
        Ok(out)
    }
}
