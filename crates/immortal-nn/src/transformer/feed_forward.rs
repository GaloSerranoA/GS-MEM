//! Position-wise feed-forward network (2-layer MLP).
//!
//! Equivalent to PyTorch's `nn.Linear(d_model, d_ff) → act → nn.Linear(d_ff, d_model)`.

use crate::blocks::{Activation, Linear};
use crate::format::SovereignModel;
use crate::tensor::Tensor;
use crate::{NnError, NnResult};

/// Standalone 2-layer feed-forward block.
#[derive(Debug)]
pub struct FeedForward {
    up: Linear,
    down: Linear,
    activation: Activation,
}

impl FeedForward {
    /// Load from `SovereignModel` using `{prefix}.up` and `{prefix}.down`.
    ///
    /// # Errors
    /// [`NnError`] if the submodule fails to load.
    pub fn load(model: &SovereignModel, prefix: &str, activation: Activation) -> NnResult<Self> {
        Ok(Self {
            up: Linear::load(model, &format!("{prefix}.up"))?,
            down: Linear::load(model, &format!("{prefix}.down"))?,
            activation,
        })
    }

    /// Construct directly from two `Linear` layers. The `up` layer's
    /// output size must match the `down` layer's input size.
    ///
    /// # Errors
    /// [`NnError::ShapeMismatch`] if `up.out_features() != down.in_features()`.
    pub fn from_parts(up: Linear, down: Linear, activation: Activation) -> NnResult<Self> {
        if up.out_features() != down.in_features() {
            return Err(NnError::ShapeMismatch {
                expected: format!(
                    "up.out_features ({}) == down.in_features ({})",
                    up.out_features(),
                    down.in_features()
                ),
                got: "mismatched".into(),
            });
        }
        Ok(Self {
            up,
            down,
            activation,
        })
    }

    /// Forward pass. Input `[B, S, d_model]` → output `[B, S, d_model]`.
    ///
    /// # Errors
    /// [`NnError`] on any shape mismatch.
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        let up = self.up.forward(x)?;
        let activated = self.activation.apply(&up);
        self.down.forward(&activated)
    }

    /// Input / output feature size (must be equal by construction).
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.up.in_features()
    }

    /// Inner dimension (`d_ff`).
    #[must_use]
    pub fn d_ff(&self) -> usize {
        self.up.out_features()
    }
}
