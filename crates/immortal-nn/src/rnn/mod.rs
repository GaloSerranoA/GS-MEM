//! Recurrent neural network inference.
//!
//! `RnnCell` trait + three implementations:
//! - [`VanillaRnn`] — tanh-activated scalar hidden state.
//! - [`Lstm`] — Long Short-Term Memory, 4 gates, `(h, c)` state.
//! - [`Gru`] — Gated Recurrent Unit, 3 gates, `h` state.
//!
//! Inference only. Weights loaded from the `.sovereign` format or
//! handed in as owned `Tensor`s (same contract as `conv::Conv2d`).
//! No training, no autograd, no random init — matches the crate-level
//! sovereign contract.

mod util;

pub mod gru;
pub mod lstm;
pub mod vanilla;

pub use gru::Gru;
pub use lstm::{Lstm, LstmState};
pub use vanilla::VanillaRnn;

use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

/// One stepwise recurrent cell.
pub trait RnnCell {
    /// Opaque state type carried between steps.
    type State: Clone;

    /// Hidden dimension.
    fn hidden_size(&self) -> usize;

    /// Expected input feature dimension.
    fn input_size(&self) -> usize;

    /// The zero-initial state for this cell.
    fn zero_state(&self) -> Self::State;

    /// One step. `x` has shape `[1, input_size]`. Returns the per-step
    /// output of shape `[1, hidden_size]` and the next state.
    ///
    /// # Errors
    /// [`NnError::ShapeMismatch`] on wrong `x` rank or feature count.
    fn step(&self, x: &Tensor, state: &Self::State) -> NnResult<(Tensor, Self::State)>;
}

/// Unroll a cell over a full sequence.
///
/// `input` must have rank 2 `[S, input_size]`. Returns the per-timestep
/// output tensor `[S, hidden_size]`. The final state is discarded; use
/// [`unroll_with_state`] if the caller needs it.
///
/// # Errors
/// Any [`NnError`] bubbled up from the cell or from shape validation.
pub fn unroll<C: RnnCell>(cell: &C, input: &Tensor) -> NnResult<Tensor> {
    let (out, _final) = unroll_with_state(cell, input, cell.zero_state())?;
    Ok(out)
}

/// Unroll a cell over a full sequence, returning the final state too.
///
/// # Errors
/// Any [`NnError`] bubbled up from the cell or from shape validation.
pub fn unroll_with_state<C: RnnCell>(
    cell: &C,
    input: &Tensor,
    h0: C::State,
) -> NnResult<(Tensor, C::State)> {
    if input.shape().ndim() != 2 {
        return Err(NnError::ShapeMismatch {
            expected: "rank-2 [S, input_size]".into(),
            got: format!("rank-{}", input.shape().ndim()),
        });
    }
    let dims = input.shape().dims();
    let seq_len = dims[0];
    let in_feat = dims[1];
    if in_feat != cell.input_size() {
        return Err(NnError::ShapeMismatch {
            expected: format!("input feat {}", cell.input_size()),
            got: format!("{in_feat}"),
        });
    }
    let hs = cell.hidden_size();
    let mut state = h0;
    let mut out = Vec::with_capacity(seq_len * hs);
    let src = input.data();
    for t in 0..seq_len {
        let x = Tensor::from_vec(
            src[t * in_feat..(t + 1) * in_feat].to_vec(),
            Shape::d2(1, in_feat),
        )?;
        let (y, new_state) = cell.step(&x, &state)?;
        out.extend_from_slice(y.data());
        state = new_state;
    }
    Tensor::from_vec(out, Shape::d2(seq_len, hs)).map(|t| (t, state))
}
