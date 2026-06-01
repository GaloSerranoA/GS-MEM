//! LSTM cell (gate order `i, f, g, o`).
//!
//! Standalone from `blocks::LstmCell` — this type implements [`RnnCell`]
//! and participates in generic [`unroll`](super::unroll). The older
//! `blocks::LstmCell` keeps its sequential forward for the `BiLstm`
//! consumer; both share the same gate semantics.
//!
//! Equations (PyTorch):
//! ```text
//! i = σ(x @ W_ii.T + b_ii + h @ W_hi.T + b_hi)
//! f = σ(x @ W_if.T + b_if + h @ W_hf.T + b_hf)
//! g = tanh(x @ W_ig.T + b_ig + h @ W_hg.T + b_hg)
//! o = σ(x @ W_io.T + b_io + h @ W_ho.T + b_ho)
//! c' = f * c + i * g
//! h' = o * tanh(c')
//! ```
//! Concatenated weight layout (PyTorch convention):
//! - `w_ih: [4*hidden, input]`  (ordered i, f, g, o)
//! - `w_hh: [4*hidden, hidden]`
//! - `b_ih: [1, 4*hidden]`
//! - `b_hh: [1, 4*hidden]`

use crate::format::SovereignModel;
use crate::ops;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

use super::util::{sigmoid_scalar, transpose_2d};
use super::RnnCell;

/// `(hidden, cell)` state carried between LSTM steps. Both tensors are
/// shape `[1, hidden_size]`.
pub type LstmState = (Tensor, Tensor);

/// Standalone LSTM cell implementing [`RnnCell`].
#[derive(Debug)]
pub struct Lstm {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    input_size: usize,
    hidden_size: usize,
}

impl Lstm {
    /// Load from `SovereignModel`.
    ///
    /// # Errors
    /// [`NnError`] on missing tensors or shape mismatch.
    pub fn load(model: &SovereignModel, prefix: &str) -> NnResult<Self> {
        let w_ih = model.tensor(&format!("{prefix}.w_ih"))?.clone();
        let w_hh = model.tensor(&format!("{prefix}.w_hh"))?.clone();
        let b_ih = model.tensor(&format!("{prefix}.b_ih"))?.clone();
        let b_hh = model.tensor(&format!("{prefix}.b_hh"))?.clone();
        Self::from_tensors(w_ih, w_hh, b_ih, b_hh)
    }

    /// Construct from owned tensors.
    ///
    /// # Errors
    /// [`NnError::ShapeMismatch`] on rank or 4×hidden mismatch.
    pub fn from_tensors(w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> NnResult<Self> {
        if w_ih.shape().ndim() != 2 || w_hh.shape().ndim() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: "w_ih, w_hh: rank-2".into(),
                got: format!("ranks {}/{}", w_ih.shape().ndim(), w_hh.shape().ndim()),
            });
        }
        let four_h = w_ih.shape().dims()[0];
        if !four_h.is_multiple_of(4) {
            return Err(NnError::DivisibilityError {
                what: "w_ih row count / 4".into(),
                numerator: four_h,
                denominator: 4,
            });
        }
        let hidden_size = four_h / 4;
        let input_size = w_ih.shape().dims()[1];
        if w_hh.shape().dims() != [four_h, hidden_size] {
            return Err(NnError::ShapeMismatch {
                expected: format!("w_hh [{four_h}, {hidden_size}]"),
                got: format!("{:?}", w_hh.shape().dims()),
            });
        }
        if b_ih.numel() != four_h || b_hh.numel() != four_h {
            return Err(NnError::ShapeMismatch {
                expected: format!("b_ih/b_hh numel {four_h}"),
                got: format!("{} / {}", b_ih.numel(), b_hh.numel()),
            });
        }
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size,
        })
    }
}

impl RnnCell for Lstm {
    type State = LstmState;

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn zero_state(&self) -> Self::State {
        (
            Tensor::zeros(Shape::d2(1, self.hidden_size)),
            Tensor::zeros(Shape::d2(1, self.hidden_size)),
        )
    }

    fn step(&self, x: &Tensor, state: &Self::State) -> NnResult<(Tensor, Self::State)> {
        let (h, c) = state;
        if x.shape().ndim() != 2 || x.shape().dims()[1] != self.input_size {
            return Err(NnError::ShapeMismatch {
                expected: format!("x: [1, {}]", self.input_size),
                got: format!("{:?}", x.shape().dims()),
            });
        }
        let hs = self.hidden_size;
        let wih_t = transpose_2d(&self.w_ih);
        let whh_t = transpose_2d(&self.w_hh);
        let xw = ops::matmul(x, &wih_t)?;
        let hw = ops::matmul(h, &whh_t)?;

        let xwd = xw.data();
        let hwd = hw.data();
        let bih = self.b_ih.data();
        let bhh = self.b_hh.data();
        let cd = c.data();

        let mut h_new = vec![0.0_f32; hs];
        let mut c_new = vec![0.0_f32; hs];
        for j in 0..hs {
            let raw_i = xwd[j] + hwd[j] + bih[j] + bhh[j];
            let raw_f = xwd[hs + j] + hwd[hs + j] + bih[hs + j] + bhh[hs + j];
            let raw_g = xwd[2 * hs + j] + hwd[2 * hs + j] + bih[2 * hs + j] + bhh[2 * hs + j];
            let raw_o = xwd[3 * hs + j] + hwd[3 * hs + j] + bih[3 * hs + j] + bhh[3 * hs + j];
            let i_g = sigmoid_scalar(raw_i);
            let f_g = sigmoid_scalar(raw_f);
            let g_g = raw_g.tanh();
            let o_g = sigmoid_scalar(raw_o);
            c_new[j] = f_g * cd[j] + i_g * g_g;
            h_new[j] = o_g * c_new[j].tanh();
        }
        let h_t = Tensor::from_vec(h_new, Shape::d2(1, hs))?;
        let c_t = Tensor::from_vec(c_new, Shape::d2(1, hs))?;
        Ok((h_t.clone(), (h_t, c_t)))
    }
}
