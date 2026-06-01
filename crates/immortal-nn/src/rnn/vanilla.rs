//! Vanilla tanh RNN cell.
//!
//! `h_t = tanh(x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh)`.
//!
//! Weight convention (PyTorch `nn.RNNCell`):
//! - `w_ih: [hidden, input]`
//! - `w_hh: [hidden, hidden]`
//! - `b_ih: [1, hidden]`
//! - `b_hh: [1, hidden]`
//!
//! `tanh` is the only supported nonlinearity. ReLU variants can be
//! added later if a pretrained model requires them.

use crate::format::SovereignModel;
use crate::ops;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

use super::util::transpose_2d;
use super::RnnCell;

/// Vanilla RNN cell.
#[derive(Debug)]
pub struct VanillaRnn {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    input_size: usize,
    hidden_size: usize,
}

impl VanillaRnn {
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

    /// Construct directly from owned tensors.
    ///
    /// # Errors
    /// [`NnError::ShapeMismatch`] if any tensor has the wrong rank or
    /// mismatching hidden/input dims.
    pub fn from_tensors(w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> NnResult<Self> {
        if w_ih.shape().ndim() != 2 || w_hh.shape().ndim() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: "w_ih, w_hh: rank-2".into(),
                got: format!("ranks {}/{}", w_ih.shape().ndim(), w_hh.shape().ndim()),
            });
        }
        let (hidden, input_size) = (w_ih.shape().dims()[0], w_ih.shape().dims()[1]);
        if w_hh.shape().dims() != [hidden, hidden] {
            return Err(NnError::ShapeMismatch {
                expected: format!("w_hh [{hidden}, {hidden}]"),
                got: format!("{:?}", w_hh.shape().dims()),
            });
        }
        if b_ih.numel() != hidden || b_hh.numel() != hidden {
            return Err(NnError::ShapeMismatch {
                expected: format!("b_ih/b_hh numel {hidden}"),
                got: format!("{} / {}", b_ih.numel(), b_hh.numel()),
            });
        }
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            input_size,
            hidden_size: hidden,
        })
    }
}

impl RnnCell for VanillaRnn {
    type State = Tensor;

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn input_size(&self) -> usize {
        self.input_size
    }

    fn zero_state(&self) -> Self::State {
        Tensor::zeros(Shape::d2(1, self.hidden_size))
    }

    fn step(&self, x: &Tensor, h: &Self::State) -> NnResult<(Tensor, Self::State)> {
        if x.shape().ndim() != 2 || x.shape().dims()[1] != self.input_size {
            return Err(NnError::ShapeMismatch {
                expected: format!("x: [1, {}]", self.input_size),
                got: format!("{:?}", x.shape().dims()),
            });
        }
        let wih_t = transpose_2d(&self.w_ih);
        let whh_t = transpose_2d(&self.w_hh);
        let xw = ops::matmul(x, &wih_t)?;
        let hw = ops::matmul(h, &whh_t)?;

        let hs = self.hidden_size;
        let xwd = xw.data();
        let hwd = hw.data();
        let bih = self.b_ih.data();
        let bhh = self.b_hh.data();

        let mut out = vec![0.0_f32; hs];
        for j in 0..hs {
            out[j] = (xwd[j] + hwd[j] + bih[j] + bhh[j]).tanh();
        }
        let h_new = Tensor::from_vec(out, Shape::d2(1, hs))?;
        Ok((h_new.clone(), h_new))
    }
}
