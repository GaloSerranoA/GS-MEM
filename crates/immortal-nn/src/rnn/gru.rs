//! GRU cell (gate order `r, z, n` — PyTorch convention).
//!
//! Equations:
//! ```text
//! r = σ(x @ W_ir.T + b_ir + h @ W_hr.T + b_hr)
//! z = σ(x @ W_iz.T + b_iz + h @ W_hz.T + b_hz)
//! n = tanh(x @ W_in.T + b_in + r * (h @ W_hn.T + b_hn))
//! h' = (1 - z) * n + z * h
//! ```
//! Weight layout:
//! - `w_ih: [3*hidden, input]`  (order r, z, n)
//! - `w_hh: [3*hidden, hidden]`
//! - `b_ih: [1, 3*hidden]`
//! - `b_hh: [1, 3*hidden]`

use crate::format::SovereignModel;
use crate::ops;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

use super::util::{sigmoid_scalar, transpose_2d};
use super::RnnCell;

/// GRU cell. Single-`h` state.
#[derive(Debug)]
pub struct Gru {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    input_size: usize,
    hidden_size: usize,
}

impl Gru {
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
    /// [`NnError::ShapeMismatch`] on rank or 3×hidden mismatch.
    pub fn from_tensors(w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor) -> NnResult<Self> {
        if w_ih.shape().ndim() != 2 || w_hh.shape().ndim() != 2 {
            return Err(NnError::ShapeMismatch {
                expected: "w_ih, w_hh: rank-2".into(),
                got: format!("ranks {}/{}", w_ih.shape().ndim(), w_hh.shape().ndim()),
            });
        }
        let three_h = w_ih.shape().dims()[0];
        if !three_h.is_multiple_of(3) {
            return Err(NnError::DivisibilityError {
                what: "w_ih row count / 3".into(),
                numerator: three_h,
                denominator: 3,
            });
        }
        let hidden_size = three_h / 3;
        let input_size = w_ih.shape().dims()[1];
        if w_hh.shape().dims() != [three_h, hidden_size] {
            return Err(NnError::ShapeMismatch {
                expected: format!("w_hh [{three_h}, {hidden_size}]"),
                got: format!("{:?}", w_hh.shape().dims()),
            });
        }
        if b_ih.numel() != three_h || b_hh.numel() != three_h {
            return Err(NnError::ShapeMismatch {
                expected: format!("b_ih/b_hh numel {three_h}"),
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

impl RnnCell for Gru {
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
        let hs = self.hidden_size;
        let wih_t = transpose_2d(&self.w_ih);
        let whh_t = transpose_2d(&self.w_hh);
        let xw = ops::matmul(x, &wih_t)?;
        let hw = ops::matmul(h, &whh_t)?;

        let xwd = xw.data();
        let hwd = hw.data();
        let bih = self.b_ih.data();
        let bhh = self.b_hh.data();
        let hd = h.data();

        let mut out = vec![0.0_f32; hs];
        for j in 0..hs {
            // r, z are additively combined before the sigmoid.
            let r_raw = xwd[j] + bih[j] + hwd[j] + bhh[j];
            let z_raw = xwd[hs + j] + bih[hs + j] + hwd[hs + j] + bhh[hs + j];
            // For n, the recurrent term is gated by r BEFORE being added
            // to the input term (PyTorch convention, not the original
            // Cho et al. 2014 paper where r is applied inside h).
            let hn_recurrent = hwd[2 * hs + j] + bhh[2 * hs + j];
            let r_gate = sigmoid_scalar(r_raw);
            let z_gate = sigmoid_scalar(z_raw);
            let n_raw = xwd[2 * hs + j] + bih[2 * hs + j] + r_gate * hn_recurrent;
            let n_gate = n_raw.tanh();
            out[j] = (1.0 - z_gate) * n_gate + z_gate * hd[j];
        }
        let h_new = Tensor::from_vec(out, Shape::d2(1, hs))?;
        Ok((h_new.clone(), h_new))
    }
}
