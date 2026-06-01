//! Small shared helpers for the `rnn` module.

use crate::tensor::{Shape, Tensor};

/// 2-D transpose. Assumes the input is rank-2.
pub(super) fn transpose_2d(t: &Tensor) -> Tensor {
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

/// Numerically stable scalar sigmoid.
#[inline]
pub(super) fn sigmoid_scalar(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
