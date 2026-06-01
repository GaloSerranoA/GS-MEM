//! 2-D pooling layers (max + average).
//!
//! Stateless — no learned parameters. NCHW layout, rank-4 input, rank-4
//! output. Pooling does not cross channel or batch dims.

use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

/// Stride + padding + kernel pair for the 2 spatial dimensions.
pub type Hw = (usize, usize);

/// 2-D max-pooling.
#[derive(Debug, Clone, Copy)]
pub struct MaxPool2d {
    kernel: Hw,
    stride: Hw,
    padding: Hw,
}

impl MaxPool2d {
    /// Construct with kernel, stride, and padding.
    ///
    /// # Errors
    /// Returns [`NnError::Format`] if any kernel or stride dimension is zero.
    pub fn new(kernel: Hw, stride: Hw, padding: Hw) -> NnResult<Self> {
        check_hw("MaxPool2d", kernel, stride)?;
        Ok(Self {
            kernel,
            stride,
            padding,
        })
    }

    /// Forward pass.
    ///
    /// # Errors
    /// Returns [`NnError::ShapeMismatch`] on wrong rank; [`NnError::Format`]
    /// if the input spatial size is smaller than the padded kernel.
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        pool2d_forward(x, self.kernel, self.stride, self.padding, PoolKind::Max)
    }

    /// Kernel size.
    #[must_use]
    pub const fn kernel(&self) -> Hw {
        self.kernel
    }

    /// Stride.
    #[must_use]
    pub const fn stride(&self) -> Hw {
        self.stride
    }

    /// Padding.
    #[must_use]
    pub const fn padding(&self) -> Hw {
        self.padding
    }
}

/// 2-D average-pooling. Padded positions contribute zero to the average
/// but DO count toward the divisor (standard PyTorch `count_include_pad`
/// default).
#[derive(Debug, Clone, Copy)]
pub struct AvgPool2d {
    kernel: Hw,
    stride: Hw,
    padding: Hw,
}

impl AvgPool2d {
    /// Construct with kernel, stride, and padding.
    ///
    /// # Errors
    /// Returns [`NnError::Format`] if any kernel or stride dimension is zero.
    pub fn new(kernel: Hw, stride: Hw, padding: Hw) -> NnResult<Self> {
        check_hw("AvgPool2d", kernel, stride)?;
        Ok(Self {
            kernel,
            stride,
            padding,
        })
    }

    /// Forward pass.
    ///
    /// # Errors
    /// Returns [`NnError::ShapeMismatch`] on wrong rank; [`NnError::Format`]
    /// if the input spatial size is smaller than the padded kernel.
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        pool2d_forward(x, self.kernel, self.stride, self.padding, PoolKind::Avg)
    }

    /// Kernel size.
    #[must_use]
    pub const fn kernel(&self) -> Hw {
        self.kernel
    }

    /// Stride.
    #[must_use]
    pub const fn stride(&self) -> Hw {
        self.stride
    }

    /// Padding.
    #[must_use]
    pub const fn padding(&self) -> Hw {
        self.padding
    }
}

#[derive(Debug, Clone, Copy)]
enum PoolKind {
    Max,
    Avg,
}

fn check_hw(what: &'static str, kernel: Hw, stride: Hw) -> NnResult<()> {
    if kernel.0 == 0 || kernel.1 == 0 {
        return Err(NnError::Format(format!("{what}: kernel must be >= 1")));
    }
    if stride.0 == 0 || stride.1 == 0 {
        return Err(NnError::Format(format!("{what}: stride must be >= 1")));
    }
    Ok(())
}

fn out_size(input: usize, pad: usize, k: usize, stride: usize) -> Option<usize> {
    let padded = input.checked_add(pad.checked_mul(2)?)?;
    let after_kernel = padded.checked_sub(k)?;
    Some(after_kernel / stride + 1)
}

fn pool2d_forward(
    x: &Tensor,
    kernel: Hw,
    stride: Hw,
    padding: Hw,
    kind: PoolKind,
) -> NnResult<Tensor> {
    if x.shape().ndim() != 4 {
        return Err(NnError::ShapeMismatch {
            expected: "rank-4 [N, C, H, W]".into(),
            got: format!("rank-{}", x.shape().ndim()),
        });
    }
    let d = x.shape().dims();
    let (n, c, h_in, w_in) = (d[0], d[1], d[2], d[3]);
    let (kh, kw) = kernel;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let h_out = out_size(h_in, ph, kh, sh).ok_or_else(|| {
        NnError::Format(format!(
            "pool2d: H_in={h_in} too small for kernel={kh} padding={ph}"
        ))
    })?;
    let w_out = out_size(w_in, pw, kw, sw).ok_or_else(|| {
        NnError::Format(format!(
            "pool2d: W_in={w_in} too small for kernel={kw} padding={pw}"
        ))
    })?;

    let out_shape = Shape::d4(n, c, h_out, w_out);
    let mut out = Tensor::zeros(out_shape);
    let x_data = x.data();
    let out_data = out.data_mut();

    let x_stride_n = c * h_in * w_in;
    let x_stride_c = h_in * w_in;
    let y_stride_n = c * h_out * w_out;
    let y_stride_c = h_out * w_out;

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut agg = match kind {
                        PoolKind::Max => f32::NEG_INFINITY,
                        PoolKind::Avg => 0.0,
                    };
                    // count_include_pad = true: divisor is always kh*kw.
                    // Padded cells contribute 0 to Avg and -inf to Max,
                    // matching PyTorch semantics.
                    for khi in 0..kh {
                        let ih = (oh * sh + khi).checked_sub(ph);
                        let ih_in = ih.filter(|&v| v < h_in);
                        for kwi in 0..kw {
                            let iw = (ow * sw + kwi).checked_sub(pw);
                            let iw_in = iw.filter(|&v| v < w_in);
                            let v = match (ih_in, iw_in) {
                                (Some(ih_v), Some(iw_v)) => {
                                    let xi = ni * x_stride_n + ci * x_stride_c + ih_v * w_in + iw_v;
                                    x_data[xi]
                                }
                                _ => match kind {
                                    PoolKind::Max => f32::NEG_INFINITY,
                                    PoolKind::Avg => 0.0,
                                },
                            };
                            match kind {
                                PoolKind::Max => {
                                    if v > agg {
                                        agg = v;
                                    }
                                }
                                PoolKind::Avg => agg += v,
                            }
                        }
                    }
                    let yi = ni * y_stride_n + ci * y_stride_c + oh * w_out + ow;
                    out_data[yi] = match kind {
                        PoolKind::Max => agg,
                        PoolKind::Avg => agg / (kh * kw) as f32,
                    };
                }
            }
        }
    }
    Ok(out)
}
