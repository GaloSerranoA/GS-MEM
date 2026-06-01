//! 2-D convolution (NCHW, `f32`).
//!
//! Naive direct convolution: O(N * OutC * OutH * OutW * InC * Kh * Kw).
//! Perf optimization (im2col + batched matmul) deferred — this engine
//! targets small-model sovereign inference where correctness + zero
//! external dependencies matter more than throughput.

use crate::format::SovereignModel;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};

/// Stride + padding pair for the 2 spatial dimensions `(height, width)`.
pub type Hw = (usize, usize);

/// 2-D convolution layer. Inference only.
///
/// Weight layout: `[out_channels, in_channels, kernel_h, kernel_w]`.
/// Optional bias: `[out_channels]` (stored as rank-2 `[1, out_channels]`
/// to match how biases are written into `.sovereign` files).
#[derive(Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
    kernel: Hw,
    stride: Hw,
    padding: Hw,
}

impl Conv2d {
    /// Load from a `SovereignModel` using the given name prefix.
    /// Expects `{prefix}.weight` of shape `[OutC, InC, Kh, Kw]` and an
    /// optional `{prefix}.bias` of shape `[1, OutC]`.
    ///
    /// # Errors
    /// Returns [`NnError::ShapeMismatch`] if the weight rank or bias
    /// length does not match the expected convention.
    pub fn load(model: &SovereignModel, prefix: &str, stride: Hw, padding: Hw) -> NnResult<Self> {
        let w = model.tensor(&format!("{prefix}.weight"))?;
        if w.shape().ndim() != 4 {
            return Err(NnError::ShapeMismatch {
                expected: format!("{prefix}.weight: rank-4 [OutC, InC, Kh, Kw]"),
                got: format!("rank-{}", w.shape().ndim()),
            });
        }
        let dims = w.shape().dims();
        let out_channels = dims[0];
        let in_channels = dims[1];
        let kernel = (dims[2], dims[3]);

        let bias = match model.tensor(&format!("{prefix}.bias")) {
            Ok(b) => Some(b.clone()),
            Err(_) => None,
        };

        Self::from_tensors(
            w.clone(),
            bias,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
        )
    }

    /// Construct directly from owned weight and optional bias tensors.
    ///
    /// # Errors
    /// Returns [`NnError::ShapeMismatch`] on rank or size mismatch between
    /// weight / bias / declared channel counts.
    pub fn from_tensors(
        weight: Tensor,
        bias: Option<Tensor>,
        in_channels: usize,
        out_channels: usize,
        kernel: Hw,
        stride: Hw,
        padding: Hw,
    ) -> NnResult<Self> {
        if weight.shape().ndim() != 4 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-4 [OutC, InC, Kh, Kw]".into(),
                got: format!("rank-{}", weight.shape().ndim()),
            });
        }
        let dims = weight.shape().dims();
        if dims[0] != out_channels
            || dims[1] != in_channels
            || dims[2] != kernel.0
            || dims[3] != kernel.1
        {
            return Err(NnError::ShapeMismatch {
                expected: format!("[{out_channels},{in_channels},{},{}]", kernel.0, kernel.1),
                got: format!("{dims:?}"),
            });
        }
        if let Some(ref b) = bias {
            let bn = b.numel();
            if bn != out_channels {
                return Err(NnError::ShapeMismatch {
                    expected: format!("bias numel {out_channels}"),
                    got: format!("bias numel {bn}"),
                });
            }
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NnError::Format("Conv2d stride must be >= 1".into()));
        }
        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
        })
    }

    /// `in_channels` of the layer.
    #[must_use]
    pub const fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// `out_channels` of the layer.
    #[must_use]
    pub const fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Kernel size `(Kh, Kw)`.
    #[must_use]
    pub const fn kernel(&self) -> Hw {
        self.kernel
    }

    /// Stride `(Sh, Sw)`.
    #[must_use]
    pub const fn stride(&self) -> Hw {
        self.stride
    }

    /// Padding `(Ph, Pw)`.
    #[must_use]
    pub const fn padding(&self) -> Hw {
        self.padding
    }

    /// Compute the output spatial size for a given input size `h_in`.
    /// `out = (h_in + 2*pad - k) / stride + 1`.
    #[must_use]
    fn out_size(input: usize, pad: usize, k: usize, stride: usize) -> Option<usize> {
        let padded = input.checked_add(pad.checked_mul(2)?)?;
        let after_kernel = padded.checked_sub(k)?;
        Some(after_kernel / stride + 1)
    }

    /// Forward pass. Input shape `[N, InC, H, W]` → output shape
    /// `[N, OutC, OutH, OutW]`.
    ///
    /// # Errors
    /// Returns [`NnError::ShapeMismatch`] if the input rank or channel
    /// count is wrong, or [`NnError::Format`] if the input spatial size
    /// is smaller than the padded kernel.
    pub fn forward(&self, x: &Tensor) -> NnResult<Tensor> {
        if x.shape().ndim() != 4 {
            return Err(NnError::ShapeMismatch {
                expected: "rank-4 [N, InC, H, W]".into(),
                got: format!("rank-{}", x.shape().ndim()),
            });
        }
        let d = x.shape().dims();
        let (n, in_c, h_in, w_in) = (d[0], d[1], d[2], d[3]);
        if in_c != self.in_channels {
            return Err(NnError::ShapeMismatch {
                expected: format!("InC={}", self.in_channels),
                got: format!("InC={in_c}"),
            });
        }
        let (kh, kw) = self.kernel;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let h_out = Self::out_size(h_in, ph, kh, sh).ok_or_else(|| {
            NnError::Format(format!(
                "Conv2d: H_in={h_in} too small for kernel={kh} padding={ph}"
            ))
        })?;
        let w_out = Self::out_size(w_in, pw, kw, sw).ok_or_else(|| {
            NnError::Format(format!(
                "Conv2d: W_in={w_in} too small for kernel={kw} padding={pw}"
            ))
        })?;

        let out_shape = Shape::d4(n, self.out_channels, h_out, w_out);
        let mut out = Tensor::zeros(out_shape);

        let x_data = x.data();
        let w_data = self.weight.data();
        let bias_data = self.bias.as_ref().map(Tensor::data);
        let out_data = out.data_mut();

        // Strides on input: [InC*H*W, H*W, W, 1]
        let x_stride_n = in_c * h_in * w_in;
        let x_stride_c = h_in * w_in;
        // Strides on weight: [InC*Kh*Kw, Kh*Kw, Kw, 1]
        let w_stride_oc = in_c * kh * kw;
        let w_stride_ic = kh * kw;
        // Strides on output: [OutC*OutH*OutW, OutH*OutW, OutW, 1]
        let y_stride_n = self.out_channels * h_out * w_out;
        let y_stride_c = h_out * w_out;

        for ni in 0..n {
            for oc in 0..self.out_channels {
                let bias_val = bias_data.map_or(0.0, |b| b[oc]);
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = bias_val;
                        for ic in 0..in_c {
                            for khi in 0..kh {
                                // Padded region has negative virtual index —
                                // express with checked_sub to stay in usize.
                                let Some(ih) = (oh * sh + khi).checked_sub(ph) else {
                                    continue;
                                };
                                if ih >= h_in {
                                    continue;
                                }
                                for kwi in 0..kw {
                                    let Some(iw) = (ow * sw + kwi).checked_sub(pw) else {
                                        continue;
                                    };
                                    if iw >= w_in {
                                        continue;
                                    }
                                    let xi = ni * x_stride_n + ic * x_stride_c + ih * w_in + iw;
                                    let wi = oc * w_stride_oc + ic * w_stride_ic + khi * kw + kwi;
                                    acc += x_data[xi] * w_data[wi];
                                }
                            }
                        }
                        let yi = ni * y_stride_n + oc * y_stride_c + oh * w_out + ow;
                        out_data[yi] = acc;
                    }
                }
            }
        }
        Ok(out)
    }
}
