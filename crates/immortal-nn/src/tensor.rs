//! Nd-Tensor core: [`Shape`] + [`Tensor`].
//!
//! Inference-only. No tape, no gradients, no graph. Tensors are standalone
//! owned values with row-major contiguous layout. Rank is fixed at 2, 3, or 4.

use crate::{NnError, NnResult};

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// Tensor shape with fixed maximum rank of 4.
///
/// Layout convention: row-major contiguous. Strides are derived from `dims`
/// on demand and never stored.
///
/// Unused dimensions (indices `>= ndim`) are required to be `1`. This is
/// validated by [`Shape::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [usize; 4],
    ndim: u8,
}

impl Shape {
    /// Construct a 2-D shape `[d0, d1]`.
    ///
    /// Use [`Shape::new`] when dimensions come from untrusted model metadata.
    /// This constructor is for already-validated internal shape literals and
    /// derived dimensions.
    #[must_use]
    pub fn d2(d0: usize, d1: usize) -> Self {
        debug_assert!(d0 > 0, "d2: d0 must be non-zero");
        debug_assert!(d1 > 0, "d2: d1 must be non-zero");
        Self {
            dims: [d0, d1, 1, 1],
            ndim: 2,
        }
    }

    /// Construct a 3-D shape `[d0, d1, d2]`.
    ///
    /// Use [`Shape::new`] when dimensions come from untrusted model metadata.
    /// This constructor is for already-validated internal shape literals and
    /// derived dimensions.
    #[must_use]
    pub fn d3(d0: usize, d1: usize, d2: usize) -> Self {
        debug_assert!(d0 > 0, "d3: d0 must be non-zero");
        debug_assert!(d1 > 0, "d3: d1 must be non-zero");
        debug_assert!(d2 > 0, "d3: d2 must be non-zero");
        Self {
            dims: [d0, d1, d2, 1],
            ndim: 3,
        }
    }

    /// Construct a 4-D shape `[d0, d1, d2, d3]`.
    ///
    /// Use [`Shape::new`] when dimensions come from untrusted model metadata.
    /// This constructor is for already-validated internal shape literals and
    /// derived dimensions.
    #[must_use]
    pub fn d4(d0: usize, d1: usize, d2: usize, d3: usize) -> Self {
        debug_assert!(d0 > 0, "d4: d0 must be non-zero");
        debug_assert!(d1 > 0, "d4: d1 must be non-zero");
        debug_assert!(d2 > 0, "d4: d2 must be non-zero");
        debug_assert!(d3 > 0, "d4: d3 must be non-zero");
        Self {
            dims: [d0, d1, d2, d3],
            ndim: 4,
        }
    }

    /// Construct a shape from raw `dims` and `ndim`, validating invariants.
    ///
    /// # Errors
    /// Returns [`NnError::UnsupportedRank`] if `ndim` is not 2, 3, or 4, or
    /// [`NnError::Format`] if any active dim is zero or any inactive dim != 1.
    pub fn new(dims: [usize; 4], ndim: u8) -> NnResult<Self> {
        if !(2..=4).contains(&ndim) {
            return Err(NnError::UnsupportedRank { rank: ndim });
        }
        for (i, &d) in dims.iter().enumerate() {
            if i < ndim as usize {
                if d == 0 {
                    return Err(NnError::Format(format!("shape dim {i} is zero")));
                }
            } else if d != 1 {
                return Err(NnError::Format(format!(
                    "inactive shape dim {i} must be 1, got {d}"
                )));
            }
        }
        Ok(Self { dims, ndim })
    }

    /// Number of active dimensions (2, 3, or 4).
    #[must_use]
    pub const fn ndim(&self) -> u8 {
        self.ndim
    }

    /// Active dimension sizes as a slice of length [`Self::ndim`].
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.ndim as usize]
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims().iter().product()
    }

    /// Row-major strides, computed on demand.
    #[must_use]
    pub fn strides(&self) -> [usize; 4] {
        let mut s = [1usize; 4];
        let n = self.ndim as usize;
        if n >= 2 {
            for i in (0..n - 1).rev() {
                s[i] = s[i + 1] * self.dims[i + 1];
            }
        }
        s
    }
}

impl core::fmt::Display for Shape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.dims())
    }
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

/// Inference-only owned tensor. Row-major contiguous, `f32` only in v1.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    /// Allocate a zeroed tensor with the given shape.
    #[must_use]
    pub fn zeros(shape: Shape) -> Self {
        Self {
            data: vec![0.0; shape.numel()],
            shape,
        }
    }

    /// Build a tensor from a `Vec<f32>` and a shape.
    ///
    /// # Errors
    /// Returns [`NnError::ElementCountMismatch`] if `data.len() != shape.numel()`.
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> NnResult<Self> {
        if data.len() != shape.numel() {
            return Err(NnError::ElementCountMismatch {
                provided: data.len(),
                required: shape.numel(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Borrow the underlying flat data.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable borrow of the underlying flat data. Used by ops internally.
    pub(crate) fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Shape of this tensor.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Reshape into a new shape with the same number of elements.
    ///
    /// Always a copy-free operation in v1 (tensors are always contiguous).
    /// Element count must match exactly — no silent broadcasting.
    ///
    /// # Errors
    /// Returns [`NnError::ElementCountMismatch`] if the new shape's element
    /// count differs from the current one.
    pub fn reshape(&self, new_shape: Shape) -> NnResult<Self> {
        if new_shape.numel() != self.shape.numel() {
            return Err(NnError::ElementCountMismatch {
                provided: self.shape.numel(),
                required: new_shape.numel(),
            });
        }
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
        })
    }

    // -- Indexing helpers (debug-checked, unchecked in release) -------------

    /// 2-D element access. Debug-checked bounds; unchecked in release.
    #[must_use]
    #[inline]
    pub fn get_2d(&self, row: usize, col: usize) -> f32 {
        debug_assert_eq!(
            self.shape.ndim, 2,
            "get_2d on rank-{} tensor",
            self.shape.ndim
        );
        let [r, c] = [self.shape.dims[0], self.shape.dims[1]];
        debug_assert!(
            row < r && col < c,
            "index ({row},{col}) out of bounds ({r},{c})"
        );
        self.data[row * c + col]
    }

    /// 3-D element access. Debug-checked bounds; unchecked in release.
    #[must_use]
    #[inline]
    pub fn get_3d(&self, b: usize, s: usize, fdim: usize) -> f32 {
        debug_assert_eq!(
            self.shape.ndim, 3,
            "get_3d on rank-{} tensor",
            self.shape.ndim
        );
        let d = self.shape.dims;
        debug_assert!(b < d[0] && s < d[1] && fdim < d[2]);
        self.data[(b * d[1] + s) * d[2] + fdim]
    }

    /// 4-D element access. Debug-checked bounds; unchecked in release.
    #[must_use]
    #[inline]
    pub fn get_4d(&self, b: usize, h: usize, s: usize, fdim: usize) -> f32 {
        debug_assert_eq!(
            self.shape.ndim, 4,
            "get_4d on rank-{} tensor",
            self.shape.ndim
        );
        let d = self.shape.dims;
        debug_assert!(b < d[0] && h < d[1] && s < d[2] && fdim < d[3]);
        self.data[((b * d[1] + h) * d[2] + s) * d[3] + fdim]
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "tensor indexing tests assert exact integer-valued floats from reshape/strided access"
)]
mod tests {
    use super::*;

    #[test]
    fn shape_new_validates_rank() {
        assert!(Shape::new([2, 3, 1, 1], 1).is_err());
        assert!(Shape::new([2, 3, 1, 1], 5).is_err());
        assert!(Shape::new([2, 3, 1, 1], 2).is_ok());
    }

    #[test]
    fn shape_new_rejects_zero_active_dim() {
        assert!(Shape::new([0, 3, 1, 1], 2).is_err());
    }

    #[test]
    fn shape_new_rejects_nonunit_inactive_dim() {
        assert!(Shape::new([2, 3, 4, 1], 2).is_err());
    }

    #[test]
    fn shape_strides_row_major() {
        let s = Shape::d3(2, 3, 4);
        let st = s.strides();
        assert_eq!(&st[..3], &[12, 4, 1]);
    }

    #[test]
    fn tensor_from_vec_checks_count() {
        assert!(Tensor::from_vec(vec![1.0; 6], Shape::d2(2, 3)).is_ok());
        assert!(Tensor::from_vec(vec![1.0; 5], Shape::d2(2, 3)).is_err());
    }

    #[test]
    fn tensor_reshape_preserves_count() {
        let t = Tensor::from_vec((0..12).map(|x| x as f32).collect(), Shape::d2(3, 4)).unwrap();
        let r = t.reshape(Shape::d3(2, 2, 3)).unwrap();
        assert_eq!(r.numel(), 12);
        assert_eq!(r.data()[0], 0.0);
        assert_eq!(r.data()[11], 11.0);
    }

    #[test]
    fn tensor_reshape_rejects_count_mismatch() {
        let t = Tensor::zeros(Shape::d2(3, 4));
        assert!(t.reshape(Shape::d2(2, 5)).is_err());
    }

    #[test]
    fn tensor_indexing_2d() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], Shape::d2(2, 3)).unwrap();
        assert_eq!(t.get_2d(0, 0), 1.0);
        assert_eq!(t.get_2d(0, 2), 3.0);
        assert_eq!(t.get_2d(1, 1), 5.0);
    }

    #[test]
    fn tensor_indexing_3d() {
        let t = Tensor::from_vec((0..24).map(|x| x as f32).collect(), Shape::d3(2, 3, 4)).unwrap();
        assert_eq!(t.get_3d(0, 0, 0), 0.0);
        assert_eq!(t.get_3d(1, 2, 3), 23.0);
        assert_eq!(t.get_3d(1, 0, 0), 12.0);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Shape::d2 round-trips through dims() — no information is lost.
            #[test]
            fn shape_d2_roundtrip(d0 in 1usize..=8, d1 in 1usize..=8) {
                let s = Shape::d2(d0, d1);
                prop_assert_eq!(s.ndim(), 2);
                prop_assert_eq!(s.dims(), &[d0, d1]);
                prop_assert_eq!(s.numel(), d0 * d1);
            }

            /// Shape::d3 round-trips through dims().
            #[test]
            fn shape_d3_roundtrip(d0 in 1usize..=8, d1 in 1usize..=8, d2 in 1usize..=8) {
                let s = Shape::d3(d0, d1, d2);
                prop_assert_eq!(s.ndim(), 3);
                prop_assert_eq!(s.dims(), &[d0, d1, d2]);
                prop_assert_eq!(s.numel(), d0 * d1 * d2);
            }

            /// Shape::d4 round-trips through dims().
            #[test]
            fn shape_d4_roundtrip(
                d0 in 1usize..=4,
                d1 in 1usize..=4,
                d2 in 1usize..=4,
                d3 in 1usize..=4,
            ) {
                let s = Shape::d4(d0, d1, d2, d3);
                prop_assert_eq!(s.ndim(), 4);
                prop_assert_eq!(s.dims(), &[d0, d1, d2, d3]);
                prop_assert_eq!(s.numel(), d0 * d1 * d2 * d3);
            }

            /// Tensor::from_vec rejects vec whose length != shape.numel().
            #[test]
            fn from_vec_rejects_wrong_count(
                rows in 2usize..=6,
                cols in 2usize..=6,
                extra in 1usize..=4,
            ) {
                let shape = Shape::d2(rows, cols);
                let correct_len = rows * cols;
                // Too short
                let short: Vec<f32> = vec![0.0; correct_len - 1];
                prop_assert!(Tensor::from_vec(short, shape).is_err());
                // Too long
                let long: Vec<f32> = vec![0.0; correct_len + extra];
                prop_assert!(Tensor::from_vec(long, shape).is_err());
            }

            /// Tensor::from_vec succeeds when length == shape.numel().
            #[test]
            fn from_vec_accepts_correct_count(rows in 1usize..=8, cols in 1usize..=8) {
                let shape = Shape::d2(rows, cols);
                let data = vec![1.0f32; rows * cols];
                prop_assert!(Tensor::from_vec(data, shape).is_ok());
            }

            /// Tensor::reshape preserves element count when shapes are compatible.
            #[test]
            fn reshape_preserves_numel(
                rows in 1usize..=8,
                cols in 1usize..=8,
            ) {
                // reshape [rows*cols] 2-D tensor into [1, rows*cols] — always valid
                let n = rows * cols;
                let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let t = Tensor::from_vec(data, Shape::d2(rows, cols)).unwrap();
                let new_shape = Shape::d2(1, n);
                let r = t.reshape(new_shape).unwrap();
                prop_assert_eq!(r.numel(), n);
                // All elements preserved in order
                for (i, &v) in r.data().iter().enumerate() {
                    prop_assert_eq!(v, i as f32);
                }
            }

            /// get_2d is panic-free inside declared bounds.
            #[test]
            fn get_2d_in_bounds_no_panic(
                rows in 1usize..=8,
                cols in 1usize..=8,
                row in 0usize..8,
                col in 0usize..8,
            ) {
                // Clamp indices to valid range
                let row = row % rows;
                let col = col % cols;
                let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
                let t = Tensor::from_vec(data, Shape::d2(rows, cols)).unwrap();
                let v = t.get_2d(row, col);
                prop_assert!(v.is_finite());
                prop_assert_eq!(v, (row * cols + col) as f32);
            }

            /// get_3d is panic-free inside declared bounds.
            #[test]
            fn get_3d_in_bounds_no_panic(
                d0 in 1usize..=4,
                d1 in 1usize..=4,
                d2 in 1usize..=4,
                i0 in 0usize..4,
                i1 in 0usize..4,
                i2 in 0usize..4,
            ) {
                let i0 = i0 % d0;
                let i1 = i1 % d1;
                let i2 = i2 % d2;
                let n = d0 * d1 * d2;
                let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
                let t = Tensor::from_vec(data, Shape::d3(d0, d1, d2)).unwrap();
                let v = t.get_3d(i0, i1, i2);
                prop_assert_eq!(v, ((i0 * d1 + i1) * d2 + i2) as f32);
            }
        }
    }
}
