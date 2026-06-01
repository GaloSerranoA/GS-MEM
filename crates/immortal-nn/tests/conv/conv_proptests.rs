//! Conv module invariants.
//!
//! 1. Output shape follows the standard formula
//!    `(h_in + 2*pad - k) / stride + 1`.
//! 2. Forward is deterministic — same input + weights ⇒ identical output.
//! 3. All-zero input with zero bias produces all-zero output.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "proptests assert exact zero/equal outputs from analytically-exact expressions on small indices"
)]

use immortal_nn::conv::{AvgPool2d, Conv2d, MaxPool2d};
use immortal_nn::tensor::{Shape, Tensor};
use proptest::prelude::*;

proptest! {
    #[test]
    fn conv2d_output_shape_matches_formula(
        n in 1usize..=2,
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        h_in in 3usize..=8,
        w_in in 3usize..=8,
        kh in 1usize..=3,
        kw in 1usize..=3,
        sh in 1usize..=2,
        sw in 1usize..=2,
    ) {
        prop_assume!(kh <= h_in);
        prop_assume!(kw <= w_in);
        let x = Tensor::zeros(Shape::d4(n, in_c, h_in, w_in));
        let w = Tensor::from_vec(
            vec![0.1; out_c * in_c * kh * kw],
            Shape::d4(out_c, in_c, kh, kw),
        ).unwrap();
        let conv = Conv2d::from_tensors(w, None, in_c, out_c, (kh, kw), (sh, sw), (0, 0)).unwrap();
        let y = conv.forward(&x).unwrap();
        let expect_h = (h_in - kh) / sh + 1;
        let expect_w = (w_in - kw) / sw + 1;
        prop_assert_eq!(y.shape().dims(), &[n, out_c, expect_h, expect_w]);
    }

    #[test]
    fn conv2d_is_deterministic(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        h in 3usize..=6,
        w in 3usize..=6,
    ) {
        let data: Vec<f32> = (0..(in_c * h * w)).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let x = Tensor::from_vec(data, Shape::d4(1, in_c, h, w)).unwrap();
        let weight_data: Vec<f32> = (0..(out_c * in_c * 2 * 2))
            .map(|i| 0.1 * (i as f32))
            .collect();
        let weight = Tensor::from_vec(weight_data, Shape::d4(out_c, in_c, 2, 2)).unwrap();
        let c1 = Conv2d::from_tensors(
            weight.clone(), None, in_c, out_c, (2, 2), (1, 1), (0, 0),
        ).unwrap();
        let c2 = Conv2d::from_tensors(
            weight, None, in_c, out_c, (2, 2), (1, 1), (0, 0),
        ).unwrap();
        let y1 = c1.forward(&x).unwrap();
        let y2 = c2.forward(&x).unwrap();
        prop_assert_eq!(y1.data(), y2.data());
    }

    #[test]
    fn conv2d_zero_input_and_no_bias_is_zero_output(
        in_c in 1usize..=3,
        out_c in 1usize..=3,
        h in 3usize..=6,
        w in 3usize..=6,
    ) {
        let x = Tensor::zeros(Shape::d4(1, in_c, h, w));
        let weight_data: Vec<f32> = (0..(out_c * in_c * 2 * 2))
            .map(|i| 0.25 * (i as f32) - 0.5)
            .collect();
        let weight = Tensor::from_vec(weight_data, Shape::d4(out_c, in_c, 2, 2)).unwrap();
        let conv = Conv2d::from_tensors(weight, None, in_c, out_c, (2, 2), (1, 1), (0, 0)).unwrap();
        let y = conv.forward(&x).unwrap();
        for v in y.data() {
            prop_assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn maxpool2d_output_shape_matches_formula(
        n in 1usize..=2,
        c in 1usize..=3,
        h in 2usize..=6,
        w in 2usize..=6,
        k in 1usize..=2,
    ) {
        prop_assume!(k <= h && k <= w);
        let x = Tensor::zeros(Shape::d4(n, c, h, w));
        let pool = MaxPool2d::new((k, k), (k, k), (0, 0)).unwrap();
        let y = pool.forward(&x).unwrap();
        let expect_h = (h - k) / k + 1;
        let expect_w = (w - k) / k + 1;
        prop_assert_eq!(y.shape().dims(), &[n, c, expect_h, expect_w]);
    }

    #[test]
    fn avgpool2d_constant_input_preserves_value(
        val in -10.0f32..=10.0,
        h in 2usize..=6,
        w in 2usize..=6,
    ) {
        // Stride = kernel = 2 so every output window is fully inside the
        // input, no padding involved.
        prop_assume!(h % 2 == 0 && w % 2 == 0);
        let x = Tensor::from_vec(vec![val; h * w], Shape::d4(1, 1, h, w)).unwrap();
        let pool = AvgPool2d::new((2, 2), (2, 2), (0, 0)).unwrap();
        let y = pool.forward(&x).unwrap();
        for &v in y.data() {
            prop_assert!((v - val).abs() < 1e-4);
        }
    }
}
