//! Hand-computed correctness tests for `Conv2d`, `MaxPool2d`, `AvgPool2d`, `BatchNorm2d`.
//!
//! Every number in this file is independently computable with a calculator.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "hand-computed integer-valued f32 results: exact equality is the point"
)]

use immortal_nn::conv::{AvgPool2d, BatchNorm2d, Conv2d, MaxPool2d};
use immortal_nn::tensor::{Shape, Tensor};

#[test]
fn conv2d_known_output_3x3_input_2x2_kernel_identity_center() {
    // Input (1, 1, 3, 3):
    //   1 2 3
    //   4 5 6
    //   7 8 9
    let x = Tensor::from_vec(
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
        Shape::d4(1, 1, 3, 3),
    )
    .unwrap();
    // Kernel (1, 1, 2, 2) = identity on top-left of window:
    //   1 0
    //   0 0
    // Conv with stride=1, padding=0 → 2×2 output, each position = top-left input.
    let w = Tensor::from_vec(vec![1., 0., 0., 0.], Shape::d4(1, 1, 2, 2)).unwrap();
    let conv = Conv2d::from_tensors(w, None, 1, 1, (2, 2), (1, 1), (0, 0)).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.shape().dims(), &[1, 1, 2, 2]);
    // Expected: y[0,0,i,j] = x[0,0,i,j] for i,j in 0..2.
    assert_eq!(y.data(), &[1.0, 2.0, 4.0, 5.0]);
}

#[test]
fn conv2d_all_ones_kernel_computes_2x2_sum() {
    // Input 3×3:
    //   1 2 3
    //   4 5 6
    //   7 8 9
    let x = Tensor::from_vec(
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
        Shape::d4(1, 1, 3, 3),
    )
    .unwrap();
    // All-ones 2×2 kernel → each output = sum of a 2×2 patch.
    let w = Tensor::from_vec(vec![1., 1., 1., 1.], Shape::d4(1, 1, 2, 2)).unwrap();
    let conv = Conv2d::from_tensors(w, None, 1, 1, (2, 2), (1, 1), (0, 0)).unwrap();
    let y = conv.forward(&x).unwrap();
    // Expected:
    //   (1+2+4+5) (2+3+5+6) = 12 16
    //   (4+5+7+8) (5+6+8+9)   24 28
    assert_eq!(y.data(), &[12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn conv2d_bias_is_added() {
    let x = Tensor::from_vec(vec![1., 1., 1., 1.], Shape::d4(1, 1, 2, 2)).unwrap();
    let w = Tensor::from_vec(vec![1., 1., 1., 1.], Shape::d4(1, 1, 2, 2)).unwrap();
    // bias=10 for the single output channel.
    let b = Tensor::from_vec(vec![10.0], Shape::d2(1, 1)).unwrap();
    let conv = Conv2d::from_tensors(w, Some(b), 1, 1, (2, 2), (1, 1), (0, 0)).unwrap();
    let y = conv.forward(&x).unwrap();
    // sum of 2x2 ones = 4, + bias 10 = 14.
    assert_eq!(y.data(), &[14.0]);
}

#[test]
fn conv2d_stride_2_halves_spatial_size() {
    // 4×4 input, 2×2 kernel, stride 2 → 2×2 output.
    let x = Tensor::from_vec((0..16).map(|i| i as f32).collect(), Shape::d4(1, 1, 4, 4)).unwrap();
    let w = Tensor::from_vec(vec![1., 0., 0., 0.], Shape::d4(1, 1, 2, 2)).unwrap();
    let conv = Conv2d::from_tensors(w, None, 1, 1, (2, 2), (2, 2), (0, 0)).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.shape().dims(), &[1, 1, 2, 2]);
    // Output picks top-left of every 2×2 stride window:
    //   row 0: x[0,0]=0  x[0,2]=2
    //   row 1: x[2,0]=8  x[2,2]=10
    assert_eq!(y.data(), &[0.0, 2.0, 8.0, 10.0]);
}

#[test]
fn conv2d_padding_1_preserves_3x3_output() {
    // 3×3 input, 3×3 all-ones kernel, padding 1, stride 1 → 3×3 output of sums.
    let x = Tensor::from_vec(
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
        Shape::d4(1, 1, 3, 3),
    )
    .unwrap();
    let w = Tensor::from_vec(vec![1.0; 9], Shape::d4(1, 1, 3, 3)).unwrap();
    let conv = Conv2d::from_tensors(w, None, 1, 1, (3, 3), (1, 1), (1, 1)).unwrap();
    let y = conv.forward(&x).unwrap();
    assert_eq!(y.shape().dims(), &[1, 1, 3, 3]);
    // Center (1,1) sums all 9 input values = 45.
    assert_eq!(y.data()[4], 45.0);
    // Top-left (0,0) sums the 2x2 sub-block x[0..2,0..2] = 1+2+4+5 = 12.
    assert_eq!(y.data()[0], 12.0);
}

#[test]
fn maxpool2d_2x2_stride2_picks_maxes() {
    // 4×4 input:
    //   1  2  5  6
    //   3  4  7  8
    //   9 10 13 14
    //  11 12 15 16
    let data = vec![
        1., 2., 5., 6., 3., 4., 7., 8., 9., 10., 13., 14., 11., 12., 15., 16.,
    ];
    let x = Tensor::from_vec(data, Shape::d4(1, 1, 4, 4)).unwrap();
    let pool = MaxPool2d::new((2, 2), (2, 2), (0, 0)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.shape().dims(), &[1, 1, 2, 2]);
    // Max of each 2×2 block: 4, 8, 12, 16.
    assert_eq!(y.data(), &[4.0, 8.0, 12.0, 16.0]);
}

#[test]
fn avgpool2d_2x2_stride2_computes_means() {
    let data = vec![
        1., 2., 5., 6., 3., 4., 7., 8., 9., 10., 13., 14., 11., 12., 15., 16.,
    ];
    let x = Tensor::from_vec(data, Shape::d4(1, 1, 4, 4)).unwrap();
    let pool = AvgPool2d::new((2, 2), (2, 2), (0, 0)).unwrap();
    let y = pool.forward(&x).unwrap();
    assert_eq!(y.shape().dims(), &[1, 1, 2, 2]);
    // Avg of each 2×2 block:
    //   (1+2+3+4)/4 = 2.5   (5+6+7+8)/4 = 6.5
    //   (9+10+11+12)/4 = 10.5  (13+14+15+16)/4 = 14.5
    assert_eq!(y.data(), &[2.5, 6.5, 10.5, 14.5]);
}

#[test]
fn batchnorm2d_with_unit_stats_is_identity() {
    // gamma=1, beta=0, mean=0, var=1, eps=0-ish → y ≈ x.
    let n = 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32 - 2.0).collect();
    let x = Tensor::from_vec(data.clone(), Shape::d4(1, 1, 2, 2)).unwrap();
    let gamma = Tensor::from_vec(vec![1.0], Shape::d2(1, 1)).unwrap();
    let beta = Tensor::from_vec(vec![0.0], Shape::d2(1, 1)).unwrap();
    let mean = Tensor::from_vec(vec![0.0], Shape::d2(1, 1)).unwrap();
    let var = Tensor::from_vec(vec![1.0], Shape::d2(1, 1)).unwrap();
    let bn = BatchNorm2d::from_tensors(gamma, beta, mean, var, 1, 1e-10).unwrap();
    let y = bn.forward(&x).unwrap();
    for (yv, xv) in y.data().iter().zip(data.iter()) {
        assert!((*yv - *xv).abs() < 1e-4, "y={yv} expected {xv}");
    }
}

#[test]
fn batchnorm2d_known_normalization() {
    // gamma=2, beta=1, mean=5, var=4, eps=0. x=[5, 7, 9, 11].
    // normalized = (x - 5) / sqrt(4) = (x-5)/2 → [0, 1, 2, 3]
    // scaled+shifted = 2 * normalized + 1 → [1, 3, 5, 7]
    let x = Tensor::from_vec(vec![5., 7., 9., 11.], Shape::d4(1, 1, 2, 2)).unwrap();
    let gamma = Tensor::from_vec(vec![2.0], Shape::d2(1, 1)).unwrap();
    let beta = Tensor::from_vec(vec![1.0], Shape::d2(1, 1)).unwrap();
    let mean = Tensor::from_vec(vec![5.0], Shape::d2(1, 1)).unwrap();
    let var = Tensor::from_vec(vec![4.0], Shape::d2(1, 1)).unwrap();
    // Tiny eps is required (BN rejects eps <= 0); negligible at this scale.
    let bn = BatchNorm2d::from_tensors(gamma, beta, mean, var, 1, 1e-8).unwrap();
    let y = bn.forward(&x).unwrap();
    let expect = [1.0, 3.0, 5.0, 7.0];
    for (yv, ev) in y.data().iter().zip(expect.iter()) {
        assert!((*yv - *ev).abs() < 1e-3, "y={yv} expected {ev}");
    }
}

#[test]
fn conv2d_wrong_input_channels_is_rejected() {
    // Conv declared as in_c=3, but input has c=1.
    let w = Tensor::from_vec(vec![1.0; 12], Shape::d4(1, 3, 2, 2)).unwrap();
    let conv = Conv2d::from_tensors(w, None, 3, 1, (2, 2), (1, 1), (0, 0)).unwrap();
    let x = Tensor::zeros(Shape::d4(1, 1, 3, 3));
    assert!(conv.forward(&x).is_err());
}

#[test]
fn conv2d_wrong_input_rank_is_rejected() {
    let w = Tensor::from_vec(vec![1., 0., 0., 0.], Shape::d4(1, 1, 2, 2)).unwrap();
    let conv = Conv2d::from_tensors(w, None, 1, 1, (2, 2), (1, 1), (0, 0)).unwrap();
    let x = Tensor::zeros(Shape::d2(3, 3));
    assert!(conv.forward(&x).is_err());
}
