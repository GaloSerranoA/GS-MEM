//! Transformer correctness + determinism tests.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::identity_op,
    reason = "hand-computed transformer tests; `1 * S * D` made explicit for tensor-shape clarity"
)]

use immortal_nn::blocks::{Activation, LayerNorm, Linear, MultiHeadAttention};
use immortal_nn::tensor::{Shape, Tensor};
use immortal_nn::transformer::{
    causal_mask, cross_attention, FeedForward, MultiHeadCrossAttention, PositionalEncoding,
    TransformerDecoderBlock,
};

// ---------------------------------------------------------------------------
// PositionalEncoding
// ---------------------------------------------------------------------------

#[test]
fn positional_encoding_pos_zero_is_sin0_cos0() {
    // At position 0 every sin term is 0 and every cos term is 1.
    let pe = PositionalEncoding::new(8, 4).unwrap();
    let row0 = &pe.table().data()[0..4];
    // Expect [sin(0), cos(0), sin(0), cos(0)] = [0, 1, 0, 1].
    assert!((row0[0] - 0.0).abs() < 1e-6);
    assert!((row0[1] - 1.0).abs() < 1e-6);
    assert!((row0[2] - 0.0).abs() < 1e-6);
    assert!((row0[3] - 1.0).abs() < 1e-6);
}

#[test]
fn positional_encoding_is_deterministic() {
    let a = PositionalEncoding::new(32, 16).unwrap();
    let b = PositionalEncoding::new(32, 16).unwrap();
    assert_eq!(a.table().data(), b.table().data());
}

#[test]
fn positional_encoding_rejects_seq_len_greater_than_max_len() {
    let pe = PositionalEncoding::new(4, 8).unwrap();
    let x = Tensor::zeros(Shape::d3(1, 5, 8));
    assert!(pe.forward(&x).is_err());
}

#[test]
fn positional_encoding_adds_to_input() {
    // With zero input, the output should equal the encoding table rows.
    let pe = PositionalEncoding::new(3, 4).unwrap();
    let x = Tensor::zeros(Shape::d3(1, 3, 4));
    let y = pe.forward(&x).unwrap();
    // y[0, pos, :] must equal pe.table()[pos, :].
    let table = pe.table().data();
    for pos in 0..3 {
        for d in 0..4 {
            let got = y.data()[(pos * 4) + d];
            let want = table[pos * 4 + d];
            assert!(
                (got - want).abs() < 1e-6,
                "pos={pos} d={d} got={got} want={want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// causal_mask
// ---------------------------------------------------------------------------

#[test]
fn causal_mask_is_lower_triangular_with_ones() {
    let m = causal_mask(4);
    assert_eq!(m.shape().dims(), &[4, 4]);
    let d = m.data();
    // Row i: cols 0..=i are 1, cols i+1..4 are 0.
    for i in 0..4 {
        for j in 0..4 {
            let v = d[i * 4 + j];
            let expected = if j <= i { 1.0 } else { 0.0 };
            assert_eq!(v, expected, "row={i} col={j}");
        }
    }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

#[test]
fn feed_forward_identity_relu_passes_positive_values() {
    // up: 2→2 identity; activation=ReLU; down: 2→2 identity → output = ReLU(x).
    let w_up = Tensor::from_vec(vec![1., 0., 0., 1.], Shape::d2(2, 2)).unwrap();
    let w_down = Tensor::from_vec(vec![1., 0., 0., 1.], Shape::d2(2, 2)).unwrap();
    let up = Linear::from_tensors(w_up, None).unwrap();
    let down = Linear::from_tensors(w_down, None).unwrap();
    let ff = FeedForward::from_parts(up, down, Activation::Relu).unwrap();
    // x = [1, -1] → ReLU → [1, 0].
    let x = Tensor::from_vec(vec![1.0, -1.0], Shape::d2(1, 2)).unwrap();
    let y = ff.forward(&x).unwrap();
    assert_eq!(y.data(), &[1.0, 0.0]);
}

#[test]
fn feed_forward_rejects_mismatched_inner_dim() {
    // up: 2→4; down: 2→2 (inner mismatch).
    let w_up = Tensor::from_vec(vec![0.0; 8], Shape::d2(4, 2)).unwrap();
    let w_down = Tensor::from_vec(vec![0.0; 4], Shape::d2(2, 2)).unwrap();
    let up = Linear::from_tensors(w_up, None).unwrap();
    let down = Linear::from_tensors(w_down, None).unwrap();
    assert!(FeedForward::from_parts(up, down, Activation::Relu).is_err());
}

// ---------------------------------------------------------------------------
// cross_attention
// ---------------------------------------------------------------------------

#[test]
fn cross_attention_output_shape_matches_query_seq() {
    // B=1, H=1, Sq=2, Skv=3, D=2.
    let q = Tensor::from_vec(
        (0..(1 * 1 * 2 * 2)).map(|i| i as f32 + 1.0).collect(),
        Shape::d4(1, 1, 2, 2),
    )
    .unwrap();
    let k = Tensor::from_vec(
        (0..(1 * 1 * 3 * 2)).map(|i| i as f32 + 1.0).collect(),
        Shape::d4(1, 1, 3, 2),
    )
    .unwrap();
    let v = Tensor::from_vec(
        (0..(1 * 1 * 3 * 2)).map(|i| i as f32 + 1.0).collect(),
        Shape::d4(1, 1, 3, 2),
    )
    .unwrap();
    let out = cross_attention(&q, &k, &v, None).unwrap();
    assert_eq!(out.shape().dims(), &[1, 1, 2, 2]);
}

#[test]
fn cross_attention_rejects_batch_mismatch() {
    let q = Tensor::zeros(Shape::d4(1, 1, 2, 2));
    let k = Tensor::zeros(Shape::d4(2, 1, 2, 2));
    let v = Tensor::zeros(Shape::d4(1, 1, 2, 2));
    assert!(cross_attention(&q, &k, &v, None).is_err());
}

// ---------------------------------------------------------------------------
// MultiHeadCrossAttention
// ---------------------------------------------------------------------------

fn mk_linear(in_f: usize, out_f: usize) -> Linear {
    let data: Vec<f32> = (0..(out_f * in_f)).map(|i| 0.05 * (i as f32)).collect();
    let w = Tensor::from_vec(data, Shape::d2(out_f, in_f)).unwrap();
    Linear::from_tensors(w, None).unwrap()
}

#[test]
fn multi_head_cross_attention_output_shape_matches_query() {
    // d_model = 4, num_heads = 2, head_dim = 2.
    let mhca = MultiHeadCrossAttention::from_parts(
        mk_linear(4, 4),
        mk_linear(4, 4),
        mk_linear(4, 4),
        mk_linear(4, 4),
        2,
    )
    .unwrap();
    let q = Tensor::from_vec(
        (0..(1 * 2 * 4)).map(|i| 0.1 * (i as f32)).collect(),
        Shape::d3(1, 2, 4),
    )
    .unwrap();
    let kv = Tensor::from_vec(
        (0..(1 * 3 * 4)).map(|i| 0.1 * (i as f32)).collect(),
        Shape::d3(1, 3, 4),
    )
    .unwrap();
    let out = mhca.forward(&q, &kv, None).unwrap();
    assert_eq!(out.shape().dims(), &[1, 2, 4]);
}

#[test]
fn multi_head_cross_attention_rejects_non_divisible_heads() {
    let res = MultiHeadCrossAttention::from_parts(
        mk_linear(4, 4),
        mk_linear(4, 4),
        mk_linear(4, 4),
        mk_linear(4, 4),
        3, // 4 % 3 != 0
    );
    assert!(res.is_err());
}

// ---------------------------------------------------------------------------
// TransformerDecoderBlock
// ---------------------------------------------------------------------------

#[test]
fn transformer_decoder_block_forward_preserves_target_shape() {
    // d_model = 4, num_heads = 2, d_ff = 8.
    let self_attn = MultiHeadAttention::from_parts(
        mk_linear(4, 4),
        mk_linear(4, 4),
        mk_linear(4, 4),
        mk_linear(4, 4),
        2,
    )
    .unwrap();
    let cross_attn = MultiHeadCrossAttention::from_parts(
        mk_linear(4, 4),
        mk_linear(4, 4),
        mk_linear(4, 4),
        mk_linear(4, 4),
        2,
    )
    .unwrap();
    let gamma = Tensor::from_vec(vec![1.0; 4], Shape::d2(1, 4)).unwrap();
    let beta = Tensor::from_vec(vec![0.0; 4], Shape::d2(1, 4)).unwrap();
    let n1 = LayerNorm::from_tensors(gamma.clone(), beta.clone(), 1e-5);
    let n2 = LayerNorm::from_tensors(gamma.clone(), beta.clone(), 1e-5);
    let n3 = LayerNorm::from_tensors(gamma, beta, 1e-5);
    let ff = FeedForward::from_parts(mk_linear(4, 8), mk_linear(8, 4), Activation::Relu).unwrap();
    let block = TransformerDecoderBlock::from_parts(self_attn, cross_attn, ff, n1, n2, n3);

    let target = Tensor::from_vec(
        (0..(1 * 2 * 4)).map(|i| 0.01 * (i as f32 + 1.0)).collect(),
        Shape::d3(1, 2, 4),
    )
    .unwrap();
    let memory = Tensor::from_vec(
        (0..(1 * 3 * 4)).map(|i| 0.02 * (i as f32 + 1.0)).collect(),
        Shape::d3(1, 3, 4),
    )
    .unwrap();
    let out = block.forward(&target, &memory).unwrap();
    assert_eq!(out.shape().dims(), &[1, 2, 4]);
    // All elements finite.
    for v in out.data() {
        assert!(v.is_finite(), "non-finite decoder output: {v}");
    }
}
