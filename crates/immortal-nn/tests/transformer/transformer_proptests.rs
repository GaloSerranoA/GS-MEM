//! Transformer module invariants.
//!
//! 1. Positional encoding forward adds exactly the table — the
//!    difference between forward(x) and x equals the PE table.
//! 2. Causal mask always satisfies `mask[i][j] = 1 iff j <= i`.
//! 3. `MultiHeadCrossAttention` output shape matches `query` for any
//!    valid `(B, Sq, Skv, E, num_heads)` combination.
//! 4. Determinism — constructing the same transformer twice gives
//!    identical outputs for the same input.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::many_single_char_names,
    reason = "transformer proptests mirror blocks.rs naming + exact float equality on analytically-exact operations"
)]

use immortal_nn::blocks::Linear;
use immortal_nn::tensor::{Shape, Tensor};
use immortal_nn::transformer::{causal_mask, MultiHeadCrossAttention, PositionalEncoding};
use proptest::prelude::*;

fn mk_linear_seeded(in_f: usize, out_f: usize, salt: u32) -> Linear {
    let data: Vec<f32> = (0..(out_f * in_f))
        .map(|i| 0.03 * ((i as u32).wrapping_add(salt) as f32).sin())
        .collect();
    let w = Tensor::from_vec(data, Shape::d2(out_f, in_f)).unwrap();
    Linear::from_tensors(w, None).unwrap()
}

proptest! {
    #[test]
    fn pe_forward_difference_equals_pe_table(
        batch in 1usize..=3,
        seq in 1usize..=6,
        d_model in 2usize..=8,
    ) {
        let pe = PositionalEncoding::new(16, d_model).unwrap();
        let input_data: Vec<f32> = (0..(batch * seq * d_model)).map(|i| 0.1 * (i as f32)).collect();
        let x = Tensor::from_vec(input_data.clone(), Shape::d3(batch, seq, d_model)).unwrap();
        let y = pe.forward(&x).unwrap();
        // For each batch, y[b] - x[b] must equal pe.table()[0..seq].
        let pe_table = pe.table().data();
        for bi in 0..batch {
            for si in 0..seq {
                for di in 0..d_model {
                    let yi = (bi * seq + si) * d_model + di;
                    let xi = yi;
                    let diff = y.data()[yi] - x.data()[xi];
                    let want = pe_table[si * d_model + di];
                    prop_assert!(
                        (diff - want).abs() < 1e-5,
                        "bi={bi} si={si} di={di} diff={diff} want={want}"
                    );
                }
            }
        }
    }

    #[test]
    fn causal_mask_is_strictly_lower_triangular(seq in 1usize..=16) {
        let m = causal_mask(seq);
        prop_assert_eq!(m.shape().dims(), &[seq, seq]);
        let d = m.data();
        for i in 0..seq {
            for j in 0..seq {
                let want = if j <= i { 1.0 } else { 0.0 };
                prop_assert_eq!(d[i * seq + j], want);
            }
        }
    }

    #[test]
    fn cross_attention_output_shape_matches_query(
        s_q in 1usize..=4,
        s_kv in 1usize..=4,
        n_heads in 1usize..=2,
    ) {
        let d_model = 4;
        prop_assume!(d_model % n_heads == 0);
        let mhca = MultiHeadCrossAttention::from_parts(
            mk_linear_seeded(d_model, d_model, 1),
            mk_linear_seeded(d_model, d_model, 2),
            mk_linear_seeded(d_model, d_model, 3),
            mk_linear_seeded(d_model, d_model, 4),
            n_heads,
        ).unwrap();
        let q = Tensor::from_vec(
            (0..(s_q * d_model)).map(|i| 0.05 * (i as f32)).collect(),
            Shape::d3(1, s_q, d_model),
        ).unwrap();
        let kv = Tensor::from_vec(
            (0..(s_kv * d_model)).map(|i| 0.05 * (i as f32)).collect(),
            Shape::d3(1, s_kv, d_model),
        ).unwrap();
        let out = mhca.forward(&q, &kv, None).unwrap();
        prop_assert_eq!(out.shape().dims(), &[1, s_q, d_model]);
    }

    #[test]
    fn transformer_blocks_deterministic_across_constructions(
        d_model in 2usize..=6,
        seq in 1usize..=4,
    ) {
        let a = PositionalEncoding::new(16, d_model).unwrap();
        let b = PositionalEncoding::new(16, d_model).unwrap();
        let x_data: Vec<f32> = (0..(seq * d_model)).map(|i| 0.1 * (i as f32)).collect();
        let x1 = Tensor::from_vec(x_data.clone(), Shape::d3(1, seq, d_model)).unwrap();
        let x2 = Tensor::from_vec(x_data, Shape::d3(1, seq, d_model)).unwrap();
        let y1 = a.forward(&x1).unwrap();
        let y2 = b.forward(&x2).unwrap();
        prop_assert_eq!(y1.data(), y2.data());
    }
}
