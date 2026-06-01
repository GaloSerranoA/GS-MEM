//! RNN invariants:
//!
//! 1. Determinism — the same weights and input produce the same output
//!    across two independent constructions.
//! 2. Output shape `[S, hidden]` always matches `input.shape[0]` ×
//!    `cell.hidden_size()`.
//! 3. Unrolling a single-step sequence equals one `step()` call.

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    reason = "rnn proptests mirror the crate's LSTM/GRU naming (w_ih/w_hh/b_ih/b_hh)"
)]

use immortal_nn::rnn::{unroll, Gru, Lstm, RnnCell, VanillaRnn};
use immortal_nn::tensor::{Shape, Tensor};
use proptest::prelude::*;

fn make_vanilla(input_size: usize, hidden: usize) -> VanillaRnn {
    let w_ih = Tensor::from_vec(
        (0..(hidden * input_size))
            .map(|i| 0.01 * (i as f32 - 1.0))
            .collect(),
        Shape::d2(hidden, input_size),
    )
    .unwrap();
    let w_hh = Tensor::from_vec(
        (0..(hidden * hidden))
            .map(|i| 0.02 * (i as f32 + 0.5))
            .collect(),
        Shape::d2(hidden, hidden),
    )
    .unwrap();
    let b_ih = Tensor::from_vec(vec![0.1; hidden], Shape::d2(1, hidden)).unwrap();
    let b_hh = Tensor::from_vec(vec![-0.05; hidden], Shape::d2(1, hidden)).unwrap();
    VanillaRnn::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap()
}

fn make_lstm(input_size: usize, hidden: usize) -> Lstm {
    let w_ih = Tensor::from_vec(
        (0..(4 * hidden * input_size))
            .map(|i| 0.01 * (i as f32))
            .collect(),
        Shape::d2(4 * hidden, input_size),
    )
    .unwrap();
    let w_hh = Tensor::from_vec(
        (0..(4 * hidden * hidden))
            .map(|i| 0.005 * (i as f32))
            .collect(),
        Shape::d2(4 * hidden, hidden),
    )
    .unwrap();
    let b_ih = Tensor::from_vec(vec![0.0; 4 * hidden], Shape::d2(1, 4 * hidden)).unwrap();
    let b_hh = Tensor::from_vec(vec![0.0; 4 * hidden], Shape::d2(1, 4 * hidden)).unwrap();
    Lstm::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap()
}

fn make_gru(input_size: usize, hidden: usize) -> Gru {
    let w_ih = Tensor::from_vec(
        (0..(3 * hidden * input_size))
            .map(|i| 0.01 * (i as f32))
            .collect(),
        Shape::d2(3 * hidden, input_size),
    )
    .unwrap();
    let w_hh = Tensor::from_vec(
        (0..(3 * hidden * hidden))
            .map(|i| 0.005 * (i as f32))
            .collect(),
        Shape::d2(3 * hidden, hidden),
    )
    .unwrap();
    let b_ih = Tensor::from_vec(vec![0.1; 3 * hidden], Shape::d2(1, 3 * hidden)).unwrap();
    let b_hh = Tensor::from_vec(vec![-0.05; 3 * hidden], Shape::d2(1, 3 * hidden)).unwrap();
    Gru::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap()
}

proptest! {
    #[test]
    fn rnn_cells_deterministic(
        in_s in 1usize..=4,
        hs in 1usize..=4,
        seq in 1usize..=5,
    ) {
        let a = make_vanilla(in_s, hs);
        let b = make_vanilla(in_s, hs);
        let input_data: Vec<f32> = (0..(seq * in_s)).map(|i| 0.1 * (i as f32)).collect();
        let x = Tensor::from_vec(input_data, Shape::d2(seq, in_s)).unwrap();
        let y1 = unroll(&a, &x).unwrap();
        let y2 = unroll(&b, &x).unwrap();
        prop_assert_eq!(y1.data(), y2.data());
    }

    #[test]
    fn lstm_unroll_shape_matches_spec(
        in_s in 1usize..=4,
        hs in 1usize..=4,
        seq in 1usize..=5,
    ) {
        let cell = make_lstm(in_s, hs);
        let input_data: Vec<f32> = (0..(seq * in_s)).map(|i| 0.05 * (i as f32)).collect();
        let x = Tensor::from_vec(input_data, Shape::d2(seq, in_s)).unwrap();
        let y = unroll(&cell, &x).unwrap();
        prop_assert_eq!(y.shape().dims(), &[seq, hs]);
    }

    #[test]
    fn gru_unroll_shape_matches_spec(
        in_s in 1usize..=4,
        hs in 1usize..=4,
        seq in 1usize..=5,
    ) {
        let cell = make_gru(in_s, hs);
        let input_data: Vec<f32> = (0..(seq * in_s)).map(|i| 0.05 * (i as f32)).collect();
        let x = Tensor::from_vec(input_data, Shape::d2(seq, in_s)).unwrap();
        let y = unroll(&cell, &x).unwrap();
        prop_assert_eq!(y.shape().dims(), &[seq, hs]);
    }

    #[test]
    fn single_step_unroll_equals_step(
        in_s in 1usize..=3,
        hs in 1usize..=3,
    ) {
        // Unrolling a seq=1 input must equal calling step() directly.
        let cell = make_vanilla(in_s, hs);
        let input_data: Vec<f32> = (0..in_s).map(|i| 0.2 * (i as f32 + 1.0)).collect();
        let x_seq = Tensor::from_vec(input_data.clone(), Shape::d2(1, in_s)).unwrap();
        let y_unrolled = unroll(&cell, &x_seq).unwrap();
        let x_step = Tensor::from_vec(input_data, Shape::d2(1, in_s)).unwrap();
        let (y_step, _) = cell.step(&x_step, &cell.zero_state()).unwrap();
        prop_assert_eq!(y_unrolled.data(), y_step.data());
    }
}
