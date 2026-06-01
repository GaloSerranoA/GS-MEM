//! Hand-computed + cross-checked correctness tests for the `rnn` module.
//!
//! Every expected number is computed either from a closed form (vanilla
//! RNN identity-weight step, LSTM identity gate passthrough) or from the
//! same equations the cell implements (cross-validation between stepwise
//! and unrolled forward).

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    dead_code,
    unused_variables,
    reason = "hand-computed rnn tests; w_ih/w_hh/b_ih/b_hh + small loop variables mirror the crate's own naming (see lib.rs)"
)]

use immortal_nn::rnn::{unroll, unroll_with_state, Gru, Lstm, LstmState, RnnCell, VanillaRnn};
use immortal_nn::tensor::{Shape, Tensor};

fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

// ---------------------------------------------------------------------------
// VanillaRnn
// ---------------------------------------------------------------------------

#[test]
fn vanilla_rnn_zero_state_zero_input_returns_zero() {
    // With zero weights + zero input the pre-activation is 0 and tanh(0)=0.
    let w_ih = Tensor::zeros(Shape::d2(3, 2));
    let w_hh = Tensor::zeros(Shape::d2(3, 3));
    let b_ih = Tensor::zeros(Shape::d2(1, 3));
    let b_hh = Tensor::zeros(Shape::d2(1, 3));
    let cell = VanillaRnn::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();
    let x = Tensor::from_vec(vec![0.0, 0.0], Shape::d2(1, 2)).unwrap();
    let (y, _h1) = cell.step(&x, &cell.zero_state()).unwrap();
    assert_eq!(y.data(), &[0.0, 0.0, 0.0]);
}

#[test]
fn vanilla_rnn_single_step_matches_closed_form() {
    // hidden=1, input=1. W_ih=[[2]], W_hh=[[0]], b_ih=[1], b_hh=[0].
    // h_0 = 0, x = 0.5 → h_1 = tanh(2*0.5 + 1 + 0 + 0) = tanh(2.0).
    let w_ih = Tensor::from_vec(vec![2.0], Shape::d2(1, 1)).unwrap();
    let w_hh = Tensor::from_vec(vec![0.0], Shape::d2(1, 1)).unwrap();
    let b_ih = Tensor::from_vec(vec![1.0], Shape::d2(1, 1)).unwrap();
    let b_hh = Tensor::from_vec(vec![0.0], Shape::d2(1, 1)).unwrap();
    let cell = VanillaRnn::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();
    let x = Tensor::from_vec(vec![0.5], Shape::d2(1, 1)).unwrap();
    let (y, _) = cell.step(&x, &cell.zero_state()).unwrap();
    let expect = 2.0_f32.tanh();
    assert!((y.data()[0] - expect).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Lstm
// ---------------------------------------------------------------------------

#[test]
fn lstm_zero_weights_zero_input_returns_zero_state() {
    let hs = 2;
    let in_s = 2;
    let w_ih = Tensor::zeros(Shape::d2(4 * hs, in_s));
    let w_hh = Tensor::zeros(Shape::d2(4 * hs, hs));
    let b_ih = Tensor::zeros(Shape::d2(1, 4 * hs));
    let b_hh = Tensor::zeros(Shape::d2(1, 4 * hs));
    let cell = Lstm::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();
    let x = Tensor::zeros(Shape::d2(1, in_s));
    let (y, (h_new, c_new)) = cell.step(&x, &cell.zero_state()).unwrap();
    // All pre-activations are 0. i=σ(0)=0.5, f=σ(0)=0.5, g=tanh(0)=0, o=σ(0)=0.5.
    // c' = 0.5*0 + 0.5*0 = 0. h' = 0.5 * tanh(0) = 0.
    assert_eq!(y.data(), &[0.0, 0.0]);
    assert_eq!(h_new.data(), &[0.0, 0.0]);
    assert_eq!(c_new.data(), &[0.0, 0.0]);
}

#[test]
fn lstm_identity_update_gate_memorizes_input() {
    // Configure gates so i=1, f=0, g=x, o=1 → c'=x, h'=tanh(x).
    // Achieve this by setting input-gate biases large positive, forget
    // bias large negative, output bias large positive, and g's weights
    // so that the g pre-activation equals the scalar input.
    let hs = 1;
    let in_s = 1;
    // gate order i, f, g, o — each has a [hs] slice of w_ih and b_ih.
    // W_ih rows: i=[0], f=[0], g=[1], o=[0]  → pre-activations x for g.
    let w_ih = Tensor::from_vec(vec![0.0, 0.0, 1.0, 0.0], Shape::d2(4, 1)).unwrap();
    let w_hh = Tensor::zeros(Shape::d2(4, 1));
    // b_ih: i=+10, f=-10, g=0, o=+10
    let b_ih = Tensor::from_vec(vec![10.0, -10.0, 0.0, 10.0], Shape::d2(1, 4)).unwrap();
    let b_hh = Tensor::zeros(Shape::d2(1, 4));
    let cell = Lstm::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();
    // x small enough that tanh(x) ≈ x and g_gate pre-activation ≈ x.
    let x_val = 0.1_f32;
    let x = Tensor::from_vec(vec![x_val], Shape::d2(1, 1)).unwrap();
    let (y, (_h, c)) = cell.step(&x, &cell.zero_state()).unwrap();
    // g_pre = x_val. i≈1, f≈0, o≈1.
    // c' ≈ 0 + 1 * tanh(x_val) = tanh(x_val).
    // h' ≈ 1 * tanh(c') = tanh(tanh(x_val)).
    let g_val = x_val.tanh();
    let h_val = g_val.tanh();
    assert!(
        (c.data()[0] - g_val).abs() < 1e-3,
        "c={} want ≈{g_val}",
        c.data()[0]
    );
    assert!((y.data()[0] - h_val).abs() < 1e-3);
}

// ---------------------------------------------------------------------------
// Gru
// ---------------------------------------------------------------------------

#[test]
fn gru_zero_weights_zero_input_returns_zero_state() {
    let hs = 2;
    let in_s = 2;
    let w_ih = Tensor::zeros(Shape::d2(3 * hs, in_s));
    let w_hh = Tensor::zeros(Shape::d2(3 * hs, hs));
    let b_ih = Tensor::zeros(Shape::d2(1, 3 * hs));
    let b_hh = Tensor::zeros(Shape::d2(1, 3 * hs));
    let cell = Gru::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();
    let x = Tensor::zeros(Shape::d2(1, in_s));
    let (y, h_new) = cell.step(&x, &cell.zero_state()).unwrap();
    // z=σ(0)=0.5, n=tanh(0)=0, h_prev=0 → h' = 0.5*0 + 0.5*0 = 0.
    assert_eq!(y.data(), &[0.0, 0.0]);
    assert_eq!(h_new.data(), &[0.0, 0.0]);
}

#[test]
fn gru_update_gate_saturated_high_preserves_hidden_state() {
    // Push z large positive so z ≈ 1; then h' ≈ z * h ≈ h regardless of n.
    let hs = 1;
    let in_s = 1;
    // gate order r, z, n. Make b_z very large.
    let w_ih = Tensor::zeros(Shape::d2(3, 1));
    let w_hh = Tensor::zeros(Shape::d2(3, 1));
    let b_ih = Tensor::from_vec(vec![0.0, 10.0, 0.0], Shape::d2(1, 3)).unwrap();
    let b_hh = Tensor::zeros(Shape::d2(1, 3));
    let cell = Gru::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();
    let x = Tensor::from_vec(vec![0.5], Shape::d2(1, 1)).unwrap();
    // Start from a nonzero h to distinguish preservation from zeroing.
    let h_prev = Tensor::from_vec(vec![0.7], Shape::d2(1, 1)).unwrap();
    let (y, h_new) = cell.step(&x, &h_prev).unwrap();
    // Exact: z = σ(10). expected h' = (1-z)*tanh(0) + z*0.7 = z*0.7 ≈ 0.7.
    let z = 1.0 / (1.0 + (-10.0_f32).exp());
    let expected = z * 0.7;
    assert!((y.data()[0] - expected).abs() < 1e-3);
    assert_eq!(y.data(), h_new.data());
}

// ---------------------------------------------------------------------------
// unroll
// ---------------------------------------------------------------------------

#[test]
fn unroll_matches_stepwise_on_lstm() {
    // A 5-step, 3-dim input, 4-dim hidden LSTM with small nonzero
    // weights. Check that `unroll` returns the same per-step outputs as
    // running `step` manually in a loop.
    let hs = 4;
    let in_s = 3;
    let w_ih = Tensor::from_vec(
        (0..(4 * hs * in_s)).map(|i| (i as f32) * 0.01).collect(),
        Shape::d2(4 * hs, in_s),
    )
    .unwrap();
    let w_hh = Tensor::from_vec(
        (0..(4 * hs * hs)).map(|i| (i as f32) * 0.005).collect(),
        Shape::d2(4 * hs, hs),
    )
    .unwrap();
    let b_ih = Tensor::from_vec(vec![0.1; 4 * hs], Shape::d2(1, 4 * hs)).unwrap();
    let b_hh = Tensor::from_vec(vec![-0.05; 4 * hs], Shape::d2(1, 4 * hs)).unwrap();
    let cell = Lstm::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();

    let seq_len = 5;
    let input_data: Vec<f32> = (0..(seq_len * in_s)).map(|i| 0.1 * (i as f32)).collect();
    let input = Tensor::from_vec(input_data.clone(), Shape::d2(seq_len, in_s)).unwrap();

    let (unrolled, _final) = unroll_with_state(&cell, &input, cell.zero_state()).unwrap();

    // Reference: step-by-step manual unroll.
    let mut state: LstmState = cell.zero_state();
    let mut expected: Vec<f32> = Vec::with_capacity(seq_len * hs);
    for t in 0..seq_len {
        let xt = Tensor::from_vec(
            input_data[t * in_s..(t + 1) * in_s].to_vec(),
            Shape::d2(1, in_s),
        )
        .unwrap();
        let (y, s) = cell.step(&xt, &state).unwrap();
        expected.extend_from_slice(y.data());
        state = s;
    }
    assert!(
        approx_eq(unrolled.data(), &expected, 1e-6),
        "unroll diverged from stepwise reference"
    );
}

#[test]
fn unroll_rejects_wrong_input_rank() {
    let hs = 2;
    let in_s = 2;
    let w_ih = Tensor::zeros(Shape::d2(3 * hs, in_s));
    let w_hh = Tensor::zeros(Shape::d2(3 * hs, hs));
    let b_ih = Tensor::zeros(Shape::d2(1, 3 * hs));
    let b_hh = Tensor::zeros(Shape::d2(1, 3 * hs));
    let cell = Gru::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();
    // 3-D input is wrong.
    let x = Tensor::zeros(Shape::d3(1, 1, 2));
    assert!(unroll(&cell, &x).is_err());
}

#[test]
fn unroll_rejects_wrong_input_feat_size() {
    let hs = 2;
    let in_s = 2;
    let w_ih = Tensor::zeros(Shape::d2(hs, in_s));
    let w_hh = Tensor::zeros(Shape::d2(hs, hs));
    let b_ih = Tensor::zeros(Shape::d2(1, hs));
    let b_hh = Tensor::zeros(Shape::d2(1, hs));
    let cell = VanillaRnn::from_tensors(w_ih, w_hh, b_ih, b_hh).unwrap();
    // Declared input=2 but passed input=3.
    let x = Tensor::zeros(Shape::d2(4, 3));
    assert!(unroll(&cell, &x).is_err());
}
