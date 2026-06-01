//! XLM-R `.sovereign` contracts for the Plan 44 BGE runtime path.

use std::path::{Path, PathBuf};

use immortal_nn::{
    Shape, SovereignWriter, Tensor, XlmRobertaEncoder, XlmRobertaSequenceClassifier,
    ARCH_XLM_ROBERTA_ENCODER, ARCH_XLM_ROBERTA_SEQUENCE_CLASSIFIER,
};

fn temp_model_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "immortal-nn-{name}-{}-{}.sovereign",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    path
}

fn tensor(data: &[f32], shape: Shape) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape).expect("test tensor shape is valid")
}

fn add_linear(writer: &mut SovereignWriter, prefix: &str, out_features: usize, in_features: usize) {
    let mut weight = Vec::with_capacity(out_features * in_features);
    for row in 0..out_features {
        for col in 0..in_features {
            let value = if row % in_features == col { 0.05 } else { 0.01 };
            weight.push(value);
        }
    }
    writer.add_tensor(
        format!("{prefix}.weight"),
        tensor(&weight, Shape::d2(out_features, in_features)),
    );
    writer.add_tensor(
        format!("{prefix}.bias"),
        tensor(&vec![0.0; out_features], Shape::d2(1, out_features)),
    );
}

fn add_layer_norm(writer: &mut SovereignWriter, prefix: &str, hidden: usize) {
    writer.add_tensor(
        format!("{prefix}.weight"),
        tensor(&vec![1.0; hidden], Shape::d2(1, hidden)),
    );
    writer.add_tensor(
        format!("{prefix}.bias"),
        tensor(&vec![0.0; hidden], Shape::d2(1, hidden)),
    );
}

fn add_tiny_xlm_roberta(path: &Path, arch: &str) {
    let mut writer = SovereignWriter::new();
    writer.set_metadata("architecture", arch);
    writer.set_metadata("num_layers", "1");
    writer.set_metadata("num_heads", "2");
    writer.set_metadata("hidden_dim", "4");
    writer.set_metadata("intermediate_dim", "6");
    writer.set_metadata("max_seq_len", "8");
    writer.set_metadata("pad_token_id", "1");
    writer.set_metadata("layer_norm_eps", "0.00001");
    writer.set_metadata("activation", "gelu");

    writer.add_tensor(
        "embedding.word_embeddings.weight",
        tensor(
            &[
                0.10, 0.00, 0.00, 0.00, // <s>
                0.00, 0.00, 0.00, 0.00, // pad
                0.00, 0.10, 0.00, 0.00, // </s>
                0.00, 0.00, 0.10, 0.00, // alpha
                0.00, 0.00, 0.00, 0.10, // beta
            ],
            Shape::d2(5, 4),
        ),
    );
    writer.add_tensor(
        "embedding.position_embeddings.weight",
        tensor(&[0.01; 8 * 4], Shape::d2(8, 4)),
    );
    writer.add_tensor(
        "embedding.token_type_embeddings.weight",
        tensor(&[0.0, 0.0, 0.0, 0.0], Shape::d2(1, 4)),
    );
    add_layer_norm(&mut writer, "embedding.norm", 4);

    let layer = "encoder.layer.0";
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        add_linear(
            &mut writer,
            &format!("{layer}.attention.{projection}"),
            4,
            4,
        );
    }
    add_layer_norm(&mut writer, &format!("{layer}.norm1"), 4);
    add_linear(&mut writer, &format!("{layer}.ffn_up"), 6, 4);
    add_linear(&mut writer, &format!("{layer}.ffn_down"), 4, 6);
    add_layer_norm(&mut writer, &format!("{layer}.norm2"), 4);

    if arch == ARCH_XLM_ROBERTA_ENCODER {
        writer.add_tensor(
            "bge_m3.sparse_linear.weight",
            tensor(&[0.5, 0.0, 0.25, 0.25], Shape::d2(1, 4)),
        );
        writer.add_tensor("bge_m3.sparse_linear.bias", tensor(&[1.0], Shape::d2(1, 1)));
        writer.add_tensor(
            "bge_m3.colbert_linear.weight",
            tensor(
                &[
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0,
                ],
                Shape::d2(4, 4),
            ),
        );
        writer.add_tensor(
            "bge_m3.colbert_linear.bias",
            tensor(&[0.0, 0.0, 0.0, 0.0], Shape::d2(1, 4)),
        );
    }

    if arch == ARCH_XLM_ROBERTA_SEQUENCE_CLASSIFIER {
        add_linear(&mut writer, "classifier.dense", 4, 4);
        writer.add_tensor(
            "classifier.out_proj.weight",
            tensor(&[0.25, -0.10, 0.15, 0.05], Shape::d2(1, 4)),
        );
        writer.add_tensor("classifier.out_proj.bias", tensor(&[0.0], Shape::d2(1, 1)));
    }

    writer.write_to(path).expect("write tiny .sovereign model");
}

#[test]
fn xlm_roberta_encoder_returns_finite_normalized_cls_embedding() {
    let path = temp_model_path("encoder");
    add_tiny_xlm_roberta(&path, ARCH_XLM_ROBERTA_ENCODER);

    let encoder = XlmRobertaEncoder::load(&path).expect("load tiny xlm-r encoder");
    let hidden = encoder.encode(&[0, 3, 4, 2], None).expect("encode tokens");
    assert_eq!(hidden.shape().dims(), [1, 4, 4]);
    assert!(hidden.data().iter().all(|value| value.is_finite()));

    let dense = encoder
        .dense_embedding(&[0, 3, 4, 2], None)
        .expect("dense embedding");
    assert_eq!(dense.len(), 4);
    assert!(dense.iter().all(|value| value.is_finite()));
    let norm = dense.iter().map(|value| value * value).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1.0e-4,
        "dense embedding must be L2 normalized, got norm={norm}"
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn xlm_roberta_sequence_classifier_returns_finite_single_logit() {
    let path = temp_model_path("classifier");
    add_tiny_xlm_roberta(&path, ARCH_XLM_ROBERTA_SEQUENCE_CLASSIFIER);

    let classifier =
        XlmRobertaSequenceClassifier::load(&path).expect("load tiny xlm-r sequence classifier");
    let score = classifier
        .score_tokens(&[0, 3, 4, 2], None)
        .expect("score pair tokens");
    assert!(score.is_finite());

    let _ = std::fs::remove_file(path);
}

#[test]
fn xlm_roberta_bge_m3_heads_return_sparse_and_colbert_vectors() {
    let path = temp_model_path("encoder-heads");
    add_tiny_xlm_roberta(&path, ARCH_XLM_ROBERTA_ENCODER);

    let encoder = XlmRobertaEncoder::load(&path).expect("load tiny xlm-r encoder");
    assert!(encoder.has_bge_m3_projection_heads());

    let input_ids = [0, 3, 4, 2, 1];
    let attention_mask = [1, 1, 1, 1, 0];
    let sparse = encoder
        .sparse_embedding(&input_ids, Some(&attention_mask))
        .expect("sparse embedding");
    assert_eq!(
        sparse
            .iter()
            .map(|(token_id, _)| *token_id)
            .collect::<Vec<_>>(),
        vec![3, 4]
    );
    assert!(sparse
        .iter()
        .all(|(_, weight)| weight.is_finite() && *weight >= 0.0));

    let colbert = encoder
        .colbert_embeddings(&input_ids, Some(&attention_mask))
        .expect("colbert embeddings");
    assert_eq!(colbert.len(), 2);
    for token_vector in &colbert {
        assert_eq!(token_vector.len(), 4);
        let norm = token_vector
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() <= 1e-4, "colbert vector norm={norm}");
    }

    let _ = std::fs::remove_file(path);
}

#[test]
fn xlm_roberta_batch_outputs_match_single_sequence_outputs() {
    let encoder_path = temp_model_path("encoder-batch");
    add_tiny_xlm_roberta(&encoder_path, ARCH_XLM_ROBERTA_ENCODER);
    let encoder = XlmRobertaEncoder::load(&encoder_path).expect("load tiny xlm-r encoder");

    let first = vec![0, 3, 4, 2];
    let second = vec![0, 3, 2];
    let first_mask = vec![1, 1, 1, 1];
    let second_mask = vec![1, 1, 1];

    let single_first = encoder
        .dense_embedding(&first, Some(&first_mask))
        .expect("first single embedding");
    let single_second = encoder
        .dense_embedding(&second, Some(&second_mask))
        .expect("second single embedding");
    let batched = encoder
        .dense_embeddings_batch(
            &[first.clone(), second.clone()],
            &[first_mask.clone(), second_mask.clone()],
        )
        .expect("batched embeddings");

    assert_eq!(batched.len(), 2);
    assert_close_vec(&single_first, &batched[0]);
    assert_close_vec(&single_second, &batched[1]);

    let classifier_path = temp_model_path("classifier-batch");
    add_tiny_xlm_roberta(&classifier_path, ARCH_XLM_ROBERTA_SEQUENCE_CLASSIFIER);
    let classifier =
        XlmRobertaSequenceClassifier::load(&classifier_path).expect("load tiny classifier");
    let single_first = classifier
        .score_tokens(&first, Some(&first_mask))
        .expect("first single score");
    let single_second = classifier
        .score_tokens(&second, Some(&second_mask))
        .expect("second single score");
    let scores = classifier
        .score_tokens_batch(&[first, second], &[first_mask, second_mask])
        .expect("batch scores");

    assert_eq!(scores.len(), 2);
    assert!((single_first - scores[0]).abs() <= 1e-5);
    assert!((single_second - scores[1]).abs() <= 1e-5);

    let _ = std::fs::remove_file(encoder_path);
    let _ = std::fs::remove_file(classifier_path);
}

fn assert_close_vec(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    for (left, right) in expected.iter().zip(actual) {
        assert!(
            (*left - *right).abs() <= 1e-5,
            "expected {left}, got {right}"
        );
    }
}
