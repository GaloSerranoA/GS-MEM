//! Task heads (Section 5).
//!
//! Each head composes [`crate::blocks`] into a complete inference pipeline for
//! one NLU task. Loaded from a single `.sovereign` file. Single `predict()`
//! entry point. Returns task-native output.

use crate::blocks::{Activation, BiLstm, Crf, Embedding, LayerNorm, Linear, TransformerBlock};
use crate::format::SovereignModel;
use crate::ops;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult};
use crate::{
    ARCH_BIAFFINE_PARSER, ARCH_SEQUENCE_CLASSIFIER_TRANSFORMER, ARCH_TOKEN_CLASSIFIER_BILSTM_CRF,
    ARCH_TOKEN_CLASSIFIER_TRANSFORMER_CRF,
};
use std::path::Path;

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// A single tagged token from a token classifier.
#[derive(Debug, Clone)]
pub struct TaggedToken {
    /// Position in the input sequence.
    pub token_idx: usize,
    /// Predicted label string.
    pub label: String,
    /// Predicted label ID.
    pub label_id: u32,
    /// Confidence score (from CRF path score, normalized).
    pub score: f32,
}

/// Prediction from a sequence classifier.
#[derive(Debug, Clone)]
pub struct ClassPrediction {
    /// Best label string.
    pub label: String,
    /// Best label ID.
    pub label_id: usize,
    /// Score of the best label (softmax probability).
    pub score: f32,
    /// All scores (softmax probabilities), indexed by label ID.
    pub all_scores: Vec<f32>,
}

/// Output from the biaffine dependency parser.
#[derive(Debug, Clone)]
pub struct BiaffineScores {
    /// Arc score matrix `[S, S]` — `scores[dep][head]`.
    pub arc_scores: Vec<Vec<f32>>,
    /// Relation score matrix `[S, S, R]` — `rel_scores[dep][head][rel]`.
    pub rel_scores: Vec<Vec<Vec<f32>>>,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of relation labels.
    pub num_relations: usize,
}

// ---------------------------------------------------------------------------
// Encoder enum (internal to TokenClassifier)
// ---------------------------------------------------------------------------

/// Encoder backend for token classifiers.
enum Encoder {
    BiLstm(Box<BiLstm>),
    Transformer(Vec<TransformerBlock>),
}

impl Encoder {
    /// `[B, S, E] -> [B, S, hidden]`
    fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> NnResult<Tensor> {
        match self {
            Self::BiLstm(bilstm) => bilstm.forward(input),
            Self::Transformer(layers) => {
                let mut x = input.clone();
                for layer in layers {
                    // BERT/DistilBERT token classifiers are post-norm.
                    x = layer.forward_post_norm(&x, mask)?;
                }
                Ok(x)
            }
        }
    }
}

/// Per-token argmax decode for token classifiers WITHOUT a CRF (plain
/// BERT/DistilBERT token heads). `logits`: `[1, S, num_tags]`.
fn argmax_tags(logits: &Tensor) -> NnResult<(Vec<u32>, f32)> {
    let dims = logits.shape().dims();
    let (s, t) = (dims[1], dims[2]);
    let data = logits.data();
    let mut tags = Vec::with_capacity(s);
    let mut total = 0.0f32;
    for i in 0..s {
        let row = &data[i * t..(i + 1) * t];
        let (best, val) =
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| {
                    if v > bv {
                        (j, v)
                    } else {
                        (bi, bv)
                    }
                });
        tags.push(best as u32);
        total += val;
    }
    Ok((tags, total))
}

// ---------------------------------------------------------------------------
// TokenClassifier
// ---------------------------------------------------------------------------

/// Token classifier for NER, POS, SRL (BIO tagging).
///
/// Pipeline: `embedding -> encoder -> classifier -> crf.decode -> label_map`.
pub struct TokenClassifier {
    embedding: Embedding,
    pos_embedding: Option<Embedding>,
    embed_norm: Option<LayerNorm>,
    encoder: Encoder,
    classifier: Linear,
    crf: Option<Crf>,
    label_map: Vec<String>,
}

impl TokenClassifier {
    /// Load from a `.sovereign` file.
    ///
    /// Metadata must contain `architecture` (one of `token_classifier_bilstm_crf`
    /// or `token_classifier_transformer_crf`) and `labels` (JSON array of label strings).
    pub fn load(path: &Path) -> NnResult<Self> {
        let model = SovereignModel::load(path)?;
        let arch = model.architecture();

        let embedding = Embedding::load(&model, "embedding")?;
        // Optional positional embeddings + embedding LayerNorm (BERT/DistilBERT).
        let pos_embedding = Embedding::load(&model, "pos_embedding").ok();
        let embed_norm = LayerNorm::load(&model, "embed_norm").ok();

        let encoder = match arch {
            ARCH_TOKEN_CLASSIFIER_BILSTM_CRF => {
                Encoder::BiLstm(Box::new(BiLstm::load(&model, "encoder")?))
            }
            ARCH_TOKEN_CLASSIFIER_TRANSFORMER_CRF => {
                let num_layers: usize = model
                    .metadata("num_layers")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(6);
                let num_heads: usize = model
                    .metadata("num_heads")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8);
                let activation = model
                    .metadata("activation")
                    .map(Activation::from_str)
                    .transpose()?
                    .unwrap_or(Activation::GeluTanh);
                let mut layers = Vec::with_capacity(num_layers);
                for i in 0..num_layers {
                    layers.push(TransformerBlock::load(
                        &model,
                        &format!("encoder.layer.{i}"),
                        num_heads,
                        activation,
                    )?);
                }
                Encoder::Transformer(layers)
            }
            _ => return Err(NnError::UnknownArchitecture(arch.to_string())),
        };

        let classifier = Linear::load(&model, "classifier")?;
        // Optional CRF: present for *-crf models; absent for plain BERT token
        // classifiers (per-token argmax is used then).
        let crf = Crf::load(&model, "crf").ok();
        let label_map = parse_label_map(&model)?;

        Ok(Self {
            embedding,
            pos_embedding,
            embed_norm,
            encoder,
            classifier,
            crf,
            label_map,
        })
    }

    /// Predict tags for a single sequence of token IDs.
    ///
    /// Returns `Vec<TaggedToken>` with label, label ID, and score per token.
    pub fn predict(&self, token_ids: &[u32]) -> NnResult<Vec<TaggedToken>> {
        let seq_len = token_ids.len();

        // Embed: [S, E]
        let embedded = self.embedding.forward(token_ids)?;
        let embed_dim = self.embedding.embed_dim();
        // Positional embeddings + embedding LayerNorm (BERT/DistilBERT) when present.
        let embedded = match &self.pos_embedding {
            Some(pe) => {
                let positions: Vec<u32> = (0..seq_len as u32).collect();
                ops::add(&embedded, &pe.forward(&positions)?)?
            }
            None => embedded,
        };
        let embedded = match &self.embed_norm {
            Some(ln) => ln.forward(&embedded)?,
            None => embedded,
        };
        // Add batch dim: [1, S, E]
        let batched = embedded.reshape(Shape::d3(1, seq_len, embed_dim))?;

        // Encode: [1, S, hidden]
        let encoded = self.encoder.forward(&batched, None)?;

        // Classify: [1, S, num_tags]
        let logits = self.classifier.forward(&encoded)?;

        // Decode: CRF Viterbi if present, else per-token argmax.
        let (tags, score): (Vec<u32>, f32) = match &self.crf {
            Some(crf) => crf.decode(&logits)?.into_iter().next().unwrap_or_default(),
            None => argmax_tags(&logits)?,
        };

        // Normalize score (simple sigmoid-style normalization for confidence)
        let norm_score = 1.0 / (1.0 + (-score / seq_len as f32).exp());

        let mut output = Vec::with_capacity(seq_len);
        for (i, &tag_id) in tags.iter().enumerate() {
            let label = self
                .label_map
                .get(tag_id as usize)
                .cloned()
                .unwrap_or_else(|| format!("TAG_{tag_id}"));
            output.push(TaggedToken {
                token_idx: i,
                label,
                label_id: tag_id,
                score: norm_score,
            });
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// SequenceClassifier
// ---------------------------------------------------------------------------

/// Sequence classifier for sentiment, WSD, intent.
///
/// Pipeline: `embedding -> transformer stack -> pool -> linear -> softmax -> argmax`.
pub struct SequenceClassifier {
    embedding: Embedding,
    pos_embedding: Option<Embedding>,
    type_embedding: Option<Embedding>,
    embed_norm: Option<LayerNorm>,
    /// Positional index offset (RoBERTa/XLM-R use `pad_idx + 1` = 2; BERT 0).
    pos_offset: usize,
    layers: Vec<TransformerBlock>,
    pre_classifier: Option<Linear>,
    pooler: Option<Linear>,
    pool_norm: Option<LayerNorm>,
    classifier: Option<Linear>,
    label_map: Vec<String>,
}

impl SequenceClassifier {
    /// Load from a `.sovereign` file.
    pub fn load(path: &Path) -> NnResult<Self> {
        let model = SovereignModel::load(path)?;
        let arch = model.architecture();
        if arch != ARCH_SEQUENCE_CLASSIFIER_TRANSFORMER {
            return Err(NnError::UnknownArchitecture(arch.to_string()));
        }

        let embedding = Embedding::load(&model, "embedding")?;
        // Optional positional embeddings + embedding LayerNorm (BERT/DistilBERT).
        // Required for order-sensitive classification; the older position-free
        // encoder exports lack them and behave exactly as before.
        let pos_embedding = Embedding::load(&model, "pos_embedding").ok();
        // Optional token-type embedding (RoBERTa/XLM-R: a single type-0 vector).
        let type_embedding = Embedding::load(&model, "type_embedding").ok();
        let embed_norm = LayerNorm::load(&model, "embed_norm").ok();
        // Positional offset: RoBERTa/XLM-R start position ids at pad_idx+1 (=2);
        // BERT/DistilBERT start at 0. Written by the converter per model_type.
        let pos_offset: usize = model
            .metadata("pos_offset")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let num_layers: usize = model
            .metadata("num_layers")
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        let num_heads: usize = model
            .metadata("num_heads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let activation = model
            .metadata("activation")
            .map(Activation::from_str)
            .transpose()?
            .unwrap_or(Activation::GeluTanh);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerBlock::load(
                &model,
                &format!("encoder.layer.{i}"),
                num_heads,
                activation,
            )?);
        }

        // Optional pooling layer norm
        let pool_norm = LayerNorm::load(&model, "pool_norm").ok();

        // Optional intermediate dense head (e.g. DistilBERT's `pre_classifier`,
        // applied with ReLU before the final classifier). Backward-compatible:
        // models written without it simply skip this step.
        let pre_classifier = Linear::load(&model, "pre_classifier").ok();
        // BERT-style pooler dense (Linear + tanh); the alternative head to
        // DistilBERT's pre_classifier (Linear + ReLU).
        let pooler = Linear::load(&model, "pooler").ok();

        // Optional: a plain encoder (feature-extraction only, no trained head)
        // loads without a classifier; `features()` still works for training a
        // downstream head on frozen sovereign features.
        let classifier = Linear::load(&model, "classifier").ok();
        let label_map = parse_label_map(&model)?;

        Ok(Self {
            embedding,
            pos_embedding,
            type_embedding,
            embed_norm,
            pos_offset,
            layers,
            pre_classifier,
            pooler,
            pool_norm,
            classifier,
            label_map,
        })
    }

    /// Encoder forward: embeddings -> transformer layers -> `[1, S, E]`.
    fn encode(&self, token_ids: &[u32]) -> NnResult<Tensor> {
        let seq_len = token_ids.len();
        let embedded = self.embedding.forward(token_ids)?;
        let embed_dim = self.embedding.embed_dim();
        // Positional embeddings (positions pos_offset..pos_offset+S) + optional
        // token-type + embedding LayerNorm. RoBERTa/XLM-R offset position ids by
        // pad_idx+1; BERT/DistilBERT use 0. Without positions the transformer is
        // permutation-invariant and order-sensitive tasks collapse.
        let embedded = match &self.pos_embedding {
            Some(pe) => {
                let positions: Vec<u32> =
                    (self.pos_offset as u32..(self.pos_offset + seq_len) as u32).collect();
                ops::add(&embedded, &pe.forward(&positions)?)?
            }
            None => embedded,
        };
        let embedded = match &self.type_embedding {
            Some(te) => ops::add(&embedded, &te.forward(&vec![0u32; seq_len])?)?,
            None => embedded,
        };
        let embedded = match &self.embed_norm {
            Some(ln) => ln.forward(&embedded)?,
            None => embedded,
        };
        // [1, S, E]
        let mut x = embedded.reshape(Shape::d3(1, seq_len, embed_dim))?;
        // Transformer layers. BERT/DistilBERT/RoBERTa/XLM-R sequence classifiers
        // are POST-norm; the pre-norm `forward` mis-scales these weights (the
        // documented MiniLM-correctness bug on TransformerBlock).
        for layer in &self.layers {
            x = layer.forward_post_norm(&x, None)?;
        }
        Ok(x)
    }

    /// Shared forward up to the raw classifier logits `[1, num_labels]`.
    fn forward_logits(&self, token_ids: &[u32]) -> NnResult<Tensor> {
        let x = self.encode(token_ids)?;

        // Pool: CLS/<s> token (position 0) -> [1, E]
        let pooled = ops::cls_pool(&x)?;

        // Optional intermediate dense head: DistilBERT uses `pre_classifier`
        // (Linear + ReLU); BERT/RoBERTa use `pooler` (Linear + tanh). Apply
        // whichever the model carries (mutually exclusive in practice).
        let pooled = if let Some(pc) = &self.pre_classifier {
            ops::relu(&pc.forward(&pooled)?)
        } else if let Some(pl) = &self.pooler {
            ops::tanh_act(&pl.forward(&pooled)?)
        } else {
            pooled
        };

        // Optional norm
        let normed = match &self.pool_norm {
            Some(ln) => ln.forward(&pooled)?,
            None => pooled,
        };

        // Classify: [1, num_labels]
        let classifier = self.classifier.as_ref().ok_or_else(|| {
            NnError::Format("classifier head absent (feature-extraction-only model)".into())
        })?;
        classifier.forward(&normed)
    }

    /// Mean-pooled encoder features `[E]` (a sentence representation), for
    /// training a downstream head on frozen sovereign features. Works without a
    /// classifier head (feature-extraction mode).
    pub fn features(&self, token_ids: &[u32]) -> NnResult<Vec<f32>> {
        let x = self.encode(token_ids)?;
        Ok(ops::mean_pool_seq(&x)?.data().to_vec())
    }

    /// Predict class for a single sequence of token IDs.
    pub fn predict(&self, token_ids: &[u32]) -> NnResult<ClassPrediction> {
        let logits = self.forward_logits(token_ids)?;

        // Softmax
        let probs = ops::softmax(&logits, 1)?;
        let scores = probs.data();

        // Argmax
        let (best_id, &best_score) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let label = self
            .label_map
            .get(best_id)
            .cloned()
            .unwrap_or_else(|| format!("CLASS_{best_id}"));

        Ok(ClassPrediction {
            label,
            label_id: best_id,
            score: best_score,
            all_scores: scores.to_vec(),
        })
    }

    /// Raw classifier logits (pre-softmax), indexed by label id. For
    /// multiple-choice models (num_labels = 1), call once per choice and argmax
    /// the scalar across choices.
    pub fn logits(&self, token_ids: &[u32]) -> NnResult<Vec<f32>> {
        Ok(self.forward_logits(token_ids)?.data().to_vec())
    }
}

// ---------------------------------------------------------------------------
// BiaffineParser
// ---------------------------------------------------------------------------

/// Biaffine dependency parser.
///
/// Pipeline: `embedding -> biLstm -> head_mlp / dep_mlp -> biaffine score -> rel_classifier`.
///
/// Returns the raw score matrices. Eisner decoding stays in `immortal-nlu`.
pub struct BiaffineParser {
    embedding: Embedding,
    encoder: BiLstm,
    head_mlp: Linear,
    dep_mlp: Linear,
    /// Biaffine weight: `[head_dim, dep_dim]` for arc scoring.
    arc_weight: Tensor,
    arc_bias: Option<Tensor>,
    /// Relation classifier head MLP.
    rel_head_mlp: Linear,
    /// Relation classifier dep MLP.
    rel_dep_mlp: Linear,
    /// Relation bilinear: `[num_relations, head_dim, dep_dim]`.
    rel_weight: Tensor,
    num_relations: usize,
}

impl BiaffineParser {
    /// Load from a `.sovereign` file.
    pub fn load(path: &Path) -> NnResult<Self> {
        let model = SovereignModel::load(path)?;
        let arch = model.architecture();
        if arch != ARCH_BIAFFINE_PARSER {
            return Err(NnError::UnknownArchitecture(arch.to_string()));
        }

        let embedding = Embedding::load(&model, "embedding")?;
        let encoder = BiLstm::load(&model, "encoder")?;
        let head_mlp = Linear::load(&model, "head_mlp")?;
        let dep_mlp = Linear::load(&model, "dep_mlp")?;
        let arc_weight = model.tensor("arc.weight")?.clone();
        let arc_bias = model.tensor("arc.bias").ok().cloned();
        let rel_head_mlp = Linear::load(&model, "rel_head_mlp")?;
        let rel_dep_mlp = Linear::load(&model, "rel_dep_mlp")?;
        let rel_weight = model.tensor("rel.weight")?.clone();

        let num_relations: usize = model
            .metadata("num_relations")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                // Infer from rel_weight shape [R, head_dim, dep_dim]
                rel_weight.shape().dims()[0]
            });

        Ok(Self {
            embedding,
            encoder,
            head_mlp,
            dep_mlp,
            arc_weight,
            arc_bias,
            rel_head_mlp,
            rel_dep_mlp,
            rel_weight,
            num_relations,
        })
    }

    /// Predict arc and relation scores for a single sequence.
    ///
    /// Returns `BiaffineScores` with `arc_scores[dep][head]` and
    /// `rel_scores[dep][head][rel]`. Eisner decoding is done by `immortal-nlu`.
    pub fn predict(&self, token_ids: &[u32]) -> NnResult<BiaffineScores> {
        let seq_len = token_ids.len();

        // Embed: [S, E]
        let embedded = self.embedding.forward(token_ids)?;
        let embed_dim = self.embedding.embed_dim();
        // [1, S, E]
        let batched = embedded.reshape(Shape::d3(1, seq_len, embed_dim))?;

        // Encode: [1, S, 2*hidden]
        let encoded = self.encoder.forward(&batched)?;
        let enc_dim = encoded.shape().dims()[2];

        // Flatten to [S, enc_dim] for MLPs
        let flat = encoded.reshape(Shape::d2(seq_len, enc_dim))?;

        // Head/dep representations: [S, mlp_dim]
        let h_repr = ops::relu(&self.head_mlp.forward(&flat)?);
        let d_repr = ops::relu(&self.dep_mlp.forward(&flat)?);
        let _head_dim = h_repr.shape().dims()[1];
        let _dep_dim = d_repr.shape().dims()[1];

        // Arc scores: biaffine(dep, head) = dep @ W @ head^T + bias
        // W: [head_dim, dep_dim], so score[i][j] = d_repr[i] @ W @ h_repr[j]^T
        let w_ht = biaffine_scores_2d(&d_repr, &self.arc_weight, &h_repr, self.arc_bias.as_ref())?;

        // Relation scores
        let rel_h = ops::relu(&self.rel_head_mlp.forward(&flat)?);
        let rel_d = ops::relu(&self.rel_dep_mlp.forward(&flat)?);
        let rel_hd = rel_h.shape().dims()[1];
        let rel_dd = rel_d.shape().dims()[1];

        // rel_weight: [R, rel_hd, rel_dd]
        // For each relation r: score[i][j] = rel_d[i] @ rel_weight[r] @ rel_h[j]^T
        let mut rel_scores = vec![vec![vec![0.0_f32; self.num_relations]; seq_len]; seq_len];
        let rw = self.rel_weight.data();

        for r in 0..self.num_relations {
            let w_off = r * rel_hd * rel_dd;
            // d_repr[i] @ W_r: [S, rel_hd]
            let mut dw = vec![0.0_f32; seq_len * rel_hd];
            let dd = rel_d.data();
            for i in 0..seq_len {
                for j in 0..rel_hd {
                    let mut sum = 0.0_f32;
                    for k in 0..rel_dd {
                        sum += dd[i * rel_dd + k] * rw[w_off + k * rel_hd + j];
                    }
                    dw[i * rel_hd + j] = sum;
                }
            }
            // (dW) @ h^T: [S, S]
            let hd = rel_h.data();
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0_f32;
                    for k in 0..rel_hd {
                        sum += dw[i * rel_hd + k] * hd[j * rel_hd + k];
                    }
                    rel_scores[i][j][r] = sum;
                }
            }
        }

        // Convert arc scores to nested Vec
        let arc_data = w_ht.data();
        let mut arc_scores = vec![vec![0.0_f32; seq_len]; seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                arc_scores[i][j] = arc_data[i * seq_len + j];
            }
        }

        Ok(BiaffineScores {
            arc_scores,
            rel_scores,
            seq_len,
            num_relations: self.num_relations,
        })
    }
}

/// Compute biaffine scores: `a @ W @ b^T + bias`.
/// `a: [N, d_a]`, `W: [d_b, d_a]`, `b: [N, d_b]` -> `[N, N]`.
fn biaffine_scores_2d(
    a: &Tensor,
    w: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
) -> NnResult<Tensor> {
    let n = a.shape().dims()[0];
    let da = a.shape().dims()[1];
    let db = b.shape().dims()[1];

    // a @ W^T: [N, d_b]
    let _aw = ops::matmul(a, w)?; // a:[N,da] @ W:[da,db] but W is [db,da]
                                  // Actually W: [head_dim, dep_dim] means we want a @ W^T for matching
                                  // Let's do it correctly: score = a @ W @ b^T where W:[da, db]
                                  // But plan says W:[head_dim, dep_dim]. Let's just matmul a with W.
                                  // a:[N, da], W:[da or db, ?] — we need to handle shape correctly.
                                  // Simplify: do it element-wise

    let ad = a.data();
    let wd = w.data();
    let bd = b.data();
    let (wd0, wd1) = (w.shape().dims()[0], w.shape().dims()[1]);

    // a @ W: [N, wd1] where a:[N, da=wd0]
    let mut aw_data = vec![0.0_f32; n * wd1];
    for i in 0..n {
        for j in 0..wd1 {
            let mut sum = 0.0_f32;
            for k in 0..wd0 {
                sum += ad[i * da + k] * wd[k * wd1 + j];
            }
            aw_data[i * wd1 + j] = sum;
        }
    }

    // (aW) @ b^T: [N, N]
    let mut out_data = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for k in 0..wd1 {
                sum += aw_data[i * wd1 + k] * bd[j * db + k];
            }
            out_data[i * n + j] = sum;
        }
    }

    // Add bias if present
    if let Some(bias_t) = bias {
        let bias_d = bias_t.data();
        // Bias is typically [N] or [1, N] — add to each row
        if bias_d.len() == n {
            for i in 0..n {
                for j in 0..n {
                    out_data[i * n + j] += bias_d[j];
                }
            }
        }
    }

    Tensor::from_vec(out_data, Shape::d2(n, n))
}

// ---------------------------------------------------------------------------
// Label map parser
// ---------------------------------------------------------------------------

/// Parse `labels` from metadata. Expects a JSON array: `["O","B-PER","I-PER",...]`.
pub(crate) fn parse_label_map(model: &SovereignModel) -> NnResult<Vec<String>> {
    let raw = model
        .metadata("labels")
        .ok_or_else(|| NnError::Format("missing 'labels' metadata".into()))?;
    parse_json_string_array(raw)
}

/// Parse a JSON array of strings: `["a","b","c"]`.
fn parse_json_string_array(s: &str) -> NnResult<Vec<String>> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return Err(NnError::Format("labels must be a JSON array".into()));
    }
    let inner = &s[1..s.len() - 1];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut labels = Vec::new();
    let mut rest = inner.trim();
    while !rest.is_empty() {
        // Skip leading quote
        if !rest.starts_with('"') {
            return Err(NnError::Format(format!(
                "expected '\"' in label array, got {:?}",
                rest.chars().next()
            )));
        }
        // Find closing quote (simple — no escapes in label names)
        let end = rest[1..]
            .find('"')
            .ok_or_else(|| NnError::Format("unterminated label string".into()))?;
        labels.push(rest[1..=end].to_string());
        rest = rest[2 + end..].trim_start();
        if rest.starts_with(',') {
            rest = rest[1..].trim_start();
        }
    }
    Ok(labels)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::SovereignWriter;

    #[test]
    fn parse_json_string_array_basic() {
        let labels = parse_json_string_array(r#"["O","B-PER","I-PER","B-LOC"]"#).unwrap();
        assert_eq!(labels, vec!["O", "B-PER", "I-PER", "B-LOC"]);
    }

    #[test]
    fn parse_json_string_array_empty() {
        assert!(parse_json_string_array("[]").unwrap().is_empty());
    }

    #[test]
    fn token_classifier_bilstm_crf_smoke() {
        // Build a tiny model in memory and write to sovereign, then load
        let dir = std::env::temp_dir().join("immortal_nn_test_token_cls");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("token_cls.sovereign");

        let vocab = 10;
        let embed_dim = 4;
        let hidden = 3;
        let num_tags = 3;

        let mut w = SovereignWriter::new();

        // Embedding [vocab, embed_dim]
        w.add_tensor(
            "embedding.weight",
            Tensor::from_vec(vec![0.01; vocab * embed_dim], Shape::d2(vocab, embed_dim)).unwrap(),
        );

        // BiLSTM forward cell
        let hs = hidden;
        let is = embed_dim;
        w.add_tensor(
            "encoder.forward.w_ih",
            Tensor::from_vec(vec![0.1; 4 * hs * is], Shape::d2(4 * hs, is)).unwrap(),
        );
        w.add_tensor(
            "encoder.forward.w_hh",
            Tensor::from_vec(vec![0.1; 4 * hs * hs], Shape::d2(4 * hs, hs)).unwrap(),
        );
        w.add_tensor(
            "encoder.forward.b_ih",
            Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap(),
        );
        w.add_tensor(
            "encoder.forward.b_hh",
            Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap(),
        );
        // BiLSTM backward cell
        w.add_tensor(
            "encoder.backward.w_ih",
            Tensor::from_vec(vec![0.1; 4 * hs * is], Shape::d2(4 * hs, is)).unwrap(),
        );
        w.add_tensor(
            "encoder.backward.w_hh",
            Tensor::from_vec(vec![0.1; 4 * hs * hs], Shape::d2(4 * hs, hs)).unwrap(),
        );
        w.add_tensor(
            "encoder.backward.b_ih",
            Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap(),
        );
        w.add_tensor(
            "encoder.backward.b_hh",
            Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap(),
        );

        // Classifier: [num_tags, 2*hidden]
        let cls_in = 2 * hidden;
        w.add_tensor(
            "classifier.weight",
            Tensor::from_vec(vec![0.1; num_tags * cls_in], Shape::d2(num_tags, cls_in)).unwrap(),
        );
        w.add_tensor(
            "classifier.bias",
            Tensor::from_vec(vec![0.0; num_tags], Shape::d2(1, num_tags)).unwrap(),
        );

        // CRF
        w.add_tensor(
            "crf.transitions",
            Tensor::from_vec(
                vec![0.0; num_tags * num_tags],
                Shape::d2(num_tags, num_tags),
            )
            .unwrap(),
        );
        w.add_tensor(
            "crf.start_transitions",
            Tensor::from_vec(vec![0.0; num_tags], Shape::d2(1, num_tags)).unwrap(),
        );
        w.add_tensor(
            "crf.end_transitions",
            Tensor::from_vec(vec![0.0; num_tags], Shape::d2(1, num_tags)).unwrap(),
        );

        // Metadata
        w.set_metadata("architecture", ARCH_TOKEN_CLASSIFIER_BILSTM_CRF);
        w.set_metadata("labels", r#"["O","B-PER","I-PER"]"#);
        w.write_to(&path).unwrap();

        // Load and predict
        let cls = TokenClassifier::load(&path).unwrap();
        let result = cls.predict(&[1, 2, 3, 4]).unwrap();
        assert_eq!(result.len(), 4);
        for t in &result {
            assert!(["O", "B-PER", "I-PER"].contains(&t.label.as_str()));
            assert!(t.score > 0.0 && t.score <= 1.0);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sequence_classifier_smoke() {
        let dir = std::env::temp_dir().join("immortal_nn_test_seq_cls");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("seq_cls.sovereign");

        let vocab = 10;
        let e = 4;
        let h = 2;
        let ffn = 8;
        let num_labels = 3;

        let mut w = SovereignWriter::new();

        // Embedding
        w.add_tensor(
            "embedding.weight",
            Tensor::from_vec(vec![0.01; vocab * e], Shape::d2(vocab, e)).unwrap(),
        );

        // 1 transformer layer
        let pfx = "encoder.layer.0";
        for (suffix, out_f, in_f) in [
            ("attention.q_proj", e, e),
            ("attention.k_proj", e, e),
            ("attention.v_proj", e, e),
            ("attention.o_proj", e, e),
            ("ffn_up", ffn, e),
            ("ffn_down", e, ffn),
        ] {
            w.add_tensor(
                format!("{pfx}.{suffix}.weight"),
                Tensor::from_vec(vec![0.01; out_f * in_f], Shape::d2(out_f, in_f)).unwrap(),
            );
        }
        for suffix in ["norm1", "norm2"] {
            w.add_tensor(
                format!("{pfx}.{suffix}.weight"),
                Tensor::from_vec(vec![1.0; e], Shape::d2(1, e)).unwrap(),
            );
            w.add_tensor(
                format!("{pfx}.{suffix}.bias"),
                Tensor::from_vec(vec![0.0; e], Shape::d2(1, e)).unwrap(),
            );
        }

        // Classifier
        w.add_tensor(
            "classifier.weight",
            Tensor::from_vec(vec![0.01; num_labels * e], Shape::d2(num_labels, e)).unwrap(),
        );

        w.set_metadata("architecture", ARCH_SEQUENCE_CLASSIFIER_TRANSFORMER);
        w.set_metadata("num_layers", "1");
        w.set_metadata("num_heads", h.to_string());
        w.set_metadata("activation", "gelu");
        w.set_metadata("labels", r#"["negative","neutral","positive"]"#);
        w.write_to(&path).unwrap();

        let cls = SequenceClassifier::load(&path).unwrap();
        let result = cls.predict(&[1, 2, 3]).unwrap();
        assert_eq!(result.all_scores.len(), num_labels);
        let sum: f32 = result.all_scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5); // softmax sums to 1
        assert!(["negative", "neutral", "positive"].contains(&result.label.as_str()));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn biaffine_parser_smoke() {
        let dir = std::env::temp_dir().join("immortal_nn_test_biaffine");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("biaffine.sovereign");

        let vocab = 10;
        let embed_dim = 4;
        let hidden = 3;
        let mlp_dim = 2;
        let num_rels = 3;

        let mut w = SovereignWriter::new();

        // Embedding
        w.add_tensor(
            "embedding.weight",
            Tensor::from_vec(vec![0.01; vocab * embed_dim], Shape::d2(vocab, embed_dim)).unwrap(),
        );

        // BiLSTM
        let hs = hidden;
        let is = embed_dim;
        for dir_name in &["forward", "backward"] {
            w.add_tensor(
                format!("encoder.{dir_name}.w_ih"),
                Tensor::from_vec(vec![0.1; 4 * hs * is], Shape::d2(4 * hs, is)).unwrap(),
            );
            w.add_tensor(
                format!("encoder.{dir_name}.w_hh"),
                Tensor::from_vec(vec![0.1; 4 * hs * hs], Shape::d2(4 * hs, hs)).unwrap(),
            );
            w.add_tensor(
                format!("encoder.{dir_name}.b_ih"),
                Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap(),
            );
            w.add_tensor(
                format!("encoder.{dir_name}.b_hh"),
                Tensor::from_vec(vec![0.0; 4 * hs], Shape::d2(1, 4 * hs)).unwrap(),
            );
        }

        let enc_out = 2 * hidden;
        // MLPs
        w.add_tensor(
            "head_mlp.weight",
            Tensor::from_vec(vec![0.1; mlp_dim * enc_out], Shape::d2(mlp_dim, enc_out)).unwrap(),
        );
        w.add_tensor(
            "dep_mlp.weight",
            Tensor::from_vec(vec![0.1; mlp_dim * enc_out], Shape::d2(mlp_dim, enc_out)).unwrap(),
        );

        // Arc biaffine: [mlp_dim, mlp_dim]
        w.add_tensor(
            "arc.weight",
            Tensor::from_vec(vec![0.1; mlp_dim * mlp_dim], Shape::d2(mlp_dim, mlp_dim)).unwrap(),
        );

        // Relation MLPs and weight
        w.add_tensor(
            "rel_head_mlp.weight",
            Tensor::from_vec(vec![0.1; mlp_dim * enc_out], Shape::d2(mlp_dim, enc_out)).unwrap(),
        );
        w.add_tensor(
            "rel_dep_mlp.weight",
            Tensor::from_vec(vec![0.1; mlp_dim * enc_out], Shape::d2(mlp_dim, enc_out)).unwrap(),
        );
        // rel.weight: [R, mlp_dim, mlp_dim]
        w.add_tensor(
            "rel.weight",
            Tensor::from_vec(
                vec![0.1; num_rels * mlp_dim * mlp_dim],
                Shape::d3(num_rels, mlp_dim, mlp_dim),
            )
            .unwrap(),
        );

        w.set_metadata("architecture", ARCH_BIAFFINE_PARSER);
        w.set_metadata("num_relations", num_rels.to_string());
        w.set_metadata("labels", r#"["root","nsubj","dobj"]"#);
        w.write_to(&path).unwrap();

        let parser = BiaffineParser::load(&path).unwrap();
        let result = parser.predict(&[1, 2, 3]).unwrap();
        assert_eq!(result.seq_len, 3);
        assert_eq!(result.arc_scores.len(), 3);
        assert_eq!(result.arc_scores[0].len(), 3);
        assert_eq!(result.rel_scores.len(), 3);
        assert_eq!(result.rel_scores[0][0].len(), num_rels);

        let _ = std::fs::remove_file(&path);
    }
}
