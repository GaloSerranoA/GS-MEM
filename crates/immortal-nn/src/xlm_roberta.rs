//! XLM-R encoder and sequence-classifier heads.
//!
//! This module is the sovereign `.sovereign` runtime path used by BGE-M3 and
//! bge-reranker-v2-m3. It keeps the `immortal-nn` contract: no external runtime
//! dependencies, no Python, no Candle, and no hidden model-side mutation.

use std::collections::BTreeMap;
use std::path::Path;

use crate::blocks::{Activation, Embedding, LayerNorm, Linear, MultiHeadAttention};
use crate::format::SovereignModel;
use crate::ops;
use crate::tensor::{Shape, Tensor};
use crate::{NnError, NnResult, ARCH_XLM_ROBERTA_ENCODER, ARCH_XLM_ROBERTA_SEQUENCE_CLASSIFIER};

type TokenBatch = Vec<Vec<u32>>;

/// XLM-R encoder configured for BGE-M3 dense embedding inference.
pub struct XlmRobertaEncoder {
    embeddings: XlmRobertaEmbeddings,
    layers: Vec<XlmRobertaLayer>,
    sparse_linear: Option<Linear>,
    colbert_linear: Option<Linear>,
    config: XlmRobertaConfig,
}

/// XLM-R sequence classifier configured for BGE reranking.
pub struct XlmRobertaSequenceClassifier {
    encoder: XlmRobertaEncoder,
    dense: Linear,
    out_proj: Linear,
}

#[derive(Debug, Clone, Copy)]
struct XlmRobertaConfig {
    num_layers: usize,
    num_heads: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    max_seq_len: usize,
    pad_token_id: u32,
    activation: Activation,
}

struct XlmRobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    norm: LayerNorm,
}

struct XlmRobertaLayer {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    ffn_up: Linear,
    ffn_down: Linear,
    norm2: LayerNorm,
    activation: Activation,
}

impl XlmRobertaEncoder {
    /// Load an XLM-R encoder from a `.sovereign` file.
    pub fn load(path: &Path) -> NnResult<Self> {
        let model = SovereignModel::load(path)?;
        Self::from_model_with_arch(&model, &[ARCH_XLM_ROBERTA_ENCODER])
    }

    /// Encode token IDs into hidden states with shape `[1, seq_len, hidden]`.
    pub fn encode(&self, input_ids: &[u32], attention_mask: Option<&[u32]>) -> NnResult<Tensor> {
        self.config.validate_input_len(input_ids.len())?;
        let mask = build_attention_mask(input_ids, attention_mask, self.config.pad_token_id)?;
        let mut hidden = self
            .embeddings
            .forward(input_ids, self.config.pad_token_id)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, Some(&mask))?;
        }
        Ok(hidden)
    }

    /// Encode a batch of token ID sequences into hidden states with shape
    /// `[batch, max_seq_len, hidden]`. Shorter sequences are right-padded with
    /// the model pad token and zero attention.
    pub fn encode_batch(
        &self,
        input_ids: &[Vec<u32>],
        attention_masks: &[Vec<u32>],
    ) -> NnResult<Tensor> {
        let (padded_ids, padded_masks) = self.prepare_padded_batch(input_ids, attention_masks)?;
        let mut hidden = self
            .embeddings
            .forward_batch(&padded_ids, self.config.pad_token_id)?;
        let mask = Tensor::from_vec(
            padded_masks
                .iter()
                .flat_map(|row| row.iter().map(|&value| if value == 0 { 0.0 } else { 1.0 }))
                .collect(),
            Shape::d2(padded_masks.len(), padded_masks[0].len()),
        )?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, Some(&mask))?;
        }
        Ok(hidden)
    }

    /// Return the L2-normalized CLS embedding for a tokenized sequence.
    pub fn dense_embedding(
        &self,
        input_ids: &[u32],
        attention_mask: Option<&[u32]>,
    ) -> NnResult<Vec<f32>> {
        let encoded = self.encode(input_ids, attention_mask)?;
        let pooled = ops::cls_pool(&encoded)?;
        l2_normalize(pooled.data())
    }

    /// Return L2-normalized CLS embeddings for a padded batch.
    pub fn dense_embeddings_batch(
        &self,
        input_ids: &[Vec<u32>],
        attention_masks: &[Vec<u32>],
    ) -> NnResult<Vec<Vec<f32>>> {
        let encoded = self.encode_batch(input_ids, attention_masks)?;
        let pooled = ops::cls_pool(&encoded)?;
        l2_normalize_rows(pooled.data(), self.config.hidden_dim)
    }

    /// Whether this model includes the optional BGE-M3 sparse and ColBERT
    /// projection heads.
    #[must_use]
    pub fn has_bge_m3_projection_heads(&self) -> bool {
        self.sparse_linear.is_some() && self.colbert_linear.is_some()
    }

    /// Return learned sparse token weights as sorted `(token_id, weight)` pairs.
    pub fn sparse_embedding(
        &self,
        input_ids: &[u32],
        attention_mask: Option<&[u32]>,
    ) -> NnResult<Vec<(u32, f32)>> {
        let sparse_linear = self
            .sparse_linear
            .as_ref()
            .ok_or_else(|| NnError::Format("BGE-M3 sparse head is not loaded".into()))?;
        let encoded = self.encode(input_ids, attention_mask)?;
        let weights = sparse_linear.forward(&encoded)?;
        let mask = build_keep_mask(input_ids, attention_mask, self.config.pad_token_id)?;
        let mut by_token = BTreeMap::<u32, f32>::new();
        for (index, (&token_id, keep)) in input_ids.iter().zip(mask).enumerate() {
            if !keep || is_special_token(token_id, self.config.pad_token_id) {
                continue;
            }
            let raw_weight = weights.data()[index].max(0.0);
            if raw_weight == 0.0 {
                continue;
            }
            by_token
                .entry(token_id)
                .and_modify(|current| *current = current.max(raw_weight))
                .or_insert(raw_weight);
        }
        Ok(by_token.into_iter().collect())
    }

    /// Return L2-normalized ColBERT token vectors for non-special input tokens.
    pub fn colbert_embeddings(
        &self,
        input_ids: &[u32],
        attention_mask: Option<&[u32]>,
    ) -> NnResult<Vec<Vec<f32>>> {
        let colbert_linear = self
            .colbert_linear
            .as_ref()
            .ok_or_else(|| NnError::Format("BGE-M3 ColBERT head is not loaded".into()))?;
        let encoded = self.encode(input_ids, attention_mask)?;
        let projected = colbert_linear.forward(&encoded)?;
        let mask = build_keep_mask(input_ids, attention_mask, self.config.pad_token_id)?;
        let mut vectors = Vec::new();
        for (index, (&token_id, keep)) in input_ids.iter().zip(mask).enumerate() {
            if !keep || is_special_token(token_id, self.config.pad_token_id) {
                continue;
            }
            let start = index * self.config.hidden_dim;
            let end = start + self.config.hidden_dim;
            vectors.push(l2_normalize(&projected.data()[start..end])?);
        }
        if vectors.is_empty() {
            return Err(NnError::Format(
                "BGE-M3 ColBERT head produced no non-special token vectors".into(),
            ));
        }
        Ok(vectors)
    }

    /// Hidden size advertised by the loaded model metadata.
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Maximum token sequence length accepted by this model.
    #[must_use]
    pub const fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    /// Plan 65-NN Phase B (2026-05-04) — pub-exposed constructor used
    /// by `immortal-gmem::SovereignEmbedder` to dispatch by architecture
    /// metadata without re-reading the .sovereign file. Accepts both
    /// `xlm_roberta_encoder` and `xlm_roberta_sequence_classifier`
    /// per the existing private contract.
    pub fn from_model_with_arch(model: &SovereignModel, allowed_arches: &[&str]) -> NnResult<Self> {
        let arch = model.architecture();
        if !allowed_arches.contains(&arch) {
            return Err(NnError::UnknownArchitecture(arch.to_string()));
        }

        let config = XlmRobertaConfig::from_model(model)?;
        let embeddings = XlmRobertaEmbeddings::load(model)?;
        let sparse_linear = optional_linear(model, "bge_m3.sparse_linear")?;
        let colbert_linear = optional_linear(model, "bge_m3.colbert_linear")?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            layers.push(XlmRobertaLayer::load(model, index, config)?);
        }
        Ok(Self {
            embeddings,
            layers,
            sparse_linear,
            colbert_linear,
            config,
        })
    }

    fn prepare_padded_batch(
        &self,
        input_ids: &[Vec<u32>],
        attention_masks: &[Vec<u32>],
    ) -> NnResult<(TokenBatch, TokenBatch)> {
        if input_ids.is_empty() {
            return Err(NnError::ShapeMismatch {
                expected: "non-empty batch".into(),
                got: "empty batch".into(),
            });
        }
        if input_ids.len() != attention_masks.len() {
            return Err(NnError::ShapeMismatch {
                expected: format!("{} attention masks", input_ids.len()),
                got: format!("{} attention masks", attention_masks.len()),
            });
        }
        let max_len = input_ids.iter().map(Vec::len).max().unwrap_or(0);
        self.config.validate_input_len(max_len)?;
        let mut padded_ids = Vec::with_capacity(input_ids.len());
        let mut padded_masks = Vec::with_capacity(attention_masks.len());
        for (ids, mask) in input_ids.iter().zip(attention_masks) {
            self.config.validate_input_len(ids.len())?;
            if ids.len() != mask.len() {
                return Err(NnError::ShapeMismatch {
                    expected: format!("attention_mask length {}", ids.len()),
                    got: format!("attention_mask length {}", mask.len()),
                });
            }
            let mut row_ids = ids.clone();
            row_ids.resize(max_len, self.config.pad_token_id);
            let mut row_mask = mask.clone();
            row_mask.resize(max_len, 0);
            padded_ids.push(row_ids);
            padded_masks.push(row_mask);
        }
        Ok((padded_ids, padded_masks))
    }
}

impl XlmRobertaSequenceClassifier {
    /// Load an XLM-R sequence classifier from a `.sovereign` file.
    pub fn load(path: &Path) -> NnResult<Self> {
        let model = SovereignModel::load(path)?;
        let encoder = XlmRobertaEncoder::from_model_with_arch(
            &model,
            &[ARCH_XLM_ROBERTA_SEQUENCE_CLASSIFIER],
        )?;
        let dense = Linear::load(&model, "classifier.dense")?;
        let out_proj = Linear::load(&model, "classifier.out_proj")?;
        Ok(Self {
            encoder,
            dense,
            out_proj,
        })
    }

    /// Score a tokenized query/document pair. For BGE reranker models this is
    /// the first classification logit.
    pub fn score_tokens(&self, input_ids: &[u32], attention_mask: Option<&[u32]>) -> NnResult<f32> {
        let encoded = self.encoder.encode(input_ids, attention_mask)?;
        let pooled = ops::cls_pool(&encoded)?;
        let dense = self.dense.forward(&pooled)?;
        let activated = ops::tanh_act(&dense);
        let logits = self.out_proj.forward(&activated)?;
        logits
            .data()
            .first()
            .copied()
            .ok_or_else(|| NnError::Format("classifier returned no logits".into()))
    }

    /// Score a batch of tokenized query/document pairs.
    pub fn score_tokens_batch(
        &self,
        input_ids: &[Vec<u32>],
        attention_masks: &[Vec<u32>],
    ) -> NnResult<Vec<f32>> {
        let encoded = self.encoder.encode_batch(input_ids, attention_masks)?;
        let pooled = ops::cls_pool(&encoded)?;
        let dense = self.dense.forward(&pooled)?;
        let activated = ops::tanh_act(&dense);
        let logits = self.out_proj.forward(&activated)?;
        Ok(logits.data().to_vec())
    }

    /// Hidden size advertised by the loaded encoder metadata.
    #[must_use]
    pub const fn hidden_dim(&self) -> usize {
        self.encoder.hidden_dim()
    }

    /// Maximum sequence length the underlying encoder accepts. Callers
    /// (e.g. `SovereignBgeReranker`) use this to configure tokenizer
    /// truncation so long documents do not exceed the encoder's
    /// position-embedding capacity.
    #[must_use]
    pub const fn max_seq_len(&self) -> usize {
        self.encoder.max_seq_len()
    }
}

impl XlmRobertaConfig {
    fn from_model(model: &SovereignModel) -> NnResult<Self> {
        let hidden_dim = metadata_usize(model, "hidden_dim", 768)?;
        let num_heads = metadata_usize(model, "num_heads", 12)?;
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(NnError::DivisibilityError {
                what: "hidden_dim % num_heads".into(),
                numerator: hidden_dim,
                denominator: num_heads,
            });
        }
        Ok(Self {
            num_layers: metadata_usize(model, "num_layers", 12)?,
            num_heads,
            hidden_dim,
            intermediate_dim: metadata_usize(model, "intermediate_dim", hidden_dim * 4)?,
            max_seq_len: metadata_usize(model, "max_seq_len", 512)?,
            pad_token_id: metadata_usize(model, "pad_token_id", 1)? as u32,
            activation: model
                .metadata("activation")
                .map(Activation::from_str)
                .transpose()?
                .unwrap_or(Activation::GeluTanh),
        })
    }

    fn validate_input_len(&self, len: usize) -> NnResult<()> {
        if len == 0 {
            return Err(NnError::ShapeMismatch {
                expected: "non-empty token sequence".into(),
                got: "empty token sequence".into(),
            });
        }
        if len > self.max_seq_len {
            return Err(NnError::ShapeMismatch {
                expected: format!("sequence length <= {}", self.max_seq_len),
                got: format!("sequence length = {len}"),
            });
        }
        Ok(())
    }
}

impl XlmRobertaEmbeddings {
    fn load(model: &SovereignModel) -> NnResult<Self> {
        Ok(Self {
            word_embeddings: Embedding::load(model, "embedding.word_embeddings")?,
            position_embeddings: Embedding::load(model, "embedding.position_embeddings")?,
            token_type_embeddings: Embedding::load(model, "embedding.token_type_embeddings")?,
            norm: LayerNorm::load(model, "embedding.norm")?,
        })
    }

    fn forward(&self, input_ids: &[u32], pad_token_id: u32) -> NnResult<Tensor> {
        let positions = position_ids_from_input(input_ids, pad_token_id);
        let token_types = vec![0_u32; input_ids.len()];
        let word = self.word_embeddings.forward(input_ids)?;
        let position = self.position_embeddings.forward(&positions)?;
        let token_type = self.token_type_embeddings.forward(&token_types)?;
        let word_position = ops::add(&word, &position)?;
        let summed = ops::add(&word_position, &token_type)?;
        let batched = summed.reshape(Shape::d3(
            1,
            input_ids.len(),
            self.word_embeddings.embed_dim(),
        ))?;
        self.norm.forward(&batched)
    }

    fn forward_batch(&self, input_ids: &[Vec<u32>], pad_token_id: u32) -> NnResult<Tensor> {
        let batch = input_ids.len();
        let seq_len = input_ids[0].len();
        let hidden = self.word_embeddings.embed_dim();
        let mut data = Vec::with_capacity(batch * seq_len * hidden);
        for row in input_ids {
            if row.len() != seq_len {
                return Err(NnError::ShapeMismatch {
                    expected: format!("sequence length {seq_len}"),
                    got: format!("sequence length {}", row.len()),
                });
            }
            let encoded = self.forward(row, pad_token_id)?;
            data.extend_from_slice(encoded.data());
        }
        Tensor::from_vec(data, Shape::d3(batch, seq_len, hidden))
    }
}

impl XlmRobertaLayer {
    fn load(model: &SovereignModel, index: usize, config: XlmRobertaConfig) -> NnResult<Self> {
        let prefix = format!("encoder.layer.{index}");
        let ffn_up = Linear::load(model, &format!("{prefix}.ffn_up"))?;
        let ffn_down = Linear::load(model, &format!("{prefix}.ffn_down"))?;
        if ffn_up.out_features() != config.intermediate_dim {
            return Err(NnError::ShapeMismatch {
                expected: format!("{prefix}.ffn_up out_features = {}", config.intermediate_dim),
                got: format!("out_features = {}", ffn_up.out_features()),
            });
        }
        if ffn_down.in_features() != config.intermediate_dim {
            return Err(NnError::ShapeMismatch {
                expected: format!(
                    "{prefix}.ffn_down in_features = {}",
                    config.intermediate_dim
                ),
                got: format!("in_features = {}", ffn_down.in_features()),
            });
        }

        Ok(Self {
            attention: MultiHeadAttention::load(
                model,
                &format!("{prefix}.attention"),
                config.num_heads,
            )?,
            norm1: LayerNorm::load(model, &format!("{prefix}.norm1"))?,
            ffn_up,
            ffn_down,
            norm2: LayerNorm::load(model, &format!("{prefix}.norm2"))?,
            activation: config.activation,
        })
    }

    fn forward(&self, input: &Tensor, mask: Option<&Tensor>) -> NnResult<Tensor> {
        let attention_output = self.attention.forward(input, mask)?;
        let attention_residual = ops::add(input, &attention_output)?;
        let attention_normed = self.norm1.forward(&attention_residual)?;

        let up = self.ffn_up.forward(&attention_normed)?;
        let activated = self.activation.apply(&up);
        let down = self.ffn_down.forward(&activated)?;
        let ffn_residual = ops::add(&attention_normed, &down)?;
        self.norm2.forward(&ffn_residual)
    }
}

fn metadata_usize(model: &SovereignModel, key: &str, default: usize) -> NnResult<usize> {
    model.metadata(key).map_or(Ok(default), |value| {
        value.parse::<usize>().map_err(|err| {
            NnError::Format(format!(
                "metadata {key:?} must be usize, got {value:?}: {err}"
            ))
        })
    })
}

fn optional_linear(model: &SovereignModel, prefix: &str) -> NnResult<Option<Linear>> {
    match model.tensor(&format!("{prefix}.weight")) {
        Ok(_) => Linear::load(model, prefix).map(Some),
        Err(_) => Ok(None),
    }
}

fn build_attention_mask(
    input_ids: &[u32],
    explicit: Option<&[u32]>,
    pad_token_id: u32,
) -> NnResult<Tensor> {
    let mask: Vec<f32> = match explicit {
        Some(values) => {
            if values.len() != input_ids.len() {
                return Err(NnError::ShapeMismatch {
                    expected: format!("attention_mask length {}", input_ids.len()),
                    got: format!("attention_mask length {}", values.len()),
                });
            }
            values
                .iter()
                .map(|&value| if value == 0 { 0.0 } else { 1.0 })
                .collect()
        }
        None => input_ids
            .iter()
            .map(|&id| if id == pad_token_id { 0.0 } else { 1.0 })
            .collect(),
    };
    Tensor::from_vec(mask, Shape::d2(1, input_ids.len()))
}

fn build_keep_mask(
    input_ids: &[u32],
    explicit: Option<&[u32]>,
    pad_token_id: u32,
) -> NnResult<Vec<bool>> {
    match explicit {
        Some(values) => {
            if values.len() != input_ids.len() {
                return Err(NnError::ShapeMismatch {
                    expected: format!("attention_mask length {}", input_ids.len()),
                    got: format!("attention_mask length {}", values.len()),
                });
            }
            Ok(values.iter().map(|&value| value != 0).collect())
        }
        None => Ok(input_ids.iter().map(|&id| id != pad_token_id).collect()),
    }
}

fn is_special_token(token_id: u32, pad_token_id: u32) -> bool {
    token_id == pad_token_id || token_id == 0 || token_id == 2
}

fn position_ids_from_input(input_ids: &[u32], pad_token_id: u32) -> Vec<u32> {
    if pad_token_id == 0 {
        // Plan 65-NN Phase B (2026-05-04) — BERT-family models
        // (`pad_token_id == 0`) use simple sequential position IDs
        // `[0, 1, 2, ..., seq_len-1]` regardless of attention mask.
        // The pad-token contribution at the embedding stage is later
        // suppressed by the attention mask. Verified vs HF
        // `BertEmbeddings.position_ids` on MiniLM-L6-v2.
        return (0..input_ids.len() as u32).collect();
    }
    // RoBERTa-family path (pad_token_id ≥ 1, typically 1): mirror HF's
    // `create_position_ids_from_input_ids(input_ids, padding_idx)` —
    // pad tokens get position = padding_idx; non-pad tokens get
    // cumulative-count + padding_idx (so first non-pad is padding_idx + 1).
    let mut current = 0_u32;
    let mut positions = Vec::with_capacity(input_ids.len());
    for &id in input_ids {
        if id == pad_token_id {
            positions.push(pad_token_id);
        } else {
            current = current.saturating_add(1);
            positions.push(current.saturating_add(pad_token_id));
        }
    }
    positions
}

fn l2_normalize(values: &[f32]) -> NnResult<Vec<f32>> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return Err(NnError::Format(
            "cannot normalize zero or non-finite embedding".into(),
        ));
    }
    Ok(values.iter().map(|value| value / norm).collect())
}

fn l2_normalize_rows(values: &[f32], row_len: usize) -> NnResult<Vec<Vec<f32>>> {
    if row_len == 0 || !values.len().is_multiple_of(row_len) {
        return Err(NnError::ShapeMismatch {
            expected: format!("flat rows divisible by {row_len}"),
            got: format!("{} values", values.len()),
        });
    }
    values
        .chunks_exact(row_len)
        .map(l2_normalize)
        .collect::<NnResult<Vec<_>>>()
}
