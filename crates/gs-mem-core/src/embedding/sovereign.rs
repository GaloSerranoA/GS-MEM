//! Sovereign embedder backed by the workspace `immortal-nn` crate.
//!
//! Adapted from `gmem/src/embedding/sovereign.rs` on 2026-04-18. Two
//! NANTAR-specific renames prevent collision with the external `gmem`
//! personal MCP tool:
//!
//! 1. `GMEM_MODEL_DIR` env var → `IMMORTAL_GMEM_MODEL_DIR`
//! 2. `%LOCALAPPDATA%/gmem/models/` fallback → `%LOCALAPPDATA%/immortal-gmem/models/`

use std::path::PathBuf;
use std::sync::Arc;

use immortal_nn::blocks::{Activation, Embedding, LayerNorm, TransformerBlock};
use immortal_nn::ops;
use immortal_nn::xlm_roberta::XlmRobertaEncoder;
use immortal_nn::{
    Shape, SovereignModel, Tensor, ARCH_SEQUENCE_CLASSIFIER_TRANSFORMER, ARCH_XLM_ROBERTA_ENCODER,
};
use immortal_tokenizer::SovereignTokenizer;
use once_cell::sync::OnceCell;

use crate::embedding::EmbeddingProvider;
use crate::error::{GmemError, Result};

const DEFAULT_DIM: usize = 384;

struct LoadedModel {
    tokenizer: SovereignTokenizer,
    arch: LoadedArch,
    dim: usize,
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("dim", &self.dim)
            .field(
                "arch",
                &match &self.arch {
                    LoadedArch::XlmRoberta(_) => "xlm_roberta_encoder",
                    LoadedArch::LegacySequenceClassifier { .. } => "legacy_sequence_classifier",
                },
            )
            .finish()
    }
}

/// Plan 65-NN Phase B (2026-05-04) — sovereign embedder dispatches over
/// architecture metadata. The `xlm_roberta_encoder` variant uses the
/// production-validated `XlmRobertaEncoder` (BERT-correct: word +
/// position + token-type embeddings + embedding-LayerNorm + post-norm
/// transformer layers). The `legacy_sequence_classifier_transformer`
/// variant is the historical pre-Plan-65-NN path; it produces
/// numerically-incorrect embeddings on BERT-family models (the
/// SOVEREIGN-EMBEDDER-MINILM-CORRECTNESS bug). Retained for
/// backwards-compat with any operator artifact that still declares the
/// old `sequence_classifier_transformer` architecture.
// XlmRoberta is the dominant runtime variant; LegacySequenceClassifier is
// kept only for backwards-compat with pre-Plan-65 artifacts. Boxing the
// hot variant for the cold one would invert the optimization tradeoff.
#[allow(clippy::large_enum_variant)]
enum LoadedArch {
    /// New (correct) path: full XLM-R / BERT encoder.
    XlmRoberta(XlmRobertaEncoder),
    /// Legacy (broken-on-BERT) path: pre-norm transformer with
    /// only word embeddings.
    LegacySequenceClassifier {
        embedding: Embedding,
        layers: Vec<TransformerBlock>,
        final_norm: Option<LayerNorm>,
    },
}

/// Sovereign-nn-backed embedder. Loads a `.sovereign` model file lazily.
#[derive(Debug)]
pub struct SovereignEmbedder {
    model_path: PathBuf,
    tokenizer_path: PathBuf,
    dim: usize,
    loaded: OnceCell<Arc<LoadedModel>>,
}

impl Clone for SovereignEmbedder {
    fn clone(&self) -> Self {
        let loaded = OnceCell::new();
        if let Some(model) = self.loaded.get() {
            let _ = loaded.set(Arc::clone(model));
        }

        Self {
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
            dim: self.dim,
            loaded,
        }
    }
}

impl SovereignEmbedder {
    /// Construct from explicit paths. Model/tokenizer load lazily on first use.
    pub fn new(model_path: PathBuf, tokenizer_path: PathBuf) -> Self {
        Self {
            model_path,
            tokenizer_path,
            dim: DEFAULT_DIM,
            loaded: OnceCell::new(),
        }
    }

    /// Override the expected output dimension. Useful for tiny test fixtures.
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Construct using the default model-file search path.
    ///
    /// Resolution order (first hit wins):
    /// 1. `IMMORTAL_GMEM_MODEL_DIR` environment variable
    /// 2. `<exe_dir>/models/` — binary shipped alongside a `models/` folder
    /// 3. `<exe_dir>/../../models/sovereign/` — cargo dev build
    ///    (`target/<profile>/<bin>.exe` → repo-root `models/sovereign/`)
    /// 4. `<cwd>/models/sovereign/` — running from the repo root
    /// 5. `%LOCALAPPDATA%/immortal-gmem/models/` (Windows) /
    ///    `$XDG_DATA_HOME/immortal-gmem/models/` fallback
    ///
    /// The first path that contains `minilm-l6-v2.sovereign` wins.
    pub fn default_paths() -> Result<Self> {
        let candidates = Self::candidate_model_dirs();
        let chosen = candidates
            .iter()
            .find(|dir| dir.join("minilm-l6-v2.sovereign").exists())
            .cloned()
            .unwrap_or_else(|| {
                candidates
                    .last()
                    .cloned()
                    .unwrap_or_else(|| PathBuf::from("models"))
            });
        Ok(Self::new(
            chosen.join("minilm-l6-v2.sovereign"),
            chosen.join("minilm-l6-v2.sovereign-tokenizer"),
        ))
    }

    fn candidate_model_dirs() -> Vec<PathBuf> {
        let mut out: Vec<PathBuf> = Vec::new();
        if let Some(env_dir) = std::env::var_os("IMMORTAL_GMEM_MODEL_DIR") {
            out.push(PathBuf::from(env_dir));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(exe_dir) = exe.parent() {
                out.push(exe_dir.join("models"));
                if let Some(repo_root) = exe_dir.parent().and_then(std::path::Path::parent) {
                    out.push(repo_root.join("models").join("sovereign"));
                }
            }
        }
        if let Ok(cwd) = std::env::current_dir() {
            out.push(cwd.join("models").join("sovereign"));
        }
        if let Some(local) = dirs::data_local_dir() {
            out.push(local.join("immortal-gmem").join("models"));
        }
        out
    }

    /// Eagerly load the tokenizer + sovereign model.
    pub fn load(&self) -> Result<()> {
        let _ = self.loaded_model()?;
        Ok(())
    }

    fn loaded_model(&self) -> Result<Arc<LoadedModel>> {
        let loaded = self.loaded.get_or_try_init(|| self.build_loaded_model())?;
        Ok(Arc::clone(loaded))
    }

    fn build_loaded_model(&self) -> Result<Arc<LoadedModel>> {
        if !self.model_path.exists() {
            return Err(GmemError::Embedding(format!(
                "sovereign model not found at {}; run `sovereign-convert` once to generate it.",
                self.model_path.display()
            )));
        }
        if !self.tokenizer_path.exists() {
            return Err(GmemError::Embedding(format!(
                "tokenizer not found at {}",
                self.tokenizer_path.display()
            )));
        }

        let tokenizer = SovereignTokenizer::load(&self.tokenizer_path)
            .map_err(|err| GmemError::Embedding(format!("tokenizer load: {err}")))?;

        let model = SovereignModel::load(&self.model_path).map_err(|err| {
            GmemError::Embedding(format!(
                "failed to load sovereign model at {}: {err}",
                self.model_path.display()
            ))
        })?;
        model
            .validate_checksums()
            .map_err(|err| GmemError::Embedding(format!("sovereign checksum validation: {err}")))?;

        let architecture = model.metadata("architecture").ok_or_else(|| {
            GmemError::Embedding("sovereign model missing architecture metadata".into())
        })?;

        let arch_key: &str = architecture;
        let arch = match arch_key {
            ARCH_XLM_ROBERTA_ENCODER => {
                // Plan 65-NN Phase B (2026-05-04) — correct path. Uses
                // the immortal-nn `XlmRobertaEncoder` which loads ALL
                // four BERT embedding components (word, position,
                // token-type, embedding-LN) and applies post-norm
                // transformer layers. The bge-reranker .sovereign uses
                // this path; verified Plan 56-LITE Phase D ρ ≥ 0.99
                // GPU vs CPU sovereign equivalence.
                let encoder =
                    XlmRobertaEncoder::from_model_with_arch(&model, &[ARCH_XLM_ROBERTA_ENCODER])
                        .map_err(|err| {
                            GmemError::Embedding(format!("xlm-r encoder load: {err}"))
                        })?;
                if encoder.hidden_dim() != self.dim {
                    return Err(GmemError::Embedding(format!(
                        "sovereign embed dim mismatch: configured {}, model {}",
                        self.dim,
                        encoder.hidden_dim()
                    )));
                }
                LoadedArch::XlmRoberta(encoder)
            }
            ARCH_SEQUENCE_CLASSIFIER_TRANSFORMER => {
                // Legacy path: kept for backwards-compat with old
                // .sovereign artifacts. Known to be broken on
                // BERT-family models (pre-norm + missing embedding
                // components); see Plan 65-NN Phase B sovereign
                // correctness audit. Operators with old artifacts
                // should re-convert via `sovereign-convert --arch
                // xlm_roberta_encoder` to migrate to the correct path.
                let embedding = Embedding::load(&model, "embedding")
                    .map_err(|err| GmemError::Embedding(format!("embedding load: {err}")))?;
                if embedding.embed_dim() != self.dim {
                    return Err(GmemError::Embedding(format!(
                        "sovereign embed dim mismatch: configured {}, model {}",
                        self.dim,
                        embedding.embed_dim()
                    )));
                }
                let num_layers = parse_usize_metadata(&model, "num_layers", 6)?;
                let num_heads = parse_usize_metadata(&model, "num_heads", 8)?;
                let activation = model
                    .metadata("activation")
                    .map(Activation::from_str)
                    .transpose()
                    .map_err(|err| GmemError::Embedding(format!("activation metadata: {err}")))?
                    .unwrap_or(Activation::GeluTanh);
                let mut layers = Vec::with_capacity(num_layers);
                for layer_idx in 0..num_layers {
                    layers.push(
                        TransformerBlock::load(
                            &model,
                            &format!("encoder.layer.{layer_idx}"),
                            num_heads,
                            activation,
                        )
                        .map_err(|err| {
                            GmemError::Embedding(format!(
                                "transformer block {layer_idx} load: {err}"
                            ))
                        })?,
                    );
                }
                let final_norm = LayerNorm::load(&model, "final_norm").ok();
                LoadedArch::LegacySequenceClassifier {
                    embedding,
                    layers,
                    final_norm,
                }
            }
            other => {
                return Err(GmemError::Embedding(format!(
                    "unsupported sovereign architecture {other:?}; expected \
                     {ARCH_XLM_ROBERTA_ENCODER:?} (preferred) or \
                     {ARCH_SEQUENCE_CLASSIFIER_TRANSFORMER:?} (legacy)"
                )));
            }
        };

        Ok(Arc::new(LoadedModel {
            tokenizer,
            arch,
            dim: self.dim,
        }))
    }
}

impl EmbeddingProvider for SovereignEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let loaded = self.loaded_model()?;
        let encoding = loaded
            .tokenizer
            .encode(text, true)
            .map_err(|err| GmemError::Embedding(format!("tokenize: {err}")))?;

        let token_ids = &encoding.ids;
        if token_ids.is_empty() {
            return Err(GmemError::Embedding("tokenizer produced no tokens".into()));
        }

        let attention_mask = encoding.attention_mask.clone();
        let mask_tensor = Tensor::from_vec(
            attention_mask.iter().map(|&value| value as f32).collect(),
            Shape::d2(1, attention_mask.len()),
        )
        .map_err(|err| GmemError::Embedding(format!("attention mask tensor: {err}")))?;

        // Plan 65-NN Phase B (2026-05-04) — dispatch over loaded
        // architecture. The XlmRoberta path is the verified-correct
        // BERT/RoBERTa path; the LegacySequenceClassifier path is
        // retained for backwards-compat with old artifacts but is
        // known to be broken on BERT-family models.
        let mut hidden = match &loaded.arch {
            LoadedArch::XlmRoberta(encoder) => encoder
                .encode(token_ids, Some(&attention_mask))
                .map_err(|err| GmemError::Embedding(format!("xlm-r encode: {err}")))?,
            LoadedArch::LegacySequenceClassifier {
                embedding,
                layers,
                final_norm,
            } => {
                let embedded = embedding
                    .forward(token_ids)
                    .map_err(|err| GmemError::Embedding(format!("embedding forward: {err}")))?;
                let mut h = embedded
                    .reshape(Shape::d3(1, token_ids.len(), loaded.dim))
                    .map_err(|err| GmemError::Embedding(format!("embedding reshape: {err}")))?;
                // Bug 1 partial fix from earlier today (2026-05-04):
                // forward_post_norm matches BERT's post-norm semantics.
                // For the legacy path on BERT-family models this is
                // closer to correct than the original pre-norm forward
                // but still produces incorrect embeddings due to the
                // missing position/type/embedding-LN tensors. Operators
                // with old artifacts should re-convert with
                // `--arch xlm_roberta_encoder` to migrate.
                for (layer_idx, layer) in layers.iter().enumerate() {
                    h = layer
                        .forward_post_norm(&h, Some(&mask_tensor))
                        .map_err(|err| {
                            GmemError::Embedding(format!("transformer layer {layer_idx}: {err}"))
                        })?;
                }
                if let Some(fn_norm) = final_norm {
                    h = fn_norm
                        .forward(&h)
                        .map_err(|err| GmemError::Embedding(format!("final layer norm: {err}")))?;
                }
                h
            }
        };

        let active_tokens = select_active_tokens(&hidden, &attention_mask, loaded.dim)?;
        let pooled = ops::mean_pool_seq(&active_tokens)
            .map_err(|err| GmemError::Embedding(format!("mean pool: {err}")))?;
        let embedding = pooled.data().to_vec();
        let _ = &mut hidden; // hidden is consumed by select_active_tokens; suppress mut-warning

        if embedding.len() != self.dim {
            return Err(GmemError::Embedding(format!(
                "pooled embedding length mismatch: expected {}, got {}",
                self.dim,
                embedding.len()
            )));
        }

        Ok(embedding)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

fn parse_usize_metadata(model: &SovereignModel, key: &str, default: usize) -> Result<usize> {
    match model.metadata(key) {
        Some(value) => value.parse::<usize>().map_err(|err| {
            GmemError::Embedding(format!("invalid {key} metadata {value:?}: {err}"))
        }),
        None => Ok(default),
    }
}

fn select_active_tokens(hidden: &Tensor, attention_mask: &[u32], dim: usize) -> Result<Tensor> {
    if hidden.shape().ndim() != 3 {
        return Err(GmemError::Embedding(format!(
            "expected hidden state rank-3 [B,S,E], got rank-{}",
            hidden.shape().ndim()
        )));
    }

    let dims = hidden.shape().dims();
    if dims[0] != 1 || dims[2] != dim {
        return Err(GmemError::Embedding(format!(
            "unexpected hidden state shape {:?}; expected [1, seq_len, {}]",
            dims, dim
        )));
    }
    if dims[1] != attention_mask.len() {
        return Err(GmemError::Embedding(format!(
            "attention mask length {} does not match hidden seq_len {}",
            attention_mask.len(),
            dims[1]
        )));
    }

    let mut active = Vec::with_capacity(dims[1] * dim);
    let mut active_tokens = 0usize;
    let data = hidden.data();

    for (token_idx, &mask_value) in attention_mask.iter().enumerate() {
        if mask_value == 0 {
            continue;
        }

        let start = token_idx * dim;
        let end = start + dim;
        active.extend_from_slice(&data[start..end]);
        active_tokens += 1;
    }

    if active_tokens == 0 {
        return Err(GmemError::Embedding(
            "attention mask contained no active tokens".into(),
        ));
    }

    Tensor::from_vec(active, Shape::d3(1, active_tokens, dim))
        .map_err(|err| GmemError::Embedding(format!("active token tensor: {err}")))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn missing_model_returns_clear_error() {
        let emb = SovereignEmbedder::new(
            PathBuf::from("/does/not/exist/model.sovereign"),
            PathBuf::from("/does/not/exist/tokenizer.json"),
        );
        let err = emb
            .embed("hello")
            .expect_err("must error when model missing");
        let msg = err.to_string();
        assert!(msg.contains("sovereign model not found"), "got: {msg}");
    }

    #[test]
    fn dim_matches_minilm() {
        let emb = SovereignEmbedder::new(PathBuf::from("x"), PathBuf::from("y"));
        assert_eq!(emb.dim(), 384);
    }
}
