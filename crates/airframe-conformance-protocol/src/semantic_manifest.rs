//! Llama-family semantic profile — derived from raw GGUF evidence with full provenance.

use crate::raw_gguf::model_identity;
use serde::{Deserialize, Serialize};

/// A semantic value with explicit provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Provenanced<T> {
    /// The derived value.
    pub value: T,
    /// Source of this value.
    pub provenance: ValueProvenance,
}

/// Provenance for a semantic value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "source")]
pub enum ValueProvenance {
    /// Directly from GGUF metadata key.
    GgufMetadata { key: String },
    /// From tensor directory (shape, type, offset).
    TensorDirectory { tensor_name: String, field: String },
    /// Computed from other values via equation.
    Derived {
        equation: String,
        inputs: Vec<String>,
    },
    /// Hard-coded default (explicitly marked as fallback).
    Default { reason: String },
    /// Explicitly unsupported/missing.
    Unsupported { reason: String },
    /// Conflicting evidence.
    Conflicting { values: Vec<Provenanced<String>> },
}

/// Llama-family semantic profile — all required semantic fields with provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LlamaProfile {
    /// Architecture identifier from GGUF.
    pub architecture: Provenanced<String>,
    /// Model parameter count (e.g., "7B", "13B").
    pub parameter_count: Provenanced<String>,
    /// Quantization format (e.g., "Q4_K_M").
    pub quantization: Provenanced<String>,
    /// Context length (n_ctx).
    pub context_length: Provenanced<u32>,
    /// Embedding dimension (n_embd).
    pub embedding_dim: Provenanced<u32>,
    /// Number of layers (n_layer).
    pub layer_count: Provenanced<u32>,
    /// Attention head count (n_head).
    pub attention_head_count: Provenanced<u32>,
    /// KV head count (n_kv_head) — for GQA.
    pub kv_head_count: Provenanced<u32>,
    /// Head dimension (head_dim).
    pub head_dim: Provenanced<u32>,
    /// RoPE dimension count.
    pub rope_dim: Provenanced<u32>,
    /// RoPE frequency base.
    pub rope_freq_base: Provenanced<f32>,
    /// RoPE scaling factor (if present).
    pub rope_scaling_factor: Provenanced<Option<f32>>,
    /// RoPE scaling type (if present).
    pub rope_scaling_type: Provenanced<Option<String>>,
    /// Attention layer norm epsilon.
    pub attn_layer_norm_eps: Provenanced<f32>,
    /// Feed-forward length (n_ff).
    pub feed_forward_length: Provenanced<u32>,
    /// Whether parallel residual is used.
    pub use_parallel_residual: Provenanced<bool>,
    /// Tensor data layout.
    pub tensor_data_layout: Provenanced<String>,
    /// Expert count (for MoE).
    pub expert_count: Provenanced<Option<u32>>,
    /// Expert used count (for MoE).
    pub expert_used_count: Provenanced<Option<u32>>,
    /// GGML file type (overall quantization).
    pub file_type: Provenanced<u32>,
    /// Whether QK normalization is used.
    pub qk_norm: Provenanced<bool>,
    /// Whether post-attention norm is present.
    pub post_attention_norm: Provenanced<bool>,
}

/// Complete semantic manifest with raw GGUF identity and derived profile.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticManifest {
    /// Schema identifier: "airframe.conformance.semantic_manifest.v1"
    #[serde(rename = "$schema")]
    pub schema: String,
    /// Raw GGUF identity (from CONF-2).
    pub raw_identity: model_identity::ModelIdentity,
    /// Llama semantic profile with provenance.
    pub profile: LlamaProfile,
    /// Shape/consistency equations that were validated.
    pub equations: Vec<ValidatedEquation>,
    /// Unsupported/ambiguous required facts.
    pub unsupported_facts: Vec<String>,
    /// Conflicting facts detected.
    pub conflicting_facts: Vec<String>,
    /// Manifest creation timestamp.
    pub created_at: String,
    /// Conformance version.
    pub conformance_version: String,
}

/// A validated shape/consistency equation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidatedEquation {
    /// Equation name/identifier.
    pub name: String,
    /// Human-readable equation (e.g., "head_dim = embedding_dim / attention_head_count").
    pub equation: String,
    /// Whether the equation held.
    pub passed: bool,
    /// Left-hand side value.
    pub lhs: Provenanced<String>,
    /// Right-hand side value.
    pub rhs: Provenanced<String>,
    /// Provenance for the equation itself.
    pub provenance: ValueProvenance,
}

impl SemanticManifest {
    pub fn new(raw_identity: model_identity::ModelIdentity) -> Self {
        Self {
            schema: "airframe.conformance.semantic_manifest.v1".to_string(),
            raw_identity,
            profile: LlamaProfile::default(),
            equations: Vec::new(),
            unsupported_facts: Vec::new(),
            conflicting_facts: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
            conformance_version: crate::CONFORMANCE_VERSION.to_string(),
        }
    }
}

impl Default for LlamaProfile {
    fn default() -> Self {
        Self {
            architecture: Provenanced {
                value: String::new(),
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            parameter_count: Provenanced {
                value: String::new(),
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            quantization: Provenanced {
                value: String::new(),
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            context_length: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            embedding_dim: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            layer_count: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            attention_head_count: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            kv_head_count: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            head_dim: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            rope_dim: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            rope_freq_base: Provenanced {
                value: 0.0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            rope_scaling_factor: Provenanced {
                value: None,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            rope_scaling_type: Provenanced {
                value: None,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            attn_layer_norm_eps: Provenanced {
                value: 0.0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            feed_forward_length: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            use_parallel_residual: Provenanced {
                value: false,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            tensor_data_layout: Provenanced {
                value: String::new(),
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            expert_count: Provenanced {
                value: None,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            expert_used_count: Provenanced {
                value: None,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            file_type: Provenanced {
                value: 0,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            qk_norm: Provenanced {
                value: false,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
            post_attention_norm: Provenanced {
                value: false,
                provenance: ValueProvenance::Unsupported {
                    reason: "not yet derived".to_string(),
                },
            },
        }
    }
}
