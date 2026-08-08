use serde::{Deserialize, Serialize};

/// Production capture protocol — telemetry-only types for conformance.
/// These types are emitted by production Airframe and consumed by conformance.
/// Conformance code MUST NOT import production capture implementation modules.

/// Point identity in the inference graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct PointIdentity {
    /// Layer index (0 = embedding, 1..n = transformer layers, n+1 = final norm, n+2 = lm_head).
    pub layer: u32,

    /// Position in the sequence.
    pub position: u32,

    /// Stage within the layer: "embedding", "residual_pre", "attn_q", "attn_k", "attn_v",
    /// "attn_post", "ffn_pre", "ffn_post", "output", "final_norm", "lm_head".
    pub stage: String,

    /// Optional sub-stage for fine-grained points.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_stage: Option<String>,
}

/// Availability state of a capture point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Availability {
    /// Real tensor data captured from GPU.
    Real,
    /// Proxy/computed value (documented approximation).
    Proxy,
    /// Point exists but data unavailable (e.g., unsupported engine).
    Unavailable,
    /// Error during capture.
    Error(String),
}

/// Coordinate plan for a capture point — describes how to locate the tensor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CoordinatePlan {
    /// Buffer binding index.
    pub binding: u32,

    /// Byte offset within the buffer.
    pub offset: u64,

    /// Element count.
    pub count: u64,

    /// Element stride in bytes.
    pub stride: u64,

    /// Data format (e.g., "f32", "f16", "bf16", "i8", "u8").
    pub format: String,

    /// Shape as [dim0, dim1, ...].
    pub shape: Vec<u64>,
}

/// Provenance fields for a capture point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// Engine that produced this capture.
    pub engine: String,

    /// Engine version/commit.
    pub engine_version: String,

    /// Capture timestamp (RFC3339).
    pub captured_at: String,

    /// Git commit of the engine (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,

    /// Build profile (debug/release).
    pub build_profile: String,

    /// Capture configuration hash.
    pub config_hash: String,
}

/// A single capture point with metadata and optional tensor data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CapturePoint {
    /// Point identity.
    pub identity: PointIdentity,

    /// Availability state.
    pub availability: Availability,

    /// Coordinate plan for locating the tensor.
    pub coordinate: CoordinatePlan,

    /// Provenance information.
    pub provenance: Provenance,

    /// Tensor statistics (RMS, NaN count, first 8 values).
    /// Only present when availability == Real.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<TensorStats>,

    /// Raw tensor data (base64 encoded).
    /// Only present when availability == Real and capture includes raw tensors.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// Tensor statistics for a capture point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TensorStats {
    /// Root mean square of the tensor.
    pub rms: f64,

    /// Number of NaN values.
    pub nan_count: u64,

    /// First 8 values (or fewer if tensor is smaller).
    pub first8: Vec<f64>,

    /// Honesty tag for the data source.
    pub sampled: SampledSource,
}

/// Source honesty classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SampledSource {
    Real,
    Proxy,
    Unavailable,
    TempLast,
    ActivationLast,
    WrongBuffer,
    ProxyResidual,
}

/// Complete capture for a single prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PromptCapture {
    /// Prompt text.
    pub prompt: String,

    /// Token IDs for the prompt.
    pub token_ids: Vec<u32>,

    /// Token pieces (optional).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_pieces: Option<Vec<String>>,

    /// Capture points in topological order.
    pub points: Vec<CapturePoint>,

    /// Model configuration at capture time.
    pub model_config: ModelConfig,
}

/// Model configuration snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Architecture identifier.
    pub arch: String,

    /// Number of layers.
    pub n_layer: u32,

    /// Embedding dimension.
    pub n_embd: u32,

    /// Number of attention heads.
    pub n_head: u32,

    /// Number of KV heads.
    pub n_kv_head: u32,

    /// Head dimension.
    pub head_dim: u32,

    /// RoPE base frequency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_base: Option<f64>,

    /// RoPE dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rope_dim: Option<u32>,

    /// RMS norm epsilon.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_eps: Option<f64>,

    /// Whether QK-norm is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qk_norm: Option<bool>,

    /// Context length cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx_capped: Option<u32>,
}

/// Full capture manifest for a conformance run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CaptureManifest {
    /// Schema identifier: "airframe.conformance.capture.v1"
    #[serde(rename = "$schema")]
    pub schema: String,

    /// Run identifier (matches ConformanceManifest.run_id).
    pub run_id: String,

    /// Conformance version.
    pub conformance_version: String,

    /// Capture timestamp.
    pub captured_at: String,

    /// All prompt captures.
    pub prompts: Vec<PromptCapture>,
}

impl CaptureManifest {
    pub fn new(run_id: String, conformance_version: String, prompts: Vec<PromptCapture>) -> Self {
        Self {
            schema: "airframe.conformance.capture.v1".to_string(),
            run_id,
            conformance_version,
            captured_at: chrono::Utc::now().to_rfc3339(),
            prompts,
        }
    }

    pub fn validate(&self) -> Result<(), crate::schemas::ValidationError> {
        crate::schemas::validate_capture(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_identity_roundtrip() {
        let id = PointIdentity {
            layer: 1,
            position: 0,
            stage: "attn_q".to_string(),
            sub_stage: Some("pre_rope".to_string()),
        };
        let json = serde_json::to_string(&id).unwrap();
        let parsed: PointIdentity = serde_json::from_str(&json).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn capture_manifest_schema() {
        let manifest = CaptureManifest::new("test-run".to_string(), "0.1.0".to_string(), vec![]);
        assert_eq!(manifest.schema, "airframe.conformance.capture.v1");
    }
}
