use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Versioned conformance manifest — describes a conformance run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConformanceManifest {
    /// Schema identifier and version: "airframe.conformance.manifest.v1"
    #[serde(rename = "$schema")]
    pub schema: String,

    /// Unique run identifier (UUID v4).
    pub run_id: String,

    /// Conformance crate version that produced this manifest.
    pub conformance_version: String,

    /// Timestamp of manifest creation (RFC3339).
    pub created_at: String,

    /// Model under test.
    pub model: ModelIdentity,

    /// Capture configuration.
    pub capture: CaptureConfig,

    /// Comparison configuration.
    pub comparison: ComparisonConfig,

    /// Evidence package output configuration.
    pub evidence: EvidenceConfig,

    /// Additional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    /// Path to the GGUF model file.
    pub model_path: String,

    /// Model architecture identifier (e.g., "llama", "qwen3", "gemma").
    pub architecture: String,

    /// Model parameter count (e.g., "4B", "7B", "14B").
    pub parameter_count: String,

    /// Quantization format (e.g., "Q4_K_M", "Q8_0").
    pub quantization: String,

    /// GGUF metadata key-value pairs relevant to conformance.
    #[serde(default)]
    pub gguf_metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CaptureConfig {
    /// Engine to capture from: "airframe_product", "airframe_sequential", "llama", "candle"
    pub engine: String,

    /// Engine-specific configuration.
    #[serde(default)]
    pub engine_config: HashMap<String, serde_json::Value>,

    /// Prompt(s) to capture.
    pub prompts: Vec<String>,

    /// Capture levels to include (0-8).
    #[serde(default = "default_capture_levels")]
    pub levels: Vec<u8>,

    /// Maximum tokens to capture per prompt.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_capture_levels() -> Vec<u8> {
    (0..=8).collect()
}

fn default_max_tokens() -> usize {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonConfig {
    /// Reference engine to compare against.
    pub reference_engine: String,

    /// Tolerance for numerical comparison (RMS ratio).
    #[serde(default = "default_rms_tolerance")]
    pub rms_tolerance: f64,

    /// Tolerance for final logits (RMS ratio).
    #[serde(default = "default_logits_tolerance")]
    pub logits_tolerance: f64,

    /// Layers to compare (empty = all).
    #[serde(default)]
    pub layers: Vec<usize>,

    /// Whether to require monotonic RMS growth.
    #[serde(default = "default_true")]
    pub require_monotonic_rms: bool,
}

fn default_rms_tolerance() -> f64 {
    2.0
}

fn default_logits_tolerance() -> f64 {
    4.0
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceConfig {
    /// Output directory for evidence package.
    pub output_dir: String,

    /// Whether to include raw tensors in evidence.
    #[serde(default = "default_false")]
    pub include_raw_tensors: bool,

    /// Whether to include per-layer comparisons.
    #[serde(default = "default_true")]
    pub include_layer_comparisons: bool,

    /// Compression format for evidence package.
    #[serde(default = "default_compression")]
    pub compression: String,
}

fn default_false() -> bool {
    false
}

fn default_compression() -> String {
    "zstd".to_string()
}

impl ConformanceManifest {
    /// Create a new manifest with default values.
    pub fn new(model_path: String, architecture: String, prompts: Vec<String>) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let run_id = uuid::Uuid::new_v4().to_string();

        Self {
            schema: "airframe.conformance.manifest.v1".to_string(),
            run_id,
            conformance_version: crate::CONFORMANCE_VERSION.to_string(),
            created_at: now,
            model: ModelIdentity {
                model_path,
                architecture,
                parameter_count: "unknown".to_string(),
                quantization: "unknown".to_string(),
                gguf_metadata: HashMap::new(),
            },
            capture: CaptureConfig {
                engine: "airframe_product".to_string(),
                engine_config: HashMap::new(),
                prompts,
                levels: default_capture_levels(),
                max_tokens: default_max_tokens(),
            },
            comparison: ComparisonConfig {
                reference_engine: "candle".to_string(),
                rms_tolerance: default_rms_tolerance(),
                logits_tolerance: default_logits_tolerance(),
                layers: Vec::new(),
                require_monotonic_rms: default_true(),
            },
            evidence: EvidenceConfig {
                output_dir: "./evidence".to_string(),
                include_raw_tensors: default_false(),
                include_layer_comparisons: default_true(),
                compression: default_compression(),
            },
            metadata: HashMap::new(),
        }
    }

    /// Validate the manifest against its JSON schema.
    pub fn validate(&self) -> Result<(), crate::schemas::ValidationError> {
        crate::schemas::validate_manifest(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrip() {
        let manifest = ConformanceManifest::new(
            "/path/model.gguf".to_string(),
            "llama".to_string(),
            vec!["test prompt".to_string()],
        );
        let json = serde_json::to_string_pretty(&manifest).unwrap();
        let parsed: ConformanceManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest, parsed);
    }

    #[test]
    fn manifest_schema_field() {
        let manifest = ConformanceManifest::new(
            "/path/model.gguf".to_string(),
            "llama".to_string(),
            vec!["test prompt".to_string()],
        );
        assert_eq!(manifest.schema, "airframe.conformance.manifest.v1");
    }
}
