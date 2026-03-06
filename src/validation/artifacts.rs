use crate::validation::errors::{SliceValidationError, ValidationResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Base artifact structure for all validation slices
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SliceArtifact {
    pub slice_id: String,
    pub model_sha256: String,
    pub model_file_size: u64,
    pub timestamp: DateTime<Utc>,
    pub validation_result: ValidationOutcome,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ValidationOutcome {
    Pass,
    Fail { reason: String },
}

/// Slice 01: Deterministic 16-token decode artifact
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DecodeArtifact {
    pub model_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub per_step_logits: Vec<LogitInfo>,
    pub determinism_proof: DeterminismProof,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LogitInfo {
    pub step: usize,
    pub max_logit_index: u32,
    pub max_logit_value: Option<f32>,
    pub finite: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DeterminismProof {
    pub run1_tokens: Vec<u32>,
    pub run2_tokens: Vec<u32>,
    pub identical: bool,
}

/// Slice 02: KV Cache validation artifact
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KVCacheArtifact {
    pub model_sha256: String,
    pub cache_len_by_step: Vec<usize>,
    pub attention_diagnostics: Option<AttentionDiagnostics>,
    pub equivalence_test: EquivalenceResult,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AttentionDiagnostics {
    pub attention_score_lengths: Vec<usize>,
    pub history_usage_confirmed: bool,
    pub single_step_illusion_detected: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EquivalenceResult {
    Pass,
    Fail {
        step: usize,
        prefill_token: u32,
        stepwise_token: u32,
    },
}

/// Slice 03: Oracle conformance artifact
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OracleArtifact {
    pub model_sha256: String,
    pub oracle_tool: String,
    pub oracle_version: String,
    pub conformance_result: ConformanceResult,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ConformanceResult {
    ExactMatch,
    Mismatch {
        first_divergence_step: usize,
        lib_token: u32,
        oracle_token: u32,
        mismatch_report: String,
    },
}

/// Artifact generation utilities
pub struct ArtifactGenerator {
    artifacts_dir: PathBuf,
}

impl ArtifactGenerator {
    pub fn new<P: AsRef<Path>>(artifacts_dir: P) -> Self {
        Self {
            artifacts_dir: artifacts_dir.as_ref().to_path_buf(),
        }
    }

    /// Generate deterministic artifact filename
    pub fn generate_filename(
        &self,
        slice_id: &str,
        model_sha256: &str,
        prompt_id: &str,
    ) -> PathBuf {
        let filename = format!("v2_{}_{}_{}.json", slice_id, &model_sha256[..8], prompt_id);
        self.artifacts_dir.join(filename)
    }

    /// Write artifact to JSON file
    pub fn write_artifact<T: Serialize>(
        &self,
        artifact: &T,
        filepath: &Path,
    ) -> ValidationResult<()> {
        // Ensure artifacts directory exists
        if let Some(parent) = filepath.parent() {
            fs::create_dir_all(parent).map_err(|e| SliceValidationError::ArtifactFailed {
                artifact_path: filepath.display().to_string(),
                error: format!("Failed to create directory: {}", e),
            })?;
        }

        let json_content = serde_json::to_string_pretty(artifact).map_err(|e| {
            SliceValidationError::ArtifactFailed {
                artifact_path: filepath.display().to_string(),
                error: format!("JSON serialization failed: {}", e),
            }
        })?;

        fs::write(filepath, json_content).map_err(|e| SliceValidationError::ArtifactFailed {
            artifact_path: filepath.display().to_string(),
            error: format!("File write failed: {}", e),
        })?;

        Ok(())
    }

    /// Read artifact from JSON file
    pub fn read_artifact<T: for<'de> Deserialize<'de>>(
        &self,
        filepath: &Path,
    ) -> ValidationResult<T> {
        let content =
            fs::read_to_string(filepath).map_err(|e| SliceValidationError::ArtifactFailed {
                artifact_path: filepath.display().to_string(),
                error: format!("File read failed: {}", e),
            })?;

        serde_json::from_str(&content).map_err(|e| SliceValidationError::ArtifactFailed {
            artifact_path: filepath.display().to_string(),
            error: format!("JSON deserialization failed: {}", e),
        })
    }
}
