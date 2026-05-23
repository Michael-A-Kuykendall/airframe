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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp_dir() -> TempDir {
        tempfile::tempdir().expect("create temp dir")
    }

    // ── ArtifactGenerator::generate_filename ─────────────────────────────────

    #[test]
    fn test_filename_format() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());
        let p = gen.generate_filename("slice01_decode16", "abcdef01234567890", "prompt_a");
        let name = p.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("v2_slice01_decode16_abcdef01"), "got: {name}");
        assert!(name.ends_with("_prompt_a.json"), "got: {name}");
    }

    #[test]
    fn test_filename_truncates_sha_to_8_chars() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());
        let sha = "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556";
        let p = gen.generate_filename("s01", sha, "p");
        let name = p.file_name().unwrap().to_string_lossy();
        // Only first 8 chars of sha appear in the name
        assert!(name.contains("da3087fb"), "got: {name}");
        assert!(!name.contains("14aede55"), "should only use first 8: {name}");
    }

    // ── write_artifact / read_artifact roundtrip ──────────────────────────────

    #[test]
    fn test_write_read_roundtrip_decode_artifact() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());

        let artifact = DecodeArtifact {
            model_sha256: "abc123".to_string(),
            prompt_token_ids: vec![1, 2, 3],
            generated_token_ids: vec![10, 11, 12, 13],
            per_step_logits: vec![
                LogitInfo { step: 0, max_logit_index: 10, max_logit_value: Some(5.0), finite: true },
                LogitInfo { step: 1, max_logit_index: 11, max_logit_value: Some(4.0), finite: true },
            ],
            determinism_proof: DeterminismProof {
                run1_tokens: vec![10, 11],
                run2_tokens: vec![10, 11],
                identical: true,
            },
        };

        let path = gen.generate_filename("slice01", "abc123ff", "p1");
        gen.write_artifact(&artifact, &path).expect("write");

        let back: DecodeArtifact = gen.read_artifact(&path).expect("read");
        assert_eq!(back.model_sha256, "abc123");
        assert_eq!(back.generated_token_ids, vec![10, 11, 12, 13]);
        assert!(back.determinism_proof.identical);
    }

    #[test]
    fn test_write_creates_intermediate_directories() {
        let td = tmp_dir();
        let nested = td.path().join("deeply").join("nested").join("dir");
        let gen = ArtifactGenerator::new(&nested);
        let path = gen.generate_filename("s01", "aaaabbbb", "p");
        let artifact = DeterminismProof {
            run1_tokens: vec![1],
            run2_tokens: vec![1],
            identical: true,
        };
        gen.write_artifact(&artifact, &path).expect("should create dirs");
        assert!(path.exists());
    }

    #[test]
    fn test_read_nonexistent_file_returns_error() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());
        let p = td.path().join("does_not_exist.json");
        let result: Result<DecodeArtifact, _> = gen.read_artifact(&p);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_kvcache_artifact_roundtrip() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());

        let artifact = KVCacheArtifact {
            model_sha256: "sha_kvcache".to_string(),
            cache_len_by_step: vec![5, 6, 7, 8],
            attention_diagnostics: Some(AttentionDiagnostics {
                attention_score_lengths: vec![5, 6, 7],
                history_usage_confirmed: true,
                single_step_illusion_detected: false,
            }),
            equivalence_test: EquivalenceResult::Pass,
        };

        let path = gen.generate_filename("slice02", "sha_kvcache00", "p2");
        gen.write_artifact(&artifact, &path).expect("write");
        let back: KVCacheArtifact = gen.read_artifact(&path).expect("read");
        assert_eq!(back.cache_len_by_step, vec![5, 6, 7, 8]);
        assert!(back.attention_diagnostics.is_some());
    }

    #[test]
    fn test_write_oracle_artifact_roundtrip() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());

        let artifact = OracleArtifact {
            model_sha256: "sha_oracle".to_string(),
            oracle_tool: "llama.cpp".to_string(),
            oracle_version: "b1234".to_string(),
            conformance_result: ConformanceResult::ExactMatch,
        };

        let path = gen.generate_filename("slice03", "sha_orac", "p3");
        gen.write_artifact(&artifact, &path).expect("write");
        let back: OracleArtifact = gen.read_artifact(&path).expect("read");
        assert_eq!(back.oracle_tool, "llama.cpp");
        assert!(matches!(back.conformance_result, ConformanceResult::ExactMatch));
    }

    #[test]
    fn test_conformance_mismatch_serialization() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());

        let artifact = OracleArtifact {
            model_sha256: "sha_mm".to_string(),
            oracle_tool: "llama.cpp".to_string(),
            oracle_version: "b1000".to_string(),
            conformance_result: ConformanceResult::Mismatch {
                first_divergence_step: 3,
                lib_token: 42,
                oracle_token: 99,
                mismatch_report: "divergence at step 3".to_string(),
            },
        };

        let path = gen.generate_filename("slice03_mm", "sha_mmXX", "p");
        gen.write_artifact(&artifact, &path).expect("write");
        let back: OracleArtifact = gen.read_artifact(&path).expect("read");
        match back.conformance_result {
            ConformanceResult::Mismatch { first_divergence_step, .. } => {
                assert_eq!(first_divergence_step, 3);
            }
            other => panic!("expected Mismatch, got {other:?}"),
        }
    }

    // ── SliceArtifact / ValidationOutcome ────────────────────────────────────

    #[test]
    fn test_slice_artifact_roundtrip() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());

        let artifact = SliceArtifact {
            slice_id: "s01".to_string(),
            model_sha256: "sha_sa".to_string(),
            model_file_size: 1234567,
            timestamp: chrono::Utc::now(),
            validation_result: ValidationOutcome::Pass,
        };

        let path = gen.generate_filename("sliceart", "sha_saXX", "pa");
        gen.write_artifact(&artifact, &path).expect("write");
        let back: SliceArtifact = gen.read_artifact(&path).expect("read");
        assert_eq!(back.slice_id, "s01");
        assert!(matches!(back.validation_result, ValidationOutcome::Pass));
    }

    #[test]
    fn test_validation_outcome_fail_serialization() {
        let td = tmp_dir();
        let gen = ArtifactGenerator::new(td.path());

        let artifact = SliceArtifact {
            slice_id: "s_fail".to_string(),
            model_sha256: "sha_fail".to_string(),
            model_file_size: 0,
            timestamp: chrono::Utc::now(),
            validation_result: ValidationOutcome::Fail { reason: "bad stuff".to_string() },
        };

        let path = gen.generate_filename("slicefail", "sha_fXXX", "pf");
        gen.write_artifact(&artifact, &path).expect("write");
        let back: SliceArtifact = gen.read_artifact(&path).expect("read");
        match back.validation_result {
            ValidationOutcome::Fail { reason } => assert_eq!(reason, "bad stuff"),
            other => panic!("expected Fail, got {other:?}"),
        }
    }
}
