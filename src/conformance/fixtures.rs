use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Conformance fixture for testing model behavior
///
/// Matches the JSON schema defined in ARCHITECT_DECISIONS.md
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformanceFixture {
    pub schema_version: u32,
    pub fixture_id: String,
    pub model: ModelInfo,
    pub oracle: OracleInfo,
    pub prompt: PromptInfo,
    pub steps: StepsInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub gguf_path_hint: String,
    pub gguf_sha256: String,
    pub spec: ModelSpecInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpecInfo {
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub ff_dim: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub rope_dim: usize,
    pub n_ctx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleInfo {
    pub source: String,
    pub version: String,
    pub build_id: String,
    pub date_utc: String,
    pub mode: String,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptInfo {
    pub text: String,
    pub token_ids: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepsInfo {
    pub prefill_last: StepInfo,
    pub decode_1: StepInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepInfo {
    pub selected_token_id: Option<usize>,
    pub topk: TopKInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKInfo {
    pub k: usize,
    pub token_ids: Vec<usize>,
    pub logits: Vec<f32>,
}

/// Oracle fixture for 1-token inference validation
///
/// Simplified fixture format for validating single-token inference
/// against known-good Shimmy oracle results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleFixture {
    pub schema_version: u32,
    pub fixture_id: String,
    pub oracle_source: String,
    pub prompt: String,
    pub token_ids: Vec<usize>,
    pub expected_next_token: usize,
    pub expected_logits: Option<Vec<f32>>, // Optional for tolerance checking
    pub metadata: OracleMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleMetadata {
    pub model_name: String,
    pub oracle_version: String,
    pub generation_date: String,
    pub sampling_method: String, // "argmax", "greedy", etc.
    pub temperature: f32,
    pub notes: Option<String>,
}

/// Load oracle fixture from JSON file
pub fn load_oracle_fixture<P: AsRef<Path>>(path: P) -> Result<OracleFixture> {
    let content = std::fs::read_to_string(path).map_err(LibshimmyError::Io)?;

    let fixture: OracleFixture =
        serde_json::from_str(&content).map_err(|e| LibshimmyError::FixtureError {
            msg: format!("Failed to parse oracle fixture JSON: {}", e),
        })?;

    validate_oracle_fixture(&fixture)?;

    Ok(fixture)
}

/// Save oracle fixture to JSON file
pub fn save_oracle_fixture<P: AsRef<Path>>(fixture: &OracleFixture, path: P) -> Result<()> {
    let json = serde_json::to_string_pretty(fixture).map_err(|e| LibshimmyError::FixtureError {
        msg: format!("Failed to serialize oracle fixture: {}", e),
    })?;

    std::fs::write(path, json).map_err(LibshimmyError::Io)?;

    Ok(())
}

/// Create oracle fixture from Shimmy (authoritative source)
///
/// This function would be used to generate oracle fixtures by calling
/// Shimmy with specific prompts and capturing the results
pub fn create_oracle_fixture_from_shimmy(
    prompt: &str,
    token_ids: Vec<usize>,
    expected_token: usize,
    expected_logits: Option<Vec<f32>>,
) -> OracleFixture {
    OracleFixture {
        schema_version: 1,
        fixture_id: format!("oracle_{}", chrono::Utc::now().timestamp()),
        oracle_source: "shimmy".to_string(),
        prompt: prompt.to_string(),
        token_ids,
        expected_next_token: expected_token,
        expected_logits,
        metadata: OracleMetadata {
            model_name: "TinyLlama-1.1B-Chat-v1.0".to_string(),
            oracle_version: "shimmy-dev".to_string(),
            generation_date: chrono::Utc::now().to_rfc3339(),
            sampling_method: "argmax".to_string(),
            temperature: 0.0, // Deterministic
            notes: Some("Generated from Shimmy oracle for V1 validation".to_string()),
        },
    }
}

/// Validate oracle fixture structure and constraints
fn validate_oracle_fixture(fixture: &OracleFixture) -> Result<()> {
    // Check schema version
    if fixture.schema_version != 1 {
        return Err(LibshimmyError::FixtureError {
            msg: format!(
                "Unsupported oracle schema version: {}",
                fixture.schema_version
            ),
        });
    }

    // Check required fields are not empty
    if fixture.fixture_id.is_empty() {
        return Err(LibshimmyError::FixtureError {
            msg: "Oracle fixture_id cannot be empty".to_string(),
        });
    }

    if fixture.prompt.is_empty() {
        return Err(LibshimmyError::FixtureError {
            msg: "Oracle prompt cannot be empty".to_string(),
        });
    }

    if fixture.token_ids.is_empty() {
        return Err(LibshimmyError::FixtureError {
            msg: "Oracle token_ids cannot be empty".to_string(),
        });
    }

    // Validate logits if provided
    if let Some(ref logits) = fixture.expected_logits {
        if logits.is_empty() {
            return Err(LibshimmyError::FixtureError {
                msg: "Oracle expected_logits cannot be empty if provided".to_string(),
            });
        }

        // Check for non-finite values
        for (i, &logit) in logits.iter().enumerate() {
            if !logit.is_finite() {
                return Err(LibshimmyError::FixtureError {
                    msg: format!("Oracle logit at index {} is not finite: {}", i, logit),
                });
            }
        }
    }

    Ok(())
}

/// Load conformance fixture from JSON file
pub fn load_fixture<P: AsRef<Path>>(path: P) -> Result<ConformanceFixture> {
    let content = std::fs::read_to_string(path).map_err(LibshimmyError::Io)?;

    let fixture: ConformanceFixture =
        serde_json::from_str(&content).map_err(|e| LibshimmyError::FixtureError {
            msg: format!("Failed to parse fixture JSON: {}", e),
        })?;

    validate_fixture(&fixture)?;

    Ok(fixture)
}

/// Save conformance fixture to JSON file
pub fn save_fixture<P: AsRef<Path>>(fixture: &ConformanceFixture, path: P) -> Result<()> {
    let json = serde_json::to_string_pretty(fixture).map_err(|e| LibshimmyError::FixtureError {
        msg: format!("Failed to serialize fixture: {}", e),
    })?;

    std::fs::write(path, json).map_err(LibshimmyError::Io)?;

    Ok(())
}

/// Validate fixture structure and constraints
fn validate_fixture(fixture: &ConformanceFixture) -> Result<()> {
    // Check schema version
    if fixture.schema_version != 1 {
        return Err(LibshimmyError::FixtureError {
            msg: format!("Unsupported schema version: {}", fixture.schema_version),
        });
    }

    // Check required fields are not empty
    if fixture.fixture_id.is_empty() {
        return Err(LibshimmyError::FixtureError {
            msg: "fixture_id cannot be empty".to_string(),
        });
    }

    if fixture.model.name.is_empty() {
        return Err(LibshimmyError::FixtureError {
            msg: "model.name cannot be empty".to_string(),
        });
    }

    // Validate top-K constraints
    if fixture.steps.prefill_last.topk.k == 0 {
        return Err(LibshimmyError::FixtureError {
            msg: "prefill_last.topk.k must be > 0".to_string(),
        });
    }

    if fixture.steps.decode_1.topk.k == 0 {
        return Err(LibshimmyError::FixtureError {
            msg: "decode_1.topk.k must be > 0".to_string(),
        });
    }

    // Validate top-K data consistency
    let prefill_topk = &fixture.steps.prefill_last.topk;
    if prefill_topk.token_ids.len() != prefill_topk.logits.len() {
        return Err(LibshimmyError::FixtureError {
            msg: "prefill_last.topk token_ids and logits length mismatch".to_string(),
        });
    }

    if prefill_topk.token_ids.len() != prefill_topk.k {
        return Err(LibshimmyError::FixtureError {
            msg: format!(
                "prefill_last.topk expected {} entries, got {}",
                prefill_topk.k,
                prefill_topk.token_ids.len()
            ),
        });
    }

    let decode_topk = &fixture.steps.decode_1.topk;
    if decode_topk.token_ids.len() != decode_topk.logits.len() {
        return Err(LibshimmyError::FixtureError {
            msg: "decode_1.topk token_ids and logits length mismatch".to_string(),
        });
    }

    if decode_topk.token_ids.len() != decode_topk.k {
        return Err(LibshimmyError::FixtureError {
            msg: format!(
                "decode_1.topk expected {} entries, got {}",
                decode_topk.k,
                decode_topk.token_ids.len()
            ),
        });
    }

    Ok(())
}

/// Create a minimal fixture for testing
pub fn create_test_fixture() -> ConformanceFixture {
    ConformanceFixture {
        schema_version: 1,
        fixture_id: "test-fixture-001".to_string(),
        model: ModelInfo {
            name: "TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf".to_string(),
            gguf_path_hint:
                "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"
                    .to_string(),
            gguf_sha256: "TBD".to_string(),
            spec: ModelSpecInfo {
                n_vocab: 32000,
                n_embd: 2048,
                n_layer: 22,
                n_head: 32,
                n_head_kv: 4,
                ff_dim: 5632,
                rms_eps: 1e-5,
                rope_base: 10000.0,
                rope_scale: 1.0,
                rope_dim: 64,
                n_ctx: 2048,
            },
        },
        oracle: OracleInfo {
            source: "shimmy".to_string(),
            version: "TBD".to_string(),
            build_id: "TBD".to_string(),
            date_utc: "2025-12-12T00:00:00Z".to_string(),
            mode: "greedy".to_string(),
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        },
        prompt: PromptInfo {
            text: "Hello".to_string(),
            token_ids: vec![15043], // Example token ID for "Hello"
        },
        steps: StepsInfo {
            prefill_last: StepInfo {
                selected_token_id: None,
                topk: TopKInfo {
                    k: 10,
                    token_ids: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    logits: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                },
            },
            decode_1: StepInfo {
                selected_token_id: Some(1234),
                topk: TopKInfo {
                    k: 10,
                    token_ids: vec![1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239],
                    logits: vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                },
            },
        },
    }
}

/// Extract top-K logits from full logits tensor
pub fn extract_topk_logits(logits: &Tensor, k: usize) -> Result<(Vec<usize>, Vec<f32>)> {
    if logits.ndim() != 1 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "topk_logits".to_string(),
            expected: vec![1],
            got: vec![logits.ndim()],
        });
    }

    let k = k.min(logits.data.len());

    // Create indexed logits
    let mut indexed_logits: Vec<(usize, f32)> = logits
        .data
        .iter()
        .enumerate()
        .map(|(i, &val)| (i, val))
        .collect();

    // Sort by logit value (descending)
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Extract top-K
    let mut token_ids = Vec::with_capacity(k);
    let mut top_logits = Vec::with_capacity(k);

    for &(token_id, logit) in indexed_logits.iter().take(k) {
        token_ids.push(token_id);
        top_logits.push(logit);
    }

    Ok((token_ids, top_logits))
}

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::tempdir;

    #[test]
    fn test_create_test_fixture() {
        let fixture = create_test_fixture();

        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.fixture_id, "test-fixture-001");
        assert_eq!(fixture.model.spec.n_vocab, 32000);
        assert_eq!(fixture.steps.prefill_last.topk.k, 10);
        assert_eq!(fixture.steps.decode_1.topk.k, 10);
    }

    #[test]
    fn test_fixture_validation() {
        let valid_fixture = create_test_fixture();
        assert!(validate_fixture(&valid_fixture).is_ok());

        // Test invalid schema version
        let mut invalid_fixture = valid_fixture.clone();
        invalid_fixture.schema_version = 999;
        assert!(validate_fixture(&invalid_fixture).is_err());

        // Test empty fixture_id
        let mut invalid_fixture = valid_fixture.clone();
        invalid_fixture.fixture_id = String::new();
        assert!(validate_fixture(&invalid_fixture).is_err());

        // Test zero top-K
        let mut invalid_fixture = valid_fixture.clone();
        invalid_fixture.steps.prefill_last.topk.k = 0;
        assert!(validate_fixture(&invalid_fixture).is_err());
    }

    #[test]
    fn test_fixture_serialization() {
        let fixture = create_test_fixture();

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&fixture).unwrap();
        assert!(json.contains("schema_version"));
        assert!(json.contains("test-fixture-001"));

        // Deserialize back
        let deserialized: ConformanceFixture = serde_json::from_str(&json).unwrap();
        assert_eq!(fixture.fixture_id, deserialized.fixture_id);
        assert_eq!(fixture.model.spec.n_vocab, deserialized.model.spec.n_vocab);
    }

    #[test]
    fn test_fixture_file_operations() {
        let fixture = create_test_fixture();
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_fixture.json");

        // Save fixture
        save_fixture(&fixture, &file_path).unwrap();
        assert!(file_path.exists());

        // Load fixture
        let loaded_fixture = load_fixture(&file_path).unwrap();
        assert_eq!(fixture.fixture_id, loaded_fixture.fixture_id);
        assert_eq!(
            fixture.model.spec.n_layer,
            loaded_fixture.model.spec.n_layer
        );
    }

    #[test]
    fn test_extract_topk_logits() {
        let logits = Tensor::new(vec![0.1, 0.9, 0.3, 0.7, 0.5], vec![5]).unwrap();
        let (token_ids, top_logits) = extract_topk_logits(&logits, 3).unwrap();

        // Should return top 3: indices [1, 3, 4] with logits [0.9, 0.7, 0.5]
        assert_eq!(token_ids, vec![1, 3, 4]);
        assert_eq!(top_logits, vec![0.9, 0.7, 0.5]);
    }

    #[test]
    fn test_extract_topk_larger_than_vocab() {
        let logits = Tensor::new(vec![0.1, 0.2], vec![2]).unwrap();
        let (token_ids, top_logits) = extract_topk_logits(&logits, 10).unwrap();

        // Should return all tokens when k > vocab_size
        assert_eq!(token_ids, vec![1, 0]);
        assert_eq!(top_logits, vec![0.2, 0.1]);
    }
}
