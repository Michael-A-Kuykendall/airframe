//! Airframe Integration Test: Deterministic 16-token decode (greedy/argmax only)
//!
//! Mirrors tests/v2_slice01_decode16.rs but targets the `airframe` crate directly
//! to ensure the "metal" maintains parity after refactoring.
//!
//! Run with: cargo test --test parity -- --ignored --nocapture

use airframe::core::model::Model;
use airframe::family::llama::{init_verbose_diagnostics, LlamaModel};
use airframe::fixtures::prompt_fixtures::PromptFixtureLoader;
use airframe::runtime::multi_token_engine::MultiTokenEngine;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// Expected model identity
const EXPECTED_SHA256: &str = "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556";
const EXPECTED_FILE_SIZE: u64 = 637_699_456;

/// Get model path from environment or use default
fn get_model_path() -> String {
    std::env::var("LIBSHIMMY_MODEL_PATH").unwrap_or_else(|_| {
        "../../../llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf".to_string()
    })
}

/// Helper: Calculate SHA256 of file
fn calculate_file_sha256(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

#[derive(Debug, Serialize)]
struct StepDiagnostic {
    step: usize,
    max_logit_index: usize,
    max_logit_value: f32,
    finite: bool,
}

#[derive(Debug, Serialize)]
struct DeterminismProof {
    run1_tokens: Vec<usize>,
    run2_tokens: Vec<usize>,
    identical: bool,
}

#[derive(Debug, Serialize)]
struct AirframeParityArtifact {
    model_sha256: String,
    model_file_size: u64,
    prompt_fixture_id: String,
    prompt_token_ids: Vec<usize>,
    generated_token_ids: Vec<usize>,
    per_step_diagnostics: Vec<StepDiagnostic>,
    determinism_proof: DeterminismProof,
    gate_status: String,
}

/// Generate 16 tokens using MultiTokenEngine
fn generate_16_tokens_greedy(
    engine: &mut MultiTokenEngine,
    prompt_ids: &[u32],
    weights: &std::collections::HashMap<
        airframe::core::weight_id::WeightId,
        airframe::core::tensor::Tensor,
    >,
) -> Result<(Vec<usize>, Vec<StepDiagnostic>), Box<dyn std::error::Error>> {
    // Use MultiTokenEngine's decode_sequence method
    let result = engine.decode_sequence(prompt_ids, weights)?;

    // Convert to expected format
    let generated_tokens: Vec<usize> = result
        .generated_tokens
        .iter()
        .map(|&t| t as usize)
        .collect();

    let diagnostics: Vec<StepDiagnostic> = result
        .per_step_logits
        .iter()
        .map(|logit_info| StepDiagnostic {
            step: logit_info.step,
            max_logit_index: logit_info.max_logit_index as usize,
            max_logit_value: logit_info.max_logit_value.unwrap_or(0.0),
            finite: logit_info.finite,
        })
        .collect();

    Ok((generated_tokens, diagnostics))
}

#[test]
#[ignore]
fn test_airframe_parity_control() {
    println!("\n🚀 AIRFRAME PARITY CHECK: Deterministic 16-token decode");
    println!("{}", "=".repeat(60));
    println!("CWD: {:?}", std::env::current_dir());

    // 1. Model Identity
    let model_path_str = get_model_path();
    let model_path = Path::new(&model_path_str);

    if !model_path.exists() {
        // Fallback check for alternative path (relative to crate root vs workspace root)
        let alt_path = "../llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"; // Try one level up?
        if Path::new(alt_path).exists() {
            // We can't change the path easily here without refactoring get_model_path logic
            // but we'll panic with clear instruction.
        }
        panic!("Model file not found at: {}", model_path_str);
    }

    let actual_sha256 = calculate_file_sha256(&model_path_str).expect("Failed SHA256");
    assert_eq!(actual_sha256, EXPECTED_SHA256, "Model SHA256 mismatch");

    let metadata = fs::metadata(&model_path_str).expect("Failed metadata");
    assert_eq!(metadata.len(), EXPECTED_FILE_SIZE, "Size mismatch");

    // 2. Load Fixture
    // Note: PromptFixtureLoader expects a path relative to CWD.
    // When running crate test, CWD is usually workspace root.
    let mut fixture_loader = PromptFixtureLoader::v2_target("../../fixtures");
    let fixture = fixture_loader
        .load_fixture("hello_world")
        .expect("Failed to load fixture");

    // 3. Load Model
    let model = Model::from_tinylama_q4_0_gguf(model_path).expect("Failed to load model");

    // 4. Initialize Engine
    let llama_model = LlamaModel::from_spec(model.spec.clone());
    let mut engine = MultiTokenEngine::new(llama_model);
    init_verbose_diagnostics();

    // 5. Run 1
    let (tokens1, diagnostics1) =
        generate_16_tokens_greedy(&mut engine, &fixture.token_ids, &model.weights)
            .expect("Run 1 failed");

    let all_finite = diagnostics1.iter().all(|d| d.finite);
    assert!(all_finite, "Non-finite logits detected");
    assert_eq!(tokens1.len(), 16);

    // 6. Run 2 (Determinism)
    let (tokens2, _) = generate_16_tokens_greedy(&mut engine, &fixture.token_ids, &model.weights)
        .expect("Run 2 failed");

    assert_eq!(tokens1, tokens2, "Non-deterministic output");

    // 7. Verify Cache
    engine.reset();
    engine
        .decode_sequence(&fixture.token_ids, &model.weights)
        .expect("Cache test failed");
    // prompt (5) + 16 generated = 21
    assert_eq!(engine.current_cache_len(), fixture.token_ids.len() + 16);

    println!("\n✅ AIRFRAME PARITY LOCKED");
}
