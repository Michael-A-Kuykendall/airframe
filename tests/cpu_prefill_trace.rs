//! Generate CPU golden trace for prefill-only sequence matching GPU dump
//!
//! GPU processes: [1, 15043] (BOS, "Hello")  
//! This test generates the matching CPU trace for validation

use airframe::core::model::Model;
use airframe::family::llama::LlamaModel;
use airframe::runtime::engine::Engine;
use std::path::Path;

#[test]
#[ignore]
fn test_cpu_prefill_hello() {
    println!("\n🔬 CPU PREFILL TRACE - HELLO");
    println!("{}", "=".repeat(70));

    // Enable L0 tracing
    std::env::set_var(
        "SHIMMY_L0_TRACE_PATH",
        "c:/Users/micha/repos/libshimmy/artifacts/cpu_prefill_hello.csv",
    );
    airframe::family::llama::init_verbose_diagnostics();

    // Load model
    let model_path =
        Path::new("D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let model = Model::from_tinylama_q4_0_gguf(model_path).expect("Failed to load model");

    // Create engine
    let llama_model = LlamaModel::from_spec(model.spec.clone());
    let mut engine = Engine::new(llama_model);

    // Process tokens ONE AT A TIME to get separate L0.1 entries
    println!("\n=== Position 0: BOS (Token 1) ===");
    let _logits1 = engine
        .prefill(&[1], &model.weights)
        .expect("Prefill BOS failed");

    println!("\n=== Position 1: Hello (Token 15043) ===");
    let _logits2 = engine
        .decode(15043, &model.weights)
        .expect("Decode Hello failed");

    println!("\n✅ CPU trace generated: artifacts/cpu_prefill_hello.csv");
    println!("This trace contains:");
    println!("  - Entry 1: BOS (1) embedding + layer states");
    println!("  - Entry 2: Hello (15043) embedding + layer states");
}
