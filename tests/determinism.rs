use airframe::core::model::Model;
use airframe::family::llama::LlamaModel;
use airframe::runtime::{engine::Engine, sampling::Sampler};
use sha2::{Digest, Sha256};
use shimmytok::Tokenizer;
use std::path::Path;

#[test]
fn test_exact_determinism_parity() -> Result<(), Box<dyn std::error::Error>> {
    // Basic determinism test without depending on local tinyllama if not present,
    // though tests/parity.rs already assumes it's present.
    let model_path = std::env::var("LIBSHIMMY_MODEL_PATH").unwrap_or_else(|_| {
        "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf".to_string()
    });

    if !Path::new(&model_path).exists() {
        println!(
            "Skipping determinism test: model not found at {}",
            model_path
        );
        return Ok(());
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path)?;
    let model = Model::from_tinylama_q4_0_gguf(Path::new(&model_path))?;
    let llama_model = LlamaModel::from_spec(model.spec.clone());

    // Run 1
    let prompt = "Determinism check:";
    let tokens = tokenizer.encode(prompt, true)?;
    let prompt_ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();

    let mut engine1 = Engine::new(llama_model.clone());
    let sampler1 = Sampler::greedy();
    let mut logits1 = engine1.prefill(&prompt_ids, &model.weights)?;
    let mut out1 = String::new();

    for _ in 0..10 {
        let next = sampler1.sample(&logits1)?;
        out1.push_str(
            &tokenizer
                .decode_single(next as u32, true)
                .unwrap_or_default(),
        );
        logits1 = engine1.decode(next, &model.weights)?;
    }

    // Run 2
    let mut engine2 = Engine::new(llama_model.clone());
    let sampler2 = Sampler::greedy();
    let mut logits2 = engine2.prefill(&prompt_ids, &model.weights)?;
    let mut out2 = String::new();

    for _ in 0..10 {
        let next = sampler2.sample(&logits2)?;
        out2.push_str(
            &tokenizer
                .decode_single(next as u32, true)
                .unwrap_or_default(),
        );
        logits2 = engine2.decode(next, &model.weights)?;
    }

    // Hash comparison
    let mut hasher1 = Sha256::new();
    hasher1.update(out1.as_bytes());
    let hash1 = format!("{:x}", hasher1.finalize());

    let mut hasher2 = Sha256::new();
    hasher2.update(out2.as_bytes());
    let hash2 = format!("{:x}", hasher2.finalize());

    assert_eq!(out1, out2, "Outputs do not match exactly");
    assert_eq!(hash1, hash2, "SHA256 hashes do not match exactly");

    println!("Determinism verified: {} == {}", hash1, hash2);

    Ok(())
}
