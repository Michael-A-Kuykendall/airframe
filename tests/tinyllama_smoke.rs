//! Bead P3 — TinyLlama end-to-end smoke (single fabric `generate` path).
//!
//! Loads TinyLlama q4_0 / q6_k and runs the single `generate()` entry point
//! (which delegates to the fabric `generate_isf`). Prints the generated text so
//! coherence can be judged by eye. This is a SMOKE, not the certification
//! authority (P2's algebraic audit is the gate).
//!
//! Run: cargo test --features isf --test tinyllama_smoke -- --nocapture

use airframe::runtime::gpu::{GpuRuntime, SamplingParams};
use std::path::Path;

const TINYLLAMA_Q4_0: &str =
    "D:/shimmy-test-models/gguf_collection/TinyLlama/TinyLlama-1.1B-Chat-v1.0/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf";
const TINYLLAMA_Q6_K: &str =
    "D:/shimmy-test-models/gguf_collection/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tinyllama-1.1b-chat-v1.0.Q6_K.gguf";

async fn smoke(path: &str, tag: &str) {
    if !Path::new(path).exists() {
        eprintln!("[smoke] SKIP {} (gguf not present)", tag);
        return;
    }
    let rt = GpuRuntime::load(Path::new(path))
        .await
        .unwrap_or_else(|e| panic!("[smoke] {} load failed: {}", tag, e));
    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        max_tokens: 32,
        seed: 0,
        extra_stop_tokens: vec![],
    };
    let out = rt
        .generate("The capital of France is", &params, None, None, None, None)
        .expect("generate");
    eprintln!("[smoke] {} output: {:?}", tag, out);
    assert!(
        !out.trim().is_empty(),
        "[smoke] {} produced empty output",
        tag
    );
}

#[tokio::test]
async fn tinyllama_q4_0_single_path_smoke() {
    smoke(TINYLLAMA_Q4_0, "tinyllama-q4_0").await;
}

#[tokio::test]
async fn tinyllama_q6_k_single_path_smoke() {
    smoke(TINYLLAMA_Q6_K, "tinyllama-q6_k").await;
}
