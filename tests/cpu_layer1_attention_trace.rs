//! CPU Layer 1 attention scalar trace (BOS + Hello)
//! Run:
//! cargo test --package airframe --test cpu_layer1_attention_trace --release -- --nocapture --ignored

use airframe::core::model::Model as CpuModel;
use airframe::core::spec::ModelSpec;
use airframe::family::llama::LlamaModel;
use airframe::runtime::engine::Engine;
use std::path::PathBuf;

#[test]
#[ignore]
fn cpu_layer1_attention_trace() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== CPU Layer1 Attention Trace (BOS + Hello) ===\n");

    let model_path =
        PathBuf::from("D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();

    let cpu_model = CpuModel::from_tinylama_q4_0_gguf(&model_path)?;
    let llama_model = LlamaModel::from_spec(spec);
    let mut engine = Engine::new(llama_model);

    std::env::set_var("LIBSHIMMY_TRACE_ATTENTION", "1");
    std::env::set_var("LIBSHIMMY_TRACE_ATTENTION_LAYER", "1");
    std::env::set_var("LIBSHIMMY_TRACE_POST_ATTN", "1");
    std::env::set_var("LIBSHIMMY_TRACE_POST_ATTN_LAYER", "1");
    std::env::set_var("LIBSHIMMY_TRACE_POST_ATTN_CACHELEN", "1");
    std::env::set_var("LIBSHIMMY_TRACE_FFN_OUT", "1");
    std::env::set_var("LIBSHIMMY_TRACE_FFN_OUT_LAYER", "1");
    std::env::set_var("LIBSHIMMY_TRACE_FFN_OUT_CACHELEN", "1");
    std::env::set_var("LIBSHIMMY_TRACE_LAYER_OUT", "1");
    std::env::set_var("LIBSHIMMY_TRACE_LAYER_OUT_LAYER", "1");
    std::env::set_var("LIBSHIMMY_TRACE_LAYER_OUT_CACHELEN", "1");

    // Canonical scenario: BOS prefill then decode Hello
    let _ = engine.prefill(&[1usize], &cpu_model.weights)?;
    let _ = engine.decode(15043usize, &cpu_model.weights)?;

    std::env::remove_var("LIBSHIMMY_TRACE_ATTENTION");
    std::env::remove_var("LIBSHIMMY_TRACE_ATTENTION_LAYER");
    std::env::remove_var("LIBSHIMMY_TRACE_POST_ATTN");
    std::env::remove_var("LIBSHIMMY_TRACE_POST_ATTN_LAYER");
    std::env::remove_var("LIBSHIMMY_TRACE_POST_ATTN_CACHELEN");
    std::env::remove_var("LIBSHIMMY_TRACE_FFN_OUT");
    std::env::remove_var("LIBSHIMMY_TRACE_FFN_OUT_LAYER");
    std::env::remove_var("LIBSHIMMY_TRACE_FFN_OUT_CACHELEN");
    std::env::remove_var("LIBSHIMMY_TRACE_LAYER_OUT");
    std::env::remove_var("LIBSHIMMY_TRACE_LAYER_OUT_LAYER");
    std::env::remove_var("LIBSHIMMY_TRACE_LAYER_OUT_CACHELEN");

    println!("\nCPU trace complete. Look for line prefix: CPU-ATTN-SCALAR L1 h0 d0");
    Ok(())
}
