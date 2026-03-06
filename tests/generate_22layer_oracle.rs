//! Generate 22-layer oracle checkpoints for GPU validation
//!
//! This test runs CPU Airframe through all 22 transformer layers
//! and captures the final hidden state after each layer.
//!
//! Output: fixtures/oracle_22layer_hello.csv
//! Format: layer_idx,RMS,first20_values
//!
//! Run: cargo test --package airframe --test generate_22layer_oracle --release -- --nocapture --ignored

use airframe::core::model::Model;
use airframe::core::spec::ModelSpec;
use airframe::core::tensor::Tensor;
use airframe::core::weight_id::WeightId;
use airframe::family::llama::LlamaBlock;
use airframe::ops::dispatch::OpDispatcher;
use airframe::runtime::kvcache::KvCache;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn calc_rms(values: &[f32]) -> f32 {
    let sum_sq: f32 = values.iter().map(|x| x * x).sum();
    (sum_sq / values.len() as f32).sqrt()
}

fn embed_token(
    token_id: usize,
    embed_weight: &Tensor,
    spec: &ModelSpec,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let hidden_size = spec.n_embd;
    let start_idx = token_id * hidden_size;
    let end_idx = start_idx + hidden_size;

    let embedding_data = embed_weight.data[start_idx..end_idx].to_vec();

    Ok(Tensor {
        data: embedding_data,
        shape: vec![1, hidden_size],
    })
}

#[test]
#[ignore] // Run with: cargo test --package airframe --test generate_22layer_oracle --release -- --nocapture --ignored
fn generate_22layer_oracle_checkpoints() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Generating 22-Layer Oracle Checkpoints ===\n");

    // Load TinyLlama model
    let model_path =
        PathBuf::from("C:/Users/micha/repos/llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    println!("[1/6] Loading model: {:?}", model_path);

    let model_data = Model::from_tinylama_q4_0_gguf(&model_path)?;
    let spec = model_data.spec.clone();

    println!(
        "[2/6] Model loaded: {} layers, {} dim",
        spec.n_layer, spec.n_embd
    );

    // Create layers
    let layers: Vec<LlamaBlock> = (0..spec.n_layer)
        .map(|i| LlamaBlock::new(i, spec.clone()))
        .collect();

    let ops = OpDispatcher::default();
    let mut kv_cache = KvCache::new(
        spec.n_ctx,                // max_seq_len
        spec.n_layer,              // n_layer
        spec.n_head_kv,            // n_head_kv
        spec.n_embd / spec.n_head, // head_dim
    );

    // Get token embedding weight
    let token_embed_weight = model_data
        .weights
        .get(&WeightId::TokenEmbed)
        .ok_or("token_embed weight not found")?;

    println!("[3/6] Processing tokens...");

    // Process BOS token (1) at position 0
    let bos_token = 1usize;
    let mut hidden_states = embed_token(bos_token, token_embed_weight, &spec)?;
    println!("      ✓ BOS embedding: shape {:?}", hidden_states.shape);

    let position_ids = vec![0usize];
    for (layer_idx, layer) in layers.iter().enumerate() {
        hidden_states = layer.forward(
            &hidden_states,
            &model_data.weights,
            &mut kv_cache,
            &position_ids,
            &ops,
        )?;
    }
    kv_cache.complete_decode()?;
    println!("      ✓ Processed BOS through all 22 layers");

    // Process "Hello" token (15043) at position 1 - CAPTURE THESE OUTPUTS
    let hello_token = 15043usize;
    let mut hidden_states = embed_token(hello_token, token_embed_weight, &spec)?;
    println!("[4/6] Processing Hello token (15043) at position 1...");

    let position_ids = vec![1usize];
    let mut layer_outputs: Vec<(usize, f32, Vec<f32>)> = Vec::new();

    for (layer_idx, layer) in layers.iter().enumerate() {
        hidden_states = layer.forward(
            &hidden_states,
            &model_data.weights,
            &mut kv_cache,
            &position_ids,
            &ops,
        )?;

        // Extract final output for this layer
        let hidden_data = &hidden_states.data;
        let rms = calc_rms(hidden_data);
        let first20: Vec<f32> = hidden_data.iter().take(20).copied().collect();

        layer_outputs.push((layer_idx, rms, first20));

        if layer_idx % 5 == 0 || layer_idx == layers.len() - 1 {
            println!(
                "      Layer {:2}: RMS={:.8}, first_val={:.8}",
                layer_idx, rms, hidden_data[0]
            );
        }
    }

    println!("[5/6] Writing oracle checkpoints...");

    // Write to CSV
    let output_path = PathBuf::from("fixtures/oracle_22layer_hello.csv");
    let mut file = File::create(&output_path)?;

    writeln!(file, "layer_idx,RMS,first20")?;

    for (layer_idx, rms, first20) in &layer_outputs {
        let first20_str = first20
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join("|");

        writeln!(file, "{},{:.8},{}", layer_idx, rms, first20_str)?;
    }

    println!("[6/6] ✅ Oracle generated: {:?}", output_path);
    println!("\nSummary:");
    println!("  Layers: {}", layer_outputs.len());
    println!("  Format: layer_idx,RMS,first20_pipe_separated");
    println!("  Token sequence: BOS (pos 0) → Hello (pos 1, captured)");
    println!("\n🎉 Ready for GPU validation!");

    Ok(())
}
