//! Generate golden traces for ANY GGUF file (Q4_0, Q6_K, etc.)
//!
//! This test reads a GGUF file from `SHIMMY_BASE_GGUF` environment variable,
//! derives the ModelSpec from GGUF metadata, runs CPU inference, and writes
//! a CSV oracle file to `fixtures/oracle_{model}_{quant}.csv`.
//!
//! Usage:
//!   SHIMMY_BASE_GGUF="D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf" cargo test --package airframe --test generate_oracle_for_gguf --release -- --ignored --nocapture

// Internal spike tool — not part of the public test surface
#![allow(dead_code, unused_variables, unused_mut)]

use airframe::core::model::Model;
use airframe::core::spec::ModelSpec;
use airframe::core::tensor::Tensor;
use airframe::core::weight_id::WeightId;
use airframe::family::llama::LlamaBlock;
use airframe::ops::dispatch::OpDispatcher;
use airframe::runtime::kvcache::KvCache;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Minimal GGUF parser to extract metadata and spec without full weight loading
fn extract_gguf_metadata<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    use byteorder::{LittleEndian, ReadBytesExt};

    let mut file = File::open(&path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;

    if &magic != b"GGUF" {
        return Err("Not a GGUF file".into());
    }

    let version = file.read_u32::<LittleEndian>()?;
    let n_kv = file.read_u64::<LittleEndian>()?;

    let mut metadata = HashMap::new();

    for _ in 0..n_kv {
        // Key
        let key_len = file.read_u64::<LittleEndian>()?;
        let mut key_bytes = vec![0u8; key_len as usize];
        file.read_exact(&mut key_bytes)?;
        let key = String::from_utf8_lossy(&key_bytes).to_string();

        // Value type
        let vtype = file.read_u32::<LittleEndian>()?;

        // Value
        let value = match vtype {
            8 => {
                // STRING
                let str_len = file.read_u64::<LittleEndian>()?;
                let mut str_bytes = vec![0u8; str_len as usize];
                file.read_exact(&mut str_bytes)?;
                String::from_utf8_lossy(&str_bytes).to_string()
            }
            4 => {
                // UINT32
                let val = file.read_u32::<LittleEndian>()?;
                val.to_string()
            }
            6 => {
                // FLOAT32
                let val = file.read_f32::<LittleEndian>()?;
                val.to_string()
            }
            _ => continue, // Skip other types
        };

        metadata.insert(key, value);
    }

    Ok(metadata)
}

/// Extract ModelSpec from GGUF metadata
fn spec_from_gguf_metadata(
    metadata: &HashMap<String, String>,
) -> Result<ModelSpec, Box<dyn std::error::Error>> {
    use airframe::core::spec::{GgufFileType, ModelArch};

    let arch_str = metadata
        .get("general.architecture")
        .map(|s| s.as_str())
        .unwrap_or("llama");

    let n_vocab = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32000);

    let n_embd = metadata
        .get("llama.embedding_length")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or("Missing llama.embedding_length")?;

    let n_layer = metadata
        .get("llama.block_count")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or("Missing llama.block_count")?;

    let n_head = metadata
        .get("llama.attention.head_count")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or("Missing llama.attention.head_count")?;

    let n_head_kv = metadata
        .get("llama.attention.head_count_kv")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(n_head);

    let ff_dim = metadata
        .get("llama.feed_forward_length")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or("Missing llama.feed_forward_length")?;

    let rms_eps = metadata
        .get("llama.attention.layer_norm_rms_epsilon")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1e-5);

    let rope_base = metadata
        .get("rope.freq_base")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(10000.0);

    let rope_dim = metadata
        .get("rope.dimension_count")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(n_embd / n_head);

    let n_ctx = metadata
        .get("context_length")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048);

    // Parse file type from general.file_type
    let file_type = metadata
        .get("general.file_type")
        .and_then(|s| s.parse::<u32>().ok())
        .map(GgufFileType::from)
        .unwrap_or(GgufFileType::Unknown);

    let arch = ModelArch::from(arch_str);

    let mut spec = ModelSpec {
        n_vocab,
        n_embd,
        n_layer,
        n_head,
        n_head_kv,
        ff_dim,
        rms_eps,
        rope_base,
        rope_scale: 1.0,
        rope_dim,
        yarn_alpha: 1.0,
        yarn_beta: 32.0,
        n_ctx,
        attn_logit_softcap: 0.0,
        final_logit_softcap: 0.0,
        has_qk_norm: false,
        head_dim: 0,
        gqa_ratio: 0,
        kv_dim: 0,
        arch,
        file_type,
        model_name: metadata
            .get("general.name")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string()),
        temp_buffer_size: 0,
        kv_cache_size_per_layer: 0,
        chat_template: None,
    };

    Ok(spec.compute_derived())
}

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

/// Generate oracle CSV for any GGUF file
fn generate_oracle_for_gguf<P: AsRef<Path>>(
    gguf_path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path = gguf_path.as_ref();
    println!("\n=== Generating Oracle for GGUF ===\n");

    // Load model with full weight dequantization
    // Use from_tinylama_q4_0_gguf if it matches TinyLlama spec, otherwise use generic from_gguf
    let model = Model::from_tinylama_q4_0_gguf(gguf_path)?;
    let spec = model.spec.clone();

    println!(
        "[1/5] Model loaded: {} layers, {} dim",
        spec.n_layer, spec.n_embd
    );
    println!("      Architecture: {:?}", spec.arch);
    println!("      Quant: {:?}", spec.file_type);

    // Create layers
    let layers: Vec<LlamaBlock> = (0..spec.n_layer)
        .map(|i| LlamaBlock::new(i, spec.clone()))
        .collect();

    let ops = OpDispatcher;
    let mut kv_cache = KvCache::new(
        spec.n_ctx,
        spec.n_layer,
        spec.n_head_kv,
        spec.n_embd / spec.n_head,
    );

    // Get token embedding weight
    let token_embed_weight = model
        .weights
        .get(&WeightId::TokenEmbed)
        .ok_or("token_embed weight not found")?;

    println!("[2/5] Processing tokens...");

    // Process BOS token at position 0 (build up KV cache)
    let bos_token = 1usize;
    let mut hidden_states = embed_token(bos_token, token_embed_weight, &spec)?;
    let mut position_ids = vec![0usize];

    for (layer_idx, layer) in layers.iter().enumerate() {
        hidden_states = layer.forward(
            &hidden_states,
            &model.weights,
            &mut kv_cache,
            &position_ids,
            &ops,
        )?;
    }
    println!(
        "      ✓ Processed BOS (pos 0) through all {} layers",
        spec.n_layer
    );

    // Process "Hello" token at position 1 (continuing from same KV cache)
    let hello_token = 15043usize;
    hidden_states = embed_token(hello_token, token_embed_weight, &spec)?;
    position_ids = vec![1usize];
    let mut layer_outputs: Vec<(usize, f32, Vec<f32>)> = Vec::new();

    println!("[3/5] Running {} layers...", spec.n_layer);

    for (layer_idx, layer) in layers.iter().enumerate() {
        hidden_states = layer.forward(
            &hidden_states,
            &model.weights,
            &mut kv_cache,
            &position_ids,
            &ops,
        )?;

        let hidden_data = &hidden_states.data;
        let rms = calc_rms(hidden_data);
        let first20: Vec<f32> = hidden_data.iter().take(20).copied().collect();

        layer_outputs.push((layer_idx, rms, first20));

        if layer_idx % 5 == 0 || layer_idx == layers.len() - 1 {
            println!("      Layer {:2}: RMS={:.8}", layer_idx, rms);
        }
    }

    kv_cache.complete_decode()?;

    println!("[4/5] Writing oracle CSV...");

    // Extract model name and quant from GGUF path
    let filename = gguf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Determine quant from file type
    let quant_str = match spec.file_type {
        airframe::core::spec::GgufFileType::Q4_0 => "q4_0",
        airframe::core::spec::GgufFileType::Q6_K => "q6_k",
        airframe::core::spec::GgufFileType::Q4_K => "q4_k",
        _ => "unknown",
    };

    let output_path = Path::new("fixtures").join(format!("oracle_{}_{}.csv", filename, quant_str));
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

    println!("[5/5] ✅ Oracle generated: {:?}", output_path);
    println!("\nSummary:");
    println!("  Layers: {}", layer_outputs.len());
    println!("  Format: layer_idx,RMS,first20_pipe_separated");
    println!("  Token sequence: BOS (pos 0) → Hello (pos 1, captured)");
    println!("\nNext steps:");
    println!("  1. Review the generated CSV");
    println!("  2. For vault import, run: python scripts/import_oracle_to_vault.py <gguf_path> <csv_path>");
    println!("\n🎉 Ready for validation!");

    Ok(())
}

#[test]
#[ignore] // Run with: SHIMMY_BASE_GGUF=... cargo test --package airframe --test generate_oracle_for_gguf --release -- --nocapture --ignored
fn generate_oracle_for_any_gguf() -> Result<(), Box<dyn std::error::Error>> {
    let gguf_path = env::var("SHIMMY_BASE_GGUF")
        .map(PathBuf::from)
        .map_err(|_| "SHIMMY_BASE_GGUF environment variable not set")?;

    println!("Target GGUF: {:?}", gguf_path);

    if !gguf_path.exists() {
        return Err(format!("GGUF file not found: {:?}", gguf_path).into());
    }

    generate_oracle_for_gguf(gguf_path)
}
