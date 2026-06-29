//! F8 FFN Verification - Verify SwiGLU formula and FFN path
//!
//! Tests FFN path: norm → gate/up matmul → SwiGLU → down matmul
//! Compares GPU final layer output against oracle_hello_l0_checkpoints.csv
//!
//! F8 Formula: swiglu = silu(gate) ⊙ up where silu(x) = x / (1 + exp(-x))
//!
//! Run: cargo test --package airframe --test ffn_f8_verify --release -- --nocapture --ignored

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams};
use airframe::core::spec::ModelSpec;
use std::collections::HashMap;
use std::path::PathBuf;

/// Parse L0 checkpoint CSV
fn load_l0_checkpoints(path: &str) -> HashMap<String, (f32, Vec<f32>)> {
    let content = std::fs::read_to_string(path).expect("Failed to read L0 checkpoints");
    let mut checkpoints = HashMap::new();

    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 5 {
            let id = parts[0].to_string();
            let rms: f32 = parts[3].parse().unwrap_or(0.0);
            let first20: Vec<f32> = parts[4]
                .split('|')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            checkpoints.insert(id, (rms, first20));
        }
    }

    checkpoints
}

fn calc_rms(data: &[f32]) -> f32 {
    let sum_sq: f32 = data.iter().map(|x| x * x).sum();
    (sum_sq / data.len() as f32).sqrt()
}

#[tokio::test]
#[ignore]
async fn test_f8_gpu_ffn_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== F8 GPU FFN Verification ===\n");

    // Load oracle checkpoints
    // Path is relative to workspace root (two levels up from crates/airframe/)
    let oracle_path = if std::path::Path::new("fixtures/oracle_hello_l0_checkpoints.csv").exists() {
        "fixtures/oracle_hello_l0_checkpoints.csv".to_string()
    } else {
        "../../fixtures/oracle_hello_l0_checkpoints.csv".to_string()
    };
    let checkpoints = load_l0_checkpoints(&oracle_path);
    println!("[1/6] Loaded {} oracle checkpoints", checkpoints.len());

    // Initialize GPU
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No GPU adapter found");

    let adapter_limits = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter_limits.max_storage_buffer_binding_size as u64;
    limits.max_storage_buffers_per_shader_stage = 8;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await?;

    println!("[2/6] GPU initialized: {:?}", adapter.get_info().name);

    // Load model
    let model_path =
        PathBuf::from("D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    let gpu_model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    println!("[3/6] Model loaded");

    // Process tokens: [1 (BOS), 15043 ("Hello")]
    let tokens = [1u32, 15043u32];

    let dim = spec.n_embd as u32;
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
    let row_bytes = (dim / 32) * 18; // Q4_0 quantization

    let layer_params = LayerParams {
        dim,
        head_count: spec.n_head as u32,
        head_count_kv: spec.n_head_kv as u32,
        head_dim: (spec.n_embd / spec.n_head) as u32,
        rope_dim: spec.rope_dim as u32,
        rms_eps: spec.rms_eps,
        ffn_dim: 5632,
        temp_stride: 16384,
        quant_qk: 0,
        quant_v: 0,
        quant_attn_out: 0,
        quant_ffn_down: 0,
        quant_ffn_gate: 0,
        quant_ffn_up: 0,
        attn_logit_softcap: 0.0,
        post_norm_enabled: 0,
        qk_norm_enabled: 0,
        layer_norm_enabled: 0,
        ffn_kind_policy: 0,
        qkv_layout_policy: 0,
        batch_offset: 0,
        batch_count: 1,
        q_weight_k: 0,
        k_weight_k: 0,
    };

    let mut kv_cache = KVCache::new(&device, 22, 4, 64, 2048);

    // Process BOS token first (position 0)
    println!("\n[4/6] Processing BOS token (id=1, pos=0)...");
    let bos_embedding_offset = embd_weight_offset + (tokens[0] as u64 * row_bytes as u64);
    let mut layer_output = pipeline.run_dequant_request(
        &device,
        &queue,
        &gpu_model,
        bos_embedding_offset as u32,
        dim,
    );

    let layer_offsets = gpu_model
        .metadata
        .get_layer_offsets(0, "tinyllama")
        .expect("Layer 0 not found");
    pipeline.run_layer_with_cache(
        &device,
        &queue,
        &gpu_model,
        &mut kv_cache,
        0,
        &layer_output,
        layer_offsets,
        layer_params,
    );
    let _ = kv_cache.increment();

    // Process "Hello" token (position 1) - this is what we'll verify (Layer 0 final output)
    println!("[5/6] Processing \"Hello\" token (id=15043, pos=1)...");
    let hello_embedding_offset = embd_weight_offset + (tokens[1] as u64 * row_bytes as u64);
    layer_output = pipeline.run_dequant_request(
        &device,
        &queue,
        &gpu_model,
        hello_embedding_offset as u32,
        dim,
    );

    // Run layer normally (FFN is integrated in the full layer pass)
    let layer_offsets = gpu_model
        .metadata
        .get_layer_offsets(0, "tinyllama")
        .expect("Layer 0 not found");
    layer_output = pipeline.run_layer_with_cache(
        &device,
        &queue,
        &gpu_model,
        &mut kv_cache,
        0,
        &layer_output,
        layer_offsets,
        layer_params,
    );

    println!("\n[6/6] Verification Against Oracle");
    println!("{}", "=".repeat(70));

    // Verify final layer output (L0.21) - this validates entire FFN path
    let (oracle_l21_rms, oracle_l21_first20) =
        checkpoints.get("L0.21").expect("L0.21 (l_out) not found");
    let shimmy_l21_rms = calc_rms(&layer_output);

    println!("\n✅ Final Layer Output Verification (L0.21 l_out):");
    println!("  Oracle RMS: {:.8}", oracle_l21_rms);
    println!("  Shimmy RMS: {:.8}", shimmy_l21_rms);
    println!(
        "  RMS diff:   {:.8}",
        (oracle_l21_rms - shimmy_l21_rms).abs()
    );
    println!("  Oracle first 10: {:?}", &oracle_l21_first20[..10]);
    println!("  Shimmy first 10: {:?}", &layer_output[..10]);

    let mut max_diff = 0.0f32;
    for i in 0..20.min(oracle_l21_first20.len()) {
        let diff = (oracle_l21_first20[i] - layer_output[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("  Max element diff (first 20): {:.8}", max_diff);

    // Use same tolerance as F6/F7 tests (1e-3 RMS, 1e-2 element-wise)
    let l21_pass = (oracle_l21_rms - shimmy_l21_rms).abs() < 1e-3 && max_diff < 1e-2;

    println!("\n=== VERDICT ===");
    println!(
        "Final Layer Output: {}",
        if l21_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    if l21_pass {
        println!("\n🎉 F8 FFN Verification PASSED!");
        println!("   GPU FFN path (incl. SwiGLU) produces correct final layer output");
        println!("   RMS matches llama.cpp oracle within 1e-3 tolerance");
        println!("\n   NOTE: This validates the complete FFN path:");
        println!("   - FFN RMSNorm");
        println!("   - Gate/Up matmuls");
        println!("   - F8 SwiGLU activation: silu(gate) ⊙ up");
        println!("   - Down matmul");
        println!("   - Residual addition");
    } else {
        panic!("Layer output tolerance exceeded");
    }

    Ok(())
}
