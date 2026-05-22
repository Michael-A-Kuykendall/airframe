//! F6/F7 Attention Verification - Extract Q/K/V from GPU Layer 0
//!
//! Compares GPU Q/K/V projections against oracle_hello_l0_checkpoints.csv
//! Verifies GQA mapping (32 Q heads → 4 KV heads)
//!
//! Run: cargo test --package airframe --test attention_f6_f7_verify --release -- --nocapture --ignored

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
async fn test_f6_f7_gpu_attention_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== F6/F7 GPU Attention Verification ===\n");

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
        PathBuf::from("C:/Users/micha/repos/llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    let gpu_model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    println!("[3/6] Model loaded");

    // Process tokens: [1 (BOS), 15043 ("Hello")]
    let tokens = vec![1u32, 15043u32];

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
        rms_eps: spec.rms_eps,
        ffn_dim: 5632,
        temp_stride: 16384,
        quant_type: 0,
        attn_logit_softcap: 0.0,
        post_norm_enabled: 0,
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
    kv_cache.increment();

    // Process "Hello" token (position 1) - this is what we'll verify
    println!("[5/6] Processing \"Hello\" token (id=15043, pos=1)...");
    let hello_embedding_offset = embd_weight_offset + (tokens[1] as u64 * row_bytes as u64);
    layer_output = pipeline.run_dequant_request(
        &device,
        &queue,
        &gpu_model,
        hello_embedding_offset as u32,
        dim,
    );

    // Run layer with debug extraction to get Q/K/V
    let layer_offsets = gpu_model
        .metadata
        .get_layer_offsets(0, "tinyllama")
        .expect("Layer 0 not found");
    let (_layer_output_vec, _post_attn_vals, _ffn_down_vals, q_vals, k_vals, v_vals) = pipeline
        .run_layer_with_cache_debug(
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

    // Verify Q (2048 values = 32 heads × 64 dims)
    let (oracle_q_rms, oracle_q_first20) =
        checkpoints.get("L0.4b").expect("L0.4b (Q_rope) not found");
    let shimmy_q_rms = calc_rms(&q_vals);

    println!("\n✅ Q Verification (L0.4b Q_rope):");
    println!("  Oracle RMS: {:.8}", oracle_q_rms);
    println!("  Shimmy RMS: {:.8}", shimmy_q_rms);
    println!("  RMS diff:   {:.8}", (oracle_q_rms - shimmy_q_rms).abs());
    println!("  Oracle first 10: {:?}", &oracle_q_first20[..10]);
    println!("  Shimmy first 10: {:?}", &q_vals[..10]);

    let mut q_max_diff = 0.0f32;
    for i in 0..20.min(oracle_q_first20.len()) {
        let diff = (oracle_q_first20[i] - q_vals[i]).abs();
        q_max_diff = q_max_diff.max(diff);
    }
    println!("  Max element diff (first 20): {:.8}", q_max_diff);

    // Verify K (256 values = 4 KV heads × 64 dims)
    let (oracle_k_rms, oracle_k_first20) =
        checkpoints.get("L0.5b").expect("L0.5b (K_rope) not found");
    let shimmy_k_rms = calc_rms(&k_vals);

    println!("\n✅ K Verification (L0.5b K_rope):");
    println!("  Oracle RMS: {:.8}", oracle_k_rms);
    println!("  Shimmy RMS: {:.8}", shimmy_k_rms);
    println!("  RMS diff:   {:.8}", (oracle_k_rms - shimmy_k_rms).abs());
    println!("  Oracle first 10: {:?}", &oracle_k_first20[..10]);
    println!("  Shimmy first 10: {:?}", &k_vals[..10]);

    let mut k_max_diff = 0.0f32;
    for i in 0..20.min(oracle_k_first20.len()) {
        let diff = (oracle_k_first20[i] - k_vals[i]).abs();
        k_max_diff = k_max_diff.max(diff);
    }
    println!("  Max element diff (first 20): {:.8}", k_max_diff);

    // Verify V (256 values = 4 KV heads × 64 dims)
    let (oracle_v_rms, oracle_v_first20) = checkpoints.get("L0.6").expect("L0.6 (Vcur) not found");
    let shimmy_v_rms = calc_rms(&v_vals);

    println!("\n✅ V Verification (L0.6 Vcur):");
    println!("  Oracle RMS: {:.8}", oracle_v_rms);
    println!("  Shimmy RMS: {:.8}", shimmy_v_rms);
    println!("  RMS diff:   {:.8}", (oracle_v_rms - shimmy_v_rms).abs());
    println!("  Oracle first 10: {:?}", &oracle_v_first20[..10]);
    println!("  Shimmy first 10: {:?}", &v_vals[..10]);

    let mut v_max_diff = 0.0f32;
    for i in 0..20.min(oracle_v_first20.len()) {
        let diff = (oracle_v_first20[i] - v_vals[i]).abs();
        v_max_diff = v_max_diff.max(diff);
    }
    println!("  Max element diff (first 20): {:.8}", v_max_diff);

    // Verify GQA Architecture
    println!("\n✅ GQA Architecture Verification:");
    println!(
        "  Q shape: {} values = {} heads × {} dims",
        q_vals.len(),
        spec.n_head,
        spec.n_embd / spec.n_head
    );
    println!(
        "  K shape: {} values = {} heads × {} dims",
        k_vals.len(),
        spec.n_head_kv,
        spec.n_embd / spec.n_head
    );
    println!(
        "  V shape: {} values = {} heads × {} dims",
        v_vals.len(),
        spec.n_head_kv,
        spec.n_embd / spec.n_head
    );
    println!(
        "  GQA ratio: {} Q heads → {} KV heads ({} Q heads per KV head)",
        spec.n_head,
        spec.n_head_kv,
        spec.n_head / spec.n_head_kv
    );

    assert_eq!(
        q_vals.len(),
        2048,
        "Q should have 2048 values (32 heads × 64 dims)"
    );
    assert_eq!(
        k_vals.len(),
        256,
        "K should have 256 values (4 heads × 64 dims)"
    );
    assert_eq!(
        v_vals.len(),
        256,
        "V should have 256 values (4 heads × 64 dims)"
    );

    // Pass/Fail Verdict
    println!("\n=== VERDICT ===");
    // Use same tolerance as CPU tests comparing to llama.cpp (tests/qkv_projection_conformance.rs)
    // GPU vs CPU llama.cpp can have small differences due to FP precision, summation order, etc.
    let q_pass = (oracle_q_rms - shimmy_q_rms).abs() < 1e-3 && q_max_diff < 1e-2;
    let k_pass = (oracle_k_rms - shimmy_k_rms).abs() < 1e-3 && k_max_diff < 1e-2;
    let v_pass = (oracle_v_rms - shimmy_v_rms).abs() < 1e-3 && v_max_diff < 1e-2;

    println!(
        "Q Projection: {}",
        if q_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "K Projection: {}",
        if k_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "V Projection: {}",
        if v_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    assert!(q_pass, "Q projection tolerance exceeded");
    assert!(k_pass, "K projection tolerance exceeded");
    assert!(v_pass, "V projection tolerance exceeded");

    println!("\n🎉 F6/F7 GPU Attention Verification PASSED!");
    println!("   Q/K/V projections match llama.cpp oracle within 1e-3 RMS tolerance");
    println!("   GQA architecture correct (32 Q heads → 4 KV heads)");

    println!("\n[6/6] Test complete!");

    Ok(())
}
