//! 22-Layer GPU Verification - Validate all transformer layers
//!
//! Tests complete multi-layer inference path on GPU against CPU oracle.
//! Compares GPU output for all 22 layers against oracle_22layer_hello.csv
//! FAIL FAST: Stops on first layer that exceeds tolerance.
//!
//! Run: cargo test --package airframe --test gpu_22layer_verify --release -- --nocapture --ignored

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams};
use airframe::core::spec::ModelSpec;
use std::collections::HashMap;
use std::path::PathBuf;

/// Parse 22-layer oracle CSV
/// Format: layer_idx,RMS,first20_pipe_separated
fn load_22layer_oracle(path: &str) -> HashMap<usize, (f32, Vec<f32>)> {
    let content = std::fs::read_to_string(path).expect("Failed to read 22-layer oracle");
    let mut oracle = HashMap::new();

    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            let layer_idx: usize = parts[0].parse().expect("Invalid layer_idx");
            let rms: f32 = parts[1].parse().expect("Invalid RMS");
            let first20: Vec<f32> = parts[2]
                .split('|')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            oracle.insert(layer_idx, (rms, first20));
        }
    }

    oracle
}

fn calc_rms(data: &[f32]) -> f32 {
    let sum_sq: f32 = data.iter().map(|x| x * x).sum();
    (sum_sq / data.len() as f32).sqrt()
}

#[tokio::test]
#[ignore]
async fn test_gpu_22layer_verification() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 22-Layer GPU Verification ===\n");

    // Load oracle checkpoints
    let oracle_path =
        if std::path::Path::new("crates/airframe/fixtures/oracle_22layer_hello.csv").exists() {
            "crates/airframe/fixtures/oracle_22layer_hello.csv".to_string()
        } else if std::path::Path::new("fixtures/oracle_22layer_hello.csv").exists() {
            "fixtures/oracle_22layer_hello.csv".to_string()
        } else {
            "../../fixtures/oracle_22layer_hello.csv".to_string()
        };
    let oracle = load_22layer_oracle(&oracle_path);
    println!("[1/7] Loaded oracle for {} layers", oracle.len());

    // Oracle provenance sanity gate (TinyLlama BOS->Hello decode fixture)
    // This catches stale/non-cache-progressed fixtures before we trust parity numbers.
    let (l2_rms, _) = oracle.get(&2).expect("Oracle layer 2 missing");
    assert!(
        *l2_rms < 1.0,
        "Oracle appears stale or generated with incorrect cache progression: layer2 RMS={:.8}",
        l2_rms
    );

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

    println!("[2/7] GPU initialized: {:?}", adapter.get_info().name);

    // Load model
    let model_path =
        PathBuf::from("C:/Users/micha/repos/llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    let gpu_model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    println!("[3/7] Model loaded");

    // Setup layer parameters
    let dim = spec.n_embd as u32;
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

    // Process tokens: [1 (BOS), 15043 ("Hello")]
    let tokens = vec![1u32, 15043u32];
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
    let row_bytes = (dim / 32) * 18; // Q4_0 quantization

    // Process BOS token (position 0) through all 22 layers
    println!("\n[4/7] Processing BOS token (id=1, pos=0) through 22 layers...");
    let bos_embedding_offset = embd_weight_offset + (tokens[0] as u64 * row_bytes as u64);
    let mut layer_output = pipeline.run_dequant_request(
        &device,
        &queue,
        &gpu_model,
        bos_embedding_offset as u32,
        dim,
    );

    for layer_idx in 0..22 {
        let layer_offsets = gpu_model
            .metadata
            .get_layer_offsets(layer_idx, "tinyllama")
            .expect(&format!("Layer {} not found", layer_idx));
        layer_output = pipeline.run_layer_with_cache(
            &device,
            &queue,
            &gpu_model,
            &mut kv_cache,
            layer_idx,
            &layer_output,
            layer_offsets,
            layer_params,
        );
    }
    kv_cache.increment(); // Increment once after all layers process the token
    println!("      ✓ BOS processed through all 22 layers");

    // Process "Hello" token (position 1) - VERIFY THESE OUTPUTS
    println!("[5/7] Processing \"Hello\" token (id=15043, pos=1) through 22 layers...");
    let hello_embedding_offset = embd_weight_offset + (tokens[1] as u64 * row_bytes as u64);
    layer_output = pipeline.run_dequant_request(
        &device,
        &queue,
        &gpu_model,
        hello_embedding_offset as u32,
        dim,
    );

    println!("\n[6/7] Layer-by-Layer Verification");
    println!("{}", "=".repeat(90));

    let mut all_pass = true;
    let tolerance_rms = 1e-5f32;
    let tolerance_element = 1e-4f32;

    for layer_idx in 0..22 {
        let layer_offsets = gpu_model
            .metadata
            .get_layer_offsets(layer_idx, "tinyllama")
            .expect(&format!("Layer {} not found", layer_idx));

        // DIAGNOSTIC: Print input state for first 2 layers
        if layer_idx <= 1 {
            let input_rms = calc_rms(&layer_output);
            println!(
                "    [PRE-LAYER {}] Input RMS: {:.8}, first 5: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                layer_idx,
                input_rms,
                layer_output[0],
                layer_output[1],
                layer_output[2],
                layer_output[3],
                layer_output[4]
            );
            println!(
                "    [PRE-LAYER {}] Offsets: attn_q={}, attn_norm={}, layer_idx={}",
                layer_idx, layer_offsets.attn_q, layer_offsets.attn_norm, layer_offsets.padding[0]
            );

            // CRITICAL DIAGNOSTIC: Check if weights are being read correctly
            // If Layer 1 reads Layer 0's weights, Q projection will be similar
            // If Layer 1 reads its own weights, Q projection will be different
            println!(
                "    [DEBUG] Testing if Layer {} reads correct weights...",
                layer_idx
            );
            println!("    [DEBUG] Expected: attn_q offset should differ from Layer 0");
            println!("    [DEBUG] Layer 0 attn_q: 114468224, Layer 1 attn_q: 139257216");
            println!("    [DEBUG] If offsets match but outputs are garbage, the BUG is in shader weight loading!");
        }

        layer_output = pipeline.run_layer_with_cache(
            &device,
            &queue,
            &gpu_model,
            &mut kv_cache,
            layer_idx,
            &layer_output,
            layer_offsets,
            layer_params,
        );

        // DIAGNOSTIC: Read cache params from shader output (last 4 elements)
        if layer_idx <= 1 {
            let seq_len_shader = layer_output[2044] as u32;
            let current_pos_shader = layer_output[2045] as u32;
            let max_seq_shader = layer_output[2046] as u32;
            let layer_idx_shader = layer_output[2047] as u32;

            println!("   [SHADER READBACK] Layer {}:", layer_idx);
            println!(
                "      seq_len = {} (expected: {})",
                seq_len_shader,
                if layer_idx == 0 { 2 } else { 2 }
            );
            println!(
                "      current_pos = {} (expected: {})",
                current_pos_shader, 1
            );
            println!("      max_seq_len = {} (expected: 2048)", max_seq_shader);
            println!(
                "      layer_idx = {} (expected: {})",
                layer_idx_shader, layer_idx
            );
        }

        // DIAGNOSTIC: Print output for first 2 layers to see if weights changed
        if layer_idx <= 1 {
            let output_rms = calc_rms(&layer_output);
            println!("    [POST-LAYER {}] Output RMS: {:.8}, first 5: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
                layer_idx, output_rms,
                layer_output[0], layer_output[1], layer_output[2], layer_output[3], layer_output[4]);

            // ALGEBRAIC CHECK: Are activation values finite?
            let has_nan = layer_output.iter().any(|v| v.is_nan());
            let has_inf = layer_output.iter().any(|v| v.is_infinite());
            let all_zero = layer_output.iter().all(|v| *v == 0.0);

            if has_nan {
                println!(
                    "    [❌ CORRUPTION] Layer {} output contains NaN!",
                    layer_idx
                );
            }
            if has_inf {
                println!(
                    "    [❌ CORRUPTION] Layer {} output contains Inf!",
                    layer_idx
                );
            }
            if all_zero {
                println!(
                    "    [❌ CORRUPTION] Layer {} output is all zeros!",
                    layer_idx
                );
            }
        }

        // Compare against oracle
        let (oracle_rms, oracle_first20) = oracle
            .get(&layer_idx)
            .expect(&format!("Oracle for layer {} not found", layer_idx));

        let shimmy_rms = calc_rms(&layer_output);
        let rms_diff = (oracle_rms - shimmy_rms).abs();

        let mut max_element_diff = 0.0f32;
        for i in 0..20.min(oracle_first20.len()) {
            let diff = (oracle_first20[i] - layer_output[i]).abs();
            max_element_diff = max_element_diff.max(diff);
        }

        let layer_pass = rms_diff < tolerance_rms && max_element_diff < tolerance_element;

        // Print every layer, highlight failures
        let status = if layer_pass { "✅" } else { "❌" };
        println!("Layer {:2}: {} | RMS diff: {:.2e} | Max elem diff: {:.2e} | Oracle RMS: {:.8} | Shimmy RMS: {:.8}",
            layer_idx, status, rms_diff, max_element_diff, oracle_rms, shimmy_rms);

        if !layer_pass {
            println!("    ❌ FAIL: Layer {} exceeded tolerance", layer_idx);
            println!("       Oracle first 10: {:?}", &oracle_first20[..10]);
            println!("       Shimmy first 10: {:?}", &layer_output[..10]);
            all_pass = false;

            // DON'T fail fast - collect all failures
            // assert!(
            //     layer_pass,
            //     "Layer {} failed:\n  RMS diff {:.2e} (tolerance {:.2e})\n  Max element diff {:.2e} (tolerance {:.2e})",
            //     layer_idx, rms_diff, tolerance_rms, max_element_diff, tolerance_element
            // );
        }
    }
    kv_cache.increment(); // Increment once after all layers process the token

    println!("\n{}", "=".repeat(90));
    println!("[7/7] === VERDICT ===");

    if all_pass {
        println!("✅ ALL 22 LAYERS PASSED!");
        println!("   GPU inference matches CPU oracle within tolerance:");
        println!("   - RMS tolerance: {:.2e}", tolerance_rms);
        println!("   - Element-wise tolerance: {:.2e}", tolerance_element);
        println!("\n🎉 Multi-layer GPU verification COMPLETE!");
        println!("   Ready for Phase 3: Scientific Validation");
    } else {
        panic!("One or more layers failed verification");
    }

    Ok(())
}
