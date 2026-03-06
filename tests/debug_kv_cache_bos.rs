//! Debug test: Verify KV cache is written correctly during BOS pass
//!
//! Run: cargo test --package airframe --test debug_kv_cache_bos --release -- --nocapture --ignored

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams};
use airframe::core::spec::ModelSpec;
use std::path::PathBuf;

#[tokio::test]
#[ignore]
async fn test_kv_cache_after_bos() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== KV Cache BOS Write Verification ===\n");

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

    println!("[1/5] GPU initialized");

    // Load model
    let model_path =
        PathBuf::from("C:/Users/micha/repos/llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    let gpu_model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    println!("[2/5] Model loaded");

    let dim = spec.n_embd as u32;
    let layer_params = LayerParams {
        dim,
        head_count: spec.n_head as u32,
        head_count_kv: spec.n_head_kv as u32,
        head_dim: (spec.n_embd / spec.n_head) as u32,
        rms_eps: spec.rms_eps,
        ffn_dim: 5632,
        temp_stride: 16384,
        padding: 0,
    };

    let mut kv_cache = KVCache::new(&device, 22, 4, 64, 2048);

    // Process BOS token (id=1, pos=0)
    let token_bos = 1u32;
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
    let row_bytes = (dim / 32) * 18;

    println!("\n[3/5] Processing BOS token (id=1) through 22 layers...");
    let bos_embedding_offset = embd_weight_offset + (token_bos as u64 * row_bytes as u64);
    let mut layer_output = pipeline.run_dequant_request(
        &device,
        &queue,
        &gpu_model,
        bos_embedding_offset as u32,
        dim,
    );

    // Process through all 22 layers
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
    kv_cache.increment();
    println!(
        "      ✓ BOS processed, cache seq_len = {}",
        kv_cache.get_seq_len()
    );

    // Read back K cache from Layer 1 position 0 to verify it's not garbage
    println!("\n[4/5] Reading back Layer 1 K cache position 0...");

    let k_layer1 = kv_cache.get_k_buffer(1);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("K Cache Staging"),
        size: 256 * 4, // First 256 elements (1 position, 4 heads, 64 dims)
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Cache Readback"),
    });
    encoder.copy_buffer_to_buffer(k_layer1, 0, &staging, 0, 256 * 4);
    let idx = queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        })
        .unwrap();

    let data = slice.get_mapped_range();
    let k_vals: &[f32] = bytemuck::cast_slice(&data);

    println!("      Layer 1 K cache position 0, first 20 elements:");
    for (i, val) in k_vals.iter().take(20).enumerate() {
        println!("        K[{}] = {:.8}", i, val);
    }

    // Check if values are reasonable (not garbage)
    let mut has_non_zero = false;
    let mut all_finite = true;
    for val in k_vals.iter().take(256) {
        if *val != 0.0 {
            has_non_zero = true;
        }
        if !val.is_finite() {
            all_finite = false;
        }
    }

    println!("\n[5/5] === VERDICT ===");
    if !all_finite {
        println!("❌ FAIL: Cache contains NaN or Inf (uninitialized garbage!)");
    } else if !has_non_zero {
        println!("⚠️  WARN: Cache is all zeros (may be unwritten)");
    } else {
        println!("✅ PASS: Cache contains finite non-zero values");
    }

    Ok(())
}
