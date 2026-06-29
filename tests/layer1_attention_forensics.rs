//! Layer 1 attention forensics on canonical 2-token case (BOS + "Hello")
//! Run:
//! cargo test --package airframe --test layer1_attention_forensics --release -- --nocapture --ignored

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams};
use airframe::core::spec::ModelSpec;
use std::collections::HashMap;
use std::path::PathBuf;

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

fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let max_v = a.max(b);
    let ea = (a - max_v).exp();
    let eb = (b - max_v).exp();
    let sum = ea + eb;
    (ea / sum, eb / sum)
}

fn calc_rms(data: &[f32]) -> f32 {
    let sum_sq: f32 = data.iter().map(|x| x * x).sum();
    (sum_sq / data.len() as f32).sqrt()
}

#[tokio::test]
#[ignore]
async fn layer1_attention_forensics() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Layer 1 Attention Forensics (BOS + Hello) ===\n");

    let oracle_path =
        if std::path::Path::new("crates/airframe/fixtures/oracle_22layer_hello.csv").exists() {
            "crates/airframe/fixtures/oracle_22layer_hello.csv".to_string()
        } else if std::path::Path::new("fixtures/oracle_22layer_hello.csv").exists() {
            "fixtures/oracle_22layer_hello.csv".to_string()
        } else {
            "../../fixtures/oracle_22layer_hello.csv".to_string()
        };
    let oracle = load_22layer_oracle(&oracle_path);

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

    let model_path =
        PathBuf::from("D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    let gpu_model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    let dim = spec.n_embd as u32;
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

    let tokens = [1u32, 15043u32];
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
    let row_bytes = (dim / 32) * 18;

    // Prefill BOS through all layers
    let bos_offset = embd_weight_offset + (tokens[0] as u64 * row_bytes as u64);
    let mut hidden =
        pipeline.run_dequant_request(&device, &queue, &gpu_model, bos_offset as u32, dim);
    for layer_idx in 0..22 {
        let layer_offsets = gpu_model
            .metadata
            .get_layer_offsets(layer_idx, "tinyllama")
            .expect("Layer offsets missing");
        hidden = pipeline.run_layer_with_cache(
            &device,
            &queue,
            &gpu_model,
            &mut kv_cache,
            layer_idx,
            &hidden,
            layer_offsets,
            layer_params,
        );
    }
    let _ = kv_cache.increment();

    // Hello embedding
    let hello_offset = embd_weight_offset + (tokens[1] as u64 * row_bytes as u64);
    let hello_hidden =
        pipeline.run_dequant_request(&device, &queue, &gpu_model, hello_offset as u32, dim);

    // Layer 0 normal (populate cache at pos=1 for layer 0)
    let layer0_offsets = gpu_model
        .metadata
        .get_layer_offsets(0, "tinyllama")
        .expect("Layer 0 offsets missing");
    let layer0_out = pipeline.run_layer_with_cache(
        &device,
        &queue,
        &gpu_model,
        &mut kv_cache,
        0,
        &hello_hidden,
        layer0_offsets,
        layer_params,
    );

    // Layer 1 debug path: output + Q + cached K/V [pos0,pos1]
    let layer1_offsets = gpu_model
        .metadata
        .get_layer_offsets(1, "tinyllama")
        .expect("Layer 1 offsets missing");
    let (layer1_out, layer1_post_attn, layer1_ffn_down, q_vals, k_cache_vals, v_cache_vals) =
        pipeline.run_layer_with_cache_debug(
            &device,
            &queue,
            &gpu_model,
            &mut kv_cache,
            1,
            &layer0_out,
            layer1_offsets,
            layer_params,
        );

    println!(
        "Q len: {} | K cache len: {} | V cache len: {}",
        q_vals.len(),
        k_cache_vals.len(),
        v_cache_vals.len()
    );
    let post_attn_rms = calc_rms(&layer1_post_attn);
    let ffn_down_rms = calc_rms(&layer1_ffn_down);
    let layer1_out_rms = calc_rms(&layer1_out);
    println!(
        "GPU-POST-ATTN-L1: RMS {:.8}, first10: {:?}",
        post_attn_rms,
        &layer1_post_attn[..10]
    );
    println!(
        "GPU-FFN-DOWN-L1: RMS {:.8}, first10: {:?}",
        ffn_down_rms,
        &layer1_ffn_down[..10]
    );
    println!(
        "GPU-LAYER-OUT-L1: RMS {:.8}, first10: {:?}",
        layer1_out_rms,
        &layer1_out[..10]
    );

    // TinyLlama constants
    let head_dim = 64usize;
    let n_head = 32usize;
    let n_head_kv = 4usize;
    let gqa_ratio = n_head / n_head_kv; // 8

    // Focus scalar: head 0, dim 0 (maps to kv head 0)
    let q_head = 0usize;
    let kv_head = q_head / gqa_ratio;

    let q_base = q_head * head_dim;
    let mut dot0 = 0.0f32;
    let mut dot1 = 0.0f32;
    for d in 0..head_dim {
        let q = q_vals[q_base + d];
        let k0 = k_cache_vals[(kv_head * head_dim) + d];
        let k1 = k_cache_vals[(n_head_kv * head_dim) + (kv_head * head_dim) + d];
        dot0 += q * k0;
        dot1 += q * k1;
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt(); // 1/sqrt(64)
    let s0 = dot0 * scale;
    let s1 = dot1 * scale;
    let (w0, w1) = softmax2(s0, s1);

    let v0 = v_cache_vals[kv_head * head_dim];
    let v1 = v_cache_vals[(n_head_kv + kv_head) * head_dim];
    let context_d0 = w0 * v0 + w1 * v1;

    // GQA sanity: head 0 and head 8 should both map to kv_head 0 (ratio=8)
    let q_head_8 = 8usize;
    let kv_head_8 = q_head_8 / gqa_ratio;
    let q8_base = q_head_8 * head_dim;
    let mut dot0_h8 = 0.0f32;
    let mut dot1_h8 = 0.0f32;
    for d in 0..head_dim {
        let q = q_vals[q8_base + d];
        let k0 = k_cache_vals[(kv_head_8 * head_dim) + d];
        let k1 = k_cache_vals[(n_head_kv * head_dim) + (kv_head_8 * head_dim) + d];
        dot0_h8 += q * k0;
        dot1_h8 += q * k1;
    }

    println!("\n[Layer1 scalar forensic | head=0 dim=0]");
    println!("dot(pos0)={:.8}, dot(pos1)={:.8}", dot0, dot1);
    println!("scaled(pos0)={:.8}, scaled(pos1)={:.8}", s0, s1);
    println!("softmax(w0,w1)=({:.8}, {:.8})", w0, w1);
    println!("v(pos0)={:.8}, v(pos1)={:.8}", v0, v1);
    println!("context(dim0)={:.8}", context_d0);
    println!(
        "K sample pos0[:4]={:?} | pos1[:4]={:?}",
        &k_cache_vals[0..4],
        &k_cache_vals[256..260]
    );
    println!(
        "V sample pos0[:4]={:?} | pos1[:4]={:?}",
        &v_cache_vals[0..4],
        &v_cache_vals[256..260]
    );
    println!(
        "GQA check: head0->kv{} dots=({:.6},{:.6}) | head8->kv{} dots=({:.6},{:.6})",
        kv_head, dot0, dot1, kv_head_8, dot0_h8, dot1_h8
    );

    if let Some((_oracle_rms, oracle_first20)) = oracle.get(&1usize) {
        println!("\nOracle L1 first 5: {:?}", &oracle_first20[..5]);
        println!("GPU L1 first 5:    {:?}", &layer1_out[..5]);
        let max_diff = (0..20)
            .map(|i| (oracle_first20[i] - layer1_out[i]).abs())
            .fold(0.0f32, f32::max);
        println!("L1 max elem diff (first20): {:.8}", max_diff);
    }

    Ok(())
}
