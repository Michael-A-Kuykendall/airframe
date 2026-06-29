// Layer Dump Tool: Capture all 22 layer outputs for algebraic verification
// Phase 2.1: Setup layer dump infrastructure

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams};
use serde::Serialize;
use shimmytok::Tokenizer;
use std::fs::File;
use std::path::PathBuf;

#[derive(Serialize)]
struct LayerOutput {
    layer_idx: usize,
    token_id: u32,
    position: usize,
    hidden_states: Vec<f32>, // 2048 dimensions
    stats: LayerStats,
}

#[derive(Serialize)]
struct LayerStats {
    min: f32,
    max: f32,
    mean: f32,
    std_dev: f32,
    first_10: Vec<f32>,
    last_10: Vec<f32>,
}

#[derive(Serialize)]
struct LayerDump {
    prompt: String,
    model: String,
    backend: String, // "gpu" or "cpu"
    layers: Vec<LayerOutput>,
}

impl LayerStats {
    fn compute(hidden_states: &[f32]) -> Self {
        let min = hidden_states.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = hidden_states
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f32 = hidden_states.iter().sum();
        let mean = sum / hidden_states.len() as f32;
        let variance: f32 = hidden_states
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / hidden_states.len() as f32;
        let std_dev = variance.sqrt();

        let first_10 = hidden_states.iter().take(10).copied().collect();
        let last_10 = hidden_states.iter().rev().take(10).rev().copied().collect();

        Self {
            min,
            max,
            mean,
            std_dev,
            first_10,
            last_10,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: layer_dump_gpu <model_path> <prompt> <output_json>");
        eprintln!("Example: layer_dump_gpu models/tinyllama.gguf \"Hello\" layers_gpu.json");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = &args[2];
    let output_path = &args[3];

    eprintln!("[Layer Dump] GPU Mode");
    eprintln!("[Layer Dump] Model: {}", model_path);
    eprintln!("[Layer Dump] Prompt: {}", prompt);

    // === GPU Initialization ===
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
    limits.max_storage_buffers_per_shader_stage =
        adapter_limits.max_storage_buffers_per_shader_stage;
    limits.max_compute_invocations_per_workgroup = 256;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("Failed to create GPU device");

    eprintln!(
        "[Layer Dump] GPU initialized: {:?}",
        adapter.get_info().name
    );

    // === Load Model ===
    let tokenizer = Tokenizer::from_gguf_file(model_path)?;

    // Read metadata first to get correct spec for preflight
    let mut meta_file = File::open(model_path)?;
    let meta = BindlessMetadata::new(&mut meta_file);
    drop(meta_file);
    let spec = meta.to_model_spec();

    eprintln!("[Layer Dump] Spec: n_embd={}, n_head={}, n_head_kv={}, head_dim={}, rope_dim={}, n_ctx={}, ffn_dim={}, rms_eps={}, has_qk_norm={}, attn_logit_softcap={}",
        spec.n_embd, spec.n_head, spec.n_head_kv, spec.n_embd / spec.n_head, spec.rope_dim, spec.n_ctx, spec.ff_dim, spec.rms_eps,
        spec.has_qk_norm, spec.attn_logit_softcap);

    let gpu_model = BindlessModel::load_from_disk(&device, &PathBuf::from(model_path), Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    let n_layers = gpu_model.metadata.compiled_layers.len();
    eprintln!("[Layer Dump] Model loaded to VRAM ({}, {} layers)", spec.model_name, n_layers);

    // === Tokenize ===
    let prompt_tokens = tokenizer.encode(prompt, true)?;
    eprintln!(
        "[Layer Dump] Tokens: {:?} ({} tokens)",
        prompt_tokens,
        prompt_tokens.len()
    );

    // === Setup ===
    let dim = spec.n_embd as u32;
    let embd_quant_type = gpu_model
        .metadata
        .get_tensor_type("token_embd.weight")
        .unwrap_or(2); // default Q4_0
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");

    let embd_row_bytes = match embd_quant_type {
        0 => dim * 4,                                  // F32
        1 => dim * 2,                                  // F16
        2 => (dim / 32) * 18,                          // Q4_0
        6 => (dim / 32) * 22,                          // Q5_0
        8 => (dim / 32) * 34,                          // Q8_0
        12 => (dim / 256) * 144,                       // Q4_K
        13 => (dim / 256) * 176,                       // Q5_K
        14 => (dim / 256) * 210,                       // Q6_K
        _ => panic!("unsupported embedding quant type: {}", embd_quant_type),
    };

    let head_dim = (spec.n_embd / spec.n_head) as u32;
    let rope_dim = if spec.rope_dim > 0 { spec.rope_dim as u32 } else { head_dim };

    let layer_params_base = LayerParams {
        dim,
        head_count: spec.n_head as u32,
        head_count_kv: spec.n_head_kv as u32,
        head_dim,
        rope_dim,
        rms_eps: spec.rms_eps,
        ffn_dim: spec.ff_dim as u32,
        temp_stride: spec.temp_buffer_size as u32,
        quant_qk: 0,
        quant_v: 0,
        quant_attn_out: 0,
        quant_ffn_down: 0,
        quant_ffn_gate: 0,
        quant_ffn_up: 0,
        attn_logit_softcap: spec.attn_logit_softcap,
        post_norm_enabled: spec.post_norm_enabled as u32,
        qk_norm_enabled: spec.has_qk_norm as u32,
        layer_norm_enabled: spec.uses_layer_norm() as u32,
        ffn_kind_policy: 0,
        qkv_layout_policy: 0,
        batch_offset: 0,
        batch_count: 0,
        q_weight_k: 0,
        k_weight_k: 0,
    };

    let mut kv_cache = KVCache::new(
        &device,
        n_layers,
        spec.n_head_kv as u32,
        head_dim,
        spec.n_ctx as u32,
    );

    let mut layers = Vec::new();

    // === Process First Token Only (for dump) ===
    let token_id = prompt_tokens[0];
    eprintln!("[Layer Dump] Processing token {} (id={})", 0, token_id);

    // Get embedding (handles all quant types via dequant_any_hot)
    let row_offset = embd_weight_offset + (token_id as u64 * embd_row_bytes as u64);
    let mut layer_output =
        pipeline.run_dequant_any_hot(&device, &queue, &gpu_model, row_offset as u32, dim, embd_quant_type);

    eprintln!("[Layer Dump] Embedding complete (min={:.6}, max={:.6}, mean={:.6})",
        layer_output.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        layer_output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        layer_output.iter().sum::<f32>() / layer_output.len() as f32);

    // Capture embedding layer (Layer 0 input)
    layers.push(LayerOutput {
        layer_idx: 0,
        token_id,
        position: 0,
        stats: LayerStats::compute(&layer_output),
        hidden_states: layer_output.clone(),
    });

    // === Run All Layers ===
    for layer_idx in 0..n_layers {
        let compiled = &gpu_model.metadata.compiled_layers[layer_idx];

        let layer_offsets = gpu_model
            .metadata
            .get_layer_offsets(layer_idx, spec.arch_string())
            .unwrap_or_else(|| panic!("Layer {} offsets not found", layer_idx));

        let layer_params = LayerParams {
            quant_qk: compiled.quant_qk,
            quant_v: compiled.quant_v,
            quant_attn_out: compiled.quant_attn_out,
            quant_ffn_down: compiled.quant_ffn_down,
            quant_ffn_gate: compiled.quant_ffn_gate,
            quant_ffn_up: compiled.quant_ffn_up,
            ..layer_params_base
        };

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

        // Capture layer output
        layers.push(LayerOutput {
            layer_idx: layer_idx + 1, // +1 because embedding is layer 0
            token_id,
            position: 0,
            stats: LayerStats::compute(&layer_output),
            hidden_states: layer_output.clone(),
        });

        eprintln!(
            "[Layer Dump] Layer {} complete (min={:.6}, max={:.6}, mean={:.6})",
            layer_idx + 1,
            layers.last().unwrap().stats.min,
            layers.last().unwrap().stats.max,
            layers.last().unwrap().stats.mean
        );
    }

    // === Save JSON ===
    let dump = LayerDump {
        prompt: prompt.to_string(),
        model: model_path.to_string(),
        backend: "gpu".to_string(),
        layers,
    };

    let json = serde_json::to_string_pretty(&dump)?;
    std::fs::write(output_path, json)?;

    eprintln!(
        "[Layer Dump] Saved {} layer outputs to {}",
        dump.layers.len(),
        output_path
    );
    eprintln!("[Layer Dump] Complete!");

    Ok(())
}
