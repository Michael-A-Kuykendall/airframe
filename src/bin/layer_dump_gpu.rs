// Cross-platform: suppress macOS clippy 1.86+ lints
#![allow(
    unknown_lints,
    clippy::manual_is_multiple_of,
    clippy::collapsible_match
)]

// Layer Dump Tool: Capture all 22 layer outputs for algebraic verification
// Phase 2.1: Setup layer dump infrastructure

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams};
use airframe::core::spec::ModelSpec;
use serde::Serialize;
use shimmytok::Tokenizer;
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
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    let gpu_model = BindlessModel::load_from_disk(&device, &PathBuf::from(model_path), Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    eprintln!("[Layer Dump] Model loaded to VRAM");

    // === Tokenize ===
    let prompt_tokens = tokenizer.encode(prompt, true)?;
    eprintln!(
        "[Layer Dump] Tokens: {:?} ({} tokens)",
        prompt_tokens,
        prompt_tokens.len()
    );

    // === Setup ===
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
        ffn_dim: spec.ff_dim as u32,
        temp_stride: spec.temp_buffer_size as u32,
        quant_type: 0,
        attn_logit_softcap: 0.0,
        post_norm_enabled: 0,
        qk_norm_enabled: 0,
        layer_norm_enabled: 0,
        ffn_kind_policy: 0,
        qkv_layout_policy: 0,
    };

    let mut kv_cache = KVCache::new(
        &device, 22,   // n_layers
        4,    // n_head_kv (GQA)
        64,   // head_dim
        2048, // max_seq_len
    );

    let mut layers = Vec::new();

    // === Process First Token Only (for dump) ===
    let token_id = prompt_tokens[0];
    eprintln!("[Layer Dump] Processing token {} (id={})", 0, token_id);

    // Get embedding
    let row_offset = embd_weight_offset + (token_id as u64 * row_bytes as u64);
    let mut layer_output =
        pipeline.run_dequant_request(&device, &queue, &gpu_model, row_offset as u32, dim);

    // Capture embedding layer (Layer 0 input)
    layers.push(LayerOutput {
        layer_idx: 0,
        token_id,
        position: 0,
        stats: LayerStats::compute(&layer_output),
        hidden_states: layer_output.clone(),
    });

    // === Run All 22 Layers ===
    for layer_idx in 0..22 {
        let layer_offsets = gpu_model
            .metadata
            .get_layer_offsets(layer_idx, "tinyllama")
            .unwrap_or_else(|| panic!("Layer {} offsets not found", layer_idx));

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
