// Layer Dump Tool: Capture all 22 layer outputs for algebraic verification
// Phase 2.1: Setup layer dump infrastructure

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::inference::layer_dump_drain;
use airframe::backend::bindless::pipeline::BindlessPipeline;
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
    let mut spec = meta.to_model_spec();
    // Cap KV VRAM like production (RTX 3060)
    if let Ok(max_ctx) = std::env::var("SHIMMY_MAX_CTX") {
        if let Ok(n) = max_ctx.parse::<usize>() {
            spec.n_ctx = n;
            spec = spec.compute_derived();
        }
    } else if spec.n_ctx > 8192 {
        spec.n_ctx = 8192;
        spec = spec.compute_derived();
    }

    // MUST use spec.head_dim (Q-weight-inferred for Qwen3/Gemma), NEVER n_embd/n_head.
    // Qwen3-4B: n_embd/n_head=80 but real head_dim=128 (attn_q [2560,4096]).
    let head_dim = spec.head_dim as u32;
    let rope_dim = if spec.rope_dim > 0 {
        spec.rope_dim as u32
    } else {
        head_dim
    };

    eprintln!(
        "[Layer Dump] Spec: n_embd={}, n_head={}, n_head_kv={}, head_dim={} (n_embd/n_head={}), rope_dim={}, n_ctx={}, ffn_dim={}, rms_eps={}, has_qk_norm={}, post_norm_enabled={}, attn_logit_softcap={}",
        spec.n_embd,
        spec.n_head,
        spec.n_head_kv,
        head_dim,
        spec.n_embd / spec.n_head.max(1),
        rope_dim,
        spec.n_ctx,
        spec.ff_dim,
        spec.rms_eps,
        spec.has_qk_norm,
        spec.post_norm_enabled,
        spec.attn_logit_softcap
    );
    if head_dim as usize != spec.n_embd / spec.n_head.max(1) {
        eprintln!(
            "[Layer Dump] NOTE: padded head_dim={} != n_embd/n_head={} (correct for Qwen3/Gemma)",
            head_dim,
            spec.n_embd / spec.n_head.max(1)
        );
    }

    let gpu_model = BindlessModel::load_from_disk(&device, &PathBuf::from(model_path), Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    let n_layers = gpu_model.metadata.compiled_layers.len();
    eprintln!(
        "[Layer Dump] Model loaded to VRAM ({}, {} layers)",
        spec.model_name, n_layers
    );

    // === Tokenize (MULTI-TOKEN — single-token hides cross-attention bugs) ===
    let prompt_tokens = tokenizer.encode(prompt, true)?;
    eprintln!(
        "[Layer Dump] Tokens: {:?} ({} tokens)",
        prompt_tokens,
        prompt_tokens.len()
    );
    if prompt_tokens.len() < 2 {
        eprintln!(
            "[Layer Dump] WARNING: prompt tokenized to {} token(s). Use a multi-token prompt.",
            prompt_tokens.len()
        );
    }

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
        0 => dim * 4,            // F32
        1 => dim * 2,            // F16
        2 => (dim / 32) * 18,    // Q4_0
        6 => (dim / 32) * 22,    // Q5_0
        8 => (dim / 32) * 34,    // Q8_0
        12 => (dim / 256) * 144, // Q4_K
        13 => (dim / 256) * 176, // Q5_K
        14 => (dim / 256) * 210, // Q6_K
        _ => panic!("unsupported embedding quant type: {}", embd_quant_type),
    };

    let kv_cache = KVCache::new(
        &device,
        n_layers,
        spec.n_head_kv as u32,
        head_dim,
        spec.n_ctx as u32,
    );
    eprintln!(
        "[Layer Dump] KVCache: head_dim={} n_head_kv={} n_ctx={}",
        head_dim, spec.n_head_kv, spec.n_ctx
    );

    // === Build fused batch embedding (all prompt tokens in ONE prefill) ===
    // Fused multi-token prefill (batch>1) is mandatory: with batch_size==1 the
    // layer-boundary capture sees the pre-layer residual (documented phantom,
    // inference.rs:2017). layer_dump must use the production prefill path, not
    // a per-token loop (q1c requirement 4).
    let last_pos = prompt_tokens.len() - 1;
    let last_token_id = prompt_tokens[last_pos];
    let mut input_embd: Vec<f32> = Vec::with_capacity(prompt_tokens.len() * dim as usize);
    let mut emb_last: Vec<f32> = Vec::new();
    for &token_id in &prompt_tokens {
        let row_offset = embd_weight_offset + (token_id as u64 * embd_row_bytes as u64);
        let row = pipeline.run_dequant_any_hot(
            &device,
            &queue,
            &gpu_model,
            row_offset as u32,
            dim,
            embd_quant_type,
        );
        input_embd.extend_from_slice(&row);
        emb_last = row;
    }
    let st_emb = LayerStats::compute(&emb_last);
    eprintln!(
        "[Layer Dump] Embedding complete (min={:.6}, max={:.6}, mean={:.6})",
        st_emb.min, st_emb.max, st_emb.mean
    );

    // === Fused multi-token prefill (production path, forces layer-boundary yield) ===
    std::env::set_var("AIRFRAME_LAYER_DUMP_CAPTURE", "1");
    pipeline.run_full_model_prefill_chunked_with_cache_state(
        &device,
        &queue,
        &gpu_model,
        &input_embd,
        None,
        0,
        Some((kv_cache.get_k_buffers(), kv_cache.get_v_buffers())),
        &spec,
        512,
    )?;
    std::env::remove_var("AIRFRAME_LAYER_DUMP_CAPTURE");

    // === Collect captures (embedding = layer 0, transformer layers 1..=N) ===
    let mut layers = Vec::new();
    let mut first_nan_layer: Option<usize> = None;
    layers.push(LayerOutput {
        layer_idx: 0,
        token_id: last_token_id,
        position: last_pos,
        stats: st_emb,
        hidden_states: emb_last,
    });
    for state in layer_dump_drain() {
        if state.position != last_pos as u32 {
            continue;
        }
        let st = LayerStats::compute(&state.hidden_states);
        if first_nan_layer.is_none() && (st.min.is_nan() || st.max.is_nan() || st.mean.is_nan()) {
            first_nan_layer = Some(state.layer_idx as usize);
        }
        layers.push(LayerOutput {
            layer_idx: state.layer_idx as usize + 1,
            token_id: last_token_id,
            position: last_pos,
            stats: st,
            hidden_states: state.hidden_states,
        });
        eprintln!(
            "[Layer Dump] Layer {} @pos{} complete (min={:.6}, max={:.6}, mean={:.6})",
            layers.last().unwrap().layer_idx,
            last_pos,
            layers.last().unwrap().stats.min,
            layers.last().unwrap().stats.max,
            layers.last().unwrap().stats.mean
        );
    }

    if let Some(l) = first_nan_layer {
        eprintln!(
            "[Layer Dump] FIRST_NAN_LAYER={} (0-based transformer index) at last prompt position",
            l
        );
    } else {
        eprintln!("[Layer Dump] No NaN in captured last-position layer outputs");
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
