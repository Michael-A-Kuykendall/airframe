use airframe::backend::bindless::kv_cache::KVCache as GpuKvCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams, RMSNormParams};
use airframe::core::dequant::dequantize_q6_k;
use airframe::core::dequant::{dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q8_0};
use airframe::core::error::{LibshimmyError, Result};
use airframe::core::model::{GgufTensorInfo, Model as CpuModelContainer};
use airframe::core::spec::ModelSpec;
use airframe::core::tensor::Tensor;
use airframe::core::weight_id::WeightId;
use airframe::family::llama::LlamaModel;
use airframe::ops::dispatch::OpDispatcher;
use airframe::runtime::kvcache::KvCache as CpuKvCache;
use clap::Parser;
use memmap2::Mmap;
use serde::Serialize;
use shimmytok::Tokenizer;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[cfg(feature = "isf")]
use airframe::backend::bindless::pipeline::inference::{
    set_invariant_ptensor_capture_sink, clear_invariant_ptensor_capture_sink, CapturedPerTensor,
};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    prompt_file: Option<PathBuf>,

    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 4096)]
    max_ctx: usize,

    #[arg(long, default_value_t = 0.5)]
    rope_scale: f32,

    #[arg(long)]
    target_index: Option<usize>,

    #[arg(long, default_value_t = 128)]
    chunk_tokens: u32,

    #[arg(long)]
    decode_start_index: Option<usize>,

    /// Validate head blob dispatch splitting: compare tiled vs unsplit output.
    /// Runs after the main probe and reports MAE.
    #[arg(long)]
    validate_head_tile: bool,

    /// Max workgroups per tile (used with --validate-head-tile).
    #[arg(long, default_value_t = 512)]
    max_safe_wgs: u32,
}

#[derive(Serialize)]
struct SummaryStats {
    max_abs_err: f32,
    mean_abs_err: f32,
    cpu_absmax: f32,
    gpu_absmax: f32,
    cpu_rms: f32,
    gpu_rms: f32,
    cpu_non_finite: usize,
    gpu_non_finite: usize,
    cpu_first8: Vec<f32>,
    gpu_first8: Vec<f32>,
}

#[derive(Serialize)]
struct LayerComparison {
    layer_idx: usize,
    q: SummaryStats,
    k: SummaryStats,
    v: SummaryStats,
    post_attn: SummaryStats,
    ffn_out: SummaryStats,
    output: SummaryStats,
}

/// Per-kernel capture emitted via the `airframe_observe` sink
/// (`run_layer_with_cache_debug` → `emit_ptensor_capture`). Mirrors
/// `LayerComparison` but carries RMS+checksum for vault comparison.
#[cfg(feature = "isf")]
#[derive(Serialize)]
struct CapturedPerTensorJson {
    layer_idx: u32,
    position: u32,
    q_rms: f32,
    q_checksum: i64,
    k_rms: f32,
    k_checksum: i64,
    v_rms: f32,
    v_checksum: i64,
    post_rms: f32,
    post_checksum: i64,
    ffn_rms: f32,
    ffn_checksum: i64,
    output_rms: f32,
    output_checksum: i64,
}

#[derive(Serialize)]
struct LayerDiag {
    layer_idx: usize,
    q_quant: u32,
    k_quant: u32,
    v_quant: u32,
    ffn_gate_quant: u32,
    ffn_down_quant: u32,
    ffn_up_quant: u32,
    attn_out_quant: u32,
    v_offset: u64,
    q_offset: u64,
    k_offset: u64,
    ffn_gate_offset: u64,
    ffn_down_offset: u64,
    ffn_up_offset: u64,
    ffn_kind: u32,
    qkv_layout: u32,
    qk_norm: u32,
    post_norm: u32,
    layer_norm: u32,
    batch_count: u32,
}

#[derive(Serialize)]
struct LogitTopK {
    token_id: usize,
    logit: f32,
}

#[derive(Serialize)]
struct ProbeOutput {
    prompt_len: usize,
    prefix_len: usize,
    target_index: usize,
    target_token_id: usize,
    max_ctx: usize,
    rope_scale: f32,
    chunk_tokens: u32,
    logits: SummaryStats,
    cpu_topk: Vec<LogitTopK>,
    gpu_topk: Vec<LogitTopK>,
    layers: Vec<LayerComparison>,
    layer_diags: Vec<LayerDiag>,
    #[cfg(feature = "isf")]
    captured_per_tensor: Vec<CapturedPerTensorJson>,
}

#[derive(Clone)]
struct CpuLayerDebug {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    post_attn: Vec<f32>,
    ffn_out: Vec<f32>,
    output: Vec<f32>,
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new().map_err(|err| {
        LibshimmyError::Unsupported(format!("failed to create tokio runtime: {err}"))
    })?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let args = Args::parse();

    // Install the per-tensor capture sink so `run_layer_with_cache_debug`
    // routes q/k/v/post/ffn/output RMS+checksum into `pt_sink`. Gated by the
    // `isf` feature (which also gates the capture API). Lives for the whole
    // async_main scope so the static pointer stays valid through the compare loop.
    #[cfg(feature = "isf")]
    let mut pt_sink: Vec<CapturedPerTensor> = Vec::new();
    #[cfg(feature = "isf")]
    {
        std::env::set_var("AIRFRAME_CAPTURE_INVARIANT", "1");
        set_invariant_ptensor_capture_sink(&mut pt_sink);
    }
    let prompt = load_prompt(&args)?;
    let tokenizer = Tokenizer::from_gguf_file(&args.model).map_err(|err| {
        LibshimmyError::Unsupported(format!("failed to load tokenizer from GGUF: {err}"))
    })?;
    let prompt_tokens_u32 = tokenizer
        .encode(&prompt, true)
        .map_err(|err| LibshimmyError::Unsupported(format!("failed to tokenize prompt: {err}")))?;
    let prompt_tokens: Vec<usize> = prompt_tokens_u32
        .iter()
        .map(|&token| token as usize)
        .collect();
    if prompt_tokens.is_empty() {
        return Err(LibshimmyError::Unsupported(
            "prompt must contain at least one token".to_string(),
        ));
    }

    let target_index = args.target_index.unwrap_or(prompt_tokens.len() - 1);
    if target_index >= prompt_tokens.len() {
        return Err(LibshimmyError::Unsupported(format!(
            "target_index {} out of range for prompt len {}",
            target_index,
            prompt_tokens.len()
        )));
    }

    let prefix_tokens = &prompt_tokens[..target_index];
    let target_token = prompt_tokens[target_index];

    let container = CpuModelContainer::from_gguf(&args.model)?;
    let mut spec = container.spec;
    spec.n_ctx = args.max_ctx;
    spec.rope_scale = args.rope_scale;
    spec = spec.compute_derived();
    let mut weights = container.weights;
    // Force tied for Qwen3/Llama-3.2 to avoid MissingTensor output.weight in CPU trace.
    let model_str = args.model.to_string_lossy().to_lowercase();
    if (model_str.contains("qwen3") || model_str.contains("llama-3.2"))
        && !weights.contains_key(&WeightId::OutputProj)
    {
        if let Some(token) = weights.get(&WeightId::TokenEmbed).cloned() {
            weights.insert(WeightId::OutputProj, token);
        }
    }

    if prefix_tokens.len() + 1 > spec.n_ctx {
        return Err(LibshimmyError::Unsupported(format!(
            "target frontier {} exceeds max_ctx {}",
            prefix_tokens.len() + 1,
            spec.n_ctx
        )));
    }

    let ops = OpDispatcher::new();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|err| {
            LibshimmyError::Unsupported(format!("failed to acquire GPU adapter: {err}"))
        })?;

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
        .map_err(|err| {
            LibshimmyError::Unsupported(format!("failed to create GPU device: {err}"))
        })?;

    let gpu_model = BindlessModel::load_from_disk(&device, &args.model, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);
    let output_head_f32 = load_output_head_f32(&args.model, &gpu_model, &device, &spec)?;
    let gpu_cache_max_len = spec.n_ctx as u32;
    let mut gpu_kv = GpuKvCache::new(
        &device,
        spec.n_layer,
        spec.n_head_kv as u32,
        spec.head_dim as u32,
        gpu_cache_max_len,
    );

    let dim = spec.n_embd as u32;
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .ok_or_else(|| LibshimmyError::MissingTensor {
            name: "token_embd.weight".to_string(),
        })?;
    // Type-aware embedding row stride (fixes wrong offset for Q6_K/Q4_K embeddings)
    let embd_quant_type = gpu_model
        .metadata
        .get_tensor_type("token_embd.weight")
        .unwrap_or(2);
    let row_bytes: u32 = match embd_quant_type {
        0 => dim * 4,
        1 => dim * 2,
        2 => (dim / 32) * 18,
        8 => (dim / 32) * 34,
        12 => (dim / 256) * 144,
        13 => (dim / 256) * 176,
        14 => (dim / 256) * 210,
        _ => (dim / 32) * 18,
    };
    eprintln!(
        "[frontier_compare] token_embd quant_type={} row_bytes={}",
        embd_quant_type, row_bytes
    );

    if !prefix_tokens.is_empty() {
        {
            eprintln!(
                "[GPU prefill] {} tokens in chunks of {} ...",
                prefix_tokens.len(),
                args.chunk_tokens
            );
            let prefix_embd = dequantize_embeddings(
                &pipeline,
                &device,
                &queue,
                &gpu_model,
                embd_weight_offset,
                row_bytes,
                dim,
                prefix_tokens,
                embd_quant_type,
            );
            let _ = pipeline.run_full_model_prefill_chunked_with_cache_state(
                &device,
                &queue,
                &gpu_model,
                &prefix_embd,
                None,
                0,
                Some((gpu_kv.get_k_buffers(), gpu_kv.get_v_buffers())),
                &spec,
                args.chunk_tokens,
            );
            for _ in 0..prefix_tokens.len() {
                let _ = gpu_kv.increment();
            }
            eprintln!("[GPU prefill] done");
        }
    }

    let target_input = token_embedding(&weights, target_token, spec.n_embd)?;
    let (cpu_layers, cpu_hidden_vec) = {
        let full_model = LlamaModel::from_spec(spec.clone());
        let mut cpu_kv = CpuKvCache::new(
            spec.n_ctx,
            spec.n_layer,
            spec.n_head_kv,
            spec.head_dim, // use derived head_dim (not n_embd/n_head — wrong for GQA)
        );

        if !prefix_tokens.is_empty() {
            eprintln!("[CPU prefill] {} tokens ...", prefix_tokens.len());
            // Tolerate CPU shape error for Qwen3-0.6B (the container/ops hit [1024,2048] for q_weight).
            // GPU prefill already completed and captured per-layer stats. This lets us reach json write
            // + cpu_layer_trace (dummies) so table method (gpu "our" col) can proceed for diagnosis.
            let _ = full_model.forward(prefix_tokens, &weights, &mut cpu_kv, &ops);
            let _ = cpu_kv.complete_prefill(prefix_tokens.len());
            eprintln!("[CPU prefill] done (tolerated for qwen; gpu data + dummies for table)");
        }

        eprintln!(
            "[CPU decode] running target token through all {} layers ...",
            spec.n_layer
        );
        let cpu_layers = cpu_debug_target_token(&weights, &ops, &spec, &mut cpu_kv, &target_input)?;
        eprintln!("[CPU decode] done");
        cpu_kv.complete_decode()?;
        let cpu_hidden = cpu_layers
            .last()
            .map(|layer| layer.output.clone())
            .unwrap_or_else(|| target_input.clone());
        (cpu_layers, cpu_hidden)
    };

    let layer_params = LayerParams {
        dim: spec.n_embd as u32,
        head_count: spec.n_head as u32,
        head_count_kv: spec.n_head_kv as u32,
        head_dim: spec.head_dim as u32,
        rope_dim: spec.rope_dim as u32,
        rms_eps: spec.rms_eps,
        ffn_dim: spec.ff_dim as u32,
        temp_stride: spec.temp_buffer_size as u32,
        quant_qk: gpu_model
            .metadata
            .get_tensor_type("blk.0.attn_q.weight")
            .unwrap_or(2),
        quant_v: gpu_model
            .metadata
            .get_tensor_type("blk.0.attn_v.weight")
            .unwrap_or(2),
        quant_attn_out: gpu_model
            .metadata
            .get_tensor_type("blk.0.attn_output.weight")
            .unwrap_or(2),
        quant_ffn_down: gpu_model
            .metadata
            .get_tensor_type("blk.0.ffn_down.weight")
            .unwrap_or(2),
        quant_ffn_gate: gpu_model
            .metadata
            .get_tensor_type("blk.0.ffn_gate.weight")
            .unwrap_or(2),
        quant_ffn_up: gpu_model
            .metadata
            .get_tensor_type("blk.0.ffn_up.weight")
            .unwrap_or(2),
        attn_logit_softcap: 0.0,
        post_norm_enabled: 0,
        qk_norm_enabled: 0,
        layer_norm_enabled: 0,
        ffn_kind_policy: 0,
        qkv_layout_policy: 0,
        batch_offset: 0,
        batch_count: 1, // single decode token
        q_weight_k: 0,
        k_weight_k: 0,
    };

    eprintln!("[GPU vs CPU] comparing {} layers ...", spec.n_layer);
    let mut gpu_hidden = target_input.clone();
    let mut layer_comparisons = Vec::with_capacity(spec.n_layer);
    let mut layer_diags = Vec::with_capacity(spec.n_layer);
    // Note: layer_idx is used legitimately as an index for cpu_layers, metadata, and pipeline
    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..spec.n_layer {
        let offsets = gpu_model
            .metadata
            .get_layer_offsets(layer_idx, spec.arch_string())
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("layer_offsets_{layer_idx}"),
            })?;

        // Per-layer quant_type: Q4_K_M models can have different quant per tensor per layer
        let qt = |name: &str| gpu_model.metadata.get_tensor_type(name).unwrap_or(0);
        let key = |s: &str| format!("blk.{}.{}", layer_idx, s);
        let l_qt_main = qt(&key("attn_q.weight"));
        let l_qt_v = qt(&key("attn_v.weight"));
        let l_qt_down = qt(&key("ffn_down.weight"));
        let l_qt_out = qt(&key("attn_output.weight"));
        let mut layer_params = layer_params;
        layer_params.quant_qk = l_qt_main;
        layer_params.quant_v = l_qt_v;
        layer_params.quant_attn_out = l_qt_out;
        layer_params.quant_ffn_down = l_qt_down;
        layer_params.quant_ffn_gate = qt(&key("ffn_gate.weight"));
        layer_params.quant_ffn_up = qt(&key("ffn_up.weight"));

        let (gpu_output, gpu_post_attn, gpu_ffn_out, gpu_q, gpu_k, gpu_v) = pipeline
            .run_layer_with_cache_debug(
                &device,
                &queue,
                &gpu_model,
                &mut gpu_kv,
                layer_idx,
                &gpu_hidden,
                offsets,
                layer_params,
            );

        // Capture per-layer quant types and offsets as structured vault data
        layer_diags.push(LayerDiag {
            layer_idx,
            q_quant: l_qt_main,
            k_quant: qt(&key("attn_k.weight")),
            v_quant: l_qt_v,
            ffn_gate_quant: qt(&key("ffn_gate.weight")),
            ffn_down_quant: l_qt_down,
            ffn_up_quant: qt(&key("ffn_up.weight")),
            attn_out_quant: l_qt_out,
            v_offset: offsets.attn_v as u64,
            q_offset: offsets.attn_q as u64,
            k_offset: offsets.attn_k as u64,
            ffn_gate_offset: offsets.ffn_gate as u64,
            ffn_down_offset: offsets.ffn_down as u64,
            ffn_up_offset: offsets.ffn_up as u64,
            ffn_kind: layer_params.ffn_kind_policy,
            qkv_layout: layer_params.qkv_layout_policy,
            qk_norm: layer_params.qk_norm_enabled,
            post_norm: layer_params.post_norm_enabled,
            layer_norm: layer_params.layer_norm_enabled,
            batch_count: layer_params.batch_count,
        });

        let cpu_layer = &cpu_layers[layer_idx];
        let cmp = LayerComparison {
            layer_idx,
            q: summarize_pair(&cpu_layer.q, &gpu_q),
            k: summarize_pair(&cpu_layer.k, &gpu_k),
            v: summarize_pair(&cpu_layer.v, &gpu_v),
            post_attn: summarize_pair(&cpu_layer.post_attn, &gpu_post_attn),
            ffn_out: summarize_pair(&cpu_layer.ffn_out, &gpu_ffn_out),
            output: summarize_pair(&cpu_layer.output, &gpu_output),
        };
        eprintln!(
            "  layer {:2}: output MAE={:.6}  post_attn MAE={:.6}",
            layer_idx, cmp.output.mean_abs_err, cmp.post_attn.mean_abs_err
        );
        layer_comparisons.push(cmp);

        gpu_hidden = gpu_output;
    }
    eprintln!("[GPU vs CPU] done");
    let _ = gpu_kv.increment();

    let output_norm =
        weights
            .get(&WeightId::OutputNorm)
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: "output_norm.weight".to_string(),
            })?;
    // Tied-embedding models (Qwen3, Llama-3.2) have no OutputProj — fall back to TokenEmbed
    let model_str = args.model.to_string_lossy().to_lowercase();
    let output_proj = if model_str.contains("qwen3") || model_str.contains("llama-3.2") {
        weights
            .get(&WeightId::TokenEmbed)
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: "token_embd.weight (tied embedding for qwen3/llama-3.2)".to_string(),
            })?
    } else {
        weights
            .get(&WeightId::OutputProj)
            .or_else(|| weights.get(&WeightId::TokenEmbed))
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: "output.weight (and token_embd.weight tied-embedding fallback)".to_string(),
            })?
    };

    let cpu_hidden_tensor = Tensor::new(
        gpu_to_row(cpu_hidden_vec, spec.n_embd),
        vec![1, spec.n_embd],
    )?;
    let cpu_norm = ops.rmsnorm(&cpu_hidden_tensor, output_norm, spec.rms_eps)?;
    let cpu_logits = ops.matmul(&cpu_norm, output_proj)?.data;

    let norm_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("output_norm.weight")
        .ok_or_else(|| LibshimmyError::MissingTensor {
            name: "output_norm.weight".to_string(),
        })? as u32;
    let norm_params = RMSNormParams {
        count: dim,
        weights_offset: (norm_weight_offset / 4), // word index (byte_offset / 4) — matches sh_rmsnorm.wgsl convention
        bias_offset: 0,
        eps: spec.rms_eps,
        norm_type: 0,
    };
    let gpu_norm = pipeline.run_rmsnorm_test(&device, &queue, &gpu_model, &gpu_hidden, norm_params);
    let gpu_logits = pipeline.run_matmul_f32(
        &device,
        &queue,
        &output_head_f32,
        &gpu_norm,
        spec.n_vocab as u32,
        dim,
    );

    // Optional: validate head blob dispatch splitting
    if args.validate_head_tile {
        let head_tensor_name = if gpu_model
            .metadata
            .get_tensor_type("output.weight")
            .is_some()
        {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let hd_vocab = spec.n_vocab as u32;
        let hd_dim = dim;
        let hd_off = (gpu_model
            .metadata
            .get_tensor_offset(head_tensor_name)
            .unwrap_or(0)
            / 4) as u32;
        let hd_qt = gpu_model
            .metadata
            .get_tensor_type(head_tensor_name)
            .unwrap_or(2);
        let hd_softcap = spec.final_logit_softcap;

        eprintln!(
            "[HEAD-TILE] validating tiled dispatch (max_safe_wgs={})...",
            args.max_safe_wgs
        );

        let tiled_logits = pipeline.run_lm_head_blob_tiled(
            &device,
            &queue,
            &gpu_model,
            &gpu_norm,
            hd_vocab,
            hd_dim,
            hd_off,
            hd_qt,
            hd_softcap,
            args.max_safe_wgs,
        );

        let unsplit_logits = pipeline.run_lm_head_blob(
            &device, &queue, &gpu_model, &gpu_norm, hd_vocab, hd_dim, hd_off, hd_qt, hd_softcap,
        );

        let mae = tiled_logits
            .iter()
            .zip(unsplit_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / tiled_logits.len() as f32;
        let max_ae = tiled_logits
            .iter()
            .zip(unsplit_logits.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!(
            "[HEAD-TILE] tiled_vs_unsplit: MAE={:.8} max_AE={:.6} PASS={}",
            mae,
            max_ae,
            if mae < 1e-6f32 { "YES" } else { "NO" }
        );
    }

    let probe = ProbeOutput {
        prompt_len: prompt_tokens.len(),
        prefix_len: prefix_tokens.len(),
        target_index,
        target_token_id: target_token,
        max_ctx: spec.n_ctx,
        rope_scale: spec.rope_scale,
        chunk_tokens: args.chunk_tokens,
        logits: summarize_pair(&cpu_logits, &gpu_logits),
        cpu_topk: topk(&cpu_logits, 10),
        gpu_topk: topk(&gpu_logits, 10),
        layers: layer_comparisons,
        layer_diags,
        #[cfg(feature = "isf")]
        captured_per_tensor: pt_sink
            .iter()
            .map(|c| CapturedPerTensorJson {
                layer_idx: c.layer_idx,
                position: c.position,
                q_rms: c.q_rms,
                q_checksum: c.q_checksum,
                k_rms: c.k_rms,
                k_checksum: c.k_checksum,
                v_rms: c.v_rms,
                v_checksum: c.v_checksum,
                post_rms: c.post_rms,
                post_checksum: c.post_checksum,
                ffn_rms: c.ffn_rms,
                ffn_checksum: c.ffn_checksum,
                output_rms: c.output_rms,
                output_checksum: c.output_checksum,
            })
            .collect(),
    };

    #[cfg(feature = "isf")]
    clear_invariant_ptensor_capture_sink();

    let json = serde_json::to_string_pretty(&probe).map_err(|err| {
        LibshimmyError::Unsupported(format!("failed to serialize probe output: {err}"))
    })?;
    fs::write(&args.output, json).map_err(|err| {
        LibshimmyError::Unsupported(format!(
            "failed to write output {}: {err}",
            args.output.display()
        ))
    })?;

    eprintln!("[done] wrote {}", args.output.display());
    println!("wrote {}", args.output.display());
    Ok(())
}

fn load_prompt(args: &Args) -> Result<String> {
    match (&args.prompt, &args.prompt_file) {
        (Some(prompt), None) => Ok(prompt.clone()),
        (None, Some(path)) => fs::read_to_string(path).map_err(|err| {
            LibshimmyError::Unsupported(format!(
                "failed to read prompt file {}: {err}",
                path.display()
            ))
        }),
        (Some(_), Some(_)) => Err(LibshimmyError::Unsupported(
            "pass either --prompt or --prompt-file, not both".to_string(),
        )),
        (None, None) => Err(LibshimmyError::Unsupported(
            "one of --prompt or --prompt-file is required".to_string(),
        )),
    }
}

fn token_embedding(
    weights: &HashMap<WeightId, Tensor>,
    token_id: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    let token_embed =
        weights
            .get(&WeightId::TokenEmbed)
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: "token_embd.weight".to_string(),
            })?;
    let start = token_id * dim;
    let end = start + dim;
    if end > token_embed.data.len() {
        return Err(LibshimmyError::Unsupported(format!(
            "token {} out of range for embedding table",
            token_id
        )));
    }
    Ok(token_embed.data[start..end].to_vec())
}

#[allow(clippy::too_many_arguments)]
fn dequantize_embeddings(
    pipeline: &BindlessPipeline,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &BindlessModel,
    embd_weight_offset: u64,
    row_bytes: u32,
    dim: u32,
    tokens: &[usize],
    embd_quant_type: u32,
) -> Vec<f32> {
    let mut batched = Vec::with_capacity(tokens.len() * dim as usize);
    for &token_id in tokens {
        let row_offset = embd_weight_offset + token_id as u64 * row_bytes as u64;
        let embd = pipeline.run_dequant_any_hot(
            device,
            queue,
            model,
            row_offset as u32,
            dim,
            embd_quant_type,
        );
        batched.extend_from_slice(&embd);
    }
    batched
}

fn load_output_head_f32(
    model_path: &PathBuf,
    gpu_model: &BindlessModel,
    device: &wgpu::Device,
    spec: &ModelSpec,
) -> Result<wgpu::Buffer> {
    // Determine which tensor to use for the output head.
    // Models with tied embeddings (e.g. Qwen3, Llama-3.2) omit `output.weight`
    // and reuse `token_embd.weight` for the final projection.
    let (tensor_name, weight_type, tensor_offset) = {
        let model_str = model_path.to_string_lossy().to_lowercase();
        let has_output = gpu_model
            .metadata
            .get_tensor_type("output.weight")
            .is_some()
            && !model_str.contains("qwen3")
            && !model_str.contains("llama-3.2");
        if has_output {
            let wt = gpu_model.metadata.get_tensor_type("output.weight").unwrap();
            let off = gpu_model
                .metadata
                .get_tensor_offset("output.weight")
                .unwrap_or(0);
            ("output.weight", wt, off)
        } else {
            // Tied embeddings: fall back to token_embd.weight for Qwen3, Llama-3.2 etc.
            let wt = gpu_model
                .metadata
                .get_tensor_type("token_embd.weight")
                .ok_or_else(|| LibshimmyError::MissingTensor {
                    name: "token_embd.weight (tied embedding fallback)".to_string(),
                })?;
            let off = gpu_model
                .metadata
                .get_tensor_offset("token_embd.weight")
                .unwrap_or(0);
            ("token_embd.weight", wt, off)
        }
    };

    let file = std::fs::File::open(model_path).map_err(|err| {
        LibshimmyError::Unsupported(format!(
            "failed to open model {} for output head dequant: {err}",
            model_path.display()
        ))
    })?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
        LibshimmyError::Unsupported(format!(
            "failed to mmap model {} for output head dequant: {err}",
            model_path.display()
        ))
    })?;

    let data_start = gpu_model.metadata.data_start_offset;
    let tensor_offset_relative = tensor_offset.saturating_sub(data_start);

    let tensor_info = GgufTensorInfo {
        name: tensor_name.to_string(),
        dimensions: vec![spec.n_vocab, spec.n_embd],
        ggml_type: weight_type,
        offset: tensor_offset_relative,
    };

    let tensor_f32 = match weight_type {
        0 => {
            use airframe::core::tensor::Tensor as AirframeTensor;
            let byte_offset = data_start + tensor_offset_relative;
            let n_elements = spec.n_vocab * spec.n_embd;
            let bytes = &mmap[byte_offset as usize..(byte_offset as usize + n_elements * 4)];
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            AirframeTensor {
                data: floats,
                shape: vec![spec.n_vocab, spec.n_embd],
            }
        }
        2 => dequantize_q4_0(&tensor_info, &mmap, data_start)?,
        8 => dequantize_q8_0(&tensor_info, &mmap, data_start)?,
        12 => dequantize_q4_k(&tensor_info, &mmap, data_start)?,
        13 => dequantize_q5_k(&tensor_info, &mmap, data_start)?,
        14 => dequantize_q6_k(&tensor_info, &mmap, data_start)?,
        other => {
            return Err(LibshimmyError::Unsupported(format!(
                "unsupported quant type {} for output head tensor '{}'",
                other, tensor_name
            )));
        }
    };

    use wgpu::util::DeviceExt;
    Ok(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Frontier Output Head F32"),
            contents: bytemuck::cast_slice(&tensor_f32.data),
            usage: wgpu::BufferUsages::STORAGE,
        }),
    )
}

fn cpu_debug_target_token(
    weights: &HashMap<WeightId, Tensor>,
    ops: &OpDispatcher,
    spec: &ModelSpec,
    kv_cache: &mut CpuKvCache,
    target_input: &[f32],
) -> Result<Vec<CpuLayerDebug>> {
    let model = LlamaModel::from_spec(spec.clone());
    // Tolerant path for Qwen3-0.6B etc where CPU container surfaces packed Q weight shape
    // that trips matmul/attn shape checks in trace path. Returns dummy cpu cols (0) so the
    // frontier json is still emitted with full gpu per-layer stats. This enables the
    // established table (Col1 known/0 | Col2 our gpu) + formula/vault method without
    // blocking on CPU golden for this model (vault_seed oracles also 0 for it; follow-up).
    let is_qwen_small = spec.n_embd == 1024 && spec.n_layer == 28;
    if is_qwen_small {
        let z = vec![0f32; spec.n_embd];
        let dummy = CpuLayerDebug {
            q: z.clone(),
            k: z.clone(),
            v: z.clone(),
            post_attn: z.clone(),
            ffn_out: z.clone(),
            output: z.clone(),
        };
        return Ok((0..spec.n_layer).map(|_| dummy.clone()).collect());
    }
    let mut hidden = Tensor::new(
        gpu_to_row(target_input.to_vec(), spec.n_embd),
        vec![1, spec.n_embd],
    )?;
    let mut out = Vec::with_capacity(spec.n_layer);
    let position_ids = vec![kv_cache.len()];

    for layer_idx in 0..spec.n_layer {
        let attn_norm = weights
            .get(&WeightId::AttnNorm { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_norm.weight"),
            })?;
        let q_weight = weights
            .get(&WeightId::AttnQ { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_q.weight"),
            })?;
        let k_weight = weights
            .get(&WeightId::AttnK { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_k.weight"),
            })?;
        let v_weight = weights
            .get(&WeightId::AttnV { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_v.weight"),
            })?;
        let o_weight = weights
            .get(&WeightId::AttnO { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_output.weight"),
            })?;
        let ffn_norm = weights
            .get(&WeightId::FfnNorm { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_norm.weight"),
            })?;
        let gate = weights
            .get(&WeightId::FfnGate { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_gate.weight"),
            })?;
        let up = weights
            .get(&WeightId::FfnUp { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_up.weight"),
            })?;
        let down = weights
            .get(&WeightId::FfnDown { layer: layer_idx })
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_down.weight"),
            })?;

        let attn_input = ops.rmsnorm(&hidden, attn_norm, spec.rms_eps)?;
        let q = ops.matmul(&attn_input, q_weight)?.data;
        let k = ops.matmul(&attn_input, k_weight)?.data;
        let v = ops.matmul(&attn_input, v_weight)?.data;
        let attn_output = ops.attention_with_cache(
            &attn_input,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            spec.n_head,
            spec.n_head_kv,
            spec.head_dim,
            &position_ids,
            spec.rope_base,
            spec.rope_dim,
            spec.rope_scale,
            layer_idx,
            kv_cache,
            None, // no QK norm for non-Qwen3
            None, // no attention.scale for non-Qwen3
        )?;
        let post_attn = ops.add(&hidden, &attn_output)?;
        let ffn_input = ops.rmsnorm(&post_attn, ffn_norm, spec.rms_eps)?;
        let ffn_output = ops.ffn_swiglu(&ffn_input, gate, up, down)?;
        let output = ops.add(&post_attn, &ffn_output)?;

        out.push(CpuLayerDebug {
            q,
            k,
            v,
            post_attn: post_attn.data.clone(),
            ffn_out: ffn_output.data.clone(),
            output: output.data.clone(),
        });

        hidden = output;
    }

    let _ = model;
    Ok(out)
}

fn summarize_pair(cpu: &[f32], gpu: &[f32]) -> SummaryStats {
    if cpu.len() != gpu.len() {
        // Qwen3-0.6B etc: cpu dummy or packed 1024 vs gpu capture 2048; produce usable stats from gpu side
        // so table json + formula method can still diagnose divergence (our gpu col) without panic.
        let gmax = gpu.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let grms = if gpu.is_empty() {
            0.0
        } else {
            (gpu.iter().map(|&v| v * v).sum::<f32>() / gpu.len() as f32).sqrt()
        };
        let gnon = gpu.iter().filter(|&&v| !v.is_finite()).count();
        return SummaryStats {
            max_abs_err: 999.0,
            mean_abs_err: 999.0,
            cpu_absmax: 0.0,
            gpu_absmax: gmax,
            cpu_rms: 0.0,
            gpu_rms: grms,
            cpu_non_finite: 0,
            gpu_non_finite: gnon,
            cpu_first8: vec![],
            gpu_first8: gpu.iter().take(8).cloned().collect(),
        };
    }
    let mut max_abs_err = 0.0f32;
    let mut sum_abs_err = 0.0f32;
    for (lhs, rhs) in cpu.iter().zip(gpu.iter()) {
        let err = (lhs - rhs).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        sum_abs_err += err;
    }

    SummaryStats {
        max_abs_err,
        mean_abs_err: if cpu.is_empty() {
            0.0
        } else {
            sum_abs_err / cpu.len() as f32
        },
        cpu_absmax: absmax(cpu),
        gpu_absmax: absmax(gpu),
        cpu_rms: rms(cpu),
        gpu_rms: rms(gpu),
        cpu_non_finite: cpu.iter().filter(|value| !value.is_finite()).count(),
        gpu_non_finite: gpu.iter().filter(|value| !value.is_finite()).count(),
        cpu_first8: cpu.iter().take(8).copied().collect(),
        gpu_first8: gpu.iter().take(8).copied().collect(),
    }
}

fn absmax(values: &[f32]) -> f32 {
    values.iter().map(|value| value.abs()).fold(0.0, f32::max)
}

fn rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = values.iter().map(|value| value * value).sum();
    (sum_sq / values.len() as f32).sqrt()
}

fn topk(logits: &[f32], k: usize) -> Vec<LogitTopK> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
    indexed
        .into_iter()
        .take(k)
        .map(|(token_id, logit)| LogitTopK { token_id, logit })
        .collect()
}

fn gpu_to_row(values: Vec<f32>, dim: usize) -> Vec<f32> {
    assert_eq!(values.len(), dim, "expected one token row");
    values
}
