use airframe::backend::bindless::kv_cache::KVCache as GpuKvCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams, RMSNormParams};
use airframe::backend::bindless::pipeline_shift::RopeShiftPipeline;
use airframe::core::dequant::dequantize_q6_k;
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

    #[arg(long)]
    helical_max_len: Option<u32>,

    #[arg(long, default_value_t = 4)]
    helical_keep_sink: u32,

    #[arg(long)]
    helical_shift_amt: Option<u32>,

    #[arg(long, default_value_t = false)]
    cpu_helical: bool,

    #[arg(long, default_value_t = false)]
    attention_mass_probe: bool,
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
    attention_mass_probe: Option<AttentionMassProbe>,
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

struct CpuHelicalLayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

struct CpuHelicalKvCache {
    layers: Vec<CpuHelicalLayerCache>,
    current_len: usize,
    max_seq_len: usize,
    window_base: usize,
    n_head_kv: usize,
    head_dim: usize,
}

#[derive(Serialize, Clone, Copy, Default)]
struct BucketMass {
    sink: f32,
    retained_window: f32,
    dropped_band: f32,
    current_token: f32,
}

#[derive(Serialize)]
struct HeadAttentionMass {
    head_idx: usize,
    mass: BucketMass,
}

#[derive(Serialize)]
struct LayerAttentionMass {
    layer_idx: usize,
    avg_dropped_band_mass: f32,
    max_dropped_band_mass: f32,
    avg_sink_mass: f32,
    avg_retained_window_mass: f32,
    avg_current_token_mass: f32,
    heads: Vec<HeadAttentionMass>,
}

#[derive(Serialize)]
struct AttentionMassProbe {
    keep_sink: usize,
    helical_max_len: usize,
    shift_amt: usize,
    window_base_before_target: usize,
    compact_query_pos_before_target: usize,
    logical_query_pos: usize,
    dropped_band_start: usize,
    dropped_band_end: usize,
    layers: Vec<LayerAttentionMass>,
}

struct HelicalPartitionState {
    keep_sink: usize,
    shift_amt: usize,
    window_base_before_target: usize,
    compact_query_pos_before_target: usize,
    logical_query_pos: usize,
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new().map_err(|err| {
        LibshimmyError::Unsupported(format!("failed to create tokio runtime: {err}"))
    })?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    let args = Args::parse();
    let prompt = load_prompt(&args)?;
    let tokenizer = Tokenizer::from_gguf_file(&args.model).map_err(|err| {
        LibshimmyError::Unsupported(format!("failed to load tokenizer from GGUF: {err}"))
    })?;
    let prompt_tokens_u32 = tokenizer.encode(&prompt, true).map_err(|err| {
        LibshimmyError::Unsupported(format!("failed to tokenize prompt: {err}"))
    })?;
    let prompt_tokens: Vec<usize> = prompt_tokens_u32.iter().map(|&token| token as usize).collect();
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
    let weights = container.weights;

    if prefix_tokens.len() + 1 > spec.n_ctx {
        return Err(LibshimmyError::Unsupported(format!(
            "target frontier {} exceeds max_ctx {}",
            prefix_tokens.len() + 1,
            spec.n_ctx
        )));
    }

    let ops = OpDispatcher::new();

    // Fast path: if only the attention mass probe is requested, skip GPU entirely.
    if args.attention_mass_probe {
        let probe = compute_attention_mass_probe(
            &weights,
            &ops,
            &spec,
            &prompt_tokens,
            target_index,
            args.helical_max_len.unwrap_or(spec.n_ctx as u32) as usize,
            args.helical_keep_sink as usize,
            args.helical_shift_amt.map(|v| v as usize),
        )?;
        let json = serde_json::to_string_pretty(&probe).map_err(|err| {
            LibshimmyError::Unsupported(format!("failed to serialize probe output: {err}"))
        })?;
        fs::write(&args.output, json).map_err(|err| {
            LibshimmyError::Unsupported(format!("failed to write output {}: {err}", args.output.display()))
        })?;
        println!("wrote {}", args.output.display());
        return Ok(());
    }

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
    limits.max_storage_buffers_per_shader_stage = 8;
    limits.max_compute_invocations_per_workgroup = 256;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .map_err(|err| LibshimmyError::Unsupported(format!("failed to create GPU device: {err}")))?;

    let gpu_model = BindlessModel::load_from_disk(&device, &args.model, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);
    let shift_pipeline = RopeShiftPipeline::new(&device);
    let output_head_f32 = load_output_head_f32(&args.model, &gpu_model, &device, &spec)?;
    let gpu_cache_max_len = args.helical_max_len.unwrap_or(spec.n_ctx as u32);
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
    let row_bytes = (dim / 32) * 18;

    if !prefix_tokens.is_empty() {
        if let Some(decode_start_index) = args.decode_start_index {
            teacher_force_gpu_prefix_helical(
                &device,
                &queue,
                &gpu_model,
                &pipeline,
                &shift_pipeline,
                &spec,
                &mut gpu_kv,
                embd_weight_offset,
                row_bytes,
                dim,
                prefix_tokens,
                decode_start_index,
                args.chunk_tokens,
                args.helical_keep_sink,
                args.helical_shift_amt,
            )?;
        } else {
            let prefix_embd = dequantize_embeddings(
                &pipeline,
                &device,
                &queue,
                &gpu_model,
                embd_weight_offset,
                row_bytes,
                dim,
                prefix_tokens,
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
                gpu_kv.increment();
            }
        }
    }

    let target_input = token_embedding(&weights, target_token, spec.n_embd)?;
    let (cpu_layers, cpu_hidden_vec) = if args.cpu_helical {
        cpu_debug_target_token_helical(
            &weights,
            &ops,
            &spec,
            &prompt_tokens,
            target_index,
            args.helical_max_len.unwrap_or(spec.n_ctx as u32) as usize,
            args.helical_keep_sink as usize,
            args.helical_shift_amt.map(|value| value as usize),
        )?
    } else {
        let full_model = LlamaModel::from_spec(spec.clone());
        let mut cpu_kv = CpuKvCache::new(
            spec.n_ctx,
            spec.n_layer,
            spec.n_head_kv,
            spec.n_embd / spec.n_head,
        );

        if !prefix_tokens.is_empty() {
            let _ = full_model.forward(prefix_tokens, &weights, &mut cpu_kv, &ops)?;
            cpu_kv.complete_prefill(prefix_tokens.len())?;
        }

        let cpu_layers = cpu_debug_target_token(&weights, &ops, &spec, &mut cpu_kv, &target_input)?;
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
        rms_eps: spec.rms_eps,
        ffn_dim: spec.ff_dim as u32,
        temp_stride: spec.temp_buffer_size as u32,
        padding: 0,
    };

    let mut gpu_hidden = target_input.clone();
    let mut layer_comparisons = Vec::with_capacity(spec.n_layer);
    for layer_idx in 0..spec.n_layer {
        let offsets = gpu_model
            .metadata
            .get_layer_offsets(layer_idx, spec.arch_string())
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("layer_offsets_{layer_idx}"),
            })?;

        let (gpu_output, gpu_post_attn, gpu_ffn_out, gpu_q, gpu_k, gpu_v) =
            pipeline.run_layer_with_cache_debug(
                &device,
                &queue,
                &gpu_model,
                &mut gpu_kv,
                layer_idx,
                &gpu_hidden,
                offsets,
                layer_params,
            );

        let cpu_layer = &cpu_layers[layer_idx];
        layer_comparisons.push(LayerComparison {
            layer_idx,
            q: summarize_pair(&cpu_layer.q, &gpu_q),
            k: summarize_pair(&cpu_layer.k, &gpu_k),
            v: summarize_pair(&cpu_layer.v, &gpu_v),
            post_attn: summarize_pair(&cpu_layer.post_attn, &gpu_post_attn),
            ffn_out: summarize_pair(&cpu_layer.ffn_out, &gpu_ffn_out),
            output: summarize_pair(&cpu_layer.output, &gpu_output),
        });

        gpu_hidden = gpu_output;
    }
    gpu_kv.increment();

    let output_norm = weights
        .get(&WeightId::OutputNorm)
        .ok_or_else(|| LibshimmyError::MissingTensor {
            name: "output_norm.weight".to_string(),
        })?;
    let output_proj = weights
        .get(&WeightId::OutputProj)
        .ok_or_else(|| LibshimmyError::MissingTensor {
            name: "output.weight".to_string(),
        })?;

    let cpu_hidden_tensor = Tensor::new(gpu_to_row(cpu_hidden_vec, spec.n_embd), vec![1, spec.n_embd])?;
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
        weights_offset: norm_weight_offset,
        eps: spec.rms_eps,
        padding: 0,
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

    let attention_mass_probe = if args.attention_mass_probe {
        Some(compute_attention_mass_probe(
            &weights,
            &ops,
            &spec,
            &prompt_tokens,
            target_index,
            args.helical_max_len.unwrap_or(spec.n_ctx as u32) as usize,
            args.helical_keep_sink as usize,
            args.helical_shift_amt.map(|value| value as usize),
        )?)
    } else {
        None
    };

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
        attention_mass_probe,
    };

    let json = serde_json::to_string_pretty(&probe).map_err(|err| {
        LibshimmyError::Unsupported(format!("failed to serialize probe output: {err}"))
    })?;
    fs::write(&args.output, json).map_err(|err| {
        LibshimmyError::Unsupported(format!("failed to write output {}: {err}", args.output.display()))
    })?;

    println!("wrote {}", args.output.display());
    Ok(())
}

fn load_prompt(args: &Args) -> Result<String> {
    match (&args.prompt, &args.prompt_file) {
        (Some(prompt), None) => Ok(prompt.clone()),
        (None, Some(path)) => fs::read_to_string(path).map_err(|err| {
            LibshimmyError::Unsupported(format!("failed to read prompt file {}: {err}", path.display()))
        }),
        (Some(_), Some(_)) => Err(LibshimmyError::Unsupported(
            "pass either --prompt or --prompt-file, not both".to_string(),
        )),
        (None, None) => Err(LibshimmyError::Unsupported(
            "one of --prompt or --prompt-file is required".to_string(),
        )),
    }
}

fn token_embedding(weights: &HashMap<WeightId, Tensor>, token_id: usize, dim: usize) -> Result<Vec<f32>> {
    let token_embed = weights
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

fn dequantize_embeddings(
    pipeline: &BindlessPipeline,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &BindlessModel,
    embd_weight_offset: u64,
    row_bytes: u32,
    dim: u32,
    tokens: &[usize],
) -> Vec<f32> {
    let mut batched = Vec::with_capacity(tokens.len() * dim as usize);
    for &token_id in tokens {
        let row_offset = embd_weight_offset + token_id as u64 * row_bytes as u64;
        let embd = pipeline.run_dequant_request(device, queue, model, row_offset as u32, dim);
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
    let output_weight_type = gpu_model
        .metadata
        .get_tensor_type("output.weight")
        .ok_or_else(|| LibshimmyError::MissingTensor {
            name: "output.weight".to_string(),
        })?;

    if output_weight_type != 14 {
        return Err(LibshimmyError::Unsupported(format!(
            "expected Q6_K (type 14) for output.weight, got type {}",
            output_weight_type
        )));
    }

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

    let tensor_info = GgufTensorInfo {
        name: "output.weight".to_string(),
        dimensions: vec![spec.n_vocab, spec.n_embd],
        ggml_type: 14,
        offset: 0,
    };

    let tensor_f32 = dequantize_q6_k(&tensor_info, &mmap, gpu_model.metadata.data_start_offset)?;

    use wgpu::util::DeviceExt;
    Ok(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Frontier Output Head F32"),
        contents: bytemuck::cast_slice(&tensor_f32.data),
        usage: wgpu::BufferUsages::STORAGE,
    }))
}

fn cpu_debug_target_token(
    weights: &HashMap<WeightId, Tensor>,
    ops: &OpDispatcher,
    spec: &ModelSpec,
    kv_cache: &mut CpuKvCache,
    target_input: &[f32],
) -> Result<Vec<CpuLayerDebug>> {
    let model = LlamaModel::from_spec(spec.clone());
    let mut hidden = Tensor::new(gpu_to_row(target_input.to_vec(), spec.n_embd), vec![1, spec.n_embd])?;
    let mut out = Vec::with_capacity(spec.n_layer);
    let position_ids = vec![kv_cache.len()];

    for layer_idx in 0..spec.n_layer {
        let attn_norm = weights.get(&WeightId::AttnNorm { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_norm.weight"),
            }
        })?;
        let q_weight = weights.get(&WeightId::AttnQ { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_q.weight"),
            }
        })?;
        let k_weight = weights.get(&WeightId::AttnK { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_k.weight"),
            }
        })?;
        let v_weight = weights.get(&WeightId::AttnV { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_v.weight"),
            }
        })?;
        let o_weight = weights.get(&WeightId::AttnO { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_output.weight"),
            }
        })?;
        let ffn_norm = weights.get(&WeightId::FfnNorm { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_norm.weight"),
            }
        })?;
        let gate = weights.get(&WeightId::FfnGate { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_gate.weight"),
            }
        })?;
        let up = weights.get(&WeightId::FfnUp { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_up.weight"),
            }
        })?;
        let down = weights.get(&WeightId::FfnDown { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_down.weight"),
            }
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
    assert_eq!(cpu.len(), gpu.len(), "mismatched compare lengths");
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
        mean_abs_err: if cpu.is_empty() { 0.0 } else { sum_abs_err / cpu.len() as f32 },
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

impl CpuHelicalKvCache {
    fn new(max_seq_len: usize, n_layer: usize, n_head_kv: usize, head_dim: usize) -> Self {
        let layer_elems = max_seq_len * n_head_kv * head_dim;
        let layers = (0..n_layer)
            .map(|_| CpuHelicalLayerCache {
                k: vec![0.0; layer_elems],
                v: vec![0.0; layer_elems],
            })
            .collect();

        Self {
            layers,
            current_len: 0,
            max_seq_len,
            window_base: 0,
            n_head_kv,
            head_dim,
        }
    }

    fn len(&self) -> usize {
        self.current_len
    }

    fn max_len(&self) -> usize {
        self.max_seq_len
    }

    fn window_base(&self) -> usize {
        self.window_base
    }

    fn write_layer(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        let head_size = self.n_head_kv * self.head_dim;
        let dst_offset = self.current_len * head_size;
        self.layers[layer_idx].k[dst_offset..dst_offset + head_size].copy_from_slice(k);
        self.layers[layer_idx].v[dst_offset..dst_offset + head_size].copy_from_slice(v);
    }

    fn finish_token(&mut self) -> Result<()> {
        if self.current_len >= self.max_seq_len {
            return Err(LibshimmyError::Unsupported(
                "CPU helical KV cache is full".to_string(),
            ));
        }
        self.current_len += 1;
        Ok(())
    }

    fn shift_all(&mut self, keep_sink: usize, shift_amt: usize) {
        if self.current_len <= keep_sink + shift_amt {
            return;
        }

        let head_size = self.n_head_kv * self.head_dim;
        for layer in &mut self.layers {
            for src_pos in (keep_sink + shift_amt)..self.current_len {
                let dst_pos = src_pos - shift_amt;
                let src = src_pos * head_size;
                let dst = dst_pos * head_size;
                layer.k.copy_within(src..src + head_size, dst);
                layer.v.copy_within(src..src + head_size, dst);
            }
        }

        self.current_len -= shift_amt;
        self.window_base += shift_amt;
    }
}

fn cpu_debug_target_token_helical(
    weights: &HashMap<WeightId, Tensor>,
    ops: &OpDispatcher,
    spec: &ModelSpec,
    prompt_tokens: &[usize],
    target_index: usize,
    helical_max_len: usize,
    keep_sink: usize,
    helical_shift_amt: Option<usize>,
) -> Result<(Vec<CpuLayerDebug>, Vec<f32>)> {
    let shift_amt = helical_shift_amt.unwrap_or(helical_max_len / 4);
    let mut cache = CpuHelicalKvCache::new(
        helical_max_len,
        spec.n_layer,
        spec.n_head_kv,
        spec.head_dim,
    );

    for &token_id in &prompt_tokens[..target_index] {
        if cache.len() >= cache.max_len().saturating_sub(4) {
            cache.shift_all(keep_sink, shift_amt);
        }
        let token_input = token_embedding(weights, token_id, spec.n_embd)?;
        let _ = cpu_helical_run_token(weights, ops, spec, &mut cache, &token_input, keep_sink, false)?;
        cache.finish_token()?;
    }

    if cache.len() >= cache.max_len().saturating_sub(4) {
        cache.shift_all(keep_sink, shift_amt);
    }

    let target_input = token_embedding(weights, prompt_tokens[target_index], spec.n_embd)?;
    cpu_helical_run_token(weights, ops, spec, &mut cache, &target_input, keep_sink, true)
}

fn cpu_helical_run_token(
    weights: &HashMap<WeightId, Tensor>,
    ops: &OpDispatcher,
    spec: &ModelSpec,
    cache: &mut CpuHelicalKvCache,
    target_input: &[f32],
    keep_sink: usize,
    capture_debug: bool,
) -> Result<(Vec<CpuLayerDebug>, Vec<f32>)> {
    let mut hidden = Tensor::new(gpu_to_row(target_input.to_vec(), spec.n_embd), vec![1, spec.n_embd])?;
    let mut out = Vec::with_capacity(spec.n_layer);

    for layer_idx in 0..spec.n_layer {
        let attn_norm = weights.get(&WeightId::AttnNorm { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_norm.weight"),
            }
        })?;
        let q_weight = weights.get(&WeightId::AttnQ { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_q.weight"),
            }
        })?;
        let k_weight = weights.get(&WeightId::AttnK { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_k.weight"),
            }
        })?;
        let v_weight = weights.get(&WeightId::AttnV { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_v.weight"),
            }
        })?;
        let o_weight = weights.get(&WeightId::AttnO { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_output.weight"),
            }
        })?;
        let ffn_norm = weights.get(&WeightId::FfnNorm { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_norm.weight"),
            }
        })?;
        let gate = weights.get(&WeightId::FfnGate { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_gate.weight"),
            }
        })?;
        let up = weights.get(&WeightId::FfnUp { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_up.weight"),
            }
        })?;
        let down = weights.get(&WeightId::FfnDown { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_down.weight"),
            }
        })?;

        let attn_input = ops.rmsnorm(&hidden, attn_norm, spec.rms_eps)?;
        let q = ops.matmul(&attn_input, q_weight)?.data;
        let k = ops.matmul(&attn_input, k_weight)?.data;
        let v = ops.matmul(&attn_input, v_weight)?.data;
        let attn_output_vec = cpu_helical_attention_single(
            ops,
            spec,
            cache,
            layer_idx,
            &q,
            &k,
            &v,
            o_weight,
            keep_sink,
        )?;
        let attn_output = Tensor::new(attn_output_vec, vec![1, spec.n_embd])?;
        let post_attn = ops.add(&hidden, &attn_output)?;
        let ffn_input = ops.rmsnorm(&post_attn, ffn_norm, spec.rms_eps)?;
        let ffn_output = ops.ffn_swiglu(&ffn_input, gate, up, down)?;
        let output = ops.add(&post_attn, &ffn_output)?;

        if capture_debug {
            out.push(CpuLayerDebug {
                q,
                k,
                v,
                post_attn: post_attn.data.clone(),
                ffn_out: ffn_output.data.clone(),
                output: output.data.clone(),
            });
        }

        hidden = output;
    }

    Ok((out, hidden.data.clone()))
}

fn cpu_helical_attention_single(
    ops: &OpDispatcher,
    spec: &ModelSpec,
    cache: &mut CpuHelicalKvCache,
    layer_idx: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o_weight: &Tensor,
    keep_sink: usize,
) -> Result<Vec<f32>> {
    let current_pos = cache.len();
    let seq_len = current_pos + 1;
    let logical_query_pos = cache.window_base() + current_pos;
    let group_size = spec.n_head / spec.n_head_kv;
    let n_pairs = spec.rope_dim / 2;
    let scale = 1.0 / (spec.head_dim as f32).sqrt();

    cache.write_layer(layer_idx, k, v);

    let mut attn_concat = vec![0.0f32; spec.n_embd];
    let head_kv_span = spec.n_head_kv * spec.head_dim;

    for head_idx in 0..spec.n_head {
        let kv_head_idx = head_idx / group_size;
        let q_base = head_idx * spec.head_dim;
        let k_head_base = kv_head_idx * spec.head_dim;
        let mut scores = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            let rel = if pos < keep_sink {
                (logical_query_pos - pos).min(cache.max_len() - 1)
            } else {
                current_pos - pos
            };

            let cache_base = pos * head_kv_span + k_head_base;
            let mut dot_qk = 0.0f32;
            for pair_idx in 0..n_pairs {
                let doff = pair_idx * 2;
                let freq = 1.0 / spec.rope_base.powf((2.0 * pair_idx as f32) / spec.rope_dim as f32);
                let angle = rel as f32 * spec.rope_scale * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let q_re = q[q_base + doff];
                let q_im = q[q_base + doff + 1];
                let k_re = cache.layers[layer_idx].k[cache_base + doff];
                let k_im = cache.layers[layer_idx].k[cache_base + doff + 1];
                dot_qk += (q_re * k_re + q_im * k_im) * cos_a
                    + (q_re * k_im - q_im * k_re) * sin_a;
            }
            for dim_idx in (n_pairs * 2)..spec.head_dim {
                dot_qk += q[q_base + dim_idx] * cache.layers[layer_idx].k[cache_base + dim_idx];
            }
            scores.push(dot_qk * scale);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = Vec::with_capacity(seq_len);
        let mut sum = 0.0f32;
        for score in scores {
            let weight = (score - max_score).exp();
            sum += weight;
            weights.push(weight);
        }
        for weight in &mut weights {
            *weight /= sum;
        }

        for dim_idx in 0..spec.head_dim {
            let mut accum = 0.0f32;
            for (pos, weight) in weights.iter().enumerate() {
                let cache_base = pos * head_kv_span + k_head_base;
                accum += *weight * cache.layers[layer_idx].v[cache_base + dim_idx];
            }
            attn_concat[q_base + dim_idx] = accum;
        }
    }

    let attn_tensor = Tensor::new(attn_concat, vec![1, spec.n_embd])?;
    Ok(ops.matmul(&attn_tensor, o_weight)?.data)
}

#[allow(clippy::too_many_arguments)]
fn teacher_force_gpu_prefix_helical(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_model: &BindlessModel,
    pipeline: &BindlessPipeline,
    shift_pipeline: &RopeShiftPipeline,
    spec: &ModelSpec,
    gpu_kv: &mut GpuKvCache,
    embd_weight_offset: u64,
    row_bytes: u32,
    dim: u32,
    prefix_tokens: &[usize],
    decode_start_index: usize,
    chunk_tokens: u32,
    keep_sink: u32,
    helical_shift_amt: Option<u32>,
) -> Result<()> {
    let prefill_len = decode_start_index.min(prefix_tokens.len());
    if prefill_len > 0 {
        let prefix_embd = dequantize_embeddings(
            pipeline,
            device,
            queue,
            gpu_model,
            embd_weight_offset,
            row_bytes,
            dim,
            &prefix_tokens[..prefill_len],
        );
        let _ = pipeline.run_full_model_prefill_chunked_with_cache_state(
            device,
            queue,
            gpu_model,
            &prefix_embd,
            None,
            0,
            Some((gpu_kv.get_k_buffers(), gpu_kv.get_v_buffers())),
            spec,
            chunk_tokens,
        );
        for _ in 0..prefill_len {
            gpu_kv.increment();
        }
    }

    if decode_start_index >= prefix_tokens.len() {
        return Ok(());
    }

    let shift_amt = helical_shift_amt.unwrap_or_else(|| gpu_kv.max_len() / 4);
    let layer_params = LayerParams {
        dim: spec.n_embd as u32,
        head_count: spec.n_head as u32,
        head_count_kv: spec.n_head_kv as u32,
        head_dim: spec.head_dim as u32,
        rms_eps: spec.rms_eps,
        ffn_dim: spec.ff_dim as u32,
        temp_stride: spec.temp_buffer_size as u32,
        padding: 0,
    };

    for &token_id in &prefix_tokens[decode_start_index..] {
        let current_len = gpu_kv.get_seq_len();
        if current_len >= gpu_kv.max_len() - 4 {
            for layer_idx in 0..spec.n_layer {
                shift_pipeline.execute(
                    device,
                    queue,
                    gpu_kv.get_k_buffer(layer_idx),
                    gpu_kv.get_v_buffer(layer_idx),
                    keep_sink,
                    shift_amt,
                    current_len,
                    spec.n_head_kv as u32,
                    spec.head_dim as u32,
                    spec.rope_dim as u32,
                    spec.rope_base,
                    gpu_kv.max_len(),
                );
            }
            gpu_kv.set_seq_len(current_len - shift_amt);
            gpu_kv.advance_window_base(shift_amt);
        }

        let row_offset = embd_weight_offset + token_id as u64 * row_bytes as u64;
        let mut layer_output =
            pipeline.run_dequant_request(device, queue, gpu_model, row_offset as u32, dim);

        for layer_idx in 0..spec.n_layer {
            let offsets = gpu_model
                .metadata
                .get_layer_offsets(layer_idx, spec.arch_string())
                .ok_or_else(|| LibshimmyError::MissingTensor {
                    name: format!("layer_offsets_{layer_idx}"),
                })?;
            layer_output = pipeline.run_layer_with_cache(
                device,
                queue,
                gpu_model,
                gpu_kv,
                layer_idx,
                &layer_output,
                offsets,
                layer_params,
            );
        }

        gpu_kv.increment();
    }

    Ok(())
}

fn simulate_helical_partition_state(
    prefix_len: usize,
    helical_max_len: usize,
    keep_sink: usize,
    helical_shift_amt: Option<usize>,
) -> HelicalPartitionState {
    let shift_amt = helical_shift_amt.unwrap_or(helical_max_len / 4);
    let mut current_len = 0usize;
    let mut window_base = 0usize;

    for _ in 0..prefix_len {
        if current_len >= helical_max_len.saturating_sub(4) {
            current_len -= shift_amt;
            window_base += shift_amt;
        }
        current_len += 1;
    }

    if current_len >= helical_max_len.saturating_sub(4) {
        current_len -= shift_amt;
        window_base += shift_amt;
    }

    HelicalPartitionState {
        keep_sink,
        shift_amt,
        window_base_before_target: window_base,
        compact_query_pos_before_target: current_len,
        logical_query_pos: window_base + current_len,
    }
}

fn classify_bucket(abs_pos: usize, partition: &HelicalPartitionState) -> BucketKind {
    if abs_pos < partition.keep_sink {
        BucketKind::Sink
    } else if abs_pos == partition.logical_query_pos {
        BucketKind::CurrentToken
    } else if abs_pos < partition.window_base_before_target + partition.keep_sink {
        BucketKind::DroppedBand
    } else {
        BucketKind::RetainedWindow
    }
}

enum BucketKind {
    Sink,
    RetainedWindow,
    DroppedBand,
    CurrentToken,
}

fn add_bucket_mass(bucket: &mut BucketMass, kind: BucketKind, weight: f32) {
    match kind {
        BucketKind::Sink => bucket.sink += weight,
        BucketKind::RetainedWindow => bucket.retained_window += weight,
        BucketKind::DroppedBand => bucket.dropped_band += weight,
        BucketKind::CurrentToken => bucket.current_token += weight,
    }
}

fn summarize_layer_attention_masses(
    layer_idx: usize,
    heads: Vec<HeadAttentionMass>,
) -> LayerAttentionMass {
    let head_count = heads.len().max(1) as f32;
    let avg_dropped_band_mass = heads.iter().map(|head| head.mass.dropped_band).sum::<f32>() / head_count;
    let max_dropped_band_mass = heads
        .iter()
        .map(|head| head.mass.dropped_band)
        .fold(0.0f32, f32::max);
    let avg_sink_mass = heads.iter().map(|head| head.mass.sink).sum::<f32>() / head_count;
    let avg_retained_window_mass = heads
        .iter()
        .map(|head| head.mass.retained_window)
        .sum::<f32>()
        / head_count;
    let avg_current_token_mass = heads
        .iter()
        .map(|head| head.mass.current_token)
        .sum::<f32>()
        / head_count;

    LayerAttentionMass {
        layer_idx,
        avg_dropped_band_mass,
        max_dropped_band_mass,
        avg_sink_mass,
        avg_retained_window_mass,
        avg_current_token_mass,
        heads,
    }
}

fn compute_attention_mass_probe(
    weights: &HashMap<WeightId, Tensor>,
    ops: &OpDispatcher,
    spec: &ModelSpec,
    prompt_tokens: &[usize],
    target_index: usize,
    helical_max_len: usize,
    keep_sink: usize,
    helical_shift_amt: Option<usize>,
) -> Result<AttentionMassProbe> {
    let partition = simulate_helical_partition_state(
        target_index,
        helical_max_len,
        keep_sink,
        helical_shift_amt,
    );
    let mut cache = CpuHelicalKvCache::new(spec.n_ctx, spec.n_layer, spec.n_head_kv, spec.head_dim);

    for &token_id in &prompt_tokens[..target_index] {
        let token_input = token_embedding(weights, token_id, spec.n_embd)?;
        let _ = cpu_full_run_token_with_mass(
            weights,
            ops,
            spec,
            &mut cache,
            &token_input,
            false,
            &partition,
        )?;
        cache.finish_token()?;
    }

    let target_input = token_embedding(weights, prompt_tokens[target_index], spec.n_embd)?;
    let layers = cpu_full_run_token_with_mass(
        weights,
        ops,
        spec,
        &mut cache,
        &target_input,
        true,
        &partition,
    )?;

    Ok(AttentionMassProbe {
        keep_sink,
        helical_max_len,
        shift_amt: partition.shift_amt,
        window_base_before_target: partition.window_base_before_target,
        compact_query_pos_before_target: partition.compact_query_pos_before_target,
        logical_query_pos: partition.logical_query_pos,
        dropped_band_start: keep_sink,
        dropped_band_end: partition.window_base_before_target + keep_sink - 1,
        layers,
    })
}

fn cpu_full_run_token_with_mass(
    weights: &HashMap<WeightId, Tensor>,
    ops: &OpDispatcher,
    spec: &ModelSpec,
    cache: &mut CpuHelicalKvCache,
    target_input: &[f32],
    capture_mass: bool,
    partition: &HelicalPartitionState,
) -> Result<Vec<LayerAttentionMass>> {
    let mut hidden = Tensor::new(gpu_to_row(target_input.to_vec(), spec.n_embd), vec![1, spec.n_embd])?;
    let mut layer_masses = Vec::with_capacity(spec.n_layer);

    for layer_idx in 0..spec.n_layer {
        let attn_norm = weights.get(&WeightId::AttnNorm { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_norm.weight"),
            }
        })?;
        let q_weight = weights.get(&WeightId::AttnQ { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_q.weight"),
            }
        })?;
        let k_weight = weights.get(&WeightId::AttnK { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_k.weight"),
            }
        })?;
        let v_weight = weights.get(&WeightId::AttnV { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_v.weight"),
            }
        })?;
        let o_weight = weights.get(&WeightId::AttnO { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.attn_output.weight"),
            }
        })?;
        let ffn_norm = weights.get(&WeightId::FfnNorm { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_norm.weight"),
            }
        })?;
        let gate = weights.get(&WeightId::FfnGate { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_gate.weight"),
            }
        })?;
        let up = weights.get(&WeightId::FfnUp { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_up.weight"),
            }
        })?;
        let down = weights.get(&WeightId::FfnDown { layer: layer_idx }).ok_or_else(|| {
            LibshimmyError::MissingTensor {
                name: format!("blk.{layer_idx}.ffn_down.weight"),
            }
        })?;

        let attn_input = ops.rmsnorm(&hidden, attn_norm, spec.rms_eps)?;
        let q = ops.matmul(&attn_input, q_weight)?.data;
        let k = ops.matmul(&attn_input, k_weight)?.data;
        let v = ops.matmul(&attn_input, v_weight)?.data;
        let (attn_output_vec, head_masses) = cpu_full_attention_single(
            ops,
            spec,
            cache,
            layer_idx,
            &q,
            &k,
            &v,
            o_weight,
            capture_mass,
            partition,
        )?;
        let attn_output = Tensor::new(attn_output_vec, vec![1, spec.n_embd])?;
        let post_attn = ops.add(&hidden, &attn_output)?;
        let ffn_input = ops.rmsnorm(&post_attn, ffn_norm, spec.rms_eps)?;
        let ffn_output = ops.ffn_swiglu(&ffn_input, gate, up, down)?;
        let output = ops.add(&post_attn, &ffn_output)?;

        if let Some(heads) = head_masses {
            layer_masses.push(summarize_layer_attention_masses(layer_idx, heads));
        }

        hidden = output;
    }

    Ok(layer_masses)
}

fn cpu_full_attention_single(
    ops: &OpDispatcher,
    spec: &ModelSpec,
    cache: &mut CpuHelicalKvCache,
    layer_idx: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o_weight: &Tensor,
    capture_mass: bool,
    partition: &HelicalPartitionState,
) -> Result<(Vec<f32>, Option<Vec<HeadAttentionMass>>)> {
    let current_pos = cache.len();
    let seq_len = current_pos + 1;
    let group_size = spec.n_head / spec.n_head_kv;
    let n_pairs = spec.rope_dim / 2;
    let scale = 1.0 / (spec.head_dim as f32).sqrt();

    cache.write_layer(layer_idx, k, v);

    let mut attn_concat = vec![0.0f32; spec.n_embd];
    let head_kv_span = spec.n_head_kv * spec.head_dim;
    let mut head_masses = if capture_mass {
        Some(Vec::with_capacity(spec.n_head))
    } else {
        None
    };

    for head_idx in 0..spec.n_head {
        let kv_head_idx = head_idx / group_size;
        let q_base = head_idx * spec.head_dim;
        let k_head_base = kv_head_idx * spec.head_dim;
        let mut scores = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            let rel = current_pos - pos;
            let cache_base = pos * head_kv_span + k_head_base;
            let mut dot_qk = 0.0f32;
            for pair_idx in 0..n_pairs {
                let doff = pair_idx * 2;
                let freq = 1.0 / spec.rope_base.powf((2.0 * pair_idx as f32) / spec.rope_dim as f32);
                let angle = rel as f32 * spec.rope_scale * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let q_re = q[q_base + doff];
                let q_im = q[q_base + doff + 1];
                let k_re = cache.layers[layer_idx].k[cache_base + doff];
                let k_im = cache.layers[layer_idx].k[cache_base + doff + 1];
                dot_qk += (q_re * k_re + q_im * k_im) * cos_a
                    + (q_re * k_im - q_im * k_re) * sin_a;
            }
            for dim_idx in (n_pairs * 2)..spec.head_dim {
                dot_qk += q[q_base + dim_idx] * cache.layers[layer_idx].k[cache_base + dim_idx];
            }
            scores.push(dot_qk * scale);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = Vec::with_capacity(seq_len);
        let mut sum = 0.0f32;
        for score in scores {
            let weight = (score - max_score).exp();
            sum += weight;
            weights.push(weight);
        }
        for weight in &mut weights {
            *weight /= sum;
        }

        if let Some(heads) = head_masses.as_mut() {
            let mut mass = BucketMass::default();
            for (pos, weight) in weights.iter().enumerate() {
                add_bucket_mass(&mut mass, classify_bucket(pos, partition), *weight);
            }
            heads.push(HeadAttentionMass { head_idx, mass });
        }

        for dim_idx in 0..spec.head_dim {
            let mut accum = 0.0f32;
            for (pos, weight) in weights.iter().enumerate() {
                let cache_base = pos * head_kv_span + k_head_base;
                accum += *weight * cache.layers[layer_idx].v[cache_base + dim_idx];
            }
            attn_concat[q_base + dim_idx] = accum;
        }
    }

    let attn_tensor = Tensor::new(attn_concat, vec![1, spec.n_embd])?;
    Ok((ops.matmul(&attn_tensor, o_weight)?.data, head_masses))
}