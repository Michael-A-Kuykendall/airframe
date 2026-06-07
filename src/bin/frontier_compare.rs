use airframe::backend::bindless::kv_cache::KVCache as GpuKvCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams, RMSNormParams};
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
    let weights = container.weights;

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
    let row_bytes = (dim / 32) * 18;

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
            spec.n_embd / spec.n_head,
        );

        if !prefix_tokens.is_empty() {
            eprintln!("[CPU prefill] {} tokens ...", prefix_tokens.len());
            let _ = full_model.forward(prefix_tokens, &weights, &mut cpu_kv, &ops)?;
            cpu_kv.complete_prefill(prefix_tokens.len())?;
            eprintln!("[CPU prefill] done");
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
        quant_type: 0,
        attn_logit_softcap: 0.0,
        post_norm_enabled: 0,
        qk_norm_enabled: 0,
        layer_norm_enabled: 0,
        ffn_kind_policy: 0,
        qkv_layout_policy: 0,
    };

    eprintln!("[GPU vs CPU] comparing {} layers ...", spec.n_layer);
    let mut gpu_hidden = target_input.clone();
    let mut layer_comparisons = Vec::with_capacity(spec.n_layer);
    // Note: layer_idx is used legitimately as an index for cpu_layers, metadata, and pipeline
    #[allow(clippy::needless_range_loop)]
    for layer_idx in 0..spec.n_layer {
        let offsets = gpu_model
            .metadata
            .get_layer_offsets(layer_idx, spec.arch_string())
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: format!("layer_offsets_{layer_idx}"),
            })?;

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
    let output_proj =
        weights
            .get(&WeightId::OutputProj)
            .ok_or_else(|| LibshimmyError::MissingTensor {
                name: "output.weight".to_string(),
            })?;

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
        weights_offset: norm_weight_offset,
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
    };

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
