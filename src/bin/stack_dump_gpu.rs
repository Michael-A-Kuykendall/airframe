//! OBS-1: Product multi-token stack dump → airframe.stack.v1 JSON.
//!
//! ```text
//! stack_dump_gpu <gguf> "<prompt>" <out.json> [--top-k N]
//! ```
//! Requires `--features isf`. Set `SHIMMY_MAX_CTX=8192` on 12GB GPUs.

use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::inference::{
    clear_stack_layer_capture_sink, set_stack_layer_capture_sink, StackLayerSnap,
};
use airframe::backend::bindless::pipeline::BindlessPipeline;
use chrono::Utc;
use serde::Serialize;
use serde_json::json;
use shimmytok::Tokenizer;
use std::path::PathBuf;

#[derive(Serialize, Clone)]
struct TokenScore {
    id: u32,
    piece: String,
    logit: f32,
}

fn rms(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let s: f32 = v.iter().map(|x| x * x).sum();
    (s / v.len() as f32).sqrt()
}

fn top_k(logits: &[f32], k: usize, tok: &Tokenizer) -> (TokenScore, Vec<TokenScore>) {
    let mut idx: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = Vec::new();
    for &i in idx.iter().take(k) {
        let piece = tok
            .token_to_piece(i as u32)
            .unwrap_or_else(|_| format!("<id:{}>", i));
        out.push(TokenScore {
            id: i as u32,
            piece,
            logit: logits[i],
        });
    }
    let argmax = out.first().cloned().unwrap_or(TokenScore {
        id: 0,
        piece: String::new(),
        logit: f32::NEG_INFINITY,
    });
    (argmax, out)
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: stack_dump_gpu <gguf> \"<prompt>\" <out.json> [--top-k N]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let prompt = &args[2];
    let out_path = &args[3];
    let mut top_k_n: usize = std::env::var("AIRFRAME_STACK_DUMP_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut i = 4;
    while i < args.len() {
        if args[i] == "--top-k" && i + 1 < args.len() {
            top_k_n = args[i + 1].parse().unwrap_or(top_k_n);
            i += 2;
        } else {
            i += 1;
        }
    }

    eprintln!("[stack_dump] model={}", model_path);
    eprintln!("[stack_dump] prompt={:?}", prompt);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        flags: wgpu::InstanceFlags::default().with_env(),
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|_| "no GPU adapter")?;
    eprintln!(
        "[stack_dump] adapter={:?} {:?}",
        adapter.get_info().name,
        adapter.get_info().device_type
    );
    let adapter_limits = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter_limits.max_buffer_size;
    limits.max_storage_buffers_per_shader_stage =
        adapter_limits.max_storage_buffers_per_shader_stage.max(14);
    limits.max_compute_invocations_per_workgroup = 256;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("stack_dump_gpu"),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await?;

    let tokenizer = Tokenizer::from_gguf_file(model_path)?;
    let mut header = std::fs::File::open(model_path)?;
    let meta = BindlessMetadata::new(&mut header);
    drop(header);
    let mut spec = meta.to_model_spec();
    if let Ok(max_ctx) = std::env::var("SHIMMY_MAX_CTX") {
        if let Ok(n) = max_ctx.parse::<usize>() {
            spec.n_ctx = n.min(spec.n_ctx);
            spec = spec.compute_derived();
        }
    } else if spec.n_ctx > 8192 {
        spec.n_ctx = 8192;
        spec = spec.compute_derived();
    }

    let prompt_tokens = tokenizer.encode(prompt, true)?;
    if prompt_tokens.len() < 2 {
        eprintln!(
            "[stack_dump] ERROR: multi-token prompt required (got {})",
            prompt_tokens.len()
        );
        std::process::exit(2);
    }
    let pieces: Vec<String> = prompt_tokens
        .iter()
        .map(|&id| {
            tokenizer
                .token_to_piece(id)
                .unwrap_or_else(|_| format!("<{}>", id))
        })
        .collect();
    eprintln!(
        "[stack_dump] tokens={:?} n={}",
        prompt_tokens,
        prompt_tokens.len()
    );

    let model = BindlessModel::load_from_disk(&device, &PathBuf::from(model_path), Some(&spec));
    let pipeline = BindlessPipeline::new(&device);
    let dim = spec.n_embd;
    let embd_quant = model
        .metadata
        .get_tensor_type("token_embd.weight")
        .unwrap_or(0);
    let embd_off = model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight");
    let row_bytes = match embd_quant {
        0 => dim * 4,
        1 => dim * 2,
        2 => (dim / 32) * 18,
        12 => (dim / 256) * 144,
        13 => (dim / 256) * 176,
        14 => (dim / 256) * 210,
        _ => (dim / 256) * 210,
    } as u64;

    let mut embd = Vec::with_capacity(prompt_tokens.len() * dim);
    for &tid in &prompt_tokens {
        let row_offset = embd_off + tid as u64 * row_bytes;
        let row = pipeline.run_dequant_any_hot(
            &device,
            &queue,
            &model,
            row_offset as u32,
            dim as u32,
            embd_quant,
        );
        embd.extend(row);
    }
    let last_pos = prompt_tokens.len() - 1;
    let emb_last = &embd[last_pos * dim..(last_pos + 1) * dim];
    let emb_first8: Vec<f32> = emb_last.iter().take(8).copied().collect();
    let emb_rms = rms(emb_last);
    let emb_nans = emb_last.iter().filter(|x| x.is_nan()).count() as u32;

    let mut snaps: Vec<StackLayerSnap> = Vec::new();
    set_stack_layer_capture_sink(&mut snaps);

    let batch = prompt_tokens.len() as u32;
    let (pre_norm, post_norm, logits) = pipeline.run_full_model_prefill_chunked_with_cache_state(
        &device,
        &queue,
        &model,
        &embd,
        None,
        0,
        None,
        &spec,
        batch.max(1),
    )?;
    clear_stack_layer_capture_sink();

    let (argmax, topk) = top_k(&logits, top_k_n, &tokenizer);
    let logits_nans = logits.iter().filter(|x| x.is_nan()).count() as u32;
    let logits_max = logits
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let layers_json: Vec<serde_json::Value> = if snaps.is_empty() {
        vec![json!({
            "layer_idx": 0,
            "position": last_pos,
            "residual": {
                "rms": rms(&post_norm),
                "first8": post_norm.iter().take(8).copied().collect::<Vec<_>>(),
                "nan_count": post_norm.iter().filter(|x| x.is_nan()).count()
            },
            "stages": {},
            "tensors": { "status": "unsupported", "reason": "no_layer_snaps" }
        })]
    } else {
        snaps
            .iter()
            .map(|s| {
                let mut stages_obj = serde_json::Map::new();
                for st in &s.stages {
                    stages_obj.insert(
                        st.name.clone(),
                        json!({
                            "rms": st.rms,
                            "first8": st.first8,
                            "nan_count": st.nan_count,
                            "count": st.count,
                            "sampled": st.sampled,
                            "buffer": st.buffer,
                            "offset_elems": st.offset_elems
                        }),
                    );
                }
                let residual_in = s.residual_in.as_ref().map(|r| {
                    json!({
                        "rms": r.rms,
                        "first8": r.first8,
                        "nan_count": r.nan_count,
                        "count": r.count,
                        "sampled": r.sampled,
                        "buffer": r.buffer,
                        "offset_elems": r.offset_elems
                    })
                });
                json!({
                    "layer_idx": s.layer_idx,
                    "position": s.position,
                    "residual_in": residual_in,
                    "residual": {
                        "rms": s.rms,
                        "first8": s.first8,
                        "nan_count": s.nan_count
                    },
                    "residual_out": {
                        "rms": s.rms,
                        "first8": s.first8,
                        "nan_count": s.nan_count
                    },
                    "stages": stages_obj,
                    "tensors": { "status": "deferred", "reason": "activation_peel_priority" }
                })
            })
            .collect()
    };

    let arch = format!("{:?}", spec.arch).to_lowercase();
    let doc = json!({
        "schema": "airframe.stack.v1",
        "engine": "airframe_product",
        "engine_detail": "run_full_model_prefill_chunked_with_cache_state",
        "model_path": model_path,
        "prompt": prompt,
        "captured_at": Utc::now().to_rfc3339(),
        "config": {
            "arch": arch,
            "n_layer": spec.n_layer,
            "n_embd": spec.n_embd,
            "n_head": spec.n_head,
            "n_kv_head": spec.n_head_kv,
            "head_dim": spec.head_dim,
            "rope_base": spec.rope_base,
            "rope_dim": spec.rope_dim,
            "rms_eps": spec.rms_eps,
            "norm_kind": format!("{:?}", spec.norm_kind).to_lowercase(),
            "uses_layer_norm": spec.uses_layer_norm(),
            "qk_norm": spec.has_qk_norm,
            "n_ctx_capped": spec.n_ctx
        },
        "tokens": {
            "ids": prompt_tokens,
            "pieces": pieces,
            "add_bos": false
        },
        "embedding": {
            "position": last_pos,
            "rms": emb_rms,
            "first8": emb_first8,
            "nan_count": emb_nans
        },
        "layers": layers_json,
        "final": {
            "post_norm_rms": rms(&post_norm),
            "pre_norm_rms": rms(&pre_norm),
            "logits": {
                "len": logits.len(),
                "rms": rms(&logits),
                "max": logits_max,
                "nan_count": logits_nans,
                "argmax": {
                    "id": argmax.id,
                    "piece": argmax.piece,
                    "logit": argmax.logit
                },
                "top_k": topk
            }
        },
        "decode": { "status": "skipped", "reason": "OBS-1_prefill_only" },
        "notes": [
            "PEEL product multi-token prefill (run_full_model_*)",
            format!("layer_snaps={}", snaps.len()),
            format!(
                "stage_counts_L0={}",
                snaps.first().map(|s| s.stages.len()).unwrap_or(0)
            )
        ]
    });

    if let Some(p) = PathBuf::from(out_path).parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&doc)?)?;
    eprintln!(
        "[stack_dump] wrote {} top1={} {:?} layers={}",
        out_path,
        argmax.id,
        argmax.piece,
        snaps.len()
    );
    Ok(())
}
