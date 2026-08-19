//! Decode ≡ re-prefill gate.
//! Prefill N-1 tokens + decode last token must match full N-token prefill logits.
//! Writes JSON; exit 0 iff max|Δ| ≤ tol and argmax matches (both head modes that run).
//!
//! Usage: decode_gate <gguf> "<prompt>" [out.json]
//! Prompt must tokenize to ≥7 tokens.

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::BindlessPipeline;
use airframe::runtime::gpu::GpuRuntime;
use serde_json::json;
use shimmytok::Tokenizer;
use std::fs::File;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: decode_gate <model.gguf> \"<prompt>\" [out.json]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let prompt = &args[2];
    let out_path = if args.len() >= 4 {
        PathBuf::from(&args[3])
    } else {
        PathBuf::from("decode_gate.json")
    };

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
        .expect("No GPU adapter");
    let adapter_limits = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter_limits.max_buffer_size;
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
        .expect("device");

    let tokenizer = Tokenizer::from_gguf_file(model_path)?;
    let mut meta_file = File::open(model_path)?;
    let meta = BindlessMetadata::new(&mut meta_file);
    drop(meta_file);
    let mut spec = meta.to_model_spec();
    spec.n_ctx = 8192;
    let gpu_model = BindlessModel::load_from_disk(&device, &PathBuf::from(model_path), Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    let f32_head = GpuRuntime::load_output_head_f32(model_path, &gpu_model, &device, &spec)
        .ok()
        .flatten();

    let n_layers = gpu_model.metadata.compiled_layers.len();
    let n_head_kv = spec.n_head_kv as u32;
    let head_dim = spec.head_dim as u32;
    let max_seq_len = 128u32;
    let dim = spec.n_embd as usize;
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
    let embd_quant_type = gpu_model
        .metadata
        .get_tensor_type("token_embd.weight")
        .unwrap_or(2);
    let embd_row_bytes: usize = match embd_quant_type {
        0 => dim * 4,
        1 => dim * 2,
        2 => (dim / 32) * 18,
        6 => (dim / 32) * 22,
        8 => (dim / 32) * 34,
        12 => (dim / 256) * 144,
        13 => (dim / 256) * 176,
        14 => (dim / 256) * 210,
        _ => panic!("unsupported embd quant {embd_quant_type}"),
    };

    let tokens = tokenizer.encode(prompt, true)?;
    if tokens.len() < 7 {
        eprintln!("prompt must yield >=7 tokens, got {}", tokens.len());
        std::process::exit(2);
    }
    let toks: Vec<u32> = tokens[0..7].to_vec();

    let mut embs = Vec::with_capacity(7);
    for &t in &toks {
        let row_offset = embd_weight_offset + (t as u64 * embd_row_bytes as u64);
        embs.push(pipeline.run_dequant_any_hot(
            &device,
            &queue,
            &gpu_model,
            row_offset as u32,
            dim as u32,
            embd_quant_type,
        ));
    }
    let concat = |slice: &[Vec<f32>]| -> Vec<f32> {
        let mut v = Vec::with_capacity(slice.len() * dim);
        for e in slice {
            v.extend_from_slice(e);
        }
        v
    };
    let emb_6 = concat(&embs[0..6]);
    let emb_7 = concat(&embs[0..7]);
    let emb_7th = &embs[6];
    let _dim_u32 = dim as u32;

    let mk_kv = || KVCache::new(&device, n_layers, n_head_kv, head_dim, max_seq_len);
    let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
    let argmax = |v: &[f32]| -> (usize, f32) {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 0.0))
    };

    let tol = 1e-2f32;
    let mut modes = Vec::new();

    let run = |head: Option<&wgpu::Buffer>,
               name: &str|
     -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let kv_ref = mk_kv();
        let rb = pipeline.run_full_model_prefill_chunked_with_cache_state(
            &device,
            &queue,
            &gpu_model,
            &emb_7,
            head,
            0,
            Some((kv_ref.get_k_buffers(), kv_ref.get_v_buffers())),
            &spec,
            512,
            None,
        )?;
        let logits_ref = rb.2;

        let kv_pf = mk_kv();
        pipeline.run_full_model_prefill_chunked_with_cache_state(
            &device,
            &queue,
            &gpu_model,
            &emb_6,
            head,
            0,
            Some((kv_pf.get_k_buffers(), kv_pf.get_v_buffers())),
            &spec,
            512,
            None,
        )?;
        let rd = pipeline.run_full_model_prefill_chunked_with_cache_state(
            &device,
            &queue,
            &gpu_model,
            emb_7th,
            head,
            6,
            Some((kv_pf.get_k_buffers(), kv_pf.get_v_buffers())),
            &spec,
            1,
            None,
        )?;
        let logits_dec = rd.2;

        let mut maxdiff = 0.0f32;
        for (a, b) in logits_dec.iter().zip(logits_ref.iter()) {
            maxdiff = maxdiff.max((a - b).abs());
        }
        let (tok_dec, _) = argmax(&logits_dec);
        let (tok_ref, _) = argmax(&logits_ref);
        let match_ok = maxdiff <= tol && tok_dec == tok_ref;
        eprintln!(
            "[{name}] max|Δ|={maxdiff:.6e} dec_arg={tok_dec} ref_arg={tok_ref} {}",
            if match_ok { "MATCH" } else { "DIVERGE" }
        );
        Ok(json!({
            "mode": name,
            "max_abs_diff": maxdiff,
            "decode_argmax": tok_dec,
            "ref_argmax": tok_ref,
            "decode_rms": rms(&logits_dec),
            "ref_rms": rms(&logits_ref),
            "match": match_ok,
        }))
    };

    modes.push(run(None, "BLOB")?);
    if let Some(ref h) = f32_head {
        modes.push(run(Some(h), "F32")?);
    }

    // Pass if every mode that ran matches (BLOB required; F32 if present).
    let ok = modes.iter().all(|m| m["match"].as_bool() == Some(true));
    let doc = json!({
        "schema": "airframe.decode_gate.v1",
        "ok": ok,
        "tol": tol,
        "token_ids": toks,
        "modes": modes,
    });
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc)?)?;
    eprintln!("wrote {} ok={ok}", out_path.display());
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
