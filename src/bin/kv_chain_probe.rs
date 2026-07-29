use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::BindlessPipeline;
use airframe::runtime::gpu::GpuRuntime;
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
        eprintln!("Usage: kv_chain_probe <model.gguf> \"<prompt>\"");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let prompt = &args[2];

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

    let tokenizer = Tokenizer::from_gguf_file(model_path)?;
    let mut meta_file = File::open(model_path)?;
    let meta = BindlessMetadata::new(&mut meta_file);
    drop(meta_file);
    let mut spec = meta.to_model_spec();
    spec.n_ctx = 8192;
    let gpu_model = BindlessModel::load_from_disk(&device, &PathBuf::from(model_path), Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    let f32_head = GpuRuntime::load_output_head_f32(model_path, &gpu_model, &device, &spec)
        .map_err(|e| {
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                as Box<dyn std::error::Error>
        })?;

    let n_layers = gpu_model.metadata.compiled_layers.len();
    let n_head_kv = spec.n_head_kv as u32;
    let head_dim = spec.head_dim as u32;
    let dim = spec.n_embd;
    let dim_u32 = dim as u32;
    let embd_quant_type = gpu_model
        .metadata
        .get_tensor_type("token_embd.weight")
        .unwrap_or(2);
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
    let embd_row_bytes = match embd_quant_type {
        0 => dim * 4, 1 => dim * 2,
        2 => (dim / 32) * 18, 6 => (dim / 32) * 22,
        8 => (dim / 32) * 34,
        12 => (dim / 256) * 144, 13 => (dim / 256) * 176,
        14 => (dim / 256) * 210,
        _ => panic!("unsupported quant type {}", embd_quant_type),
    };

    let tokens = tokenizer.encode(prompt, true)?;
    let need = 12;
    assert!(tokens.len() >= need, "prompt must tokenize to >={} tokens, got {}", need, tokens.len());
    let toks: Vec<u32> = tokens[0..need].to_vec();
    eprintln!("[chain] prompt_len={} tokens[0..{}]={:?}", tokens.len(), need, toks);

    let mut embs = Vec::with_capacity(need);
    for &t in &toks {
        let row_offset = embd_weight_offset + (t as u64 * embd_row_bytes as u64);
        let e = pipeline.run_dequant_any_hot(
            &device, &queue, &gpu_model, row_offset as u32, dim_u32, embd_quant_type,
        );
        embs.push(e);
    }
    let concat = |s: &[Vec<f32>]| -> Vec<f32> {
        let mut v = Vec::with_capacity(s.len() * dim);
        for e in s { v.extend_from_slice(e); }
        v
    };
    let emb_prefill = concat(&embs[0..need]);
    let emb_first_decode = embs[need - 1].clone();

    let rms = |v: &[f32]| -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    };
    let argmax = |logits: &[f32]| -> (usize, String) {
        let (idx, _) = logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap_or((0, &0.0));
        let piece = tokenizer.decode_single(idx as u32, true).unwrap_or_default();
        (idx, piece)
    };

    let decode_steps = 6;

    for mode_name in ["BLOB", "F32"] {
        let head_override: Option<&wgpu::Buffer> = if mode_name == "F32" { Some(&f32_head) } else { None };

        eprintln!("\n=== MODE: {} ===", mode_name);

        // Reference: full N-token prefill -> logits at last position
        let kv_ref = KVCache::new(&device, n_layers, n_head_kv, head_dim, 8192);
        let rb_ref = pipeline.run_full_model_prefill_chunked_with_cache_state(
            &device, &queue, &gpu_model, &emb_prefill, head_override, 0,
            Some((kv_ref.get_k_buffers(), kv_ref.get_v_buffers())),
            &spec, 512,
        )?;
        let logits_ref = rb_ref.2;
        let (ref_idx, ref_piece) = argmax(&logits_ref);
        eprintln!("  REF: last-position logits -> argmax={} piece='{}' rms={:.6}", ref_idx, ref_piece, rms(&logits_ref));

        // Chain: prefill (need-1) tokens, then N decode steps
        let kv_chain = KVCache::new(&device, n_layers, n_head_kv, head_dim, 8192);
        let emb_prefill_short = concat(&embs[0..need - 1]);

        // Prefill (need-1) tokens
        pipeline.run_full_model_prefill_chunked_with_cache_state(
            &device, &queue, &gpu_model, &emb_prefill_short, head_override, 0,
            Some((kv_chain.get_k_buffers(), kv_chain.get_v_buffers())),
            &spec, 512,
        )?;

        // Sequential decode using greedy argmax feedback
        let mut cur_emb = emb_first_decode.clone();
        let mut pos = (need - 1) as u32;

        for step in 0..decode_steps {
            let (_, _, logits) = pipeline.run_full_model_prefill_chunked_with_cache_state(
                &device, &queue, &gpu_model, &cur_emb, head_override, pos,
                Some((kv_chain.get_k_buffers(), kv_chain.get_v_buffers())),
                &spec, 1,
            )?;
            let (tok_idx, tok_piece) = argmax(&logits);
            let diff: f32 = logits.iter().zip(logits_ref.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("  step {}: current_pos={} argmax={} piece={:?} max|Δ|_ref={:.6e}", step, pos, tok_idx, tok_piece, diff);

            // Build embedding for this argmax token (re-dequant)
            let row_offset = embd_weight_offset + (tok_idx as u64 * embd_row_bytes as u64);
            let offset32 = row_offset as u32;
            cur_emb = pipeline.run_dequant_any_hot(
                &device, &queue, &gpu_model, offset32, dim_u32, embd_quant_type,
            );
            pos += 1;
        }
    }

    Ok(())
}
