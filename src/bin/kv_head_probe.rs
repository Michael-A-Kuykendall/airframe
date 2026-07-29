// KV-Head Probe (decisive test for Qwen3 decode bug, playbook §3)
//
// Reproduces shimmy's EXACT decode orchestration:
//   - run_full_model_prefill_chunked_with_cache_state (shimmy's wrapper)
//   - head_override = Some(f32_head)  [shimmy's default for Qwen3]
//   - embeddings via run_dequant_any_hot (untied Qwen3 == shimmy path)
//   - KVCache with seq_len bookkeeping through a real prefill(6)+decode(6) flow
//
// For each head mode (BLOB=None, F32=Some) we compute:
//   P) prefill 6 tokens (positions 0..5) into KV, leaving buffers warm
//   D) decode step 0 at current_pos=6 with the 7th token embedding
//   R) 7-token single-pass prefill (positions 0..6) -> reference logits at pos 6
// The decode (D) must equal the reference (R). Whichever mode diverges is the bug.

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
        eprintln!("Usage: kv_head_probe <model.gguf> \"<prompt>\"");
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

    // Build F32 output head (shimmy's path)
    let f32_head = GpuRuntime::load_output_head_f32(model_path, &gpu_model, &device, &spec)
        .map_err(|e| {
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                as Box<dyn std::error::Error>
        })?;

    let n_layers = gpu_model.metadata.compiled_layers.len();
    let n_head_kv = spec.n_head_kv as u32;
    let head_dim = spec.head_dim as u32;
    let max_seq_len = 8192;

    // Embedding extraction (untied Qwen3 == shimmy: run_dequant_any_hot from token_embd)
    let embd_quant_type = gpu_model
        .metadata
        .get_tensor_type("token_embd.weight")
        .unwrap_or(2);
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
    let dim = spec.n_embd;
    let embd_row_bytes = match embd_quant_type {
        0 => dim * 4,
        1 => dim * 2,
        2 => (dim / 32) * 18,
        6 => (dim / 32) * 22,
        8 => (dim / 32) * 34,
        12 => (dim / 256) * 144,
        13 => (dim / 256) * 176,
        14 => (dim / 256) * 210,
        _ => panic!("unsupported embedding quant type: {}", embd_quant_type),
    };

    let tokens = tokenizer.encode(prompt, true)?;
    assert!(tokens.len() >= 7, "prompt must tokenize to >=7 tokens");
    let toks: Vec<u32> = tokens[0..7].to_vec();
    eprintln!("[kv_head_probe] tokens[0..7] = {:?}", toks);

    // Build embedding vectors
    let mut embs = Vec::with_capacity(7);
    for &t in &toks {
        let row_offset = embd_weight_offset + (t as u64 * embd_row_bytes as u64);
        let e = pipeline.run_dequant_any_hot(
            &device, &queue, &gpu_model, row_offset as u32, dim as u32, embd_quant_type,
        );
        embs.push(e);
    }
    let concat = |slice: &[Vec<f32>]| -> Vec<f32> {
        let mut v = Vec::with_capacity(slice.len() * dim);
        for e in slice { v.extend_from_slice(e); }
        v
    };
    let emb_6 = concat(&embs[0..6]);  // positions 0..5 (prefill)
    let emb_7 = concat(&embs[0..7]);  // positions 0..6 (full prefill reference)
    let emb_7th = &embs[6];           // 7th token embedding (decode input at pos 6)
    let dim_u32 = dim as u32;

    let mk_kv = || -> KVCache {
        KVCache::new(&device, n_layers, n_head_kv, head_dim, max_seq_len)
    };

    let rms = |v: &[f32]| -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    };
    let argmax = |v: &[f32]| -> (usize, f32) {
        v.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, &v)| (i, v)).unwrap_or((0, 0.0))
    };

    eprintln!("\n=== HEAD MODE TEST: BLOB (None) ===");
    let (maxdiff_blob, tok_dec_blob, tok_ref_blob) = run_mode(
        &device, &queue, &gpu_model, &pipeline, &spec,
        None,
        &emb_6, &emb_7, &emb_7th,
        dim_u32, mk_kv, &rms, &argmax,
    )?;

    eprintln!("\n=== HEAD MODE TEST: F32 (Some) ===");
    let (maxdiff_f32, tok_dec_f32, tok_ref_f32) = run_mode(
        &device, &queue, &gpu_model, &pipeline, &spec,
        Some(&f32_head),
        &emb_6, &emb_7, &emb_7th,
        dim_u32, mk_kv, &rms, &argmax,
    )?;

    eprintln!("\n=== SUMMARY ===");
    eprintln!("BLOB:  max|Δ|={:.6e}  decode_argmax={}  ref_argmax={}  {}", maxdiff_blob, tok_dec_blob, tok_ref_blob, if maxdiff_blob <= 1e-2 { "MATCH" } else { "DIVERGE" });
    eprintln!("F32:   max|Δ|={:.6e}  decode_argmax={}  ref_argmax={}  {}", maxdiff_f32, tok_dec_f32, tok_ref_f32, if maxdiff_f32 <= 1e-2 { "MATCH" } else { "DIVERGE" });

    Ok(())
}

fn run_mode(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    gpu_model: &BindlessModel,
    pipeline: &BindlessPipeline,
    spec: &airframe::core::spec::ModelSpec,
    head_override: Option<&wgpu::Buffer>,
    emb_6: &[f32],
    emb_7: &[f32],
    emb_7th: &[f32],
    dim: u32,
    mk_kv: impl Fn() -> KVCache,
    rms: &dyn Fn(&[f32]) -> f32,
    argmax: &dyn Fn(&[f32]) -> (usize, f32),
) -> Result<(f32, usize, usize), Box<dyn std::error::Error>> {
    // Reference: 7-token single-pass prefill -> logits at position 6
    let kv_ref = mk_kv();
    let rb = pipeline.run_full_model_prefill_chunked_with_cache_state(
        device, queue, gpu_model, emb_7, head_override, 0,
        Some((kv_ref.get_k_buffers(), kv_ref.get_v_buffers())),
        spec, 512,
    )?;
    let logits_ref = rb.2;

    // Prefill 6 tokens (0..5)
    let kv_pf = mk_kv();
    pipeline.run_full_model_prefill_chunked_with_cache_state(
        device, queue, gpu_model, emb_6, head_override, 0,
        Some((kv_pf.get_k_buffers(), kv_pf.get_v_buffers())),
        spec, 512,
    )?;

    // Decode at pos 6 with 7th token embedding
    let rd = pipeline.run_full_model_prefill_chunked_with_cache_state(
        device, queue, gpu_model, emb_7th, head_override, 6,
        Some((kv_pf.get_k_buffers(), kv_pf.get_v_buffers())),
        spec, 1,
    )?;
    let logits_dec = rd.2;

    let mut maxdiff = 0.0f32;
    for (a, b) in logits_dec.iter().zip(logits_ref.iter()) {
        let d = (a - b).abs();
        if d > maxdiff { maxdiff = d; }
    }
    let (tok_dec, _) = argmax(&logits_dec);
    let (tok_ref, _) = argmax(&logits_ref);

    eprintln!("  ref_logits_rms={:.6}  dec_logits_rms={:.6}  max|Δ|={:.6e}", rms(&logits_ref), rms(&logits_dec), maxdiff);
    eprintln!("  ref_argmax={}  dec_argmax={}", tok_ref, tok_dec);

    Ok((maxdiff, tok_dec, tok_ref))
}