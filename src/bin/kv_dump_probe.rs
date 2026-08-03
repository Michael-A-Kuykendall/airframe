// KV-Dump Probe (EP-4 localization)
//
// Reproduces the exact prefill->decode scenario that collapses in production
// (run_full_model_with_cache_state path), and dumps layer-0 K-cache at fixed
// positions after each stage so we can pin the corrupted slot/path.
//
// Three scenarios share ONE model + ONE embedding-extraction path so all
// comparisons are self-consistent:
//   A) 5-token prefill  -> writes K[0..4] into buffers X
//   C) decode step 0    -> reads K[0..4] from X, writes K[5] into X (same buffers)
//   B) 6-token prefill  -> writes K[0..5] into buffers Y (ground-truth reference)
//
// Verdict logic:
//   carry : X[K 0..4] after A  ==  X[K 0..4] after C   (decode must not clobber carry)
//   write : X[K 5]    after C  ==  Y[K 5]    after B   (decode writes identical K to prefill)
//   pref  : X[K 0..4] after A  ==  Y[K 0..4] after B   (prefill carry is internally consistent)
//
// Run: cargo run --bin kv_dump_probe -- <model.gguf> "<prompt>"  (first 6 tokens used)

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::BindlessPipeline;
use airframe::runtime::gpu::GpuRuntime;
use shimmytok::Tokenizer;
use std::fs::File;
use std::path::PathBuf;

fn readback_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len_bytes: u64,
) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("KV Staging"),
        size: len_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("KV Readback"),
    });
    enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, len_bytes);
    let idx = queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        })
        .unwrap();
    let data = slice.get_mapped_range();
    let vals: &[f32] = bytemuck::cast_slice(&data);
    let out = vals.to_vec();
    drop(data);
    staging.unmap();
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: kv_dump_probe <model.gguf> \"<prompt>\"");
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
    spec.n_ctx = 8192; // cap KV buffer VRAM on the RTX 3060; only positions 0..5 are used
                       // [B5-debug] Force the CORRECT padded head_dim for Qwen3: attn_q.weight is
                       // [2560, 4096] = [n_embd, n_head*head_dim] -> 4096/32 = 128. The default
                       // n_embd/n_head = 80 is wrong for this padded attention dimension.
    spec.head_dim = 128;
    spec = spec.compute_derived();
    let gpu_model = BindlessModel::load_from_disk(&device, &PathBuf::from(model_path), Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    let n_layers = gpu_model.metadata.compiled_layers.len();
    let dim = spec.n_embd as u32;
    let n_head_kv = spec.n_head_kv as u32;
    let head_dim = (spec.n_embd / spec.n_head) as u32;
    let kv_bytes = spec.kv_cache_size_per_layer as u64;

    eprintln!(
        "[kv_dump] layers={} dim={} n_head_kv={} head_dim={} kv_bytes/layer={}",
        n_layers, dim, n_head_kv, head_dim, kv_bytes
    );

    // Tokenize -> first 6 tokens
    let tokens = tokenizer.encode(prompt, true)?;
    assert!(tokens.len() >= 6, "prompt must tokenize to >=6 tokens");
    let toks: Vec<u32> = tokens[0..6].to_vec();
    eprintln!("[kv_dump] tokens[0..6] = {:?}", toks);

    // Embedding extraction (handles all quant types, identical path for all scenarios)
    let embd_quant_type = gpu_model
        .metadata
        .get_tensor_type("token_embd.weight")
        .unwrap_or(2);
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
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
    let mut emb = Vec::with_capacity(6);
    for &t in &toks {
        let row_offset = embd_weight_offset + (t as u64 * embd_row_bytes as u64);
        let e = pipeline.run_dequant_any_hot(
            &device,
            &queue,
            &gpu_model,
            row_offset as u32,
            dim,
            embd_quant_type,
        );
        emb.push(e);
    }

    // Build input embeddings
    let concat = |slice: &[Vec<f32>]| -> Vec<f32> {
        let mut v = Vec::with_capacity(slice.len() * dim as usize);
        for e in slice {
            v.extend_from_slice(e);
        }
        v
    };
    let emb_5 = concat(&emb[0..5]); // positions 0..4
    let emb_6 = concat(&emb[0..6]); // positions 0..5
    let emb_1 = &emb[5]; // decode token at position 5

    // Allocate KV buffers (persistent across prefill->decode)
    let make_kv = || -> (Vec<wgpu::Buffer>, Vec<wgpu::Buffer>) {
        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let _ = i;
            k.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("KV K L{}", i)),
                size: kv_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            v.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("KV V L{}", i)),
                size: kv_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }
        (k, v)
    };

    let pos_len = (n_head_kv * head_dim) as usize; // f32s per position in K
    let grab = |buf: &[f32], pos: usize| -> Vec<f32> {
        let start = pos * pos_len;
        buf[start..start + pos_len].to_vec()
    };

    // ---- Scenario A: 5-token prefill into X ----
    let (kx, vx) = make_kv();
    eprintln!("[kv_dump] A: 5-token prefill (current_pos=0, seq_len=5)");
    let _ = pipeline.run_full_model_with_cache_state(
        &device,
        &queue,
        &gpu_model,
        &emb_5,
        None,
        0,
        5,
        Some((&kx, &vx)),
        &spec,
    );
    let ka_after_a = readback_f32(&device, &queue, &kx[0], kv_bytes);

    // ---- Scenario C: decode step 0 into SAME X (current_pos=5, seq_len=6) ----
    eprintln!("[kv_dump] C: decode (current_pos=5, seq_len=6)");
    let rc = pipeline
        .run_full_model_with_cache_state(
            &device,
            &queue,
            &gpu_model,
            emb_1,
            None,
            5,
            6,
            Some((&kx, &vx)),
            &spec,
        )
        .expect("decode forward failed");
    let logits_c = rc.2;
    let kc_after_c = readback_f32(&device, &queue, &kx[0], kv_bytes);

    // ---- Scenario B: 6-token prefill into Y (ground truth) ----
    let (ky, vy) = make_kv();
    eprintln!("[kv_dump] B: 6-token prefill (current_pos=0, seq_len=6)");
    let rb = pipeline
        .run_full_model_with_cache_state(
            &device,
            &queue,
            &gpu_model,
            &emb_6,
            None,
            0,
            6,
            Some((&ky, &vy)),
            &spec,
        )
        .expect("prefill forward failed");
    let logits_b = rb.2;
    let kb_after_b = readback_f32(&device, &queue, &ky[0], kv_bytes);
    drop(vy);

    // ---- Final logits comparison (the actual decode-vs-prefill output) ----
    let mut logits_maxdiff = 0.0f32;
    for (a, b) in logits_c.iter().zip(logits_b.iter()) {
        let d = (a - b).abs();
        if d > logits_maxdiff {
            logits_maxdiff = d;
        }
    }
    let rms = |v: &[f32]| -> f32 { (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt() };
    println!(
        "\n=== FINAL LOGITS: decode(C) vs 6-tok-prefill(B) ===\n  logits_c_rms={:.6} logits_b_rms={:.6} max|Δ|={:.6e} {}",
        rms(&logits_c),
        rms(&logits_b),
        logits_maxdiff,
        if logits_maxdiff <= 1e-2 { "MATCH -> airframe decode forward is CORRECT" } else { "DIVERGE -> bug is in airframe forward" }
    );

    // ---- Compare ----
    let max_diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    let mut verdict = Vec::new();
    // carry: X[0..4] after A == X[0..4] after C
    let mut carry_ok = true;
    for p in 0..5usize {
        let a = grab(&ka_after_a, p);
        let c = grab(&kc_after_c, p);
        let d = max_diff(&a, &c);
        if d > 1e-3 {
            carry_ok = false;
        }
        verdict.push(format!(
            "  carry  pos{}: max|Δ|={:.6e} {}",
            p,
            d,
            if d <= 1e-3 { "OK" } else { "CORRUPTED" }
        ));
    }
    // write: X[5] after C == Y[5] after B
    let wc = grab(&kc_after_c, 5);
    let wb = grab(&kb_after_b, 5);
    let dw = max_diff(&wc, &wb);
    let write_ok = dw <= 1e-3;
    verdict.push(format!(
        "  write  pos5: max|Δ|={:.6e} {}",
        dw,
        if write_ok { "OK" } else { "MISMATCH" }
    ));
    // pref: X[0..4] after A == Y[0..4] after B
    let mut pref_ok = true;
    for p in 0..5usize {
        let a = grab(&ka_after_a, p);
        let b = grab(&kb_after_b, p);
        let d = max_diff(&a, &b);
        if d > 1e-3 {
            pref_ok = false;
        }
        verdict.push(format!(
            "  pref   pos{}: max|Δ|={:.6e} {}",
            p,
            d,
            if d <= 1e-3 { "OK" } else { "MISMATCH" }
        ));
    }

    // Also report RMS of each position for intuition
    let rms = |v: &[f32]| -> f32 { (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt() };
    println!("\n=== KV-DUMP VERDICT (layer 0 K cache) ===");
    println!(
        "RMS  A[pos0..4]={:?}",
        (0..5)
            .map(|p| format!("{:.4}", rms(&grab(&ka_after_a, p))))
            .collect::<Vec<_>>()
    );
    println!(
        "RMS  C[pos0..4]={:?}",
        (0..5)
            .map(|p| format!("{:.4}", rms(&grab(&kc_after_c, p))))
            .collect::<Vec<_>>()
    );
    println!("RMS  C[pos5]={:.4}  B[pos5]={:.4}", rms(&wc), rms(&wb));
    println!("{}", verdict.join("\n"));
    println!("---");
    println!(
        "carry_ok={} write_ok={} pref_ok={}",
        carry_ok, write_ok, pref_ok
    );
    let overall = if carry_ok && write_ok && pref_ok {
        "ALL MATCH -> decode KV identical to prefill: bug is NOT in KV cache contents"
    } else if !carry_ok {
        "CARRY CORRUPTED: decode clobbers prefill KV (index/buffer binding bug)"
    } else if !write_ok {
        "WRITE MISMATCH: decode computes wrong K for the new token (QKV/QK-norm/RoPE batch=1 path)"
    } else {
        "PREFILL INCONSISTENT: prefill carry differs between 5- and 6-token runs"
    };
    println!("CONCLUSION: {}", overall);

    // ===== B5: delta bisection for D2 (f32 output head) =====
    // Reference = blob head (None) — proven correct by prior probe (decode logits
    // bit-matched the 6-tok prefill). shimmy instead passes
    // head_override = Some(&f32_head) built by load_output_head_f32. If the
    // f32-head logits diverge from the blob-head logits, the f32 output-head
    // projection (load_output_head_f32) is THE bug.
    eprintln!("[B5] building f32 output head via load_output_head_f32");
    let f32_head = match GpuRuntime::load_output_head_f32(model_path, &gpu_model, &device, &spec) {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    let run_head = |head: Option<&wgpu::Buffer>,
                    toks: &[f32],
                    pos: u32,
                    kv: Option<(&[wgpu::Buffer], &[wgpu::Buffer])>|
     -> Vec<f32> {
        let n = (toks.len() / spec.n_embd as usize) as u32;
        pipeline
            .run_full_model_with_cache_state(
                &device,
                &queue,
                &gpu_model,
                toks,
                head,
                pos,
                pos + n,
                kv,
                &spec,
            )
            .expect("B5 forward failed")
            .2
    };

    // Fresh KV cache for the bisection (faithful to shimmy's shared cache).
    let kvb = KVCache::new(&device, n_layers, n_head_kv, head_dim, 8192);
    let kb = kvb.get_k_buffers();
    let vb = kvb.get_v_buffers();

    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    };
    let dt = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    // Prefill 6 tokens: blob head vs f32 head
    let lp_none = run_head(None, &emb_6, 0, Some((kb, vb)));
    let lp_f32 = run_head(Some(&f32_head), &emb_6, 0, Some((kb, vb)));
    let tk_none = tokenizer
        .decode_single(argmax(&lp_none) as u32, true)
        .unwrap_or_default();
    let tk_f32 = tokenizer
        .decode_single(argmax(&lp_f32) as u32, true)
        .unwrap_or_default();
    println!(
        "\n=== B5 PREFILL(6-tok) lm_head: blob vs f32 ===\n  blob argmax={} '{}'\n  f32  argmax={} '{}'\n  max|Δ|={:.6e} {}",
        argmax(&lp_none),
        tk_none,
        argmax(&lp_f32),
        tk_f32,
        dt(&lp_none, &lp_f32),
        if dt(&lp_none, &lp_f32) <= 1e-1 {
            "CLOSE -> f32 head OK"
        } else {
            "DIVERGE -> D2 bug (f32 head)"
        }
    );

    // Decode 1 token at pos5 (uses the kv carry from the 6-tok prefill above)
    let ld_none = run_head(None, emb_1, 5, Some((kb, vb)));
    let ld_f32 = run_head(Some(&f32_head), emb_1, 5, Some((kb, vb)));
    let tkd_none = tokenizer
        .decode_single(argmax(&ld_none) as u32, true)
        .unwrap_or_default();
    let tkd_f32 = tokenizer
        .decode_single(argmax(&ld_f32) as u32, true)
        .unwrap_or_default();
    println!(
        "\n=== B5 DECODE(1-tok @pos5) lm_head: blob vs f32 ===\n  blob argmax={} '{}'\n  f32  argmax={} '{}'\n  max|Δ|={:.6e} {}",
        argmax(&ld_none),
        tkd_none,
        argmax(&ld_f32),
        tkd_f32,
        dt(&ld_none, &ld_f32),
        if dt(&ld_none, &ld_f32) <= 1e-1 {
            "CLOSE -> f32 head OK"
        } else {
            "DIVERGE -> D2 bug (f32 head)"
        }
    );

    Ok(())
}
