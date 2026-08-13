//! PLE per-layer latent input construction — GPU vs CPU reference gate (f41.2.3 / 2il).
//!
//! Loads gemma-4-E4B, runs the PLE input GPU pass, readbacks the per-layer
//! input buffer, and compares element-wise against a CPU reference computed
//! with `airframe_observe::quant_formula` (the math authority). Skips cleanly
//! when the model file or a GPU adapter is unavailable.
#![cfg(feature = "isf")]

use std::path::Path;

use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::BindlessPipeline;
use airframe::core::spec::ModelSpec;

const MODEL: &str = "/home/michael/models/Gemma-4/gemma-4-E4B-it-Q4_K_M/gemma-4-E4B-it-Q4_K_M.gguf";

fn load_raw(path: &Path) -> Vec<u8> {
    std::fs::read(path).expect("read model for CPU reference")
}

// Dequant one element of per_layer_token_embd (Q6_K, 256 elems / 210 bytes).
fn dequant_q6k(raw: &[u8], base: usize, elem: usize) -> f32 {
    let block = elem / 256;
    let e = elem % 256;
    let boff = base + block * 210;
    // Reuse the spec-registry dequant on the 210-byte block.
    airframe_observe::quant_formula::dequant_elem(14, &raw[boff..boff + 210], e).expect("q6k")
}

// Dequant one element of per_layer_model_proj (IQ4_XS, 128 elems / 128 bytes).
fn dequant_iq4xs(raw: &[u8], base: usize, elem: usize) -> f32 {
    let block = elem / 128;
    let e = elem % 128;
    let boff = base + block * 128;
    airframe_observe::quant_formula::dequant_elem(30, &raw[boff..boff + 128], e).expect("iq4xs")
}

fn f32_at(raw: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
}

fn cpu_reference(raw: &[u8], spec: &ModelSpec, tokens: &[u32], input_embd: &[f32]) -> Vec<f32> {
    let latent = spec.ple_latent_dim;
    let n_layer = spec.n_layer;
    let n_embd = spec.n_embd;
    let n_tokens = tokens.len();
    let mut out = vec![0.0f32; n_layer * n_tokens * latent];

    let token_embd_base = spec.per_layer_token_embd_offset as usize;
    let model_proj_base = spec.per_layer_model_proj_offset as usize;
    let proj_norm_base = spec.per_layer_proj_norm_offset as usize;

    let token_row_bytes = latent * n_layer * 210 / 256;

    for (t, &tok) in tokens.iter().enumerate() {
        // gather = per_layer_token_embd[tokens[t]][j] * sqrt(latent)
        // (column `tok` of [latent*n_layer, vocab]; contiguous latent*n_layer elems)
        for il in 0..n_layer {
            let mut proj = vec![0.0f32; latent];
            let mut proj_sq_sum = 0.0f32;
            for (k, slot) in proj.iter_mut().enumerate() {
                let j = il * latent + k;
                // mm(per_layer_model_proj, inp_batch) / sqrt(n_embd)
                // element (i, j) at byte i + j*n_embd (column-major, IQ4_XS 1B/elem)
                let mut dot = 0.0f32;
                for i in 0..n_embd {
                    let elem = i + j * n_embd;
                    let w = dequant_iq4xs(raw, model_proj_base, elem);
                    dot += w * input_embd[t * n_embd + i];
                }
                *slot = dot * (1.0 / (n_embd as f32).sqrt());
                proj_sq_sum += (*slot) * (*slot);
            }
            let rms_inv = 1.0 / (proj_sq_sum / latent as f32 + spec.rms_eps).sqrt();
            for (k, proj_k) in proj.iter().enumerate() {
                let norm_w = f32_at(raw, proj_norm_base + k * 4);
                let proj_normed = proj_k * rms_inv * norm_w;
                let gather = dequant_q6k(
                    raw,
                    token_embd_base + tok as usize * token_row_bytes,
                    il * latent + k,
                ) * (latent as f32).sqrt();
                let out_val = (proj_normed + gather) * (1.0 / 2.0f32.sqrt());
                out[(il * n_tokens + t) * latent + k] = out_val;
            }
        }
    }
    out
}

#[tokio::test]
async fn ple_input_gpu_matches_cpu_reference() {
    let model_path = Path::new(MODEL);
    if !model_path.exists() {
        eprintln!("SKIP: model {} absent", MODEL);
        return;
    }

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No adapter");
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage =
        adapter.limits().max_storage_buffers_per_shader_stage;
    limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("No device");

    let raw = load_raw(model_path);
    let model = BindlessModel::load_from_disk(&device, model_path, None);
    let spec = model.metadata.to_model_spec();
    let pipeline = BindlessPipeline::new(&device);

    // 3 distinct tokens, scaled embeddings like the prefill path.
    let tokens: Vec<u32> = vec![563, 837, 15043];
    let n_embd = spec.n_embd;
    let mut input_embd: Vec<f32> = Vec::new();
    for t in 0..3 {
        let base = (t as f32 * 0.01).sin();
        for i in 0..n_embd {
            input_embd.push(base + (i as f32 * 0.0001).cos());
        }
    }
    if spec.scale_embeddings_by_sqrt_dim {
        let s = (n_embd as f32).sqrt();
        for v in input_embd.iter_mut() {
            *v *= s;
        }
    }

    let gpu_buf = pipeline
        .run_ple_input_pass(&device, &queue, &model, &input_embd, &tokens, &spec)
        .expect("PLE input pass");
    let out_bytes = spec.n_layer * tokens.len() * spec.ple_latent_dim * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PLE staging"),
        size: out_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&gpu_buf, 0, &staging, 0, out_bytes as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    rx.recv().expect("map").expect("map failed");
    let gpu_vals: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();

    let cpu_vals = cpu_reference(&raw, &spec, &tokens, &input_embd);

    assert_eq!(gpu_vals.len(), cpu_vals.len(), "size mismatch");
    let mut max_err = 0.0f32;
    for (g, c) in gpu_vals.iter().zip(cpu_vals.iter()) {
        let err = (g - c).abs();
        max_err = max_err.max(err);
    }
    eprintln!("PLE input max_err = {:.6e} (tokens {:?})", max_err, tokens);
    assert!(
        max_err <= 1e-3,
        "PLE input GPU vs CPU max_err {max_err:e} > 1e-3"
    );
}
