// quant_verify — GPU dequant validation binary.
//
// Loads a GGUF file, finds tensors of each supported quant type, dequantizes
// them on CPU (reference) and GPU, and reports max/mean absolute error.
//
// Usage:
//   LIBSHIMMY_MODEL_PATH=/path/to/model.gguf cargo run --release --bin quant_verify
//   or: cargo run --release --bin quant_verify -- --model-path /path/to/model.gguf
//
// Exit code: 0 if all tested types pass, 1 if any fail.

use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::BindlessPipeline;
use airframe::core::dequant::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};
use airframe::core::model::GgufTensorInfo;
use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;

fn main() {
    // --- Argument parsing (minimal) ---
    let args: Vec<String> = std::env::args().collect();
    let model_path = if let Some(p) = args.windows(2).find(|w| w[0] == "--model-path") {
        p[1].clone()
    } else if let Ok(p) = std::env::var("LIBSHIMMY_MODEL_PATH") {
        p
    } else {
        eprintln!("Usage: quant_verify --model-path <gguf>");
        std::process::exit(2);
    };

    println!("[quant_verify] Model: {}", model_path);

    // --- GPU initialisation (synchronous via pollster) ---
    let (device, queue) = pollster::block_on(async {
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
        limits.max_storage_buffer_binding_size =
            adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_storage_buffers_per_shader_stage = 8;
        limits.max_compute_invocations_per_workgroup = 256;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("Failed to create GPU device");
        println!("[quant_verify] GPU: {}", adapter.get_info().name);
        (device, queue)
    });

    // --- Parse GGUF metadata ---
    let mut hdr = File::open(&model_path).expect("cannot open model");
    let meta = BindlessMetadata::new(&mut hdr);
    drop(hdr);

    // --- Load model to GPU ---
    let path = PathBuf::from(&model_path);
    let gpu_model = BindlessModel::load_from_disk(&device, &path, None);
    let pipeline = BindlessPipeline::new(&device);

    // --- Open mmap for CPU dequant ---
    let file = File::open(&model_path).expect("mmap open");
    let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };

    // --- Types to test: (ggml_type, name) ---
    let quant_types: &[(u32, &str)] = &[
        (0,  "F32"),
        (1,  "F16"),
        (2,  "Q4_0"),
        (8,  "Q8_0"),
        (12, "Q4_K"),
        (13, "Q5_K"),
        (14, "Q6_K"),
    ];

    // How many elements to test per tensor (keep small to stay fast).
    // Must be a multiple of the block size (256 for K-quants, 32 for others).
    const TEST_ELEMS: u32 = 512;

    let mut any_fail = false;

    for &(qt, qt_name) in quant_types {
        // Find any tensor with this type.
        let tensor_name = meta
            .tensor_types
            .iter()
            .find(|(_, &t)| t == qt)
            .map(|(n, _)| n.clone());

        let Some(name) = tensor_name else {
            println!("[quant_verify] {qt_name:5} (type {qt:2}) — not present in model, skipping");
            continue;
        };

        let offset_abs = *meta.tensor_offsets.get(&name).unwrap();
        let dims = meta.tensor_dims.get(&name).unwrap().clone();
        let total_elements: u64 = dims.iter().product();

        let elems = TEST_ELEMS.min(total_elements as u32);

        println!(
            "[quant_verify] {qt_name:5} (type {qt:2}) — tensor={name}  total_elems={total_elements}  testing={elems}"
        );

        // --- CPU reference ---
        let tensor_info = GgufTensorInfo {
            name: name.clone(),
            dimensions: dims.iter().map(|&d| d as usize).collect(),
            ggml_type: qt,
            offset: offset_abs - meta.data_start_offset,
        };

        let cpu_full = match qt {
            0 => cpu_dequant_f32(&mmap, offset_abs as usize, elems as usize),
            1 => cpu_dequant_f16(&mmap, offset_abs as usize, elems as usize),
            2 => {
                let t = dequantize_q4_0(&tensor_info, &mmap, meta.data_start_offset)
                    .expect("cpu q4_0 failed");
                t.data[..elems as usize].to_vec()
            }
            8 => {
                let t = dequantize_q8_0(&tensor_info, &mmap, meta.data_start_offset)
                    .expect("cpu q8_0 failed");
                t.data[..elems as usize].to_vec()
            }
            12 => {
                let t = dequantize_q4_k(&tensor_info, &mmap, meta.data_start_offset)
                    .expect("cpu q4_k failed");
                t.data[..elems as usize].to_vec()
            }
            13 => {
                let t = dequantize_q5_k(&tensor_info, &mmap, meta.data_start_offset)
                    .expect("cpu q5_k failed");
                t.data[..elems as usize].to_vec()
            }
            14 => {
                let t = dequantize_q6_k(&tensor_info, &mmap, meta.data_start_offset)
                    .expect("cpu q6_k failed");
                t.data[..elems as usize].to_vec()
            }
            _ => unreachable!(),
        };

        // --- GPU dequant ---
        let gpu_out = pipeline.run_dequant_any_request(
            &device,
            &queue,
            &gpu_model,
            offset_abs as u32,
            elems,
            qt,
        );

        // --- Compare ---
        let n = cpu_full.len().min(gpu_out.len());
        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f32;
        let mut nan_count = 0u32;

        for j in 0..n {
            let c = cpu_full[j];
            let g = gpu_out[j];
            if !c.is_finite() || !g.is_finite() {
                nan_count += 1;
                continue;
            }
            let err = (c - g).abs();
            if err > max_err { max_err = err; }
            sum_err += err;
        }
        let mean_err = if n > 0 { sum_err / n as f32 } else { 0.0 };

        // Tolerance: fp16 round-trip introduces ~1e-3 max error
        let tolerance = 1e-2_f32;
        let pass = max_err <= tolerance && nan_count == 0;

        if pass {
            println!(
                "  PASS  max_err={max_err:.2e}  mean_err={mean_err:.2e}  nan={nan_count}"
            );
        } else {
            println!(
                "  FAIL  max_err={max_err:.2e}  mean_err={mean_err:.2e}  nan={nan_count}  (tolerance={tolerance:.2e})"
            );
            any_fail = true;

            // Print first 8 mismatches for debugging
            let mut printed = 0;
            for j in 0..n {
                let err = (cpu_full[j] - gpu_out[j]).abs();
                if err > tolerance || !gpu_out[j].is_finite() {
                    println!(
                        "    elem[{j}]: cpu={:.6}  gpu={:.6}  err={:.2e}",
                        cpu_full[j], gpu_out[j], err
                    );
                    printed += 1;
                    if printed >= 8 { break; }
                }
            }
        }
    }

    if any_fail {
        println!("\n[quant_verify] FAILED — one or more quant types have GPU/CPU mismatch");
        std::process::exit(1);
    } else {
        println!("\n[quant_verify] ALL PASS");
    }
}

// ---------------------------------------------------------------------------
// Simple CPU helpers for F32 and F16 (no GgufTensorInfo needed)
// ---------------------------------------------------------------------------

fn cpu_dequant_f32(mmap: &Mmap, byte_offset: usize, count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = byte_offset + i * 4;
        let bits = u32::from_le_bytes([mmap[b], mmap[b+1], mmap[b+2], mmap[b+3]]);
        out.push(f32::from_bits(bits));
    }
    out
}

fn cpu_dequant_f16(mmap: &Mmap, byte_offset: usize, count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = byte_offset + i * 2;
        let bits = u16::from_le_bytes([mmap[b], mmap[b+1]]);
        out.push(f16_bits_to_f32(bits));
    }
    out
}

/// Minimal IEEE 754 fp16 → fp32 conversion (no dependency on internal crate module).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    let sign = ((bits >> 15) as u32) << 31;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal
        let leading = mant.leading_zeros() - 22; // leading zeros above 10 bits
        let f32_exp = (127 - 14 - leading) as u32;
        let f32_mant = (mant << (leading + 1)) & 0x7FFFFF;
        return f32::from_bits(sign | (f32_exp << 23) | f32_mant);
    } else if exp == 31 {
        // Inf or NaN
        return f32::from_bits(sign | 0x7F800000 | (mant << 13));
    }
    let f32_exp = ((exp + 127 - 15) as u32) << 23;
    let f32_mant = (mant as u32) << 13;
    f32::from_bits(sign | f32_exp | f32_mant)
}
