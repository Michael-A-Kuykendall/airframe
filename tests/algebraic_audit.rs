//! Bead P2 — Algebraic audit (THE math gate).
//!
//! Authoritative certification = spec math vs shader, element-wise. The reference
//! is `airframe_observe::quant_formula` — a transcription of the GGML/GGUF
//! quantization spec (validated by hand-computed vectors in that module). The
//! implementation under test is the real production dequant shader
//! (`sh_dequant_any.wgsl`), executed on the GPU via `run_dequant_any_blob`.
//!
//! For every supported quant type we:
//!   1. synthesize a deterministic block of raw GGUF bytes,
//!   2. compute the expected dequant per element from the spec formula,
//!   3. run the shader on those exact bytes,
//!   4. assert every element matches within a tight f32 tolerance.
//!
//! No DuckDB, no traces, no external engine. Pure math vs implementation.
//! Requires a GPU adapter (this is the gate that runs on the RTX 3060 CI/workspace).

use airframe::backend::bindless::pipeline::BindlessPipeline;
use airframe_observe::quant_formula::{
    block_bytes, block_elems, dequant_elem, formula_for_type, slot_for_type,
};

#[allow(dead_code)]
fn f16_to_f32_dbg(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let sign_bit = (sign as u32) << 31;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign_bit)
        } else {
            let val = (mant as f32) * 2f32.powi(-24);
            if sign == 1 {
                -val
            } else {
                val
            }
        }
    } else if exp == 0x1f {
        let m = (mant as u32) << 13;
        f32::from_bits(sign_bit | 0x7f80_0000 | m)
    } else {
        let e = (exp as i32 + (127 - 15)) as u32;
        let m = (mant as u32) << 13;
        f32::from_bits(sign_bit | (e << 23) | m)
    }
}

/// Deterministic PRNG (LCG) — no external deps, reproducible across runs.
fn lcg(state: &mut u64) -> u8 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u8
}

/// Build a `n_blocks`-worth blob of deterministic bytes for the given quant type.
///
/// Scale/dmin f16 fields have their exponent MSB cleared so they stay finite:
/// real GGUF weight scales are always finite, so this keeps the gate
/// deterministic and avoids inf/NaN scale edge cases the shader is not required
/// to carry. The dequant *formula* (nibble extraction, group-scale application,
/// etc.) is still exercised in full.
fn synth_block(type_id: u32, n_blocks: usize, seed: u64) -> Vec<u8> {
    let be = block_elems(type_id).expect("supported type");
    let bb = block_bytes(type_id).expect("supported type");
    let mut state = seed;
    let mut v = vec![0u8; bb * n_blocks];
    // Fill with pseudo-random bytes; for F16 keep even total length.
    for b in v.iter_mut() {
        *b = lcg(&mut state);
    }
    // F32 block must be a real f32 bit pattern so the spec dequant reads it back
    // identically; random bytes are fine (any 4 bytes is a valid f32).
    let _ = be;

    // Clear the f16 exponent MSB (bit 6 of the LE high byte) for every scale /
    // dmin field so exp can never reach 0x1F (inf/NaN). Offsets are the high
    // bytes of each scale/dmin f16 within one (super)block.
    let scale_hi_offsets: &[usize] = match type_id {
        2 | 6 | 8 => &[1],  // Q4_0 / Q5_0 / Q8_0: single f16 scale @ [0..1]
        12 | 13 => &[1, 3], // Q4_K / Q5_K: d @ [0..1], dmin @ [2..3]
        14 => &[209],       // Q6_K: d @ [208..210]
        _ => &[],
    };
    for blk in 0..n_blocks {
        let base = blk * bb;
        for &hi in scale_hi_offsets {
            v[base + hi] &= 0xBF;
        }
    }
    v
}

fn approx_eq(a: f32, b: f32) -> bool {
    if a == b {
        return true;
    }
    let diff = (a - b).abs();
    diff < 1e-2 || diff <= (a.abs().max(b.abs()) * 1e-3)
}

#[tokio::test]
async fn algebraic_audit_dequant_shader_vs_spec() {
    // 1. Acquire GPU (the gate runs on hardware; skip gracefully if none).
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("P2 gate requires a GPU adapter (RTX 3060 CI/workspace)");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: adapter
                    .limits()
                    .max_storage_buffers_per_shader_stage,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("device request failed");

    let pipeline = BindlessPipeline::new(&device);

    // 2. Every supported quant type (all 8). F16 uses 2-byte elements; the
    //    special path below handles it.
    let types: [(u32, usize); 8] = [
        (0, 1),  // F32  (1 elem / 4 bytes)
        (1, 4),  // F16  (2 bytes/elem) — dedicated path below
        (2, 2),  // Q4_0
        (6, 2),  // Q5_0
        (8, 2),  // Q8_0
        (12, 1), // Q4_K
        (13, 1), // Q5_K
        (14, 1), // Q6_K
    ];

    for (type_id, n_blocks) in types {
        let slot = slot_for_type(type_id).expect("slot mapped");
        let be = block_elems(type_id).unwrap();
        let bb = block_bytes(type_id).unwrap();
        let count = (be * n_blocks) as u32;

        // F16 needs an even-length blob (2 bytes/elem). Use a dedicated path.
        if type_id == 1 {
            let mut state = 0x9e3779b97f4a7c15u64;
            let mut blob = vec![0u8; 2 * n_blocks];
            for b in blob.iter_mut() {
                *b = lcg(&mut state);
            }
            // Keep each f16 element finite (clear exp MSB) so the gate stays
            // deterministic — real weights never carry inf/NaN.
            for i in 0..n_blocks {
                blob[i * 2 + 1] &= 0xBF;
            }
            let expected: Vec<f32> = (0..(n_blocks as u32))
                .map(|i| dequant_elem(1, &blob, i as usize).unwrap())
                .collect();
            let got = pipeline.run_dequant_any_blob(&device, &queue, &blob, 0, n_blocks as u32, 1);
            for i in 0..n_blocks {
                assert!(
                    approx_eq(expected[i], got[i]),
                    "F16 elem {i}: spec={} shader={}",
                    expected[i],
                    got[i]
                );
            }
            println!(
                "[P2] type 1 (F16) OK: {} elements match spec within tolerance",
                n_blocks
            );
            continue;
        }

        let blob = synth_block(type_id, n_blocks, 0x1234_0000u64 + (type_id as u64) * 7919);
        assert_eq!(blob.len(), bb * n_blocks);

        // Spec expectation: dequant every element from the same bytes.
        // `dequant_elem` takes a SINGLE (super)block slice + a within-block
        // index (its contract), so slice the blob per block — the shader also
        // dequantizes per-block (GGML semantics).
        let expected: Vec<f32> = (0..count)
            .map(|e| {
                let e = e as usize;
                let b = e / be;
                let within = e % be;
                let start = b * bb;
                dequant_elem(type_id, &blob[start..start + bb], within).unwrap()
            })
            .collect();

        // Shader (real production path).
        let got = pipeline.run_dequant_any_blob(&device, &queue, &blob, 0, count, type_id);

        assert_eq!(
            got.len(),
            expected.len(),
            "type {} length mismatch",
            type_id
        );
        for (i, (exp, act)) in expected.iter().zip(got.iter()).enumerate() {
            assert!(
                approx_eq(*exp, *act),
                "type {} ({}) elem {}/{}: spec={:.6} shader={:.6} (slot={})",
                type_id,
                formula_for_type(type_id).unwrap().name,
                i,
                count,
                exp,
                act,
                slot.as_u32()
            );
        }
        println!(
            "[P2] type {} ({}) OK: {} elements match spec within tolerance",
            type_id,
            formula_for_type(type_id).unwrap().name,
            count
        );
    }
}
