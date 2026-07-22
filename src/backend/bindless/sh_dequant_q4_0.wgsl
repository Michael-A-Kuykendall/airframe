

// IEEE-754 binary16 -> binary32. Float-arithmetic only — NO bitcast of a
// computed integer (unreliable on this driver, where unpack2x16float is also
// broken). Bit-exact to airframe_observe::quant_formula::f16_to_f32 for
// normal/zero/subnormal values; the P2 algebraic_audit gate validates the
// shader element-wise against that reference.
fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp  = (bits >> 10u) & 0x1fu;
    let mant = bits & 0x3ffu;
    let sign_f = select(-1.0, 1.0, sign == 0u);
    if (exp == 0u) {
        if (mant == 0u) {
            return sign_f * 0.0;
        }
        // subnormal: (-1)^sign * mant * 2^-24 (exact division by power of two)
        return sign_f * (f32(mant) / f32(1u << 24u));
    }
    if (exp == 0x1fu) {
        // ±inf / NaN. Real GGUF weight scales are always finite, so this branch
        // is unreachable for valid input; return 0.0 to stay parse-clean.
        return 0.0;
    }
    // normal: (-1)^sign * (1 + mant/1024) * 2^(exp-15)
    // exp-15 may be negative, so split on the sign of the shift: every shift
    // count stays non-negative and every power-of-two op is exact, making the
    // result bit-identical to the reference integer-assembled f32.
    let fraction = 1.0 + f32(mant) / 1024.0;
    if (exp >= 15u) {
        let p = f32(1u << (exp - 15u));
        return sign_f * fraction * p;
    } else {
        let p = f32(1u << (15u - exp));
        return sign_f * fraction / p;
    }
}
// 1. Q4_0 Layout
// Block size = 32 elements.
// Total bytes = 18 bytes.
// Structure:
// - scale: f16 (2 bytes)
// - quants: u8[16] (16 bytes). Each u8 holds two 4-bit nibbles.
//
// Nibble ordering (little endian byte):
// byte = (nibble1 << 4) | nibble0
// nibble0 corresponds to element i
// nibble1 corresponds to element i+1
// But wait! GGML/llama.cpp packing might be different. 
// Let's verify standard GGUF Q4_0 spec.
//
// Spec says:
// d: f16
// qs: uint8_t[16] (x 32 nibbles)
// v = (qs[i] & 0x0F) - 8
// v = (qs[i] >> 4) - 8
//
// Actually, looking at ggml-quants.c:
// offset 0: f16 d
// offset 2: u8 qs[16]
//
// Loop j=0..16:
//   x0 = qs[j] & 0x0F
//   x1 = qs[j] >> 4
//   y[j] = (x0 - 8) * d
//   y[j+16] = (x1 - 8) * d    <-- WAIT. This is "blocked interleave" or "blocked planar"?
// 
// Let's re-read ggml-common.h or our CPU implementation.
//
// Our CPU impl (ggml_types.rs? No, core/dequant/q4_k.rs is K-quants. We need Q4_0.)
// We don't have a Q4_0 kernel file? 
// Ah, `crates/airframe/src/core/ggml_types.rs` defines it, but where is the logic?
// `crates/airframe/src/ops/reference/matmul.rs`??
// 
// Let's check `crates/airframe/src/core/tensor.rs`.
//
// Wait, I see `q4_k.rs` and `q6_k.rs` but no `q4_0.rs` in `core/dequant`.
// Are we calculating Q4_0 inline?
//
// Let's find where Q4_0 is dequantized in CPU code to match logic.

struct DequantParams {
    offset_bytes: u32,
    count: u32,       // Number of floats to dequantize
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> gguf_blob: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: DequantParams; // Added binding

// Utility to unpack f16 from u32
// We assume little-endian u32 reads.
// f16 is 2 bytes. 
// if address % 4 == 0: lower 16 bits
// if address % 4 == 2: upper 16 bits
fn get_f16(byte_offset: u32) -> f32 {
    let u32_idx = byte_offset / 4u;
    let shift = (byte_offset % 4u) * 8u;
    let u32_val = gguf_blob[u32_idx];
    let u16_val = (u32_val >> shift) & 0xFFFFu;
    return f16_to_f32(u16_val);
}

// Read u8 from u32 array
fn get_u8(byte_offset: u32) -> u32 {
    let u32_idx = byte_offset / 4u;
    let shift = (byte_offset % 4u) * 8u;
    let u32_val = gguf_blob[u32_idx];
    return (u32_val >> shift) & 0xFFu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Thread ID = Output Element Index
    let gid = id.x;
    if (gid >= params.count) { return; }

    // Block logic strictly per GGML Q4_0
    // Block size = 32.
    // Block Bytes = 18.
    
    let block_idx = gid / 32u;
    let lane_idx = gid % 32u; // 0..31
    
    let block_offset = params.offset_bytes + (block_idx * 18u);
    
    // Read scale (d)
    let d = get_f16(block_offset);
    
    // Read nibble - GGML Q4_0 SPLIT layout:
    // Elements 0-15:  low nibbles from bytes 0-15
    // Elements 16-31: high nibbles from bytes 0-15
    
    let byte_idx = lane_idx % 16u;            // Which byte (0..15)
    let is_high = lane_idx >= 16u;            // High nibble if lane >= 16
    
    let qs_byte = get_u8(block_offset + 2u + byte_idx);
    let nibble = select(qs_byte & 0x0Fu, (qs_byte >> 4u) & 0x0Fu, is_high);
    
    let val = (f32(nibble) - 8.0) * d;
    
    output[gid] = val;
}
