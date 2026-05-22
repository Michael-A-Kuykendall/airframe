// sh_output_proj_q6k.wgsl
// LM-head output projection using Q6_K weights read directly from the GGUF blob.
//
// Avoids allocating a ≥2 GB F32 weight buffer for large-vocab models (e.g. Gemma-2:
// 256k × 2304 × 4 bytes = 2.36 GB, which exceeds wgpu's 2 GB max buffer limit).
//
// Each thread computes one output logit:
//   output_y[row] = dot(input_x, weight_row[row])
// where weight_row is stored in Q6_K format inside the GGUF blob.
//
// Q6_K superblock layout (210 bytes, 256 elements):
//   [  0..128)  ql    – low 4 bits of each quantized value (nibble-packed)
//   [128..192)  qh    – high 2 bits of each quantized value (2-bits per element)
//   [192..208)  scales – 16 int8 scale values (one per group of 16 elements)
//   [208..210)  d      – FP16 global scale

struct OutputProjParams {
    n_vocab:              u32,  // number of output logits  (e.g. 256000)
    n_embd:               u32,  // hidden dimension         (e.g. 2304)
    weights_byte_offset:  u32,  // absolute byte offset of weight data in gguf_blob
    pad:                  u32,
}

@group(0) @binding(0) var<storage, read>       gguf_blob : array<u32>;
@group(0) @binding(1) var<storage, read>       input_x   : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y  : array<f32>;
@group(0) @binding(3) var<uniform>             params    : OutputProjParams;

// ─── helpers ────────────────────────────────────────────────────────────────

// Read one byte from gguf_blob at the given absolute byte offset.
fn read_byte(byte_pos: u32) -> u32 {
    return (gguf_blob[byte_pos >> 2u] >> ((byte_pos & 3u) << 3u)) & 0xFFu;
}

// FP16 bits → FP32 (IEEE 754 half precision, no inf/nan special-casing needed
// for weights – just clamp to ±65504).
fn f16_to_f32(bits: u32) -> f32 {
    let s: u32 = bits & 0x8000u;
    let e: u32 = (bits >> 10u) & 0x1Fu;
    let m: u32 = bits & 0x3FFu;
    var fval: f32;
    if e == 0u {
        // subnormal: ±m × 2^{-24}
        fval = f32(m) * exp2(-24.0);
    } else if e == 31u {
        // infinity / NaN → clamp
        fval = 65504.0;
    } else {
        // normal: ±(1 + m·2^{-10}) × 2^{e−15}
        fval = (1.0 + f32(m) * exp2(-10.0)) * exp2(f32(e) - 15.0);
    }
    return select(fval, -fval, s != 0u);
}

// Reinterpret a u8 value (0–255) as int8 (-128–127).
fn u8_to_i8(v: u32) -> i32 {
    // Arithmetic-shift-right from bit 7 sign-extends correctly.
    return i32(v << 24u) >> 24;
}

// ─── Q6_K dot-product ───────────────────────────────────────────────────────

// Dequantize one 256-element Q6_K superblock and accumulate
//   sum(weight[k] * input_x[k_base + k])  for k in 0..256
// superblock_base: absolute byte offset of the 210-byte superblock in gguf_blob.
// k_base:          starting index in input_x for this superblock.
fn dot_q6k_superblock(superblock_base: u32, k_base: u32) -> f32 {
    let d_lo = read_byte(superblock_base + 208u);
    let d_hi = read_byte(superblock_base + 209u);
    let d    = f16_to_f32(d_lo | (d_hi << 8u));

    var sum = 0.0;

    // Two 128-element halves (n = 0, 1).
    for (var n = 0u; n < 2u; n = n + 1u) {
        let y_off   = n * 128u;
        let ql_base = superblock_base + n * 64u;         // low-nibble bytes
        let qh_base = superblock_base + 128u + n * 32u;  // high-2-bit bytes
        let sc_base = superblock_base + 192u + n * 8u;   // int8 scale bytes

        // 32 iterations, each contributing 4 output values.
        for (var l = 0u; l < 32u; l = l + 1u) {
            let ql_lo  = read_byte(ql_base + l);        // covers l and l+64
            let ql_hi  = read_byte(ql_base + l + 32u);  // covers l+32 and l+96
            let qh_val = read_byte(qh_base + l);

            // Reconstruct 6-bit signed quantized values (offset by −32).
            let q1 = i32((ql_lo  & 0x0Fu) | ((qh_val        & 0x03u) << 4u)) - 32;
            let q2 = i32((ql_hi  & 0x0Fu) | (((qh_val >> 2u) & 0x03u) << 4u)) - 32;
            let q3 = i32((ql_lo  >> 4u  ) | (((qh_val >> 4u) & 0x03u) << 4u)) - 32;
            let q4 = i32((ql_hi  >> 4u  ) | (((qh_val >> 6u) & 0x03u) << 4u)) - 32;

            // Scale indices: one scale per 16 elements.
            let is = l / 16u;
            let sc1 = u8_to_i8(read_byte(sc_base + is      ));
            let sc2 = u8_to_i8(read_byte(sc_base + is + 2u));
            let sc3 = u8_to_i8(read_byte(sc_base + is + 4u));
            let sc4 = u8_to_i8(read_byte(sc_base + is + 6u));

            let k1 = k_base + y_off + l;
            let k2 = k_base + y_off + l + 32u;
            let k3 = k_base + y_off + l + 64u;
            let k4 = k_base + y_off + l + 96u;

            let n_embd = params.n_embd;
            if k1 < n_embd { sum = sum + d * f32(sc1) * f32(q1) * input_x[k1]; }
            if k2 < n_embd { sum = sum + d * f32(sc2) * f32(q2) * input_x[k2]; }
            if k3 < n_embd { sum = sum + d * f32(sc3) * f32(q3) * input_x[k3]; }
            if k4 < n_embd { sum = sum + d * f32(sc4) * f32(q4) * input_x[k4]; }
        }
    }
    return sum;
}

// ─── entry point ────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= params.n_vocab) { return; }

    // Number of Q6_K superblocks per weight row.
    let n_sb      = (params.n_embd + 255u) / 256u;
    // Byte offset of this row's weight data.
    let row_start = params.weights_byte_offset + row * n_sb * 210u;

    var total = 0.0;
    for (var sb = 0u; sb < n_sb; sb = sb + 1u) {
        total = total + dot_q6k_superblock(row_start + sb * 210u, sb * 256u);
    }

    output_y[row] = total;
}
