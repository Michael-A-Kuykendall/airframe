// sh_dequant_q6k.wgsl
// Dequantize one row of Q6_K quantized weights into F32.
//
// Used for embedding lookups (token_embd.weight is Q6_K in Q4_K_M models).
//
// Bindings:
//   0 — gguf_blob (u32[], the entire GGUF file)
//   1 — output   (f32[], destination for dequantized values)
//   2 — params   (DequantParams uniform: offset_bytes, count)

struct DequantParams {
    offset_bytes: u32,   // byte offset of the row start in gguf_blob
    count: u32,          // number of output elements (must be a multiple of 256)
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read>       gguf_blob : array<u32>;
@group(0) @binding(1) var<storage, read_write> output    : array<f32>;
@group(0) @binding(2) var<uniform>             params    : DequantParams;

// ----- helpers -----

fn get_byte(byte_pos: u32) -> u32 {
    let word = gguf_blob[byte_pos / 4u];
    return (word >> ((byte_pos % 4u) * 8u)) & 0xFFu;
}

fn f16_from_bytes(lo: u32, hi: u32) -> f32 {
    let bits = lo | (hi << 8u);
    let s = bits & 0x8000u;
    let e = (bits >> 10u) & 0x1Fu;
    let m = bits & 0x3FFu;
    var fval: f32;
    if (e == 0u) {
        fval = f32(m) * exp2(-24.0);
    } else if (e == 31u) {
        fval = 65504.0;
    } else {
        fval = (1.0 + f32(m) * exp2(-10.0)) * exp2(f32(e) - 15.0);
    }
    return select(fval, -fval, s != 0u);
}

fn u8_to_i8(v: u32) -> i32 {
    return i32(v << 24u) >> 24;
}

// ----- kernel -----
// Each invocation decodes one output element.
// Q6_K superblock layout (210 bytes, 256 elements):
//   ql[128]     — low 4 bits of each quantized value
//   qh[64]      — high 2 bits  (2 bits per element, packed)
//   scales[16]  — int8 scale for each group of 16 elements
//   d[2]        — FP16 global scale

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;       // element index within the row
    if (j >= params.count) { return; }

    // Superblock index and position within superblock
    let sb_idx   = j / 256u;
    let j_in_sb  = j % 256u;

    let bb = params.offset_bytes + sb_idx * 210u;

    // FP16 global scale at bytes 208–209
    let d_val = f16_from_bytes(get_byte(bb + 208u), get_byte(bb + 209u));

    // Decompose j_in_sb into half / piece / l
    let half      = j_in_sb / 128u;
    let j_in_half = j_in_sb % 128u;
    let piece     = j_in_half / 32u;
    let l         = j_in_half % 32u;

    // Byte bases for this half
    let ql_base = bb + half * 64u;          // ql layout: first 64 bytes low nibbles
    let qh_base = bb + 128u + half * 32u;   // qh layout: next 32 bytes per half
    let sc_base = bb + 192u + half * 8u;    // scales: 8 int8 per half

    let ql_lo  = get_byte(ql_base + l);
    let ql_hi  = get_byte(ql_base + l + 32u);
    let qh_val = get_byte(qh_base + l);

    // Reconstruct 6-bit unsigned value, then subtract 32 for signed offset
    var q: i32;
    if (piece == 0u) {
        q = i32((ql_lo  & 0x0Fu) | ((qh_val         & 0x03u) << 4u)) - 32;
    } else if (piece == 1u) {
        q = i32((ql_hi  & 0x0Fu) | (((qh_val >> 2u) & 0x03u) << 4u)) - 32;
    } else if (piece == 2u) {
        q = i32((ql_lo  >> 4u  ) | (((qh_val >> 4u) & 0x03u) << 4u)) - 32;
    } else {
        q = i32((ql_hi  >> 4u  ) | (((qh_val >> 6u) & 0x03u) << 4u)) - 32;
    }

    // Scale index: each scale covers 16 elements.
    // Within the half, the offset is is + piece*2.
    let is  = l / 16u;
    let sc  = u8_to_i8(get_byte(sc_base + is + piece * 2u));

    output[j] = d_val * f32(sc) * f32(q);
}
