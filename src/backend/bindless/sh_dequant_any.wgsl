// sh_dequant_any.wgsl — Multi-type GPU dequantization shader.
//
// Dispatches on quant_type to dequantize any supported GGML tensor type.
// Binding layout is identical to sh_dequant_q4_0.wgsl:
//   0: gguf_blob (StorageRead)
//   1: output    (StorageReadWrite, array<f32>)
//   2: params    (Uniform, DequantAnyParams)
//
// Supported quant_type values:
//   0  = F32
//   1  = F16
//   2  = Q4_0  (32-elem blocks, 18 bytes each)
//   8  = Q8_0  (32-elem blocks, 34 bytes each)
//   12 = Q4_K  (256-elem superblocks, 144 bytes each)
//   13 = Q5_K  (256-elem superblocks, 176 bytes each)
//   14 = Q6_K  (256-elem superblocks, 210 bytes each)

struct DequantAnyParams {
    offset_bytes: u32,  // Absolute byte offset of tensor in gguf_blob
    count: u32,         // Number of f32 elements to produce
    quant_type: u32,    // GGML type (see above)
    pad: u32,
};

@group(0) @binding(0) var<storage, read>       gguf_blob : array<u32>;
@group(0) @binding(1) var<storage, read_write> output    : array<f32>;
@group(0) @binding(2) var<uniform>             params    : DequantAnyParams;

// ---------------------------------------------------------------------------
// Byte-level read helper
// ---------------------------------------------------------------------------
fn read_byte(byte_idx: u32) -> u32 {
    let word = gguf_blob[byte_idx / 4u];
    return (word >> ((byte_idx % 4u) * 8u)) & 0xFFu;
}

// ---------------------------------------------------------------------------
// get_scale_min_k4 — exact llama.cpp port (shared by Q4_K, Q5_K, Q6_K)
// Returns vec2(sc, m) for scale index j into a 12-byte packed scale array.
// ---------------------------------------------------------------------------
fn get_scale_min_k4(j: u32, scales_base_byte: u32) -> vec2<u32> {
    if (j < 4u) {
        let sc = read_byte(scales_base_byte + j) & 63u;
        let m  = read_byte(scales_base_byte + j + 4u) & 63u;
        return vec2<u32>(sc, m);
    } else {
        let sc = (read_byte(scales_base_byte + j + 4u) & 0x0Fu)
               | (((read_byte(scales_base_byte + j - 4u) >> 6u) & 0x03u) << 4u);
        let m  = ((read_byte(scales_base_byte + j + 4u) >> 4u) & 0x0Fu)
               | (((read_byte(scales_base_byte + j) >> 6u) & 0x03u) << 4u);
        return vec2<u32>(sc, m);
    }
}

// ---------------------------------------------------------------------------
// Q4_0 element dequant (18-byte blocks, 32 elements)
// ---------------------------------------------------------------------------
fn dequant_q4_0_elem(block_base: u32, e: u32) -> f32 {
    let scale_packed = extractBits(gguf_blob[block_base / 4u],
                                   (block_base % 4u) * 8u, 16u);
    let scale = unpack2x16float(scale_packed).x;
    let qs_byte = block_base + 2u + (e % 16u);
    let qs = read_byte(qs_byte);
    let nib = select(qs & 0x0Fu, qs >> 4u, e >= 16u);
    return (f32(nib) - 8.0) * scale;
}

// ---------------------------------------------------------------------------
// Q8_0 element dequant (34-byte blocks, 32 elements)
// ---------------------------------------------------------------------------
fn dequant_q8_0_elem(block_base: u32, e: u32) -> f32 {
    let scale_packed = extractBits(gguf_blob[block_base / 4u],
                                   (block_base % 4u) * 8u, 16u);
    let scale = unpack2x16float(scale_packed).x;
    let qs_byte = block_base + 2u + e;
    let raw = read_byte(qs_byte);
    let signed_val = select(i32(raw), i32(raw) - 256, raw >= 128u);
    return scale * f32(signed_val);
}

// ---------------------------------------------------------------------------
// Q4_K element dequant (144-byte superblocks, 256 elements)
// ---------------------------------------------------------------------------
fn dequant_q4k_elem(block_base: u32, e: u32) -> f32 {
    let d_packed = extractBits(gguf_blob[block_base / 4u],
                               (block_base % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;
    let dmin_byte = block_base + 2u;
    let dmin_packed = extractBits(gguf_blob[dmin_byte / 4u],
                                  (dmin_byte % 4u) * 8u, 16u);
    let dmin_val = unpack2x16float(dmin_packed).x;
    let scales_base = block_base + 4u;
    let qs_base     = block_base + 16u;
    let group    = e / 64u;
    let in_group = e % 64u;
    let sub      = in_group / 32u;
    let l        = in_group % 32u;
    let is = group * 2u + sub;
    let sm = get_scale_min_k4(is, scales_base);
    let sc_val = d * f32(sm.x);
    let m_val  = dmin_val * f32(sm.y);
    let ql_byte = qs_base + group * 32u + l;
    var nibble: u32;
    if (sub == 0u) {
        nibble = read_byte(ql_byte) & 0x0Fu;
    } else {
        nibble = read_byte(ql_byte) >> 4u;
    }
    return sc_val * (f32(nibble) - 8.0) - m_val;
}

// ---------------------------------------------------------------------------
// Q5_K element dequant (176-byte superblocks, 256 elements)
// ---------------------------------------------------------------------------
fn dequant_q5k_elem(block_base: u32, e: u32) -> f32 {
    let d_packed = extractBits(gguf_blob[block_base / 4u],
                               (block_base % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;
    let dmin_byte = block_base + 2u;
    let dmin_packed = extractBits(gguf_blob[dmin_byte / 4u],
                                  (dmin_byte % 4u) * 8u, 16u);
    let dmin_val = unpack2x16float(dmin_packed).x;
    let scales_base = block_base + 4u;
    let qh_base     = block_base + 16u;
    let qs_base     = block_base + 48u;
    let group    = e / 64u;
    let in_group = e % 64u;
    let sub      = in_group / 32u;
    let l        = in_group % 32u;
    let is = group * 2u + sub;
    let sm = get_scale_min_k4(is, scales_base);
    let sc_val = d * f32(sm.x);
    let m_val  = dmin_val * f32(sm.y);
    let ql_byte = qs_base + group * 32u + l;
    var nibble: u32;
    if (sub == 0u) {
        nibble = read_byte(ql_byte) & 0x0Fu;
    } else {
        nibble = read_byte(ql_byte) >> 4u;
    }
    let bit_pos = e / 32u;  // 0..7
    let high_bit = (read_byte(qh_base + l) >> bit_pos) & 1u;
    let q5 = nibble | (high_bit << 4u);
    return sc_val * f32(q5) - m_val;
}

// ---------------------------------------------------------------------------
// Q6_K element dequant (210-byte superblocks, 256 elements)
// ---------------------------------------------------------------------------
fn dequant_q6k_elem(block_base: u32, e: u32) -> f32 {
    // Layout: ql(128) + qh(64) + scales(16) + d(2) = 210 bytes
    let ql_base     = block_base;
    let qh_base     = block_base + 128u;
    let scales_base = block_base + 192u;
    let d_byte      = block_base + 208u;
    let d_packed = extractBits(gguf_blob[d_byte / 4u],
                               (d_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;

    // Decompose element index (mirrors sh_layer_v1.wgsl dequant_q6k_elem)
    let half    = e / 128u;          // 0 or 1
    let half_e  = e % 128u;
    let quarter = half_e / 32u;      // 0..3
    let l       = half_e % 32u;      // 0..31 within quarter

    // ql: quarters 0&2 share the same ql byte (ql[half*64+l]);
    //     quarters 1&3 share ql[half*64+32+l].
    let ql_idx = half * 64u + select(l + 32u, l, quarter == 0u || quarter == 2u);
    let ql     = read_byte(ql_base + ql_idx);
    // low nibble for quarters 0,1 (half=sub=0); high nibble for quarters 2,3 (half=sub=1)
    let ql_val = select((ql >> 4u) & 0x0Fu, ql & 0x0Fu, quarter < 2u);

    // qh: one byte per l, covers all 4 quarters
    let qh_val  = read_byte(qh_base + half * 32u + l);
    let upper2  = (qh_val >> (quarter * 2u)) & 3u;

    let q6       = ql_val | (upper2 << 4u);
    let signed_q = i32(q6) - 32;

    // scale index = half*8 + (l/16) + quarter*2  (16 scales per block)
    let sc_idx = half * 8u + (l / 16u) + quarter * 2u;
    let sc_raw = read_byte(scales_base + sc_idx);
    let sc     = select(i32(sc_raw), i32(sc_raw) - 256, sc_raw >= 128u);

    return d * f32(sc) * f32(signed_q);
}

// ---------------------------------------------------------------------------
// F16 dequant
// ---------------------------------------------------------------------------
fn dequant_f16_at(byte_offset: u32) -> f32 {
    let packed = extractBits(gguf_blob[byte_offset / 4u],
                             (byte_offset % 4u) * 8u, 16u);
    return unpack2x16float(packed).x;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }

    let qt  = params.quant_type;
    let off = params.offset_bytes;
    var val: f32;

    if (qt == 14u) { // Q6_K — 256-elem superblocks, 210 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q6k_elem(off + b * 210u, e);
    } else if (qt == 13u) { // Q5_K — 256-elem superblocks, 176 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q5k_elem(off + b * 176u, e);
    } else if (qt == 12u) { // Q4_K — 256-elem superblocks, 144 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q4k_elem(off + b * 144u, e);
    } else if (qt == 8u) { // Q8_0 — 32-elem blocks, 34 bytes
        let b = i / 32u;
        let e = i % 32u;
        val = dequant_q8_0_elem(off + b * 34u, e);
    } else if (qt == 1u) { // F16 — 2 bytes per element
        val = dequant_f16_at(off + i * 2u);
    } else if (qt == 0u) { // F32 — 4 bytes per element
        val = bitcast<f32>(gguf_blob[(off / 4u) + i]);
    } else { // Q4_0 (qt == 2) and fallback — 32-elem blocks, 18 bytes
        let b = i / 32u;
        let e = i % 32u;
        val = dequant_q4_0_elem(off + b * 18u, e);
    }

    output[i] = val;
}
