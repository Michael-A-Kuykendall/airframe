

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
    formula_index: u32, // B1 registry slot (0..7) — shader switches on this, not raw GGML type
    pad: u32,
};

@group(0) @binding(0)  var<storage, read> blob_0: array<u32>;
@group(0) @binding(10) var<storage, read> blob_1: array<u32>;
@group(0) @binding(11) var<storage, read> blob_2: array<u32>;
@group(0) @binding(1) var<storage, read_write> output    : array<f32>;
@group(0) @binding(2) var<uniform>             params    : DequantAnyParams;

// ---------------------------------------------------------------------------
// Byte-level read helper
// ---------------------------------------------------------------------------
const BLOB_SPLIT_0: u32 = 500000000u;
const BLOB_SPLIT_1: u32 = 1000000000u;

fn read_blob(word_idx: u32) -> u32 {
    if word_idx < BLOB_SPLIT_0 {
        return blob_0[word_idx];
    } else if word_idx < BLOB_SPLIT_1 {
        return blob_1[word_idx - BLOB_SPLIT_0];
    } else {
        return blob_2[word_idx - BLOB_SPLIT_1];
    }
}

fn read_byte(byte_idx: u32) -> u32 {
    return extractBits(read_blob(byte_idx / 4u), (byte_idx % 4u) * 8u, 8u);
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
    let scale_packed = extractBits(read_blob(block_base / 4u),
                                    (block_base % 4u) * 8u, 16u);
    let scale = f16_to_f32(scale_packed);
    let qs_byte = block_base + 2u + (e % 16u);
    let qs = read_byte(qs_byte);
    let nib = select(qs & 0x0Fu, qs >> 4u, e >= 16u);
    return (f32(nib) - 8.0) * scale;
}

// ---------------------------------------------------------------------------
// Q5_0 element dequant (22-byte blocks, 32 elements)
// ---------------------------------------------------------------------------
fn dequant_q5_0_elem(block_base: u32, e: u32) -> f32 {
    let d_packed = extractBits(read_blob(block_base / 4u),
                                (block_base % 4u) * 8u, 16u);
    let d = f16_to_f32(d_packed);
    // qh: u32 at byte offset block_base+2 (may be sub-word aligned).
    let qh_word  = read_blob((block_base + 2u) / 4u);
    let qh_shift = ((block_base + 2u) % 4u) * 8u;
    let qh       = extractBits(qh_word, qh_shift, 32u);
    let high_bit = (qh >> e) & 1u;
    let qs_byte  = block_base + 6u + (e % 16u);
    let qs       = read_byte(qs_byte);
    let low      = select(qs >> 4u, qs & 0x0Fu, e < 16u);
    let val5     = low | (high_bit << 4u);
    return (f32(val5) - 16.0) * d;
}

// ---------------------------------------------------------------------------
// Q8_0 element dequant (34-byte blocks, 32 elements)
// ---------------------------------------------------------------------------
fn dequant_q8_0_elem(block_base: u32, e: u32) -> f32 {
    let scale_packed = extractBits(read_blob(block_base / 4u),
                                   (block_base % 4u) * 8u, 16u);
    let scale = f16_to_f32(scale_packed);
    let qs_byte = block_base + 2u + e;
    let raw = read_byte(qs_byte);
    let signed_val = select(i32(raw), i32(raw) - 256, raw >= 128u);
    return scale * f32(signed_val);
}

// ---------------------------------------------------------------------------
// Q4_K element dequant (144-byte superblocks, 256 elements)
// ---------------------------------------------------------------------------
fn dequant_q4k_elem(block_base: u32, e: u32) -> f32 {
    let d_packed = extractBits(read_blob(block_base / 4u),
                               (block_base % 4u) * 8u, 16u);
    let d = f16_to_f32(d_packed);
    let dmin_byte = block_base + 2u;
    let dmin_packed = extractBits(read_blob(dmin_byte / 4u),
                                  (dmin_byte % 4u) * 8u, 16u);
    let dmin_val = f16_to_f32(dmin_packed);
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
    return sc_val * f32(nibble) - m_val;
}

// ---------------------------------------------------------------------------
// Q5_K element dequant (176-byte superblocks, 256 elements)
// ---------------------------------------------------------------------------
fn dequant_q5k_elem(block_base: u32, e: u32) -> f32 {
    let d_packed = extractBits(read_blob(block_base / 4u),
                               (block_base % 4u) * 8u, 16u);
    let d = f16_to_f32(d_packed);
    let dmin_byte = block_base + 2u;
    let dmin_packed = extractBits(read_blob(dmin_byte / 4u),
                                  (dmin_byte % 4u) * 8u, 16u);
    let dmin_val = f16_to_f32(dmin_packed);
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
    let d_packed = extractBits(read_blob(d_byte / 4u),
                               (d_byte % 4u) * 8u, 16u);
    let d = f16_to_f32(d_packed);

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
    let packed = extractBits(read_blob(byte_offset / 4u),
                             (byte_offset % 4u) * 8u, 16u);
    return f16_to_f32(packed);
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }

    let slot = params.formula_index;
    let off = params.offset_bytes;
    var val: f32;

    if (slot == 7u) { // Q6_K — 256-elem superblocks, 210 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q6k_elem(off + b * 210u, e);
    } else if (slot == 6u) { // Q5_K — 256-elem superblocks, 176 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q5k_elem(off + b * 176u, e);
    } else if (slot == 5u) { // Q4_K — 256-elem superblocks, 144 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q4k_elem(off + b * 144u, e);
    } else if (slot == 4u) { // Q8_0 — 32-elem blocks, 34 bytes
        let b = i / 32u;
        let e = i % 32u;
        val = dequant_q8_0_elem(off + b * 34u, e);
    } else if (slot == 3u) { // Q5_0 — 32-elem blocks, 22 bytes
        let b = i / 32u;
        let e = i % 32u;
        val = dequant_q5_0_elem(off + b * 22u, e);
    } else if (slot == 1u) { // F16 — 2 bytes per element
        val = dequant_f16_at(off + i * 2u);
    } else if (slot == 0u) { // F32 — 4 bytes per element
        val = bitcast<f32>(read_blob((off / 4u) + i));
    } else { // Q4_0 (slot == 2) and fallback — 32-elem blocks, 18 bytes
        let b = i / 32u;
        let e = i % 32u;
        val = dequant_q4_0_elem(off + b * 18u, e);
    }

    output[i] = val;
}
