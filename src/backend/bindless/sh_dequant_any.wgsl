

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
    blob_base_words: u32, // base_byte/4 for reconstructing absolute word index
    offset_words: u32,    // word offset relative to base (internal dispatch)
    count: u32,           // Number of f32 elements to produce
    formula_index: u32,   // B1 registry slot (0..7) — shader switches on this, not raw GGML type
    chunk_words: u32,     // words per blob chunk (effective_chunk / 4) — for chunk dispatch
};

@group(0) @binding(0)  var<storage, read> blob_0: array<u32>;
@group(0) @binding(10) var<storage, read> blob_1: array<u32>;
@group(0) @binding(11) var<storage, read> blob_2: array<u32>;
@group(0) @binding(12) var<storage, read> blob_3: array<u32>;
@group(0) @binding(13) var<storage, read> blob_4: array<u32>;
@group(0) @binding(14) var<storage, read> blob_5: array<u32>;
@group(0) @binding(15) var<storage, read> blob_6: array<u32>;
@group(0) @binding(16) var<storage, read> blob_7: array<u32>;
@group(0) @binding(1) var<storage, read_write> output    : array<f32>;
@group(0) @binding(2) var<uniform>             params    : DequantAnyParams;

// ---------------------------------------------------------------------------
// Byte-level read helper
// ---------------------------------------------------------------------------
fn read_blob(word_idx: u32) -> u32 {
    let chunk = word_idx / params.chunk_words;
    let off = word_idx % params.chunk_words;
    if chunk == 0u { return blob_0[off]; }
    if chunk == 1u { return blob_1[off]; }
    if chunk == 2u { return blob_2[off]; }
    if chunk == 3u { return blob_3[off]; }
    if chunk == 4u { return blob_4[off]; }
    if chunk == 5u { return blob_5[off]; }
    if chunk == 6u { return blob_6[off]; }
    return blob_7[off];
}

// Packed absolute offsets (host pack_blob_offset = byte/2). Never pack*2 in u32.
fn gow(pack: u32) -> u32 { return pack / 2u + params.blob_base_words; }
fn add_pack(pack: u32, even_bytes: u32) -> u32 { return pack + even_bytes / 2u; }
fn read_byte_rel(pack: u32, rel: u32) -> u32 {
    let adj = 2u * (pack % 2u) + rel;
    let word = pack / 2u + adj / 4u;
    return extractBits(read_blob(word), (adj % 4u) * 8u, 8u);
}
fn read_f16_rel(pack: u32, rel: u32) -> f32 {
    let adj = 2u * (pack % 2u) + rel;
    let word = pack / 2u + adj / 4u;
    let bits = extractBits(read_blob(word), (adj % 4u) * 8u, 16u);
    return f16_to_f32(bits);
}
fn read_u32_rel(pack: u32, rel: u32) -> u32 {
    let b0 = read_byte_rel(pack, rel);
    let b1 = read_byte_rel(pack, rel + 1u);
    let b2 = read_byte_rel(pack, rel + 2u);
    let b3 = read_byte_rel(pack, rel + 3u);
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

// ---------------------------------------------------------------------------
// get_scale_min_k4 — exact llama.cpp port (shared by Q4_K, Q5_K, Q6_K)
// ---------------------------------------------------------------------------
fn get_scale_min_k4(j: u32, block_pack: u32, scales_rel: u32) -> vec2<u32> {
    if (j < 4u) {
        let sc = read_byte_rel(block_pack, scales_rel + j) & 63u;
        let m  = read_byte_rel(block_pack, scales_rel + j + 4u) & 63u;
        return vec2<u32>(sc, m);
    } else {
        let sc = (read_byte_rel(block_pack, scales_rel + j + 4u) & 0x0Fu)
               | (((read_byte_rel(block_pack, scales_rel + j - 4u) >> 6u) & 0x03u) << 4u);
        let m  = ((read_byte_rel(block_pack, scales_rel + j + 4u) >> 4u) & 0x0Fu)
               | (((read_byte_rel(block_pack, scales_rel + j) >> 6u) & 0x03u) << 4u);
        return vec2<u32>(sc, m);
    }
}

// ---------------------------------------------------------------------------
// Q4_0 element dequant (18-byte blocks, 32 elements)
// ---------------------------------------------------------------------------
fn dequant_q4_0_elem(block_pack: u32, e: u32) -> f32 {
    let scale = read_f16_rel(block_pack, 0u);
    let qs = read_byte_rel(block_pack, 2u + (e % 16u));
    let nib = select(qs & 0x0Fu, qs >> 4u, e >= 16u);
    return (f32(nib) - 8.0) * scale;
}

// ---------------------------------------------------------------------------
// Q5_0 element dequant (22-byte blocks, 32 elements)
// ---------------------------------------------------------------------------
fn dequant_q5_0_elem(block_pack: u32, e: u32) -> f32 {
    let d = read_f16_rel(block_pack, 0u);
    let qh = read_u32_rel(block_pack, 2u);
    let high_bit = (qh >> e) & 1u;
    let qs = read_byte_rel(block_pack, 6u + (e % 16u));
    let low = select(qs >> 4u, qs & 0x0Fu, e < 16u);
    let val5 = low | (high_bit << 4u);
    return (f32(val5) - 16.0) * d;
}

// ---------------------------------------------------------------------------
// Q8_0 element dequant (34-byte blocks, 32 elements)
// ---------------------------------------------------------------------------
fn dequant_q8_0_elem(block_pack: u32, e: u32) -> f32 {
    let scale = read_f16_rel(block_pack, 0u);
    let raw = read_byte_rel(block_pack, 2u + e);
    let signed_val = select(i32(raw), i32(raw) - 256, raw >= 128u);
    return scale * f32(signed_val);
}

// ---------------------------------------------------------------------------
// Q4_K element dequant (144-byte superblocks, 256 elements)
// ---------------------------------------------------------------------------
fn dequant_q4k_elem(block_pack: u32, e: u32) -> f32 {
    let d = read_f16_rel(block_pack, 0u);
    let dmin_val = read_f16_rel(block_pack, 2u);
    let group    = e / 64u;
    let in_group = e % 64u;
    let sub      = in_group / 32u;
    let l        = in_group % 32u;
    let is = group * 2u + sub;
    let sm = get_scale_min_k4(is, block_pack, 4u);
    let sc_val = d * f32(sm.x);
    let m_val  = dmin_val * f32(sm.y);
    var nibble: u32;
    if (sub == 0u) {
        nibble = read_byte_rel(block_pack, 16u + group * 32u + l) & 0x0Fu;
    } else {
        nibble = read_byte_rel(block_pack, 16u + group * 32u + l) >> 4u;
    }
    return sc_val * f32(nibble) - m_val;
}

// ---------------------------------------------------------------------------
// Q5_K element dequant (176-byte superblocks, 256 elements)
// ---------------------------------------------------------------------------
fn dequant_q5k_elem(block_pack: u32, e: u32) -> f32 {
    let d = read_f16_rel(block_pack, 0u);
    let dmin_val = read_f16_rel(block_pack, 2u);
    let group    = e / 64u;
    let in_group = e % 64u;
    let sub      = in_group / 32u;
    let l        = in_group % 32u;
    let is = group * 2u + sub;
    let sm = get_scale_min_k4(is, block_pack, 4u);
    let sc_val = d * f32(sm.x);
    let m_val  = dmin_val * f32(sm.y);
    var nibble: u32;
    if (sub == 0u) {
        nibble = read_byte_rel(block_pack, 48u + group * 32u + l) & 0x0Fu;
    } else {
        nibble = read_byte_rel(block_pack, 48u + group * 32u + l) >> 4u;
    }
    let bit_pos = e / 32u;
    let high_bit = (read_byte_rel(block_pack, 16u + l) >> bit_pos) & 1u;
    let q5 = nibble | (high_bit << 4u);
    return sc_val * f32(q5) - m_val;
}

// ---------------------------------------------------------------------------
// Q6_K element dequant (210-byte superblocks, 256 elements)
// ---------------------------------------------------------------------------
fn dequant_q6k_elem(block_pack: u32, e: u32) -> f32 {
    let d = read_f16_rel(block_pack, 208u);
    let half    = e / 128u;
    let half_e  = e % 128u;
    let quarter = half_e / 32u;
    let l       = half_e % 32u;
    let ql_idx = half * 64u + select(l + 32u, l, quarter == 0u || quarter == 2u);
    let ql     = read_byte_rel(block_pack, ql_idx);
    let ql_val = select((ql >> 4u) & 0x0Fu, ql & 0x0Fu, quarter < 2u);
    let qh_val  = read_byte_rel(block_pack, 128u + half * 32u + l);
    let upper2  = (qh_val >> (quarter * 2u)) & 3u;
    let q6       = ql_val | (upper2 << 4u);
    let signed_q = i32(q6) - 32;
    let sc_idx = half * 8u + (l / 16u) + quarter * 2u;
    let sc_raw = read_byte_rel(block_pack, 192u + sc_idx);
    let sc     = select(i32(sc_raw), i32(sc_raw) - 256, sc_raw >= 128u);
    return d * f32(sc) * f32(signed_q);
}

// ---------------------------------------------------------------------------
// F16 dequant — pack is packed absolute (byte/2)
// ---------------------------------------------------------------------------
fn dequant_f16_at(pack: u32) -> f32 {
    return read_f16_rel(pack, 0u);
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }

    let slot = params.formula_index;
    let off_pack = params.offset_words;
    var val: f32;

    if (slot == 7u) { // Q6_K — 256-elem superblocks, 210 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q6k_elem(add_pack(off_pack, b * 210u), e);
    } else if (slot == 6u) { // Q5_K — 256-elem superblocks, 176 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q5k_elem(add_pack(off_pack, b * 176u), e);
    } else if (slot == 5u) { // Q4_K — 256-elem superblocks, 144 bytes
        let b = i / 256u;
        let e = i % 256u;
        val = dequant_q4k_elem(add_pack(off_pack, b * 144u), e);
    } else if (slot == 4u) { // Q8_0 — 32-elem blocks, 34 bytes
        let b = i / 32u;
        let e = i % 32u;
        val = dequant_q8_0_elem(add_pack(off_pack, b * 34u), e);
    } else if (slot == 3u) { // Q5_0 — 32-elem blocks, 22 bytes
        let b = i / 32u;
        let e = i % 32u;
        val = dequant_q5_0_elem(add_pack(off_pack, b * 22u), e);
    } else if (slot == 1u) { // F16 — 2 bytes per element
        val = dequant_f16_at(add_pack(off_pack, i * 2u));
    } else if (slot == 0u) { // F32 — 4 bytes per element
        val = bitcast<f32>(read_blob(gow(off_pack) + i));
    } else { // Q4_0 (slot == 2) and fallback — 32-elem blocks, 18 bytes
        let b = i / 32u;
        let e = i % 32u;
        val = dequant_q4_0_elem(add_pack(off_pack, b * 18u), e);
    }

    output[i] = val;
}
