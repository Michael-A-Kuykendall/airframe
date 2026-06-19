// sh_head_blob.wgsl
// LM Head matmul — reads quantized weights directly from the GGUF blob.
// One GPU thread per output vocabulary row.  No F32 dequant buffer needed.
//
// Supported quant types (matching llama.cpp GGML_TYPE_* values):
//   0  = F32
//   1  = F16
//   2  = Q4_0  (default / fallback)
//   8  = Q8_0
//  12  = Q4_K
//  13  = Q5_K
//  14  = Q6_K  ← primary path for Qwen2/MiniCPM-V output.weight

// ---------------------------------------------------------------------------
// Blob split constants — must match loader.rs (BLOB_CHUNK_BYTES = 2_000_000_000)
// ---------------------------------------------------------------------------
const BLOB_SPLIT_0: u32 = 500000000u;   // 2,000,000,000 bytes / 4 = 500 M words
const BLOB_SPLIT_1: u32 = 1000000000u;  // 4,000,000,000 bytes / 4 = 1 B words

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------
@group(0) @binding(0)  var<storage, read>       blob_0:  array<u32>;  // GGUF blob chunk 0 [0, 2 GB)
@group(0) @binding(1)  var<storage, read>        act_in:  array<f32>;  // normed activation [dim]
@group(0) @binding(2)  var<storage, read_write>  logits:  array<f32>;  // output [vocab_size]
@group(0) @binding(3)  var<uniform>              params:  HeadBlobParams;
@group(0) @binding(10) var<storage, read>        blob_1:  array<u32>;  // GGUF blob chunk 1 [2 GB, 4 GB)
@group(0) @binding(11) var<storage, read>        blob_2:  array<u32>;  // GGUF blob chunk 2 [4 GB, end)

// ---------------------------------------------------------------------------
// Uniform params
// ---------------------------------------------------------------------------
struct HeadBlobParams {
    vocab_size: u32,   // number of output tokens (rows of output.weight)
    dim:        u32,   // hidden dim (columns of output.weight = n_embd)
    weight_off: u32,   // word offset (byte_offset / 4) of output.weight inside the GGUF blob
    quant_type: u32,   // GGML quant type: 0=F32 1=F16 2=Q4_0 8=Q8_0 12=Q4_K 13=Q5_K 14=Q6_K
    softcap:    f32,   // final_logit_softcap (0.0 = disabled, Gemma-2 uses 30.0)
    base_row:   u32,   // output row offset for dispatch splitting (TDR tiles)
    _pad:       u32,
}

// ---------------------------------------------------------------------------
// Blob read helper
// ---------------------------------------------------------------------------
fn read_blob(word_idx: u32) -> u32 {
    if word_idx < BLOB_SPLIT_0 {
        return blob_0[word_idx];
    } else if word_idx < BLOB_SPLIT_1 {
        return blob_1[word_idx - BLOB_SPLIT_0];
    } else {
        return blob_2[word_idx - BLOB_SPLIT_1];
    }
}

fn read_byte_gguf(byte_idx: u32) -> u32 {
    return extractBits(read_blob(byte_idx / 4u), (byte_idx % 4u) * 8u, 8u);
}

// Weight-tensor-relative helpers.
// params.weight_off is a word index (absolute_byte_offset / 4).  All rel_byte / rel_word
// args are offsets from the start of the weight tensor, keeping all arithmetic safely
// within u32 range for models up to ~16 GB (max rel_word for Q6_K vocab ~112 M words).
fn read_wt_blob(rel_word: u32) -> u32 {
    return read_blob(params.weight_off + rel_word);
}
fn read_wt_byte(rel_byte: u32) -> u32 {
    return extractBits(read_wt_blob(rel_byte / 4u), (rel_byte % 4u) * 8u, 8u);
}

// ---------------------------------------------------------------------------
// Q4_K helpers (copied verbatim from sh_layer_v1.wgsl)
// ---------------------------------------------------------------------------
fn get_scale_min_k4(j: u32, scales_base_byte: u32) -> vec2<u32> {
    if (j < 4u) {
        let sc = read_wt_byte(scales_base_byte + j) & 63u;
        let m  = read_wt_byte(scales_base_byte + j + 4u) & 63u;
        return vec2<u32>(sc, m);
    } else {
        let sc = (read_wt_byte(scales_base_byte + j + 4u) & 0x0Fu)
               | (((read_wt_byte(scales_base_byte + j - 4u) >> 6u) & 3u) << 4u);
        let m  = ((read_wt_byte(scales_base_byte + j + 4u) >> 4u) & 0x0Fu)
               | (((read_wt_byte(scales_base_byte + j) >> 6u) & 3u) << 4u);
        return vec2<u32>(sc, m);
    }
}

fn dequant_q4k_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    let d_packed = extractBits(read_wt_blob(block_base_byte / 4u),
                               (block_base_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;
    let dmin_byte = block_base_byte + 2u;
    let dmin_packed = extractBits(read_wt_blob(dmin_byte / 4u),
                                  (dmin_byte % 4u) * 8u, 16u);
    let dmin_val = unpack2x16float(dmin_packed).x;
    let scales_base = block_base_byte + 4u;
    let qs_base     = block_base_byte + 16u;
    let group        = elem_in_block / 64u;
    let elem_in_grp  = elem_in_block % 64u;
    let is           = group * 2u;
    var sc_val: f32;
    var m_val:  f32;
    var nibble: u32;
    if (elem_in_grp < 32u) {
        let sm = get_scale_min_k4(is, scales_base);
        sc_val = d * f32(sm.x);
        m_val  = dmin_val * f32(sm.y);
        nibble = read_wt_byte(qs_base + group * 32u + elem_in_grp) & 0x0Fu;
    } else {
        let sm = get_scale_min_k4(is + 1u, scales_base);
        sc_val = d * f32(sm.x);
        m_val  = dmin_val * f32(sm.y);
        nibble = read_wt_byte(qs_base + group * 32u + (elem_in_grp - 32u)) >> 4u;
    }
    return sc_val * f32(nibble) - m_val;
}

// ---------------------------------------------------------------------------
// Q6_K helper
// ---------------------------------------------------------------------------
fn dequant_q6k_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    let d_byte = block_base_byte + 208u;
    let d_packed = extractBits(read_wt_blob(d_byte / 4u), (d_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;
    let half    = elem_in_block / 128u;
    let half_e  = elem_in_block % 128u;
    let l       = half_e % 32u;
    let quarter = half_e / 32u;
    let ql_rel = select(half * 64u + l + 32u, half * 64u + l, quarter == 0u || quarter == 2u);
    let ql_byte_val = read_wt_byte(block_base_byte + ql_rel);
    let lower4 = select(ql_byte_val >> 4u, ql_byte_val & 0xFu, quarter < 2u);
    let qh_byte_val = read_wt_byte(block_base_byte + 128u + half * 32u + l);
    let upper2 = (qh_byte_val >> (quarter * 2u)) & 3u;
    let q6 = lower4 | (upper2 << 4u);
    let signed_q = i32(q6) - 32;
    let sc_idx = 192u + half * 8u + (l / 16u) + quarter * 2u;
    let sc_raw = read_wt_byte(block_base_byte + sc_idx);
    let sc_signed = select(i32(sc_raw), i32(sc_raw) - 256, sc_raw >= 128u);
    return d * f32(sc_signed) * f32(signed_q);
}

// ---------------------------------------------------------------------------
// Q8_0 helper
// ---------------------------------------------------------------------------
fn dequant_q8_0_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    let scale_packed = extractBits(read_wt_blob(block_base_byte / 4u),
                                   (block_base_byte % 4u) * 8u, 16u);
    let scale = unpack2x16float(scale_packed).x;
    let qs_byte = block_base_byte + 2u + elem_in_block;
    let raw = read_wt_byte(qs_byte);
    let signed_val = select(i32(raw), i32(raw) - 256, raw >= 128u);
    return scale * f32(signed_val);
}

// ---------------------------------------------------------------------------
// F16 helper
// ---------------------------------------------------------------------------
fn dequant_f16_at(byte_offset: u32) -> f32 {
    let packed = extractBits(read_wt_blob(byte_offset / 4u),
                             (byte_offset % 4u) * 8u, 16u);
    return unpack2x16float(packed).x;
}

// ---------------------------------------------------------------------------
// Q5_K helper
// ---------------------------------------------------------------------------
fn dequant_q5k_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    let d_packed = extractBits(read_wt_blob(block_base_byte / 4u),
                               (block_base_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;
    let dmin_byte = block_base_byte + 2u;
    let dmin_packed = extractBits(read_wt_blob(dmin_byte / 4u),
                                  (dmin_byte % 4u) * 8u, 16u);
    let dmin_val = unpack2x16float(dmin_packed).x;
    let scales_base = block_base_byte + 4u;
    let qh_base     = block_base_byte + 16u;
    let qs_base     = block_base_byte + 48u;
    let group    = elem_in_block / 64u;
    let in_group = elem_in_block % 64u;
    let sub      = in_group / 32u;
    let l        = in_group % 32u;
    let is = group * 2u + sub;
    let sm = get_scale_min_k4(is, scales_base);
    let sc_val = d * f32(sm.x);
    let m_val  = dmin_val * f32(sm.y);
    let ql_byte = qs_base + group * 32u + l;
    var nibble: u32;
    if (sub == 0u) {
        nibble = read_wt_byte(ql_byte) & 0x0Fu;
    } else {
        nibble = read_wt_byte(ql_byte) >> 4u;
    }
    let bit_pos = elem_in_block / 32u;
    let high_bit = (read_wt_byte(qh_base + l) >> bit_pos) & 1u;
    let q5 = nibble | (high_bit << 4u);
    return sc_val * f32(q5) - m_val;
}

// ---------------------------------------------------------------------------
// Q5_0 helper (22 bytes/block, 32 elems/block)
// ---------------------------------------------------------------------------
fn dequant_q5_0_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    let d_packed = extractBits(read_wt_blob(block_base_byte / 4u),
                               (block_base_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;
    let qh_word = read_wt_blob((block_base_byte + 2u) / 4u);
    let qh_shift = (block_base_byte + 2u) % 4u;
    let qh = extractBits(qh_word, qh_shift * 8u, 32u);
    let high_bit = (qh >> elem_in_block) & 1u;
    let qs_byte = block_base_byte + 6u + (elem_in_block % 16u);
    let raw = read_wt_byte(qs_byte);
    let low_nibble = select(raw >> 4u, raw & 0x0Fu, elem_in_block < 16u);
    let val_5bit = low_nibble | (high_bit << 4u);
    return (f32(val_5bit) - 16.0) * d;
}

// ---------------------------------------------------------------------------
// Main kernel — one thread per output vocab row
// ---------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn main_lm_head(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = params.base_row + global_id.x;   // vocab row index, offset by tile base
    if (idx >= params.vocab_size) { return; }

    let dim = params.dim;
    var dot = 0.0f;

    if (params.quant_type == 14u) { // Q6_K  (210 bytes/block, 256 elems/block)
        let bpr       = dim / 256u;                         // blocks per row
        let row_start = idx * bpr * 210u;                   // relative byte offset from tensor start
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 210u;
            for (var e = 0u; e < 256u; e++) {
                dot += act_in[b * 256u + e] * dequant_q6k_elem(bb, e);
            }
        }
    } else if (params.quant_type == 13u) { // Q5_K  (176 bytes/block, 256 elems/block)
        let bpr       = dim / 256u;
        let row_start = idx * bpr * 176u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 176u;
            for (var e = 0u; e < 256u; e++) {
                dot += act_in[b * 256u + e] * dequant_q5k_elem(bb, e);
            }
        }
    } else if (params.quant_type == 12u) { // Q4_K  (144 bytes/block, 256 elems/block)
        let bpr       = dim / 256u;
        let row_start = idx * bpr * 144u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 144u;
            for (var e = 0u; e < 256u; e++) {
                dot += act_in[b * 256u + e] * dequant_q4k_elem(bb, e);
            }
        }
    } else if (params.quant_type == 6u) { // Q5_0  (22 bytes/block, 32 elems/block)
        let bpr       = dim / 32u;
        let row_start = idx * bpr * 22u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 22u;
            for (var e = 0u; e < 32u; e++) {
                dot += act_in[b * 32u + e] * dequant_q5_0_elem(bb, e);
            }
        }
    } else if (params.quant_type == 8u) { // Q8_0  (34 bytes/block, 32 elems/block)
        let bpr       = dim / 32u;
        let row_start = idx * bpr * 34u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 34u;
            for (var e = 0u; e < 32u; e++) {
                dot += act_in[b * 32u + e] * dequant_q8_0_elem(bb, e);
            }
        }
    } else if (params.quant_type == 1u) { // F16
        for (var col = 0u; col < dim; col++) {
            let w_byte = (idx * dim + col) * 2u;
            dot += act_in[col] * dequant_f16_at(w_byte);
        }
    } else if (params.quant_type == 0u) { // F32
        for (var col = 0u; col < dim; col++) {
            dot += act_in[col] * bitcast<f32>(read_wt_blob(idx * dim + col));
        }
    } else { // Q4_0 (default / fallback)  (18 bytes/block, 32 elems/block)
        let bpr          = dim / 32u;
        let row_start_b  = idx * bpr * 18u;
        for (var b = 0u; b < bpr; b++) {
            let block_base   = row_start_b + b * 18u;
            let scale_packed = extractBits(read_wt_blob(block_base / 4u),
                                           (block_base % 4u) * 8u, 16u);
            let scale        = unpack2x16float(scale_packed).x;
            let qs_byte_start = block_base + 2u;
            for (var i = 0u; i < 32u; i++) {
                let col      = b * 32u + i;
                let byte_idx = qs_byte_start + (i % 16u);
                let qs_word  = read_wt_blob(byte_idx / 4u);
                let qs_byte  = extractBits(qs_word, (byte_idx % 4u) * 8u, 8u);
                let nib      = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
                dot         += act_in[col] * ((f32(nib) - 8.0) * scale);
            }
        }
    }

    // Final logit softcap (Gemma-2: 30.0; disabled with 0.0)
    if (params.softcap > 0.0) {
        dot = tanh(dot / params.softcap) * params.softcap;
    }

    logits[idx] = dot;
}
