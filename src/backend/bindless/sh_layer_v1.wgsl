// sh_layer_v1.wgsl
// Full Transformer Layer (TinyLlama Q4_0) - Split Kernels
// Revised for Bindless Architecture V2.3 (Preflight Fused Resources)

// Constants
const BLOCK_SIZE: u32 = 32u; // Q4_0 Block Size

struct LayerOffsets {
    attn_norm: u32,
    attn_norm_bias: u32,
    attn_q: u32,
    attn_k: u32,
    attn_v: u32,
    attn_out: u32,
    ffn_norm: u32,
    ffn_norm_bias: u32,
    ffn_gate: u32,
    ffn_down: u32,
    ffn_up: u32,
    layer_idx: u32,     // was padding[0] — layer index for norm_bank lookup
    attn_q_norm: u32,   // byte offset of Q-norm weights in GGUF blob (0 = disabled)
    attn_k_norm: u32,   // byte offset of K-norm weights in GGUF blob (0 = disabled)
    attn_q_bias: u32,   // byte offset of Q bias (F32) in GGUF blob (0 = disabled; Qwen2)
    attn_k_bias: u32,   // byte offset of K bias (F32) in GGUF blob (0 = disabled; Qwen2)
    attn_v_bias: u32,   // byte offset of V bias (F32) in GGUF blob (0 = disabled; Qwen2)
};

struct LayerParams {
    dim: u32,           // 2048
    head_count: u32,    // 32
    head_count_kv: u32, // 4 (GQA)
    head_dim: u32,      // 64
    rope_dim: u32,      // rotary sub-dimension (may be < head_dim for partial RoPE)
    rms_eps: f32,       // 1e-5
    ffn_dim: u32,       // Feed-forward intermediate dim (e.g. 5632)
    temp_stride: u32,   // Per-token temp buffer stride in floats (e.g. 16384)
    quant_type: u32,    // GGML type: 0=F32, 1=F16, 2=Q4_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
    attn_logit_softcap: f32, // 0.0 = disabled; Gemma-2 uses 50.0
    post_norm_enabled: u32,   // 1 = apply post-attn/post-ffw norms (Gemma-2); 0 = disabled
    qk_norm_enabled: u32,     // 1 = apply per-head Q/K RMSNorm before attention (Qwen3); 0 = disabled
    layer_norm_enabled: u32,  // 1 = LayerNorm (mean+variance), 0 = RMSNorm
    ffn_kind_policy: u32,     // 0 = infer from offsets, 1 = gated, 2 = non-gated
    qkv_layout_policy: u32,   // 0 = infer from offsets, 1 = separate, 2 = fused
    batch_offset: u32,        // first token index in this QKV micro-batch chunk
    batch_count: u32,         // number of tokens in this chunk (guard for last chunk)
    q_weight_k: u32,          // stored K for attn_q.weight (packed Qwen3 etc; 0=dim)
    k_weight_k: u32,          // for attn_k
};

struct CacheParams {
    current_pos: u32,   // Position to write new K/V (0-based)
    seq_len: u32,       // Total cached positions (current_pos + batch_size)
    max_seq_len: u32,   // Context window (2048)
    batch_size: u32,    // Number of tokens in this dispatch
    logical_pos_base: u32, // Logical base of the compacted sliding window
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

// Bindings
@group(0) @binding(0)  var<storage, read> blob_0: array<u32>;
@group(0) @binding(10) var<storage, read> blob_1: array<u32>;
@group(0) @binding(11) var<storage, read> blob_2: array<u32>;

const BLOB_SPLIT_0: u32 = 500000000u;  // 2,000,000,000 bytes / 4 = 500M words
const BLOB_SPLIT_1: u32 = 1000000000u; // 4,000,000,000 bytes / 4 = 1B words

fn read_blob(word_idx: u32) -> u32 {
    if word_idx < BLOB_SPLIT_0 {
        return blob_0[word_idx];
    } else if word_idx < BLOB_SPLIT_1 {
        return blob_1[word_idx - BLOB_SPLIT_0];
    } else {
        return blob_2[word_idx - BLOB_SPLIT_1];
    }
}
@group(0) @binding(1) var<storage, read_write> activation_in: array<f32>; // The "Residual" stream
@group(0) @binding(2) var<storage, read_write> temp_state: array<f32>;    // Scratchpad
@group(0) @binding(3) var<uniform> offsets: LayerOffsets;
@group(0) @binding(4) var<uniform> params: LayerParams;
@group(0) @binding(5) var<storage, read> norm_bank: array<f32>;           // [n_layer * dim * 4 + dim]
@group(0) @binding(6) var<storage, read> rope_table: array<f32>;           // [2048 × head_dim/2 × 2] pre-computed (cos, sin)
@group(0) @binding(7) var<storage, read_write> kv_cache_k: array<f32>;    // K cache [max_seq * n_head_kv * head_dim]
@group(0) @binding(8) var<storage, read_write> kv_cache_v: array<f32>;    // V cache [max_seq * n_head_kv * head_dim]
@group(0) @binding(9) var<uniform> cache_params: CacheParams;             // Sequence position tracking

fn is_non_gated() -> bool {
    if (params.ffn_kind_policy == 2u) {
        return true;
    }
    if (params.ffn_kind_policy == 1u) {
        return false;
    }
    return offsets.ffn_gate == 0u;
}

// Helper functions for Q4_0 dequant
fn unpack_q4_0(block_val: u32, idx_in_block: u32) -> f32 {
    let shift = (idx_in_block % 8u) * 4u;
    return f32((block_val >> shift) & 0xFu) - 8.0;
}

// -------------------------------------------------------------------------
// Q4_K dequant helpers
// Q4_K block layout (144 bytes per 256-element superblock):
//   [0..1]   d     (fp16)
//   [2..3]   dmin  (fp16)
//   [4..15]  scales (12 bytes, 6-bit packed sub-block scale/min factors)
//   [16..143] qs   (128 bytes, 4-bit nibbles)
// -------------------------------------------------------------------------
fn read_byte_gguf(byte_idx: u32) -> u32 {
    return extractBits(read_blob(byte_idx / 4u), (byte_idx % 4u) * 8u, 8u);
}

// Returns vec2(sc, m) — 6-bit unsigned scale and min for sub-block j.
// Exact port of llama.cpp get_scale_min_k4.
fn get_scale_min_k4(j: u32, scales_base_byte: u32) -> vec2<u32> {
    if (j < 4u) {
        let sc = read_byte_gguf(scales_base_byte + j) & 63u;
        let m  = read_byte_gguf(scales_base_byte + j + 4u) & 63u;
        return vec2<u32>(sc, m);
    } else {
        let sc = (read_byte_gguf(scales_base_byte + j + 4u) & 0x0Fu)
               | (((read_byte_gguf(scales_base_byte + j - 4u) >> 6u) & 3u) << 4u);
        let m  = ((read_byte_gguf(scales_base_byte + j + 4u) >> 4u) & 0x0Fu)
               | (((read_byte_gguf(scales_base_byte + j) >> 6u) & 3u) << 4u);
        return vec2<u32>(sc, m);
    }
}

// Dequantize one element from a Q4_K superblock.
// block_base_byte: byte offset of the 144-byte superblock in gguf_blob
// elem_in_block:   0..255
fn dequant_q4k_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    // Read d (fp16 at offset 0)
    let d_packed = extractBits(read_blob(block_base_byte / 4u),
                               (block_base_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;

    // Read dmin (fp16 at offset 2)
    let dmin_byte = block_base_byte + 2u;
    let dmin_packed = extractBits(read_blob(dmin_byte / 4u),
                                  (dmin_byte % 4u) * 8u, 16u);
    let dmin_val = unpack2x16float(dmin_packed).x;

    let scales_base = block_base_byte + 4u;   // 12 scale bytes
    let qs_base     = block_base_byte + 16u;  // 128 nibble bytes

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
        nibble = read_byte_gguf(qs_base + group * 32u + elem_in_grp) & 0x0Fu;
    } else {
        let sm = get_scale_min_k4(is + 1u, scales_base);
        sc_val = d * f32(sm.x);
        m_val  = dmin_val * f32(sm.y);
        nibble = read_byte_gguf(qs_base + group * 32u + (elem_in_grp - 32u)) >> 4u;
    }

    return sc_val * f32(nibble) - m_val;
}

// -------------------------------------------------------------------------
// Q6_K dequant helper
// Q6_K block layout (210 bytes per 256-element superblock):
//   [0..127]   ql[128]    - lower 4 bits of 6-bit quants
//   [128..191] qh[64]     - upper 2 bits of 6-bit quants
//   [192..207] scales[16] - int8 sub-block scales (one per 16 elements)
//   [208..209] d          - fp16 super-block scale
// Exact port of llama.cpp dequantize_row_q6_K.
// -------------------------------------------------------------------------
fn dequant_q6k_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    // Read d (fp16 at byte offset 208)
    let d_byte = block_base_byte + 208u;
    let d_packed = extractBits(read_blob(d_byte / 4u), (d_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;

    let half    = elem_in_block / 128u;   // 0 or 1
    let half_e  = elem_in_block % 128u;   // 0..127
    let l       = half_e % 32u;           // position within quarter
    let quarter = half_e / 32u;           // 0..3

    // ql index: quarters 0,2 use ql[half*64 + l], quarters 1,3 use ql[half*64 + l + 32]
    let ql_rel = select(half * 64u + l + 32u, half * 64u + l, quarter == 0u || quarter == 2u);
    let ql_byte_val = read_byte_gguf(block_base_byte + ql_rel);

    // lower4: quarters 0,1 use low nibble; quarters 2,3 use high nibble
    let lower4 = select(ql_byte_val >> 4u, ql_byte_val & 0xFu, quarter < 2u);

    // qh: one byte per l within a half (at block offset 128 + half*32 + l)
    let qh_byte_val = read_byte_gguf(block_base_byte + 128u + half * 32u + l);
    let upper2 = (qh_byte_val >> (quarter * 2u)) & 3u;

    // 6-bit value, signed (range -32..31)
    let q6 = lower4 | (upper2 << 4u);
    let signed_q = i32(q6) - 32;

    // int8 scale: block offset 192 + half*8 + (l/16) + quarter*2
    let sc_idx = 192u + half * 8u + (l / 16u) + quarter * 2u;
    let sc_raw = read_byte_gguf(block_base_byte + sc_idx);
    let sc_signed = select(i32(sc_raw), i32(sc_raw) - 256, sc_raw >= 128u);

    return d * f32(sc_signed) * f32(signed_q);
}

// -------------------------------------------------------------------------
// Q8_0 dequant helper
// Q8_0 block layout (34 bytes per 32-element block):
//   [0..1]  d   (fp16 scale)
//   [2..33] qs  (32 int8 values)
// -------------------------------------------------------------------------
fn dequant_q8_0_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    let scale_packed = extractBits(read_blob(block_base_byte / 4u),
                                   (block_base_byte % 4u) * 8u, 16u);
    let scale = unpack2x16float(scale_packed).x;
    let qs_byte = block_base_byte + 2u + elem_in_block;
    let raw = read_byte_gguf(qs_byte);
    let signed_val = select(i32(raw), i32(raw) - 256, raw >= 128u);
    return scale * f32(signed_val);
}

// -------------------------------------------------------------------------
// Q5_0 dequant helper (type 6)
// Q5_0 block layout (22 bytes per 32-element block):
//   [0..1]   d      (fp16 scale)
//   [2..5]   qh     (uint32: bit i = 5th bit of element i)
//   [6..21]  qs     (16 nibble bytes: low nibble=elements 0..15, high nibble=elements 16..31)
// dequant: val = (nibble | (high_bit<<4) - 16) * d
// -------------------------------------------------------------------------
fn dequant_q5_0_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    let d_packed = extractBits(read_blob(block_base_byte / 4u),
                               (block_base_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;

    // qh: uint32 at byte offset 2
    let qh_word = read_blob((block_base_byte + 2u) / 4u);
    let qh_shift = (block_base_byte + 2u) % 4u;
    let qh = extractBits(qh_word, qh_shift * 8u, 32u);
    let high_bit = (qh >> elem_in_block) & 1u;

    // qs: nibbles at byte offset 6, packed 2 per byte (low nibble = elem byte, high nibble = elem byte+16)
    let qs_byte = block_base_byte + 6u + (elem_in_block % 16u);
    let raw = read_byte_gguf(qs_byte);
    let low_nibble = select(raw >> 4u, raw & 0x0Fu, elem_in_block < 16u);

    let val_5bit = low_nibble | (high_bit << 4u);
    return (f32(val_5bit) - 16.0) * d;
}

// Read a single fp16 value from an arbitrary byte offset in gguf_blob.
fn dequant_f16_at(byte_offset: u32) -> f32 {
    let packed = extractBits(read_blob(byte_offset / 4u),
                             (byte_offset % 4u) * 8u, 16u);
    return unpack2x16float(packed).x;
}

// -------------------------------------------------------------------------
// Q5_K dequant helper
// Q5_K block layout (176 bytes per 256-element superblock):
//   [0..1]   d       (fp16)
//   [2..3]   dmin    (fp16)
//   [4..15]  scales  (12 bytes, same 6-bit packed format as Q4_K)
//   [16..47] qh      (32 bytes: for element i, high_bit = (qh[i%32] >> (i/32)) & 1)
//   [48..175] qs     (128 bytes: low 4 bits per element)
// 5-bit value: q5 = nibble | (high_bit << 4) → range 0..31
// dequant:     val = d * sc * q5 - dmin * m
// -------------------------------------------------------------------------
fn dequant_q5k_elem(block_base_byte: u32, elem_in_block: u32) -> f32 {
    // d and dmin (fp16)
    let d_packed = extractBits(read_blob(block_base_byte / 4u),
                               (block_base_byte % 4u) * 8u, 16u);
    let d = unpack2x16float(d_packed).x;
    let dmin_byte = block_base_byte + 2u;
    let dmin_packed = extractBits(read_blob(dmin_byte / 4u),
                                  (dmin_byte % 4u) * 8u, 16u);
    let dmin_val = unpack2x16float(dmin_packed).x;

    let scales_base = block_base_byte + 4u;
    let qh_base     = block_base_byte + 16u;
    let qs_base     = block_base_byte + 48u;  // NOTE: 48, not 16 like Q4_K

    let group    = elem_in_block / 64u;
    let in_group = elem_in_block % 64u;
    let sub      = in_group / 32u;
    let l        = in_group % 32u;

    let is = group * 2u + sub;
    let sm = get_scale_min_k4(is, scales_base);
    let sc_val = d * f32(sm.x);
    let m_val  = dmin_val * f32(sm.y);

    // Low nibble: qs[group*32 + l]
    let ql_byte = qs_base + group * 32u + l;
    var nibble: u32;
    if (sub == 0u) {
        nibble = read_byte_gguf(ql_byte) & 0x0Fu;
    } else {
        nibble = read_byte_gguf(ql_byte) >> 4u;
    }

    // High bit: qh[l] bit (elem_in_block/32)
    // elem_in_block/32 = group*2 + sub, which cycles 0..7 over the 256 elements
    let bit_pos = elem_in_block / 32u;  // 0..7
    let high_bit = (read_byte_gguf(qh_base + l) >> bit_pos) & 1u;

    let q5 = nibble | (high_bit << 4u);  // 0..31
    return sc_val * f32(q5) - m_val;
}

// -------------------------------------------------------------------------
// Kernel 0: Attention RMSNorm Provider
// -------------------------------------------------------------------------
// Computes the shared attention-normalized activation stream once per token.
// Writes normalized activations into temp_state[0..dim) for Q/K/V consumers.
@compute @workgroup_size(256, 1, 1)
fn main_attn_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = global_id.y;

    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }

    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    var sum_sq = 0.0;
    var sum = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let val = activation_in[act_base + i];
        sum_sq += val * val;
        sum += val;
    }
    let mean = sum / f32(params.dim);
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    var inv_std = rms;
    if (params.layer_norm_enabled != 0u) {
        var var_sum = 0.0;
        for (var i = 0u; i < params.dim; i++) {
            let d = activation_in[act_base + i] - mean;
            var_sum += d * d;
        }
        inv_std = inverseSqrt(var_sum / f32(params.dim) + params.rms_eps);
    }

    let norm_offset_base = offsets.layer_idx * 4u * params.dim;
    let norm_w = norm_bank[norm_offset_base + idx];
    let norm_b = select(0.0, bitcast<f32>(read_blob(offsets.attn_norm_bias / 4u + idx)), offsets.attn_norm_bias != 0u);
    let centered = select(activation_in[act_base + idx], activation_in[act_base + idx] - mean, params.layer_norm_enabled != 0u);
    let normed = centered * inv_std * norm_w + norm_b;
    temp_state[temp_base + idx] = normed;

    // Phi uses a shared pre-attention normalized input for both branches.
    // Keep existing non-gated behavior as well.
    if (params.layer_norm_enabled != 0u || is_non_gated()) {
        temp_state[temp_base + params.ffn_dim * 2u + idx] = normed;
    }
}

// -------------------------------------------------------------------------
// Kernel 1: QKV Generation + RoPE + Cache Update
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_qkv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = global_id.y + params.batch_offset;
    
    if (global_id.y >= params.batch_count) { return; }
    if (token_idx >= cache_params.batch_size) { return; }
    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    
    let dim_q = params.head_count * params.head_dim;           
    let dim_k = params.head_count_kv * params.head_dim;        
    let dim_v = params.head_count_kv * params.head_dim;        
    let total_out = dim_q + dim_k + dim_v;                     
    
    if (idx >= total_out) { return; }

    // 1. Select Weight Matrix & Row
    var weight_off_roots: array<u32, 3>;
    weight_off_roots[0] = offsets.attn_q;
    weight_off_roots[1] = offsets.attn_k;
    weight_off_roots[2] = offsets.attn_v;

    var target_stage = 0u; // 0=Q, 1=K, 2=V
    var row_idx = idx;
    
    if (idx >= dim_q) {
        if (idx < dim_q + dim_k) {
            target_stage = 1u;
            row_idx = idx - dim_q;
        } else {
            target_stage = 2u;
            row_idx = idx - (dim_q + dim_k);
        }
    }

    let weight_byte_offset = weight_off_roots[target_stage];
    
    // 2. MatMul (Row `row_idx`) against the staged attention-normalized provider.
    var dot: f32 = 0.0;

    // Per-component quant type: bits 0-7 = main (Q/K), bits 8-15 = V, bits 16-23 = ffn_down
    let qt_main = params.quant_type & 0xFFu;
    let qt_v    = (params.quant_type >> 8u) & 0xFFu;
    let qt      = select(qt_main, qt_v, target_stage == 2u);

    if (qt == 14u) { // Q6_K
        let bpr = params.dim / 256u;
        let row_start = weight_byte_offset + (row_idx * bpr * 210u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 210u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                dot += temp_state[temp_base + col] * dequant_q6k_elem(bb, e);
            }
        }
    } else if (qt == 13u) { // Q5_K
        let bpr = params.dim / 256u;
        let row_start = weight_byte_offset + (row_idx * bpr * 176u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 176u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                dot += temp_state[temp_base + col] * dequant_q5k_elem(bb, e);
            }
        }
    } else if (qt == 12u) { // Q4_K
        let blocks_per_row_k = params.dim / 256u;
        let row_start_byte_k = weight_byte_offset + (row_idx * blocks_per_row_k * 144u);
        for (var b = 0u; b < blocks_per_row_k; b++) {
            let block_base_k = row_start_byte_k + (b * 144u);
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_x = temp_state[temp_base + col];
                let val_w = dequant_q4k_elem(block_base_k, e);
                dot += val_x * val_w;
            }
        }
    } else if (qt == 6u) { // Q5_0
        let bpr = params.dim / 32u;
        let row_start = weight_byte_offset + row_idx * bpr * 22u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 22u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                dot += temp_state[temp_base + col] * dequant_q5_0_elem(bb, e);
            }
        }
    } else if (qt == 8u) { // Q8_0
        let bpr = params.dim / 32u;
        let row_start = weight_byte_offset + row_idx * bpr * 34u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 34u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                dot += temp_state[temp_base + col] * dequant_q8_0_elem(bb, e);
            }
        }
    } else if (qt == 1u) { // F16
        for (var col = 0u; col < params.dim; col++) {
            let w_byte = weight_byte_offset + (row_idx * params.dim + col) * 2u;
            dot += temp_state[temp_base + col] * dequant_f16_at(w_byte);
        }
    } else if (qt == 0u) { // F32
        for (var col = 0u; col < params.dim; col++) {
            let w_idx = weight_byte_offset / 4u + row_idx * params.dim + col;
            dot += temp_state[temp_base + col] * bitcast<f32>(read_blob(w_idx));
        }
    } else { // Q4_0 (quant_type == 2)
        let blocks_per_row = params.dim / 32u;
        let row_start_byte = weight_byte_offset + (row_idx * blocks_per_row * 18u);
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_base = row_start_byte + (b * 18u);
            let scale_idx = block_base / 4u;
            let scale_packed = extractBits(read_blob(scale_idx), (block_base % 4u) * 8u, 16u);
            let scale = unpack2x16float(scale_packed).x;
            let qs_byte_start = block_base + 2u;
            for (var i = 0u; i < 32u; i++) {
                let col = b * 32u + i;
                let val_x = temp_state[temp_base + col];
                let byte_idx = i % 16u;
                let qs_idx = qs_byte_start + byte_idx;
                let qs_word = read_blob(qs_idx / 4u);
                let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
                let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
                let val_w = (f32(nib) - 8.0) * scale;
                dot += val_x * val_w;
            }
        }
    }

    // 4. QKV Bias addition (Qwen2 uses F32 biases on Q, K, V projections)
    // When the bias offset is non-zero, add the stored F32 value to the dot product.
    var qkv_bias_off: u32;
    if (target_stage == 0u) { qkv_bias_off = offsets.attn_q_bias; }
    else if (target_stage == 1u) { qkv_bias_off = offsets.attn_k_bias; }
    else { qkv_bias_off = offsets.attn_v_bias; }
    if (qkv_bias_off != 0u) {
        dot += bitcast<f32>(read_blob(qkv_bias_off / 4u + row_idx));
    }

    // 5. RoPE - REMOVED (Relative RoPE Architecture)
    // Q and K are stored RAW without absolute position encoding.
    // Position-dependent rotation is applied as relative RoPE(i-j)
    // during the Q·K dot product in main_attn_out.
    // This enables infinite context via sliding window:
    // - Only relative distances matter (observer-relative frame)
    // - Max relative distance bounded by window size (always in training range)
    // - No position counter overflow, no extrapolation artifacts

    // 6. Store Output
    // Q -> Temp State (offset dim)
    // K, V -> KV Cache at current position
    // 
    // Cache layout per buffer: [max_seq, n_head_kv, head_dim]
    // Element offset: (pos * n_head_kv * head_dim) + (head * head_dim) + dim
    
    if (target_stage == 0u) {
        // Q goes to temp_state for attention computation
        temp_state[temp_base + params.dim + row_idx] = dot;
    } else if (target_stage == 1u) {
        // K -> K cache
        // row_idx = head * head_dim + dim_in_head (0..255 for 4 heads * 64 dims)
        let head = row_idx / params.head_dim;
        let dim_in_head = row_idx % params.head_dim;
        // batch_offset positions this chunk correctly within the prefill sequence.
        let cache_idx = ((cache_params.current_pos + params.batch_offset + token_idx) * params.head_count_kv * params.head_dim)
                      + (head * params.head_dim)
                      + dim_in_head;
        kv_cache_k[cache_idx] = dot; 
    } else {
        // V -> V cache
        let head = row_idx / params.head_dim;
        let dim_in_head = row_idx % params.head_dim;
        let cache_idx = ((cache_params.current_pos + params.batch_offset + token_idx) * params.head_count_kv * params.head_dim)
                      + (head * params.head_dim)
                      + dim_in_head;
        kv_cache_v[cache_idx] = dot;
    }
}

// -------------------------------------------------------------------------
// Kernel 1.5: QK Norm — per-head RMSNorm on Q and K (Qwen3 only)
// -------------------------------------------------------------------------
// When qk_norm_enabled == 1: normalizes each Q head (in temp_state) and each
// K head (in kv_cache_k at the freshly-written position) using per-element
// F32 weights stored in the GGUF blob at offsets.attn_q_norm / attn_k_norm.
//
// When qk_norm_enabled == 0: no-op (early return).
//
// Layout:
//   idx 0..dim_q-1           -> Q heads in temp_state[temp_base + dim + idx]
//   idx dim_q..dim_q+dim_k-1 -> K heads in kv_cache_k[...] at current_pos
//
// Each thread owns one output element. To compute RMS it reads all head_dim
// elements of its head (scalar loop, same approach as main_attn_norm).
@compute @workgroup_size(256, 1, 1)
fn main_qk_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx        = global_id.x;
    let token_idx  = global_id.y;

    if (token_idx >= cache_params.batch_size) { return; }
    if (params.qk_norm_enabled == 0u) { return; }

    let temp_base = token_idx * params.temp_stride;
    let dim_q     = params.head_count    * params.head_dim;
    let dim_k     = params.head_count_kv * params.head_dim;

    if (idx >= dim_q + dim_k) { return; }

    let is_k       = idx >= dim_q;
    let elem_idx   = select(idx, idx - dim_q, is_k);
    let head_idx   = elem_idx / params.head_dim;
    let dim_in_head = elem_idx % params.head_dim;

    // Norm weight byte offset in GGUF blob (F32, one per head_dim element)
    let norm_off = select(offsets.attn_q_norm, offsets.attn_k_norm, is_k);

    // Compute RMS across the full head (all head_dim elements)
    var sum_sq = 0.0;
    if (!is_k) {
        let q_base = temp_base + params.dim + head_idx * params.head_dim;
        for (var i = 0u; i < params.head_dim; i++) {
            let v = temp_state[q_base + i];
            sum_sq += v * v;
        }
    } else {
        let cache_base = (cache_params.current_pos + token_idx) * params.head_count_kv * params.head_dim
                       + head_idx * params.head_dim;
        for (var i = 0u; i < params.head_dim; i++) {
            let v = kv_cache_k[cache_base + i];
            sum_sq += v * v;
        }
    }

    let rms = inverseSqrt(sum_sq / f32(params.head_dim) + params.rms_eps);

    // Read norm weight for this dimension from GGUF blob (F32)
    let w = bitcast<f32>(read_blob(norm_off / 4u + dim_in_head));

    // Apply norm and write back
    if (!is_k) {
        let q_base = temp_base + params.dim + head_idx * params.head_dim;
        temp_state[q_base + dim_in_head] = temp_state[q_base + dim_in_head] * rms * w;
    } else {
        let cache_base = (cache_params.current_pos + token_idx) * params.head_count_kv * params.head_dim
                       + head_idx * params.head_dim;
        kv_cache_k[cache_base + dim_in_head] = kv_cache_k[cache_base + dim_in_head] * rms * w;
    }
}

// -------------------------------------------------------------------------
// Kernel 2: Attention — Flash Attention online softmax (no scores buffer)
// -------------------------------------------------------------------------
// Computes attention output in a single pass using online softmax
// (Flash Attention style — Milakov & Gimelshein 2018).
// Benefits:
//   - O(1) auxiliary state per thread (running_max, running_sum, running_out)
//   - No external scratch buffer required
//   - Handles arbitrary max_seq_len — correct for ctx > 2048
//   - Works for batched prefill (batch_size > 1) and decode (batch_size == 1)
@compute @workgroup_size(256, 1, 1)
fn main_attn_out(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;       // Output dimension index (0..attn_dim-1)
    let token_idx = global_id.y; // Batch token index

    if (token_idx >= cache_params.batch_size) { return; }
    // Guard against models where dim > n_head * head_dim (e.g. Gemma-2: 2304 vs 2048)
    let attn_dim = params.head_count * params.head_dim;
    if (idx >= attn_dim) { return; }

    let temp_base  = token_idx * params.temp_stride;
    let gqa_ratio  = params.head_count / params.head_count_kv;
    let scale      = 1.0 / sqrt(f32(params.head_dim));
    let head_idx   = idx / params.head_dim;
    let head_offset = idx % params.head_dim;
    let kv_head_idx = head_idx / gqa_ratio;
    let q_base     = temp_base + params.dim + head_idx * params.head_dim;
    let n_pairs    = params.head_dim / 2u;
    let rope_pairs = params.rope_dim / 2u;
    let compact_query_pos = cache_params.current_pos + token_idx;
    let logical_query_pos = cache_params.logical_pos_base + compact_query_pos;

    const SINK_COUNT: u32 = 4u;

    // Online softmax accumulators (Flash Attention — O(1) memory)
    var running_max: f32 = -1e10;
    var running_sum: f32 = 0.0;
    var running_out: f32 = 0.0;

    for (var pos = 0u; pos < cache_params.seq_len; pos++) {
        // Causal mask: skip positions the current query cannot attend to
        if (pos > cache_params.current_pos + token_idx) { continue; }

        var rel: u32 = compact_query_pos - pos;
        if (pos < SINK_COUNT) {
            // Sink positions are pinned to absolute slots 0..SINK_COUNT-1 across compactions.
            // Use the logical query position so sink-relative distance survives helical shift.
            rel = logical_query_pos - pos;
            rel = min(rel, cache_params.max_seq_len - 1u);
        } else if (rel >= cache_params.max_seq_len) {
            // Beyond sliding window context horizon: skip
            continue;
        }

        // Compute Q · RoPE(rel) · K[pos]  (grouped-query attention, GQA)
        var dot_qk: f32 = 0.0;
        let k_base = pos * params.head_count_kv * params.head_dim
                   + kv_head_idx * params.head_dim;
        for (var p = 0u; p < n_pairs; p++) {
            var cos_a = 1.0;
            var sin_a = 0.0;
            if (p < rope_pairs) {
                let tbl = rel * rope_pairs * 2u + p * 2u;
                cos_a = rope_table[tbl];
                sin_a = rope_table[tbl + 1u];
            }
            let doff  = p * 2u;
            let q_re  = temp_state[q_base + doff];
            let q_im  = temp_state[q_base + doff + 1u];
            let k_re  = kv_cache_k[k_base + doff];
            let k_im  = kv_cache_k[k_base + doff + 1u];
            dot_qk += (q_re * k_re + q_im * k_im) * cos_a
                    + (q_re * k_im - q_im * k_re) * sin_a;
        }
        let score_raw = dot_qk * scale;
        // Gemma-2 attention logit soft-cap: tanh(score / cap) * cap
        let score = select(score_raw,
                           tanh(score_raw / params.attn_logit_softcap) * params.attn_logit_softcap,
                           params.attn_logit_softcap > 0.0);

        // V value for this position / kv-head / output element
        let v_val = kv_cache_v[
            pos * params.head_count_kv * params.head_dim
            + kv_head_idx * params.head_dim
            + head_offset
        ];

        // Online softmax update (numerically stable)
        let m_new     = max(running_max, score);
        let exp_diff  = exp(running_max - m_new);
        let exp_score = exp(score - m_new);
        running_sum = running_sum * exp_diff + exp_score;
        running_out = running_out * exp_diff + exp_score * v_val;
        running_max = m_new;
    }

    // Finalize: divide accumulated output by softmax denominator
    var context_val = 0.0;
    if (running_sum > 0.0) {
        context_val = running_out / running_sum;
    }
    temp_state[temp_base + idx] = context_val;
}

// -------------------------------------------------------------------------
// Kernel 2.5: Output Projection (NEW - Split from attention)
// -------------------------------------------------------------------------
// Apply output projection matrix to attention context
// NOTE: attn_dim = head_count * head_dim may differ from params.dim (e.g. Gemma-2:
// n_head=8, head_dim=256 → attn_dim=2048 but params.dim=2304). The W_o matrix has
// attn_dim columns (input) and params.dim rows (output). Use attn_dim for inner loops.
@compute @workgroup_size(256, 1, 1)
fn main_attn_proj(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // Output dimension (0..params.dim-1)
    let token_idx = global_id.y;
    
    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }
    
    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    // Attention output dimension: may be < params.dim for GQA models like Gemma-2
    let attn_dim = params.head_count * params.head_dim;
    
    var dot = 0.0;
    let weight_byte_offset = offsets.attn_out;

    if (((params.quant_type >> 24u) & 0xFFu) == 14u) { // Q6_K
        let bpr = attn_dim / 256u;
        let row_start = weight_byte_offset + idx * bpr * 210u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 210u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                dot += temp_state[temp_base + col] * dequant_q6k_elem(bb, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 13u) { // Q5_K
        let bpr = attn_dim / 256u;
        let row_start = weight_byte_offset + idx * bpr * 176u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 176u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                dot += temp_state[temp_base + col] * dequant_q5k_elem(bb, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 12u) { // Q4_K
        let blocks_per_row_k = attn_dim / 256u;
        let row_start_byte_k = weight_byte_offset + (idx * blocks_per_row_k * 144u);
        for (var b = 0u; b < blocks_per_row_k; b++) {
            let block_base_k = row_start_byte_k + (b * 144u);
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_ctx = temp_state[temp_base + col];
                dot += val_ctx * dequant_q4k_elem(block_base_k, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 6u) { // Q5_0
        let bpr = attn_dim / 32u;
        let row_start = weight_byte_offset + idx * bpr * 22u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 22u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                dot += temp_state[temp_base + col] * dequant_q5_0_elem(bb, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 8u) { // Q8_0
        let bpr = attn_dim / 32u;
        let row_start = weight_byte_offset + idx * bpr * 34u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 34u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                dot += temp_state[temp_base + col] * dequant_q8_0_elem(bb, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 1u) { // F16
        for (var col = 0u; col < attn_dim; col++) {
            let w_byte = weight_byte_offset + (idx * attn_dim + col) * 2u;
            dot += temp_state[temp_base + col] * dequant_f16_at(w_byte);
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 0u) { // F32
        for (var col = 0u; col < attn_dim; col++) {
            let w_idx = weight_byte_offset / 4u + idx * attn_dim + col;
            dot += temp_state[temp_base + col] * bitcast<f32>(read_blob(w_idx));
        }
    } else { // Q4_0
        let blocks_per_row = attn_dim / 32u;
        let row_start_byte = weight_byte_offset + (idx * blocks_per_row * 18u);
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_base = row_start_byte + (b * 18u);
            let scale_idx = block_base / 4u;
            let scale_packed = extractBits(read_blob(scale_idx), (block_base % 4u) * 8u, 16u);
            let w_scale = unpack2x16float(scale_packed).x;
            let qs_byte_start = block_base + 2u;
            for (var i = 0u; i < 32u; i++) {
                let col = b * 32u + i;
                let val_ctx = temp_state[temp_base + col];
                let byte_idx = i % 16u;
                let qs_idx = qs_byte_start + byte_idx;
                let qs_word = read_blob(qs_idx / 4u);
                let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
                let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
                let val_w = (f32(nib) - 8.0) * w_scale;
                dot += val_ctx * val_w;
            }
        }
    }
    
    // Stash raw attn_proj dot for post-attn norm correction (Gemma-2)
    // Uses Q-area + head_count*head_dim offset which is free after Kernel 2
    let attn_stash_base = params.dim + params.head_count * params.head_dim;
    temp_state[temp_base + attn_stash_base + idx] = dot;

    // Add residual connection
    let residual = activation_in[act_base + idx];
    activation_in[act_base + idx] = residual + dot;
}

// -------------------------------------------------------------------------
// Kernel 2.6: Post-Attention RMSNorm correction (Gemma-2 only)
// -------------------------------------------------------------------------
// When post_norm_enabled == 1: reads the stashed attn_proj dot, normalizes it
// with the post-attn norm weights (slot 2 in norm_bank), then corrects the
// residual: activation_in -= dot; activation_in += rms_normed_dot.
@compute @workgroup_size(256, 1, 1)
fn main_post_attn_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = global_id.y;

    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }
    if (params.post_norm_enabled == 0u) { return; }

    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    let attn_stash_base = params.dim + params.head_count * params.head_dim;

    // Compute RMS over the stashed attn_proj output
    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let v = temp_state[temp_base + attn_stash_base + i];
        sum_sq += v * v;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    // Apply post-attn norm weight (slot 2 per layer: layer_idx * 4 + 2)
    let norm_offset_base = (offsets.layer_idx * 4u + 2u) * params.dim;
    let norm_w = norm_bank[norm_offset_base + idx];
    let dot = temp_state[temp_base + attn_stash_base + idx];
    let normed_dot = dot * rms * norm_w;

    // Correct residual: activation_in was (residual + dot), should be (residual + normed_dot)
    activation_in[act_base + idx] += normed_dot - dot;
}

// -------------------------------------------------------------------------
// Kernel 4.5: Post-FFW RMSNorm correction (Gemma-2 only)
// -------------------------------------------------------------------------
// When post_norm_enabled == 1: reads the stashed ffn_down dot (already stored
// at temp_base + ffn_dim*2 + idx by main_ffn_down), normalizes it with post-ffw
// norm weights (slot 3 in norm_bank), then corrects the residual.
@compute @workgroup_size(256, 1, 1)
fn main_post_ffw_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = global_id.y;

    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }
    if (params.post_norm_enabled == 0u) { return; }

    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    // FFN down stash is at temp_base + ffn_dim*2 + idx (written by main_ffn_down)
    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let v = temp_state[temp_base + params.ffn_dim * 2u + i];
        sum_sq += v * v;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    // Apply post-ffw norm weight (slot 3 per layer: layer_idx * 4 + 3)
    let norm_offset_base = (offsets.layer_idx * 4u + 3u) * params.dim;
    let norm_w = norm_bank[norm_offset_base + idx];
    let dot = temp_state[temp_base + params.ffn_dim * 2u + idx];
    let normed_dot = dot * rms * norm_w;

    // Correct residual: activation_in was (residual + dot), should be (residual + normed_dot)
    activation_in[act_base + idx] += normed_dot - dot;
}

// Workgroup-shared partial sums for the FFN norm cooperative reduction.
// 256 slots × 4 bytes = 1 KB — well within the 16 KB workgroup memory limit.
var<workgroup> wg_rms_partial: array<f32, 256>;

// -------------------------------------------------------------------------
// Kernel 3: FFN Norm Broadcast — cooperative parallel reduction
// -------------------------------------------------------------------------
// One workgroup per token (dispatch X=1, Y=batch_size).  All 256 threads
// cooperate to compute the RMS scalar in a single O(dim) pass, then
// broadcast the normalized+weighted activations into the scratch slot.
//
// FSE invariant: activation_in is traversed ONCE per token, not once per
// output element (the old per-thread O(dim) loop was O(dim²) total).
@compute @workgroup_size(256, 1, 1)
fn main_ffn_norm(
    @builtin(local_invocation_id)  lid:       vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let lx        = lid.x;
    let token_idx = global_id.y;

    // token_idx is uniform across all 256 threads in this workgroup (same Y),
    // so this early-return is uniform and does not break workgroupBarrier.
    if (token_idx >= cache_params.batch_size) { return; }

    // For Phi (layer_norm_enabled) and non-gated FFN blocks, use the staged
    // pre-attention normalized stream from main_attn_norm.
    if (params.layer_norm_enabled != 0u || is_non_gated()) { return; }

    let act_base  = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    // ── 1. Each thread accumulates its strided partial sum-of-squares ──────
    var partial = 0.0;
    var partial_sum = 0.0;
    for (var i = lx; i < params.dim; i += 256u) {
        let v = activation_in[act_base + i];
        partial += v * v;
        partial_sum += v;
    }
    wg_rms_partial[lx] = partial;
    workgroupBarrier();

    // Reduce mean sum in-place when LayerNorm is enabled.
    if (params.layer_norm_enabled != 0u) {
        wg_rms_partial[lx] = partial_sum;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
            if (lx < stride) {
                wg_rms_partial[lx] += wg_rms_partial[lx + stride];
            }
            workgroupBarrier();
        }
    }
    let mean = select(0.0, wg_rms_partial[0] / f32(params.dim), params.layer_norm_enabled != 0u);

    // Restore sum-of-squares for RMS/variance path.
    wg_rms_partial[lx] = partial;
    workgroupBarrier();

    // ── 2. Parallel reduction: 256 → 1 in shared memory ────────────────────
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lx < stride) {
            wg_rms_partial[lx] += wg_rms_partial[lx + stride];
        }
        workgroupBarrier();
    }

    var inv_std = inverseSqrt(wg_rms_partial[0] / f32(params.dim) + params.rms_eps);
    if (params.layer_norm_enabled != 0u) {
        var var_partial = 0.0;
        for (var i = lx; i < params.dim; i += 256u) {
            let d = activation_in[act_base + i] - mean;
            var_partial += d * d;
        }
        wg_rms_partial[lx] = var_partial;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
            if (lx < stride) {
                wg_rms_partial[lx] += wg_rms_partial[lx + stride];
            }
            workgroupBarrier();
        }
        inv_std = inverseSqrt(wg_rms_partial[0] / f32(params.dim) + params.rms_eps);
    }

    // ── 3. Write normalized + norm-weight-scaled activations ────────────────
    let norm_offset_base = (offsets.layer_idx * 4u + 1u) * params.dim;
    for (var j = lx; j < params.dim; j += 256u) {
        let norm_w = norm_bank[norm_offset_base + j];
        let norm_b = select(0.0, bitcast<f32>(read_blob(offsets.ffn_norm_bias / 4u + j)), offsets.ffn_norm_bias != 0u);
        let centered = select(activation_in[act_base + j], activation_in[act_base + j] - mean, params.layer_norm_enabled != 0u);
        temp_state[temp_base + params.ffn_dim * 2u + j] =
            centered * inv_std * norm_w + norm_b;
    }
}

// -------------------------------------------------------------------------
// Kernel 4: FFN Proj (Gate/Up -> SiLu)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_ffn_proj(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ffn_dim = params.ffn_dim;
    let idx = global_id.x;
    let token_idx = global_id.y;

    if (idx >= ffn_dim * 2u || token_idx >= cache_params.batch_size) { return; }

    // Non-gated FFN (phi-2, GPT-2): up slot is unused; write 1.0 so gate*up = GELU(gate).
    let non_gated = is_non_gated();

    if (non_gated && idx >= ffn_dim) {
        temp_state[token_idx * params.temp_stride + idx] = 1.0;
        return;
    }

    let act_base  = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    var rms = 1.0;
    var norm_offset_base: u32 = 0u;
    if (!non_gated) {
        // Gated models: FFN RMSNorm over post-attention residual stream.
        var sum_sq = 0.0;
        for (var i = 0u; i < params.dim; i++) {
            let val = activation_in[act_base + i];
            sum_sq += val * val;
        }
        rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);
        norm_offset_base = (offsets.layer_idx * 4u + 1u) * params.dim;
    }

    // Select Matrix
    var weight_off: u32;
    var row_idx = idx;

    if (idx < ffn_dim) {
        // Non-gated (phi-2, GPT-2): use ffn_up for the single proj that feeds the gate slot.
        weight_off = select(offsets.ffn_gate, offsets.ffn_up, non_gated);
    } else {
        weight_off = offsets.ffn_up; // Up
        row_idx = idx - ffn_dim;
    }

    // MatMul
    var dot = 0.0;

    if (((params.quant_type >> 24u) & 0xFFu) == 14u) { // Q6_K
        let bpr = params.dim / 256u;
        let row_start = weight_off + (row_idx * bpr * 210u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 210u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q6k_elem(bb, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 13u) { // Q5_K
        let bpr = params.dim / 256u;
        let row_start = weight_off + (row_idx * bpr * 176u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 176u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q5k_elem(bb, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 12u) { // Q4_K
        let blocks_per_row_k = params.dim / 256u;
        let row_start_byte_k = weight_off + (row_idx * blocks_per_row_k * 144u);
        for (var b = 0u; b < blocks_per_row_k; b++) {
            let block_base_k = row_start_byte_k + (b * 144u);
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q4k_elem(block_base_k, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 6u) { // Q5_0
        let bpr = params.dim / 32u;
        let row_start = weight_off + (row_idx * bpr * 22u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 22u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q5_0_elem(bb, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 8u) { // Q8_0
        let bpr = params.dim / 32u;
        let row_start = weight_off + (row_idx * bpr * 34u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 34u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q8_0_elem(bb, e);
            }
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 1u) { // F16
        for (var col = 0u; col < params.dim; col++) {
            let w_byte = weight_off + (row_idx * params.dim + col) * 2u;
            let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
            dot += val_x * dequant_f16_at(w_byte);
        }
    } else if (((params.quant_type >> 24u) & 0xFFu) == 0u) { // F32
        for (var col = 0u; col < params.dim; col++) {
            let w_idx = weight_off / 4u + row_idx * params.dim + col;
            let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
            dot += val_x * bitcast<f32>(read_blob(w_idx));
        }
    } else { // Q4_0
        let blocks_per_row = params.dim / 32u;
        let row_start_byte = weight_off + (row_idx * blocks_per_row * 18u);
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_base = row_start_byte + (b * 18u);
            let scale_idx = block_base / 4u;
            let scale_packed = extractBits(read_blob(scale_idx), (block_base % 4u) * 8u, 16u);
            let scale = unpack2x16float(scale_packed).x;
            let qs_byte_start = block_base + 2u;
            for (var i = 0u; i < 32u; i++) {
                let col = b * 32u + i;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                let byte_idx = i % 16u;
                let qs_idx = qs_byte_start + byte_idx;
                let qs_word = read_blob(qs_idx / 4u);
                let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
                let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
                let val_w = (f32(nib) - 8.0) * scale;
                dot += val_x * val_w;
            }
        }
    }
    
    // If Gate, apply activation (GELU for Gemma-2, SiLU for LLaMA-style).
    // Gemma-2 uses GeGLU; detected by attn_logit_softcap > 0 (only Gemma-2 uses it).
    if (idx < ffn_dim) {
        var activated: f32;
        // Non-gated uses GELU; Gemma-2 (GeGLU) also uses GELU; gated uses SiLU.
        if (non_gated || params.attn_logit_softcap > 0.0) {
            // GELU approximate (PyTorch tanh variant): 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))
            activated = 0.5 * dot * (1.0 + tanh(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
        } else {
            // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            activated = dot / (1.0 + exp(-dot));
        }
        temp_state[temp_base + idx] = activated;
    } else {
        temp_state[temp_base + idx] = dot;
    }
}

// -------------------------------------------------------------------------
// Kernel 4: FFN Down (Multiply Gate*Up -> Down -> Residual)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_ffn_down(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x; // 0..2047 (Dim)
    let token_idx = global_id.y;
    
    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }
    
    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    
    let ffn_dim = params.ffn_dim;
    var dot = 0.0;

    let weight_off = offsets.ffn_down;

    let qt_down = (params.quant_type >> 16u) & 0xFFu;
    if (qt_down == 14u) { // Q6_K
        let bpr = ffn_dim / 256u;
        let row_start = weight_off + (idx * bpr * 210u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 210u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_gate = temp_state[temp_base + col];
                let val_up   = temp_state[temp_base + ffn_dim + col];
                dot += (val_gate * val_up) * dequant_q6k_elem(bb, e);
            }
        }
    } else if (qt_down == 13u) { // Q5_K
        let bpr = ffn_dim / 256u;
        let row_start = weight_off + (idx * bpr * 176u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 176u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_gate = temp_state[temp_base + col];
                let val_up   = temp_state[temp_base + ffn_dim + col];
                dot += (val_gate * val_up) * dequant_q5k_elem(bb, e);
            }
        }
    } else if (qt_down == 12u) { // Q4_K
        let blocks_per_row_k = ffn_dim / 256u;
        let row_start_byte_k = weight_off + (idx * blocks_per_row_k * 144u);
        for (var b = 0u; b < blocks_per_row_k; b++) {
            let block_base_k = row_start_byte_k + (b * 144u);
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_gate = temp_state[temp_base + col];
                let val_up   = temp_state[temp_base + ffn_dim + col];
                dot += (val_gate * val_up) * dequant_q4k_elem(block_base_k, e);
            }
        }
    } else if (qt_down == 6u) { // Q5_0
        let bpr = ffn_dim / 32u;
        let row_start = weight_off + idx * bpr * 22u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 22u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                let val_gate = temp_state[temp_base + col];
                let val_up   = temp_state[temp_base + ffn_dim + col];
                dot += (val_gate * val_up) * dequant_q5_0_elem(bb, e);
            }
        }
    } else if (qt_down == 8u) { // Q8_0
        let bpr = ffn_dim / 32u;
        let row_start = weight_off + idx * bpr * 34u;
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 34u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                let val_gate = temp_state[temp_base + col];
                let val_up   = temp_state[temp_base + ffn_dim + col];
                dot += (val_gate * val_up) * dequant_q8_0_elem(bb, e);
            }
        }
    } else if (qt_down == 1u) { // F16
        for (var col = 0u; col < ffn_dim; col++) {
            let w_byte = weight_off + (idx * ffn_dim + col) * 2u;
            let val_gate = temp_state[temp_base + col];
            let val_up   = temp_state[temp_base + ffn_dim + col];
            dot += (val_gate * val_up) * dequant_f16_at(w_byte);
        }
    } else if (qt_down == 0u) { // F32
        for (var col = 0u; col < ffn_dim; col++) {
            let w_idx = weight_off / 4u + idx * ffn_dim + col;
            let val_gate = temp_state[temp_base + col];
            let val_up   = temp_state[temp_base + ffn_dim + col];
            dot += (val_gate * val_up) * bitcast<f32>(read_blob(w_idx));
        }
    } else { // Q4_0
        let blocks_per_row = ffn_dim / 32u;
        let row_start_byte = weight_off + (idx * blocks_per_row * 18u);
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_base = row_start_byte + (b * 18u);
            let scale_idx = block_base / 4u;
            let scale_packed = extractBits(read_blob(scale_idx), (block_base % 4u) * 8u, 16u);
            let scale = unpack2x16float(scale_packed).x;
            let qs_byte_start = block_base + 2u;
            for (var i = 0u; i < 32u; i++) {
                let col = b * 32u + i;
                let val_gate = temp_state[temp_base + col];
                let val_up   = temp_state[temp_base + ffn_dim + col];
                let val_x    = val_gate * val_up;
                let byte_idx = i % 16u;
                let qs_idx = qs_byte_start + byte_idx;
                let qs_word = read_blob(qs_idx / 4u);
                let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
                let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
                let val_w = (f32(nib) - 8.0) * scale;
                dot += val_x * val_w;
            }
        }
    }
    
    // Debug stash: store pure FFN down output before residual
    // temp_state layout uses 0..ffn_dim*2-1; ffn_dim*2..temp_stride is free for dim floats
    temp_state[temp_base + params.ffn_dim * 2u + idx] = dot;

    // Residual Add
    let residual = activation_in[act_base + idx];
    activation_in[act_base + idx] = residual + dot;
}
