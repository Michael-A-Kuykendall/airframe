// sh_layer_q4k.wgsl
// Full Transformer Layer — Mixed Q4_K/Q6_K weights (as used in Q4_K_M models).
// Most weights are Q4_K (144-byte superblocks); attn_v and ffn_down are Q6_K (210-byte superblocks).
// Identical bindings and entry points as sh_layer_v1.wgsl (Q4_0).

struct LayerOffsets {
    attn_norm: u32,
    attn_norm_bias: u32,     // byte offset of attn norm bias (F32); 0 = disabled
    attn_q: u32,
    attn_k: u32,
    attn_v: u32,
    attn_out: u32,
    ffn_norm: u32,
    ffn_norm_bias: u32,      // byte offset of ffn norm bias (F32); 0 = disabled
    ffn_gate: u32,
    ffn_down: u32,
    ffn_up: u32,
    layer_idx: u32,          // layer index (used by shader)
    attn_q_norm: u32,        // byte offset of Q-norm weights (Qwen3; 0 = disabled)
    attn_k_norm: u32,        // byte offset of K-norm weights (Qwen3; 0 = disabled)
    attn_q_bias: u32,        // byte offset of Q bias F32 (Qwen2; 0 = disabled)
    attn_k_bias: u32,        // byte offset of K bias F32 (Qwen2; 0 = disabled)
    attn_v_bias: u32,        // byte offset of V bias F32 (Qwen2; 0 = disabled)
    v_is_q4k: u32,           // 1 if attn_v uses Q4_K (144-byte blocks), 0 for Q6_K (210-byte)
    ffn_down_is_q4k: u32,    // 1 if ffn_down uses Q4_K (144-byte blocks), 0 for Q6_K (210-byte)
};

struct LayerParams {
    dim: u32,
    head_count: u32,
    head_count_kv: u32,
    head_dim: u32,
    rope_dim: u32,             // rotary sub-dimension (must match V1 struct offset)
    rms_eps: f32,
    ffn_dim: u32,
    temp_stride: u32,
    quant_type: u32,           // GGML quant type (packed: bits[7:0]=main, [15:8]=v, [23:16]=ffn_down)
    attn_logit_softcap: f32,   // Attention logit soft-cap (Gemma-2: 50.0, 0.0 = disabled)
    post_norm_enabled: u32,    // 1 = apply post-attn/post-ffw norms (Gemma-2)
    qk_norm_enabled: u32,      // 1 = per-head Q/K RMSNorm (Qwen3)
    layer_norm_enabled: u32,   // 1 = LayerNorm (Phi-family)
    ffn_kind_policy: u32,      // 0 = infer, 1 = gated, 2 = non-gated
    qkv_layout_policy: u32,    // 0 = infer, 1 = separate, 2 = fused
    batch_offset: u32,         // first token index in this QKV micro-batch chunk
    batch_count: u32,          // number of tokens in this chunk
    q_weight_k: u32,           // stored K (in dim) for attn_q.weight (packed Qwen3 etc; 0=dim)
    k_weight_k: u32,           // for attn_k.weight (packed)
};

struct CacheParams {
    current_pos: u32,
    seq_len: u32,
    max_seq_len: u32,
    batch_size: u32,
};

@group(0) @binding(0) var<storage, read> gguf_blob: array<u32>;
@group(0) @binding(1) var<storage, read_write> activation_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_state: array<f32>;
@group(0) @binding(3) var<uniform> offsets: LayerOffsets;
@group(0) @binding(4) var<uniform> params: LayerParams;
@group(0) @binding(5) var<storage, read> norm_bank: array<f32>;
@group(0) @binding(6) var<storage, read> rope_table: array<f32>;
@group(0) @binding(7) var<storage, read_write> kv_cache_k: array<f32>;
@group(0) @binding(8) var<storage, read_write> kv_cache_v: array<f32>;
@group(0) @binding(9) var<uniform> cache_params: CacheParams;
@group(0) @binding(10) var<storage, read> blob_1: array<u32>;
@group(0) @binding(11) var<storage, read> blob_2: array<u32>;

// Multi-chunk blob split constants (matches sh_layer_v1.wgsl and sh_rmsnorm.wgsl).
// BLOB_CHUNK_BYTES = 2_000_000_000; words = bytes / 4.
const BLOB_SPLIT_0: u32 = 500000000u;  // 2,000,000,000 bytes / 4 = 500M words
const BLOB_SPLIT_1: u32 = 1000000000u; // 4,000,000,000 bytes / 4 = 1B words

// Read one u32 word from the correct chunk of the split blob.
fn read_blob_word(word_idx: u32) -> u32 {
    if word_idx < BLOB_SPLIT_0 {
        return gguf_blob[word_idx];
    } else if word_idx < BLOB_SPLIT_1 {
        return blob_1[word_idx - BLOB_SPLIT_0];
    } else {
        return blob_2[word_idx - BLOB_SPLIT_1];
    }
}

// -------------------------------------------------------------------------
// Q4_K helper functions
// -------------------------------------------------------------------------
// Read one byte from the gguf_blob u32 array (multi-chunk aware).
fn get_byte(byte_pos: u32) -> u32 {
    let word = read_blob_word(byte_pos / 4u);
    return (word >> ((byte_pos % 4u) * 8u)) & 0xFFu;
}

// Read one f16 (2 bytes, little-endian) from gguf_blob.
// byte_pos must be within a single u32 word (i.e. byte_pos % 4 <= 2).
// This holds for Q4_K block starts (144-byte blocks are 4-byte aligned).
fn get_f16_at(byte_pos: u32) -> f32 {
    let word = read_blob_word(byte_pos / 4u);
    let bits16 = (word >> ((byte_pos % 4u) * 8u)) & 0xFFFFu;
    return unpack2x16float(bits16).x;
}

// Read one F32 (4 bytes) from gguf_blob at a 4-byte aligned byte offset.
// Used for F32 norm weights stored in the GGUF blob.
fn get_f32_at(byte_pos: u32) -> f32 {
    return bitcast<f32>(read_blob_word(byte_pos / 4u));
}

// Extract the 6-bit scale for sub-block j (0..7) from the 12-byte scale array.
// Mirrors llama.cpp get_scale_min_k4 exactly.
fn q4k_sc(j: u32, sb: u32) -> f32 {
    var raw: u32;
    if (j < 4u) {
        raw = get_byte(sb + j) & 0x3Fu;
    } else {
        let sA = get_byte(sb + j + 4u);
        let sB = get_byte(sb + j - 4u);
        raw = (sA & 0x0Fu) | (((sB >> 6u) & 0x03u) << 4u);
    }
    return f32(raw);
}

// Extract the 6-bit min for sub-block j (0..7) from the 12-byte scale array.
fn q4k_mn(j: u32, sb: u32) -> f32 {
    var raw: u32;
    if (j < 4u) {
        raw = get_byte(sb + j + 4u) & 0x3Fu;
    } else {
        let sA = get_byte(sb + j + 4u);
        let sC = get_byte(sb + j);
        raw = ((sA >> 4u) & 0x0Fu) | (((sC >> 6u) & 0x03u) << 4u);
    }
    return f32(raw);
}

// -------------------------------------------------------------------------
// Q6_K helper functions
// -------------------------------------------------------------------------
// Safe FP16 → F32 from two bytes (no dependency on alignment).
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

// Reinterpret a u8 value (0–255) as int8 (-128–127).
fn u8_to_i8(v: u32) -> i32 {
    return i32(v << 24u) >> 24;
}

// -------------------------------------------------------------------------
// Kernel 0: Attention RMSNorm Provider (stub - uses V1 shader at runtime)
// -------------------------------------------------------------------------
// This is a placeholder since Q4_K uses V1's attn_norm at runtime.
// Required for pipeline compilation to succeed.
@compute @workgroup_size(256, 1, 1)
fn main_attn_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // No-op: actual attn_norm is handled by V1 pipeline
    // This function exists only to satisfy pipeline compilation
}

// -------------------------------------------------------------------------
// Kernel 1: QKV Generation + Cache Update  (Q4_K for Q/K, Q6_K for V)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_qkv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = global_id.y;

    if (token_idx >= cache_params.batch_size) { return; }
    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    let dim_q = params.head_count * params.head_dim;
    let dim_k = params.head_count_kv * params.head_dim;
    let dim_v = params.head_count_kv * params.head_dim;
    let total_out = dim_q + dim_k + dim_v;

    if (idx >= total_out) { return; }

    // RMS Norm
    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let v = activation_in[act_base + i];
        sum_sq += v * v;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    // Select weight matrix and row
    var weight_off_roots: array<u32, 3>;
    weight_off_roots[0] = offsets.attn_q;
    weight_off_roots[1] = offsets.attn_k;
    weight_off_roots[2] = offsets.attn_v;

    var target_stage = 0u;
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
    let layer_idx = offsets.layer_idx;
    let norm_base = layer_idx * 4u * params.dim; // AttnNorm is slot 0 in 4-slot per-layer layout (attn, ffn, post_attn, post_ffn)

    // Dispatch Q4_K matmul for attn_q and attn_k; Q6_K for attn_v
    var dot = 0.0;
    let blocks_per_row = params.dim / 256u;

    if (target_stage < 2u) {
        // ---- Q4_K (attn_q, attn_k) ----
        let row_start_byte = weight_byte_offset + row_idx * blocks_per_row * 144u;

        for (var b = 0u; b < blocks_per_row; b++) {
            let bb = row_start_byte + b * 144u;
            let d_val   = get_f16_at(bb);
            let dm_val  = get_f16_at(bb + 2u);
            let sb      = bb + 4u;    // scales base (12 bytes)
            let qs_base = bb + 16u;   // quantized nibbles base (128 bytes);

            for (var group = 0u; group < 4u; group++) {
                let is0 = group * 2u;
                let is1 = group * 2u + 1u;
                let d1 = d_val  * q4k_sc(is0, sb);
                let m1 = dm_val * q4k_mn(is0, sb);
                let d2 = d_val  * q4k_sc(is1, sb);
                let m2 = dm_val * q4k_mn(is1, sb);
                let qs_grp = qs_base + group * 32u;

                for (var qi = 0u; qi < 32u; qi++) {
                    let qb = get_byte(qs_grp + qi);
                    let col_base = b * 256u + group * 64u;

                    let col_lo = col_base + qi;
                    let nw_lo  = norm_bank[norm_base + col_lo];
                    let vx_lo  = activation_in[act_base + col_lo] * rms * nw_lo;
                    dot += vx_lo * (d1 * f32(qb & 0x0Fu) - m1);

                    let col_hi = col_base + 32u + qi;
                    let nw_hi  = norm_bank[norm_base + col_hi];
                    let vx_hi  = activation_in[act_base + col_hi] * rms * nw_hi;
                    dot += vx_hi * (d2 * f32((qb >> 4u) & 0x0Fu) - m2);
                }
            }
        }
    } else if (offsets.v_is_q4k == 0u) {
        // ---- Q6_K (attn_v): 210-byte superblocks ----
        // Layout: ql[128] | qh[64] | scales[16] | d[2]
        let blocks_per_row = params.dim / 256u;
        let row_start_byte = weight_byte_offset + row_idx * blocks_per_row * 210u;

        for (var b = 0u; b < blocks_per_row; b++) {
            let bb = row_start_byte + b * 210u;
            let d_val = f16_from_bytes(get_byte(bb + 208u), get_byte(bb + 209u));

            // Two 128-element halves
            for (var half = 0u; half < 2u; half++) {
                let ql_base = bb + half * 64u;          // 64 bytes of low nibbles
                let qh_base = bb + 128u + half * 32u;   // 32 bytes of high 2-bit pairs
                let sc_base = bb + 192u + half * 8u;    // 8 bytes of int8 scales
                let y_off   = b * 256u + half * 128u;   // output element base

                for (var l = 0u; l < 32u; l++) {
                    let ql_lo  = get_byte(ql_base + l);
                    let ql_hi  = get_byte(ql_base + l + 32u);
                    let qh_val = get_byte(qh_base + l);

                    // Reconstruct 6-bit quantized values (offset -32)
                    let q1 = i32((ql_lo  & 0x0Fu) | ((qh_val         & 0x03u) << 4u)) - 32;
                    let q2 = i32((ql_hi  & 0x0Fu) | (((qh_val >> 2u) & 0x03u) << 4u)) - 32;
                    let q3 = i32((ql_lo  >> 4u  ) | (((qh_val >> 4u) & 0x03u) << 4u)) - 32;
                    let q4 = i32((ql_hi  >> 4u  ) | (((qh_val >> 6u) & 0x03u) << 4u)) - 32;

                    let is = l / 16u;  // scale index within this half (0 or 1)
                    let sc1 = u8_to_i8(get_byte(sc_base + is));
                    let sc2 = u8_to_i8(get_byte(sc_base + is + 2u));
                    let sc3 = u8_to_i8(get_byte(sc_base + is + 4u));
                    let sc4 = u8_to_i8(get_byte(sc_base + is + 6u));

                    let k1 = y_off + l;
                    let k2 = y_off + l + 32u;
                    let k3 = y_off + l + 64u;
                    let k4 = y_off + l + 96u;

                    let nw1 = norm_bank[norm_base + k1];
                    let nw2 = norm_bank[norm_base + k2];
                    let nw3 = norm_bank[norm_base + k3];
                    let nw4 = norm_bank[norm_base + k4];

                    let vx1 = activation_in[act_base + k1] * rms * nw1;
                    let vx2 = activation_in[act_base + k2] * rms * nw2;
                    let vx3 = activation_in[act_base + k3] * rms * nw3;
                    let vx4 = activation_in[act_base + k4] * rms * nw4;

                    dot += d_val * f32(sc1) * f32(q1) * vx1;
                    dot += d_val * f32(sc2) * f32(q2) * vx2;
                    dot += d_val * f32(sc3) * f32(q3) * vx3;
                    dot += d_val * f32(sc4) * f32(q4) * vx4;
                }
            }
        }
    } else {
        // ---- Q4_K (attn_v): 144-byte superblocks ----
        let row_start_byte = weight_byte_offset + row_idx * blocks_per_row * 144u;

        for (var b = 0u; b < blocks_per_row; b++) {
            let bb = row_start_byte + b * 144u;
            let d_val   = get_f16_at(bb);
            let dm_val  = get_f16_at(bb + 2u);
            let sb      = bb + 4u;    // scales base (12 bytes)
            let qs_base = bb + 16u;   // quantized nibbles base (128 bytes)

            for (var group = 0u; group < 4u; group++) {
                let is0 = group * 2u;
                let is1 = group * 2u + 1u;
                let d1 = d_val  * q4k_sc(is0, sb);
                let m1 = dm_val * q4k_mn(is0, sb);
                let d2 = d_val  * q4k_sc(is1, sb);
                let m2 = dm_val * q4k_mn(is1, sb);
                let qs_grp = qs_base + group * 32u;

                for (var qi = 0u; qi < 32u; qi++) {
                    let qb = get_byte(qs_grp + qi);
                    let col_base = b * 256u + group * 64u;

                    let col_lo = col_base + qi;
                    let nw_lo  = norm_bank[norm_base + col_lo];
                    let vx_lo  = activation_in[act_base + col_lo] * rms * nw_lo;
                    dot += vx_lo * (d1 * f32(qb & 0x0Fu) - m1);

                    let col_hi = col_base + 32u + qi;
                    let nw_hi  = norm_bank[norm_base + col_hi];
                    let vx_hi  = activation_in[act_base + col_hi] * rms * nw_hi;
                    dot += vx_hi * (d2 * f32((qb >> 4u) & 0x0Fu) - m2);
                }
            }
        }
    }

    // Store output
    if (target_stage == 0u) {
        // Write Q to V1-expected position so V1 qk_norm can see it for Qwen3 etc.
        temp_state[temp_base + params.dim + row_idx] = dot;
    } else if (target_stage == 1u) {
        let head        = row_idx / params.head_dim;
        let dim_in_head = row_idx % params.head_dim;
        // batch_offset positions this chunk correctly within the prefill sequence.
        // Without it, every chunk writes to current_pos+0..N overwriting previous chunks.
        let cache_idx   = ((cache_params.current_pos + params.batch_offset + token_idx) * params.head_count_kv * params.head_dim)
                        + (head * params.head_dim) + dim_in_head;
        kv_cache_k[cache_idx] = dot;
    } else {
        let head        = row_idx / params.head_dim;
        let dim_in_head = row_idx % params.head_dim;
        let cache_idx   = ((cache_params.current_pos + params.batch_offset + token_idx) * params.head_count_kv * params.head_dim)
                        + (head * params.head_dim) + dim_in_head;
        kv_cache_v[cache_idx] = dot;
    }
}

// -------------------------------------------------------------------------
// Kernel 1.5: QK Norm (Q4K path uses V1 impl via pipe selection for now)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_qk_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Q4K models dispatch qk_norm via the V1 pipeline entry (real impl) for hybrid
    // compatibility with Q write positions established for Qwen3 etc.
}

// -------------------------------------------------------------------------
// Kernel 2: Attention (unchanged — no weight access, pure KV-cache arithmetic)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_attn_out(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let token_idx = global_id.y;

    if (token_idx >= cache_params.batch_size) { return; }
    let temp_base = token_idx * params.temp_stride;

    if (idx >= params.dim) { return; }

    let gqa_ratio   = params.head_count / params.head_count_kv;
    let scale       = 1.0 / sqrt(f32(params.head_dim));
    let head_idx    = idx / params.head_dim;
    if (head_idx >= params.head_count) { return; }  // guard: skip threads beyond valid heads
    let head_offset = idx % params.head_dim;
    let kv_head_idx = head_idx / gqa_ratio;
    // Read Q from V1-expected position (after activation dim) so qk_norm from V1 applies.
    let q_base      = temp_base + params.dim + head_idx * params.head_dim;
    let causal_pos  = cache_params.current_pos + token_idx;
    let n_pairs     = params.head_dim / 2u;

    const SINK_COUNT: u32 = 4u;
    var running_max = -1e30;
    var running_sum = 0.0;
    var context_val = 0.0;

    for (var pos = 0u; pos <= causal_pos; pos++) {
        if (pos >= cache_params.seq_len) { break; }

        var rel_pos = causal_pos - pos;
        if (pos < SINK_COUNT) {
            rel_pos = min(rel_pos, 2047u);
        } else if (rel_pos > 2047u) {
            continue;
        }

        var dot_qk = 0.0;
        let k_idx_base = (pos * params.head_count_kv * params.head_dim)
                       + (kv_head_idx * params.head_dim);

        for (var p = 0u; p < n_pairs; p++) {
            let table_idx = rel_pos * n_pairs * 2u + p * 2u;
            let cos_a = rope_table[table_idx];
            let sin_a = rope_table[table_idx + 1u];
            let d      = p * 2u;
            let q_re   = temp_state[q_base + d];
            let q_im   = temp_state[q_base + d + 1u];
            let k_re   = kv_cache_k[k_idx_base + d];
            let k_im   = kv_cache_k[k_idx_base + d + 1u];
            dot_qk += (q_re * k_re + q_im * k_im) * cos_a
                    + (q_re * k_im - q_im * k_re) * sin_a;
        }

        var score = dot_qk * scale;
        if (params.attn_logit_softcap > 0.0) {
            score = params.attn_logit_softcap * tanh(score / params.attn_logit_softcap);
        }
        let v_idx = (pos * params.head_count_kv * params.head_dim)
                  + (kv_head_idx * params.head_dim) + head_offset;
        let v_val = kv_cache_v[v_idx];

        if (score > running_max) {
            let correction = exp(running_max - score);
            running_sum    = running_sum * correction + 1.0;
            context_val    = context_val * correction + v_val;
            running_max    = score;
        } else {
            let exp_score = exp(score - running_max);
            running_sum  += exp_score;
            context_val  += exp_score * v_val;
        }
    }

    if (running_sum > 0.0) { context_val /= running_sum; }
    temp_state[temp_base + idx] = context_val;
    if (global_id.x == 0u && global_id.y == 0u) {
        let s0 = temp_state[temp_base + 0];
        let s1 = temp_state[temp_base + 1];
        let s2 = temp_state[temp_base + 2];
        let s3 = temp_state[temp_base + 3];
        /* DIAG attn_out via temp_state */
    }
}

// -------------------------------------------------------------------------
// Kernel 2.5: Attention Output Projection  (Q4_K weights)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_attn_proj(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx       = global_id.x;
    let token_idx = global_id.y;

    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }

    let act_base  = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    var dot = 0.0;
    let weight_byte_offset = offsets.attn_out;
    // attn_out weight shape: [n_embd, n_head * head_dim] — input cols = n_head * head_dim, not dim
    let blocks_per_row = (params.head_count * params.head_dim) / 256u;
    let row_start_byte = weight_byte_offset + idx * blocks_per_row * 144u;

    for (var b = 0u; b < blocks_per_row; b++) {
        let bb = row_start_byte + b * 144u;
        let d_val   = get_f16_at(bb);
        let dm_val  = get_f16_at(bb + 2u);
        let sb      = bb + 4u;
        let qs_base = bb + 16u;

        for (var group = 0u; group < 4u; group++) {
            let is0 = group * 2u;
            let is1 = group * 2u + 1u;
            let d1 = d_val  * q4k_sc(is0, sb);
            let m1 = dm_val * q4k_mn(is0, sb);
            let d2 = d_val  * q4k_sc(is1, sb);
            let m2 = dm_val * q4k_mn(is1, sb);
            let qs_grp = qs_base + group * 32u;

            for (var qi = 0u; qi < 32u; qi++) {
                let qb       = get_byte(qs_grp + qi);
                let col_base = b * 256u + group * 64u;

                let col_lo  = col_base + qi;
                let ctx_lo  = temp_state[temp_base + col_lo];
                dot += ctx_lo * (d1 * f32(qb & 0x0Fu) - m1);

                let col_hi  = col_base + 32u + qi;
                let ctx_hi  = temp_state[temp_base + col_hi];
                dot += ctx_hi * (d2 * f32((qb >> 4u) & 0x0Fu) - m2);
            }
        }
    }

    // Add attention output to residual stream.
    // Gemma-2: store to scratch area — main_post_attn_norm will apply post-norm then add.
    // All other models: add directly to residual (no post-norm).
    if (params.post_norm_enabled != 0u) {
        temp_state[temp_base + params.ffn_dim * 2u + idx] = dot;
    } else {
        activation_in[act_base + idx] += dot;
    }
}

// -------------------------------------------------------------------------
// Kernel 2.75: Post-Attention Norm + Residual Add  (Gemma-2)
// Applies RMS norm to attention output stored in scratch, adds to residual.
// Matches V1 shader implementation — uses norm_bank slot 2 and post_norm_enabled flag.
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_post_attn_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx       = global_id.x;
    let token_idx = global_id.y;

    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }
    if (params.post_norm_enabled == 0u) { return; }

    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    let attn_stash_base = params.dim + params.head_count * params.head_dim;

    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let v = temp_state[temp_base + attn_stash_base + i];
        sum_sq += v * v;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    let norm_offset_base = (offsets.layer_idx * 4u + 2u) * params.dim;
    let norm_w = norm_bank[norm_offset_base + idx];
    let dot = temp_state[temp_base + attn_stash_base + idx];
    let normed_dot = dot * rms * norm_w;
    activation_in[act_base + idx] += normed_dot - dot;
}

// -------------------------------------------------------------------------
// Kernel 2.5: FFN Norm (stub - uses V1 shader at runtime)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_ffn_norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // No-op: actual ffn_norm is handled by V1 pipeline
}

// -------------------------------------------------------------------------
// Kernel 3: FFN Gate + Up Projection + SiLU  (Q4_K weights)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_ffn_proj(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ffn_dim   = params.ffn_dim;
    let idx       = global_id.x;
    let token_idx = global_id.y;

    if (idx >= ffn_dim * 2u || token_idx >= cache_params.batch_size) { return; }

    let act_base  = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    // RMS Norm (FFN)
    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let v = activation_in[act_base + i];
        sum_sq += v * v;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    var weight_off: u32;
    var row_idx = idx;
    if (idx < ffn_dim) {
        weight_off = offsets.ffn_gate;
    } else {
        weight_off = offsets.ffn_up;
        row_idx    = idx - ffn_dim;
    }

    var dot = 0.0;
    let blocks_per_row   = params.dim / 256u;
    let row_start_byte   = weight_off + row_idx * blocks_per_row * 144u;
    let norm_offset_base = (offsets.layer_idx * 4u + 1u) * params.dim;  // Fixed for Q4K norm_bank layout (4 slots/layer from preflight)

    for (var b = 0u; b < blocks_per_row; b++) {
        let bb = row_start_byte + b * 144u;
        let d_val   = get_f16_at(bb);
        let dm_val  = get_f16_at(bb + 2u);
        let sb      = bb + 4u;
        let qs_base = bb + 16u;

        for (var group = 0u; group < 4u; group++) {
            let is0 = group * 2u;
            let is1 = group * 2u + 1u;
            let d1 = d_val  * q4k_sc(is0, sb);
            let m1 = dm_val * q4k_mn(is0, sb);
            let d2 = d_val  * q4k_sc(is1, sb);
            let m2 = dm_val * q4k_mn(is1, sb);
            let qs_grp = qs_base + group * 32u;

            for (var qi = 0u; qi < 32u; qi++) {
                let qb       = get_byte(qs_grp + qi);
                let col_base = b * 256u + group * 64u;

                let col_lo = col_base + qi;
                let nw_lo  = norm_bank[norm_offset_base + col_lo];
                let vx_lo  = activation_in[act_base + col_lo] * rms * nw_lo;
                dot += vx_lo * (d1 * f32(qb & 0x0Fu) - m1);

                let col_hi = col_base + 32u + qi;
                let nw_hi  = norm_bank[norm_offset_base + col_hi];
                let vx_hi  = activation_in[act_base + col_hi] * rms * nw_hi;
                dot += vx_hi * (d2 * f32((qb >> 4u) & 0x0Fu) - m2);
            }
        }
    }

    if (idx < ffn_dim) {
        // Gemma-2 uses GeGLU (GELU activation); LLaMA-style models use SwiGLU (SiLU).
        // We detect Gemma-2 by attn_softcap > 0 (only Gemma-2 uses attention logit softcapping).
        var activated: f32;
        if (params.attn_logit_softcap > 0.0) {
            // GELU approximate (PyTorch tanh variant): 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))
            let gelu = 0.5 * dot * (1.0 + tanh(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
            activated = gelu;
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
// Kernel 4: FFN Down Projection + Residual  (Q6_K or Q4_K weights for ffn_down)
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_ffn_down(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx       = global_id.x;
    let token_idx = global_id.y;

    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }

    let act_base  = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;
    let ffn_dim   = params.ffn_dim;

    var dot = 0.0;
    let weight_off     = offsets.ffn_down;
    let blocks_per_row = ffn_dim / 256u;

    if (offsets.ffn_down_is_q4k == 0u) {
        // ---- Q6_K (ffn_down): 210-byte superblocks ----
        let row_start_byte = weight_off + idx * blocks_per_row * 210u;

        for (var b = 0u; b < blocks_per_row; b++) {
            let bb = row_start_byte + b * 210u;
            let d_val = f16_from_bytes(get_byte(bb + 208u), get_byte(bb + 209u));

            for (var half = 0u; half < 2u; half++) {
                let ql_base = bb + half * 64u;
                let qh_base = bb + 128u + half * 32u;
                let sc_base = bb + 192u + half * 8u;
                let y_off   = b * 256u + half * 128u;

                for (var l = 0u; l < 32u; l++) {
                    let ql_lo  = get_byte(ql_base + l);
                    let ql_hi  = get_byte(ql_base + l + 32u);
                    let qh_val = get_byte(qh_base + l);

                    let q1 = i32((ql_lo  & 0x0Fu) | ((qh_val         & 0x03u) << 4u)) - 32;
                    let q2 = i32((ql_hi  & 0x0Fu) | (((qh_val >> 2u) & 0x03u) << 4u)) - 32;
                    let q3 = i32((ql_lo  >> 4u  ) | (((qh_val >> 4u) & 0x03u) << 4u)) - 32;
                    let q4 = i32((ql_hi  >> 4u  ) | (((qh_val >> 6u) & 0x03u) << 4u)) - 32;

                    let is = l / 16u;
                    let sc1 = u8_to_i8(get_byte(sc_base + is));
                    let sc2 = u8_to_i8(get_byte(sc_base + is + 2u));
                    let sc3 = u8_to_i8(get_byte(sc_base + is + 4u));
                    let sc4 = u8_to_i8(get_byte(sc_base + is + 6u));

                    let k1 = y_off + l;
                    let k2 = y_off + l + 32u;
                    let k3 = y_off + l + 64u;
                    let k4 = y_off + l + 96u;

                    let gate1 = temp_state[temp_base + k1];
                    let up1   = temp_state[temp_base + ffn_dim + k1];
                    let gate2 = temp_state[temp_base + k2];
                    let up2   = temp_state[temp_base + ffn_dim + k2];
                    let gate3 = temp_state[temp_base + k3];
                    let up3   = temp_state[temp_base + ffn_dim + k3];
                    let gate4 = temp_state[temp_base + k4];
                    let up4   = temp_state[temp_base + ffn_dim + k4];

                    dot += d_val * f32(sc1) * f32(q1) * (gate1 * up1);
                    dot += d_val * f32(sc2) * f32(q2) * (gate2 * up2);
                    dot += d_val * f32(sc3) * f32(q3) * (gate3 * up3);
                    dot += d_val * f32(sc4) * f32(q4) * (gate4 * up4);
                }
            }
        }
    } else {
        // ---- Q4_K (ffn_down): 144-byte superblocks ----
        let row_start_byte = weight_off + idx * blocks_per_row * 144u;

        for (var b = 0u; b < blocks_per_row; b++) {
            let bb = row_start_byte + b * 144u;
            let d_val   = get_f16_at(bb);
            let dm_val  = get_f16_at(bb + 2u);
            let sb      = bb + 4u;    // scales base (12 bytes)
            let qs_base = bb + 16u;   // quantized nibbles base (128 bytes)

            for (var group = 0u; group < 4u; group++) {
                let is0 = group * 2u;
                let is1 = group * 2u + 1u;
                let d1 = d_val  * q4k_sc(is0, sb);
                let m1 = dm_val * q4k_mn(is0, sb);
                let d2 = d_val  * q4k_sc(is1, sb);
                let m2 = dm_val * q4k_mn(is1, sb);
                let qs_grp = qs_base + group * 32u;

                for (var qi = 0u; qi < 32u; qi++) {
                    let qb = get_byte(qs_grp + qi);
                    let col_base = b * 256u + group * 64u;

                    let k_lo  = col_base + qi;
                    let gate_lo = temp_state[temp_base + k_lo];
                    let up_lo   = temp_state[temp_base + ffn_dim + k_lo];
                    dot += (gate_lo * up_lo) * (d1 * f32(qb & 0x0Fu) - m1);

                    let k_hi  = col_base + 32u + qi;
                    let gate_hi = temp_state[temp_base + k_hi];
                    let up_hi   = temp_state[temp_base + ffn_dim + k_hi];
                    dot += (gate_hi * up_hi) * (d2 * f32((qb >> 4u) & 0x0Fu) - m2);
                }
            }
        }
    }
    // Add FFN output to residual stream.
    // Gemma-2: store to scratch area — main_post_ffn_norm will apply post-norm then add.
    // All other models: add directly to residual (no post-norm).
    if (params.post_norm_enabled != 0u) {
        temp_state[temp_base + params.ffn_dim * 2u + idx] = dot;
    } else {
        activation_in[act_base + idx] += dot;
    }
}

// -------------------------------------------------------------------------
// Kernel 4.5: Post-FFW Norm + Residual Add  (Gemma-2)
// -------------------------------------------------------------------------
// Kernel 5: Post-FFN Norm + Residual Add  (Gemma-2)
// Matches V1 shader implementation — uses norm_bank slot 3 and post_norm_enabled flag.
// -------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn main_post_ffn_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx       = global_id.x;
    let token_idx = global_id.y;

    if (idx >= params.dim || token_idx >= cache_params.batch_size) { return; }
    if (params.post_norm_enabled == 0u) { return; }

    let act_base = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let v = temp_state[temp_base + params.ffn_dim * 2u + i];
        sum_sq += v * v;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    let norm_offset_base = (offsets.layer_idx * 4u + 3u) * params.dim;
    let norm_w = norm_bank[norm_offset_base + idx];
    let dot = temp_state[temp_base + params.ffn_dim * 2u + idx];
    let normed_dot = dot * rms * norm_w;
    activation_in[act_base + idx] += normed_dot - dot;
}
