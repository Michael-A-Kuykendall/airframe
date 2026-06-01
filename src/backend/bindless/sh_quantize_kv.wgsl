// sh_quantize_kv.wgsl
// TurboQuant: Per-head-vector INT4 quantization of the F32 KV cache.
//
// One GPU invocation per (head, position) pair.
// Each invocation:
//   1. Scans head_dim F32 values to find max_abs
//   2. Writes scale = max_abs / 7.0 to the scale buffer
//   3. Packs head_dim nibbles (bias-8, range 1-15) into head_dim/8 U32s
//
// Encoding: q = clamp(round(val / scale) + 8, 1, 15)
//   - 8 represents 0.0
//   - 1 represents -7*scale (negative floor)
//   - 15 represents +7*scale (positive ceil)
//   - 0 reserved (never written)
//
// U32 packing: nibble[0] in bits[3:0], nibble[1] in bits[7:4], ..., nibble[7] in bits[31:28]
//
// Dispatch for decode (single new position):
//   dispatch(n_head_kv, 1, 1), qparams.pos_offset = current_pos
//
// Dispatch for full requantization (after helical shift):
//   dispatch(n_head_kv, seq_len, 1), qparams.pos_offset = 0

struct QuantizeKvParams {
    n_head_kv:  u32,  // Number of KV heads
    head_dim:   u32,  // Elements per head-vector (must be multiple of 8)
    pos_offset: u32,  // Base position: actual pos = pos_offset + global_id.y
    _pad:       u32,
};

@group(0) @binding(0) var<storage, read>       f32_k_cache:    array<f32>;  // F32 K staging [max_seq, n_head_kv, head_dim]
@group(0) @binding(1) var<storage, read>       f32_v_cache:    array<f32>;  // F32 V staging [max_seq, n_head_kv, head_dim]
@group(0) @binding(2) var<storage, read_write> packed_k_cache: array<u32>;  // INT4 K packed [max_seq, n_head_kv, head_dim/8]
@group(0) @binding(3) var<storage, read_write> packed_v_cache: array<u32>;  // INT4 V packed [max_seq, n_head_kv, head_dim/8]
@group(0) @binding(4) var<storage, read_write> scale_k_cache:  array<f32>;  // K scales [max_seq, n_head_kv]
@group(0) @binding(5) var<storage, read_write> scale_v_cache:  array<f32>;  // V scales [max_seq, n_head_kv]
@group(0) @binding(6) var<uniform>             qparams:        QuantizeKvParams;

@compute @workgroup_size(1, 1, 1)
fn quantize_kv(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let head = global_id.x;
    let pos  = qparams.pos_offset + global_id.y;

    if (head >= qparams.n_head_kv) { return; }

    let head_dim = qparams.head_dim;
    let n_head   = qparams.n_head_kv;
    let hd8      = head_dim / 8u;  // U32s per head-vector

    // Flat base into F32 layout: [pos, head, dim]
    let f32_base   = pos * n_head * head_dim + head * head_dim;
    let pack_base  = pos * n_head * hd8      + head * hd8;
    let scale_idx  = pos * n_head + head;

    // ---- K head-vector ----
    var max_abs_k = 0.0;
    for (var d = 0u; d < head_dim; d++) {
        max_abs_k = max(max_abs_k, abs(f32_k_cache[f32_base + d]));
    }
    let scale_k = select(max_abs_k / 7.0, 1.0, max_abs_k == 0.0);
    scale_k_cache[scale_idx] = scale_k;

    for (var u = 0u; u < hd8; u++) {
        var packed = 0u;
        for (var n = 0u; n < 8u; n++) {
            let val  = f32_k_cache[f32_base + u * 8u + n];
            let q_i  = clamp(i32(round(val / scale_k)) + 8, 1, 15);
            packed  |= (u32(q_i) << (n * 4u));
        }
        packed_k_cache[pack_base + u] = packed;
    }

    // ---- V head-vector ----
    var max_abs_v = 0.0;
    for (var d = 0u; d < head_dim; d++) {
        max_abs_v = max(max_abs_v, abs(f32_v_cache[f32_base + d]));
    }
    let scale_v = select(max_abs_v / 7.0, 1.0, max_abs_v == 0.0);
    scale_v_cache[scale_idx] = scale_v;

    for (var u = 0u; u < hd8; u++) {
        var packed = 0u;
        for (var n = 0u; n < 8u; n++) {
            let val  = f32_v_cache[f32_base + u * 8u + n];
            let q_i  = clamp(i32(round(val / scale_v)) + 8, 1, 15);
            packed  |= (u32(q_i) << (n * 4u));
        }
        packed_v_cache[pack_base + u] = packed;
    }
}
