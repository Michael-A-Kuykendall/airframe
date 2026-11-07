// Opt-in row-cooperative QKV; each workgroup owns one row and one microbatch token.
var<workgroup> qkv_partial: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main_qkv_cooperative(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let idx = group.x;
    let token = group.y + params.batch_offset;
    let lane = local.x;
    let q_len = params.head_count * params.head_dim;
    let kv_len = params.head_count_kv * params.head_dim;
    if (group.y >= params.batch_count || token >= cache_params.batch_size ||
        idx >= q_len + 2u * kv_len) { return; }
    var row = idx;
    var pack = offsets.attn_q;
    var bias = offsets.attn_q_bias;
    var slot = params.formula_qk;
    if (idx >= q_len + kv_len) {
        row = idx - q_len - kv_len;
        pack = offsets.attn_v;
        bias = offsets.attn_v_bias;
        slot = params.formula_v;
    } else if (idx >= q_len) {
        row = idx - q_len;
        pack = offsets.attn_k;
        bias = offsets.attn_k_bias;
    }
    let base = token * params.temp_stride;
    var partial = 0.0;
    for (var col = lane; col < params.dim; col += 256u) {
        var weight = 0.0;
        if (slot == 0u) {
            weight = bitcast<f32>(read_blob(gow(pack) + row * params.dim + col));
        } else if (slot == 1u) {
            weight = dequant_f16_at(add_pack(pack, (row * params.dim + col) * 2u));
        } else {
            let epb = QUANT_ELEMS[slot];
            let bpb = QUANT_BYTES[slot];
            let row_pack = add_pack(pack, row * (params.dim / epb) * bpb);
            weight = dequant_by_formula(slot, add_pack(row_pack, (col / epb) * bpb), col % epb);
        }
        partial += temp_state[base + col] * weight;
    }
    qkv_partial[lane] = partial;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { qkv_partial[lane] += qkv_partial[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) {
        var dot = qkv_partial[0];
        if (bias != 0u) { dot += bitcast<f32>(read_blob(gow(bias) + row)); }
        if (idx < q_len) {
            temp_state[base + max(params.dim, q_len) + row] = dot;
        } else {
            let cache_idx = (cache_params.current_pos + token) * kv_len + row;
            if (idx < q_len + kv_len) { kv_cache_k[cache_idx] = dot; }
            else { kv_cache_v[cache_idx] = dot; }
        }
    }
}
