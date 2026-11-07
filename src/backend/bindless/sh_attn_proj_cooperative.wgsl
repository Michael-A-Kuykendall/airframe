// Opt-in row-cooperative attention output projection; attention itself is unchanged.
var<workgroup> attn_proj_partial: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main_attn_proj_cooperative(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = group.x;
    let token = group.y;
    let lane = local.x;
    if (row >= params.dim || token >= cache_params.batch_size) { return; }
    let cols = params.head_count * params.head_dim;
    let base = token * params.temp_stride;
    let slot = params.formula_attn_out;
    var partial = 0.0;
    for (var col = lane; col < cols; col += 256u) {
        var weight = 0.0;
        if (slot == 0u) {
            weight = bitcast<f32>(read_blob(gow(offsets.attn_out) + row * cols + col));
        } else if (slot == 1u) {
            weight = dequant_f16_at(add_pack(offsets.attn_out, (row * cols + col) * 2u));
        } else {
            let epb = QUANT_ELEMS[slot];
            let bpb = QUANT_BYTES[slot];
            let row_pack = add_pack(offsets.attn_out, row * (cols / epb) * bpb);
            weight = dequant_by_formula(slot, add_pack(row_pack, (col / epb) * bpb), col % epb);
        }
        partial += temp_state[base + col] * weight;
    }
    attn_proj_partial[lane] = partial;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { attn_proj_partial[lane] += attn_proj_partial[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) {
        let dot = attn_proj_partial[0];
        temp_state[base + params.dim + cols + row] = dot;
        activation_in[token * params.dim + row] += dot;
    }
}
