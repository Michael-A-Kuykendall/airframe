// Experimental staged, row-cooperative gate/up projection.
// Each workgroup owns one output row; normalized input is disjoint from outputs.
var<workgroup> ffn_proj_partial: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main_ffn_proj_staged(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let output_row = group.x;
    let token = group.y;
    let lane = local.x;
    if (output_row >= params.ffn_dim * 2u || token >= cache_params.batch_size) { return; }
    let gate = output_row < params.ffn_dim;
    var row = output_row;
    if (!gate) { row -= params.ffn_dim; }
    let weight_off = select(offsets.ffn_up, offsets.ffn_gate, gate);
    let slot = select(params.formula_ffn_up, params.formula_ffn_gate, gate);
    let cols = params.dim;
    let temp_base = token * params.temp_stride;
    var partial = 0.0;
    for (var col = lane; col < cols; col += 256u) {
        var weight = 0.0;
        if (slot == 0u) {
            weight = bitcast<f32>(read_blob(gow(weight_off) + row * cols + col));
        } else if (slot == 1u) {
            weight = dequant_f16_at(add_pack(weight_off, (row * cols + col) * 2u));
        } else {
            let epb = QUANT_ELEMS[slot];
            let bpb = QUANT_BYTES[slot];
            let row_pack = add_pack(weight_off, row * (cols / epb) * bpb);
            let block = add_pack(row_pack, (col / epb) * bpb);
            weight = dequant_by_formula(slot, block, col % epb);
        }
        partial += temp_state[temp_base + params.ffn_dim * 2u + col] * weight;
    }
    ffn_proj_partial[lane] = partial;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { ffn_proj_partial[lane] += ffn_proj_partial[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) {
        let dot = ffn_proj_partial[0];
        var value = dot;
        if (gate) {
            // Preserve the reference gate activation predicate and formula.
            if (params.attn_logit_softcap > 0.0) {
                value = 0.5 * dot * (1.0 + tanh(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
            } else {
                value = dot / (1.0 + exp(-dot));
            }
        }
        temp_state[temp_base + output_row] = value;
    }
}
