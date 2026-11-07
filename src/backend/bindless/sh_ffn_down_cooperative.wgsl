// Experimental row-cooperative matvec. One workgroup owns one output row.
// Every lane reaches every barrier; row/token early returns are workgroup-uniform.
var<workgroup> ffn_down_partial: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main_ffn_down_cooperative(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = group.x;
    let token = group.y;
    let lane = local.x;
    if (row >= params.dim || token >= cache_params.batch_size) { return; }
    let temp_base = token * params.temp_stride;
    let slot = params.formula_ffn_down;
    let cols = params.ffn_dim;
    var partial = 0.0;
    for (var col = lane; col < cols; col += 256u) {
        var weight = 0.0;
        if (slot == 0u) {
            weight = bitcast<f32>(read_blob(gow(offsets.ffn_down) + row * cols + col));
        } else if (slot == 1u) {
            weight = dequant_f16_at(add_pack(offsets.ffn_down, (row * cols + col) * 2u));
        } else {
            let epb = QUANT_ELEMS[slot];
            let bpb = QUANT_BYTES[slot];
            let row_pack = add_pack(offsets.ffn_down, row * (cols / epb) * bpb);
            let block = add_pack(row_pack, (col / epb) * bpb);
            weight = dequant_by_formula(slot, block, col % epb);
        }
        partial += (temp_state[temp_base + col] * temp_state[temp_base + cols + col]) * weight;
    }
    ffn_down_partial[lane] = partial;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            ffn_down_partial[lane] += ffn_down_partial[lane + stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        let dot = ffn_down_partial[0];
        temp_state[temp_base + cols * 2u + row] = dot;
        activation_in[token * params.dim + row] += dot;
    }
}
