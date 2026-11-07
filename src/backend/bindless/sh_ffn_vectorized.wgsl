// Experimental four-value Q4_K/Q6_K gate/up projection, no weight cache.
// All quad starts are multiples of four, never crossing a 16-element scale group.
fn ffn_packed_bytes4(pack: u32, rel: u32) -> vec4<u32> {
    let adj = 2u * (pack % 2u) + rel;
    let word = pack / 2u + adj / 4u + params.blob_base_words;
    let shift = (adj % 4u) * 8u;
    var bits = read_blob(word);
    if (shift != 0u) {
        bits = (bits >> shift) | (read_blob(word + 1u) << (32u - shift));
    }
    return (vec4<u32>(bits) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u);
}
fn ffn_q4k_quad(block: u32, e: u32) -> vec4<f32> {
    let group = e / 64u;
    let sub = (e % 64u) / 32u;
    let sm = get_scale_min_k4(group * 2u + sub, block, 4u);
    let scale = read_f16_rel(block, 0u) * f32(sm.x);
    let minimum = read_f16_rel(block, 2u) * f32(sm.y);
    let bytes = ffn_packed_bytes4(block, 16u + group * 32u + e % 32u);
    let q = (bytes >> vec4<u32>(sub * 4u)) & vec4<u32>(15u);
    return vec4<f32>(scale) * vec4<f32>(q) - vec4<f32>(minimum);
}
fn ffn_q6k_quad(block: u32, e: u32) -> vec4<f32> {
    let half = e / 128u;
    let quarter = (e % 128u) / 32u;
    let l = e % 32u;
    let ql_rel = half * 64u + l + select(32u, 0u, quarter == 0u || quarter == 2u);
    let ql = ffn_packed_bytes4(block, ql_rel);
    let lower = (ql >> vec4<u32>(select(4u, 0u, quarter < 2u))) & vec4<u32>(15u);
    let qh = ffn_packed_bytes4(block, 128u + half * 32u + l);
    let upper = (qh >> vec4<u32>(quarter * 2u)) & vec4<u32>(3u);
    let q = vec4<i32>(lower | (upper << vec4<u32>(4u))) - vec4<i32>(32);
    let raw = read_byte_rel(block, 192u + half * 8u + l / 16u + quarter * 2u);
    let signed_scale = select(i32(raw), i32(raw) - 256, raw >= 128u);
    let scale = read_f16_rel(block, 208u) * f32(signed_scale);
    return vec4<f32>(scale) * vec4<f32>(q);
}
var<workgroup> ffn_vector_partial: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main_ffn_proj_vectorized(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let output_row = group.x;
    let token = group.y;
    let lane = local.x;
    if (output_row >= params.ffn_dim * 2u || token >= cache_params.batch_size) { return; }
    let gate = output_row < params.ffn_dim;
    let row = select(output_row - params.ffn_dim, output_row, gate);
    let weight_off = select(offsets.ffn_up, offsets.ffn_gate, gate);
    let slot = select(params.formula_ffn_up, params.formula_ffn_gate, gate);
    let cols = params.dim;
    let row_pack = add_pack(weight_off, row * (cols / 256u) * QUANT_BYTES[slot]);
    let temp_base = token * params.temp_stride;
    let input_base = temp_base + params.ffn_dim * 2u;
    var partial = 0.0;
    for (var col = lane * 4u; col < cols; col += 1024u) {
        let block = add_pack(row_pack, (col / 256u) * QUANT_BYTES[slot]);
        var weights: vec4<f32>;
        if (slot == 5u) {
            weights = ffn_q4k_quad(block, col % 256u);
        } else {
            weights = ffn_q6k_quad(block, col % 256u);
        }
        let x = vec4<f32>(temp_state[input_base + col], temp_state[input_base + col + 1u],
                          temp_state[input_base + col + 2u], temp_state[input_base + col + 3u]);
        partial += dot(x, weights);
    }
    ffn_vector_partial[lane] = partial;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { ffn_vector_partial[lane] += ffn_vector_partial[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) {
        let dot_value = ffn_vector_partial[0];
        var value = dot_value;
        if (gate) {
            if (params.attn_logit_softcap > 0.0) {
                value = 0.5 * dot_value * (1.0 + tanh(0.7978845608f * (dot_value + 0.044715f * dot_value * dot_value * dot_value)));
            } else {
                value = dot_value / (1.0 + exp(-dot_value));
            }
        }
        temp_state[temp_base + output_row] = value;
    }
}
