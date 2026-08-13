// sh_ple_block.wgsl
// gemma-4 dense-latent PLE block residual (llama.cpp gemma4.cpp "per-layer
// embedding" branch @ 49f35421):
//
//   pe_in = cur                        (activation after FFN residual + post-ffw norm)
//   cur   = mm(per_layer_inp_gate, cur)   // inp_gate [n_embd, latent] Q4_K
//   cur   = gelu(cur)
//   cur   = cur * inp_this_layer          // inp_this_layer = ple_input[il][t][k]
//   cur   = mm(per_layer_proj, cur)       // proj [latent, n_embd] Q4_K
//   cur   = RMSNorm(cur, per_layer_post_norm)   // post_norm F32 [n_embd]
//   cur   = pe_in + cur
//   cur   = cur * out_scale               // layer_scales[layer_idx]
//
// Dedicated bind group layout + pipeline, only dispatched when ple_enabled.
// Workgroup 256 = one token; thread j handles latent element j (j < latent)
// for the gate stage and output element j (strided over dim) for the proj stage.

fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 1u;
    let exp  = (bits >> 10u) & 0x1fu;
    let mant = bits & 0x3ffu;
    let sign_f = select(-1.0, 1.0, sign == 0u);
    if (exp == 0u) {
        if (mant == 0u) {
            return sign_f * 0.0;
        }
        return sign_f * (f32(mant) / f32(1u << 24u));
    }
    if (exp == 0x1fu) {
        return 0.0;
    }
    let fraction = 1.0 + f32(mant) / 1024.0;
    if (exp >= 15u) {
        let p = f32(1u << (exp - 15u));
        return sign_f * fraction * p;
    } else {
        let p = f32(1u << (15u - exp));
        return sign_f * fraction / p;
    }
}

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
    layer_idx: u32,
    attn_q_norm: u32,
    attn_k_norm: u32,
    attn_q_bias: u32,
    attn_k_bias: u32,
    attn_v_bias: u32,
    v_is_q4k: u32,
    ffn_down_is_q4k: u32,
    ple_inp_gate: u32,
    ple_proj: u32,
    ple_layer_output_scale: u32,
    ple_rope_freqs: u32,
    ple_attn_post_norm: u32,
    ple_ffn_post_norm: u32,
    ple_post_norm: u32,
    ple_enabled: u32,
}

struct PleBlockParams {
    dim: u32,            // n_embd (2560)
    latent: u32,         // 256
    temp_stride: u32,    // per-token temp stride in floats
    rms_eps: f32,
    out_scale_enabled: u32,
    scratch_base: u32,   // temp offset for proj scratch (ffn_dim*2, dead slot)
    blob_base_words: u32,
    chunk_words: u32,
}

struct CacheParams {
    current_pos: u32,
    seq_len: u32,
    max_seq_len: u32,
    batch_size: u32,
    logical_pos_base: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0)  var<storage, read> blob_0: array<u32>;
@group(0) @binding(10) var<storage, read> blob_1: array<u32>;
@group(0) @binding(11) var<storage, read> blob_2: array<u32>;
@group(0) @binding(12) var<storage, read> blob_3: array<u32>;
@group(0) @binding(13) var<storage, read> blob_4: array<u32>;
@group(0) @binding(14) var<storage, read> blob_5: array<u32>;
@group(0) @binding(15) var<storage, read> blob_6: array<u32>;
@group(0) @binding(16) var<storage, read> blob_7: array<u32>;

@group(0) @binding(1)  var<storage, read_write> activation_in: array<f32>; // residual stream
@group(0) @binding(2)  var<storage, read_write> temp_state: array<f32>;    // scratch (gate stage)
@group(0) @binding(3)  var<uniform> offsets: LayerOffsets;
@group(0) @binding(4)  var<uniform> params: PleBlockParams;
@group(0) @binding(9)  var<uniform> cache_params: CacheParams;
@group(0) @binding(17) var<storage, read> layer_scales: array<f32>;        // per-block out_scale
@group(0) @binding(18) var<storage, read> ple_input: array<f32>;           // [n_layer*n_tokens*latent]

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

fn gow(pack: u32) -> u32 {
    return pack / 2u + params.blob_base_words;
}
fn add_pack(pack: u32, even_bytes: u32) -> u32 {
    return pack + even_bytes / 2u;
}
fn read_byte_rel(pack: u32, rel: u32) -> u32 {
    let adj = 2u * (pack % 2u) + rel;
    let word = pack / 2u + adj / 4u + params.blob_base_words;
    return extractBits(read_blob(word), (adj % 4u) * 8u, 8u);
}
fn read_f16_rel(pack: u32, rel: u32) -> f32 {
    let adj = 2u * (pack % 2u) + rel;
    let word = pack / 2u + adj / 4u + params.blob_base_words;
    let bits = extractBits(read_blob(word), (adj % 4u) * 8u, 16u);
    return f16_to_f32(bits);
}

// Q4_K dequant (256-elem superblock, 144 bytes) — inp_gate / proj quant type.
fn dequant_q4k_elem(block_pack: u32, elem_in_block: u32) -> f32 {
    let d = read_f16_rel(block_pack, 0u);
    let dmin = read_f16_rel(block_pack, 2u);
    let quarter = elem_in_block / 64u;       // 0..3
    let qe = elem_in_block % 64u;            // 0..63
    let s_idx = 16u + quarter;
    let sc_raw = read_byte_rel(block_pack, s_idx);
    let sc = select(f32(sc_raw) - 32.0, f32(sc_raw & 0x3Fu) - 32.0, sc_raw >= 128u);
    let m_idx = 24u + quarter;
    let m_raw = read_byte_rel(block_pack, m_idx);
    let m = select(f32(m_raw) - 32.0, f32(m_raw & 0x3Fu) - 32.0, m_raw >= 128u);
    let ql = read_byte_rel(block_pack, 32u + qe);
    let l = select(ql >> 4u, ql & 0x0Fu, qe % 2u == 0u);
    let qh_raw = read_byte_rel(block_pack, 96u + qe / 2u);
    let h = select(qh_raw >> 4u, qh_raw & 0x0Fu, qe % 2u == 0u);
    let q = l | ((h & 1u) << 4u);
    let q2 = l | ((h & 2u) << 3u);
    let qval = select(f32(i32(q2) - 8), f32(i32(q) - 8), (h & 3u) < 2u);
    return d * qval + m * dmin;
}

var<workgroup> wg_gate: array<f32, 256>;     // latent gate values (j < latent)
var<workgroup> wg_partial: array<f32, 256>;

// One workgroup per token. Thread j (0..255):
//   stage 1: if j < latent, compute gate[j] = GELU(dot(inp_gate[:, j], act)) * ple_input[il][t][j]
//   stage 2: compute proj[j] = dot(prog[j, :], gate) over latent
//   stage 3: RMSNorm over dim (strided) -> add to activation, then out_scale.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.y;
    let j = lid.x;
    if (t >= cache_params.batch_size) { return; }
    let latent = params.latent;
    let dim = params.dim;
    let act_base = t * dim;
    let temp_base = t * params.temp_stride;
    let il = offsets.layer_idx;

    // ── Stage 1: gate = GELU(inp_gate^T act) * ple_input slice ──
    if (j < latent) {
        var dot = 0.0;
        // inp_gate [dim, latent] Q4_K (256 elems/block, 144 B/block)
        for (var i = 0u; i < dim; i++) {
            let elem = i + j * dim;
            let block = elem / 256u;
            let e_in_block = elem % 256u;
            let bp = add_pack(offsets.ple_inp_gate, block * 144u);
            let w = dequant_q4k_elem(bp, e_in_block);
            dot += w * activation_in[act_base + i];
        }
        let gate = 0.5 * dot * (1.0 + tanh(0.7978845608 * (dot + 0.044715 * dot * dot * dot)));
        let slice = ple_input[il * cache_params.batch_size * latent + t * latent + j];
        wg_gate[j] = gate * slice;
    }
    workgroupBarrier();

    // ── Stage 2: proj = proj[latent, dim]^T gate ──
    // proj [latent, dim] Q4_K: element (k, i) at byte k + i*latent.
    // out[i] = sum_k proj[k + i*latent] * gate[k]. Store to temp for the RMS pass.
    for (var i = j; i < dim; i += 256u) {
        var dot = 0.0;
        for (var k = 0u; k < latent; k++) {
            let elem = k + i * latent;
            let block = elem / 256u;
            let e_in_block = elem % 256u;
            let bp = add_pack(offsets.ple_proj, block * 144u);
            let w = dequant_q4k_elem(bp, e_in_block);
            dot += w * wg_gate[k];
        }
        temp_state[temp_base + params.scratch_base + i] = dot;
    }
    workgroupBarrier();

    // Cooperative RMS over temp_state[temp_base + scratch_base .. +dim].
    var partial = 0.0;
    for (var i = j; i < dim; i += 256u) {
        let v = temp_state[temp_base + params.scratch_base + i];
        partial += v * v;
    }
    wg_partial[j] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (j < s) {
            wg_partial[j] += wg_partial[j + s];
        }
        workgroupBarrier();
    }
    let rms_inv = inverseSqrt(wg_partial[0] / f32(dim) + params.rms_eps);

    // Apply norm weight, add residual, then out_scale.
    for (var i = j; i < dim; i += 256u) {
        // post_norm F32 [dim] at offsets.ple_post_norm
        let rel = i * 4u;
        let b0 = read_byte_rel(offsets.ple_post_norm, rel);
        let b1 = read_byte_rel(offsets.ple_post_norm, rel + 1u);
        let b2 = read_byte_rel(offsets.ple_post_norm, rel + 2u);
        let b3 = read_byte_rel(offsets.ple_post_norm, rel + 3u);
        let norm_w = bitcast<f32>(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
        let dot = temp_state[temp_base + params.scratch_base + i];
        let normed = dot * rms_inv * norm_w;
        var resid = activation_in[act_base + i] + normed;
        if (params.out_scale_enabled != 0u) {
            resid = resid * layer_scales[il];
        }
        activation_in[act_base + i] = resid;
    }
}
