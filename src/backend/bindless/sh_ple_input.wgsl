
// sh_ple_input.wgsl
// gemma-4 dense-latent PLE per-layer latent input construction
// (llama.cpp gemma4.cpp build_inp_per_layer + project_per_layer_inputs @ 49f35421):
//
//   inp_per_layer = get_rows(per_layer_token_embd, tokens) * sqrt(n_embd_per_layer)
//   per_layer_proj = mm(per_layer_model_proj, inp_batch) * (1/sqrt(n_embd))
//   per_layer_proj = RMSNorm(per_layer_proj, per_layer_proj_norm)   // per 256-slice
//   inp_per_layer = (per_layer_proj + inp_per_layer) * (1/sqrt(2))
//
// Output layout: ple_input[il * n_tokens * latent + t * latent + k]
//   -> slice il (a [latent, n_tokens] matrix) is what the PLE block kernel reads.
//
// Self-contained: dedicated bind group layout + pipeline, built only when
// spec.ple_enabled. No other model ever binds or dispatches this pass.

// IEEE-754 binary16 -> binary32 (matches quant_formula f16_to_f32).
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

struct PleInputParams {
    latent: u32,           // 256 (n_embd_per_layer)
    n_layer: u32,          // 42
    n_tokens: u32,         // batch_size
    n_embd: u32,           // 2560 (model input dim)
    token_embd_off: u32,   // packed offset of per_layer_token_embd (Q6_K [latent*n_layer, vocab])
    token_embd_row_bytes: u32, // bytes per token row (Q6_K 10752 elems = 42 blocks * 210)
    model_proj_off: u32,   // packed offset of per_layer_model_proj (IQ4_XS [n_embd, latent*n_layer])
    proj_norm_off: u32,    // packed offset of per_layer_proj_norm (F32 [latent])
    rms_eps: f32,
    blob_base_words: u32,  // window-local base (rebase absolute -> window)
    chunk_words: u32,
}

// Blob bindings: 0 + 10..16 (same read_blob convention as sh_layer_v1).
@group(0) @binding(0)  var<storage, read> blob_0: array<u32>;
@group(0) @binding(10) var<storage, read> blob_1: array<u32>;
@group(0) @binding(11) var<storage, read> blob_2: array<u32>;
@group(0) @binding(12) var<storage, read> blob_3: array<u32>;
@group(0) @binding(13) var<storage, read> blob_4: array<u32>;
@group(0) @binding(14) var<storage, read> blob_5: array<u32>;
@group(0) @binding(15) var<storage, read> blob_6: array<u32>;
@group(0) @binding(16) var<storage, read> blob_7: array<u32>;

@group(0) @binding(1) var<storage, read> token_ids: array<u32>;   // [n_tokens]
@group(0) @binding(2) var<storage, read> inp_batch: array<f32>;   // [n_tokens * n_embd] (scaled embeddings)
@group(0) @binding(3) var<uniform> params: PleInputParams;
@group(0) @binding(4) var<storage, read_write> ple_input: array<f32>; // [n_layer * n_tokens * latent]

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

// Q6_K dequant (256-elem superblock, 210 bytes) — per_layer_token_embd type.
fn dequant_q6k_elem(block_pack: u32, elem_in_block: u32) -> f32 {
    let d = read_f16_rel(block_pack, 208u);
    let half    = elem_in_block / 128u;
    let half_e  = elem_in_block % 128u;
    let l       = half_e % 32u;
    let quarter = half_e / 32u;
    let ql_rel = select(half * 64u + l + 32u, half * 64u + l, quarter == 0u || quarter == 2u);
    let ql_byte_val = read_byte_rel(block_pack, ql_rel);
    let lower4 = select(ql_byte_val >> 4u, ql_byte_val & 0xFu, quarter < 2u);
    let qh_byte_val = read_byte_rel(block_pack, 128u + half * 32u + l);
    let upper2 = (qh_byte_val >> (quarter * 2u)) & 3u;
    let q6 = lower4 | (upper2 << 4u);
    let signed_q = i32(q6) - 32;
    let sc_idx = 192u + half * 8u + (l / 16u) + quarter * 2u;
    let sc_raw = read_byte_rel(block_pack, sc_idx);
    let sc_signed = select(i32(sc_raw), i32(sc_raw) - 256, sc_raw >= 128u);
    return d * f32(sc_signed) * f32(signed_q);
}

// IQ4_XS dequant (128-elem superblock, 128 bytes) — per_layer_model_proj type.
fn dequant_iq4_xs_elem(block_pack: u32, e: u32) -> f32 {
    let scale_idx = e / 4u;
    let scale = read_f16_rel(block_pack, scale_idx * 2u);
    let byte_idx = e / 2u;
    let qs_byte = read_byte_rel(block_pack, 64u + byte_idx);
    let nibble = select(qs_byte >> 4u, qs_byte & 0x0Fu, e % 2u == 0u);
    return f32(i32(nibble) - 8) * scale;
}

// F32 element read at absolute byte offset (per_layer_proj_norm).
fn read_f32_rel(pack: u32, rel: u32) -> f32 {
    let b0 = read_byte_rel(pack, rel);
    let b1 = read_byte_rel(pack, rel + 1u);
    let b2 = read_byte_rel(pack, rel + 2u);
    let b3 = read_byte_rel(pack, rel + 3u);
    return bitcast<f32>(b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u));
}

var<workgroup> wg_proj_sq: array<f32, 256>;

// One workgroup per (token, layer): 256 threads = one 256-element latent slice.
// Thread k computes proj[k] (a 2560-length dot over model_proj column j),
// workgroup-reduces sum(proj^2) for the slice RMS, then writes
// (proj*rms*proj_norm[k] + gather[k]) * (1/sqrt(2)).
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    let k  = lid.x;          // 0..latent-1 (256 threads per workgroup = one slice)
    let t  = gid.x / 256u;   // token (workgroup covers 256 consecutive X)
    let il = gid.y;          // layer

    if (t >= params.n_tokens || il >= params.n_layer) { return; }
    let latent = params.latent;
    let j = il * latent + k; // global latent index (0..latent*n_layer)

    // ── per_layer_proj[j] = sum_i model_proj[i][j] * inp[i][t] / sqrt(n_embd) ──
    var proj = 0.0;
    let proj_blocks_per_col = params.n_embd / 128u;
    for (var i = 0u; i < params.n_embd; i++) {
        // model_proj stored column-major: element (i, j) at byte i + j*n_embd.
        let elem = i + j * params.n_embd;
        let block = elem / 128u;
        let e_in_block = elem % 128u;
        let bp = add_pack(params.model_proj_off, block * 128u);
        let w = dequant_iq4_xs_elem(bp, e_in_block);
        proj += w * inp_batch[t * params.n_embd + i];
    }
    proj = proj * (1.0 / sqrt(f32(params.n_embd)));

    // ── workgroup reduction of proj^2 over the 256-slice ──
    wg_proj_sq[k] = proj * proj;
    workgroupBarrier();
    var sum_sq = 0.0;
    for (var s = 0u; s < latent; s++) {
        sum_sq += wg_proj_sq[s];
    }
    let rms_inv = inverseSqrt(sum_sq / f32(latent) + params.rms_eps);

    // proj_norm[k] — F32 [latent], same weight for every layer slice.
    let norm_w = read_f32_rel(params.proj_norm_off, k * 4u);
    let proj_normed = proj * rms_inv * norm_w;

    // ── gather = per_layer_token_embd[tokens[t]][j] * sqrt(latent) ──
    // get_rows picks the token COLUMN of per_layer_token_embd [latent*n_layer, vocab];
    // stored column-major, so the token row is contiguous `latent*n_layer` elems.
    let token = token_ids[t];
    let row_base = add_pack(params.token_embd_off, token * params.token_embd_row_bytes);
    let q6_block = j / 256u;
    let q6_e = j % 256u;
    let gather = dequant_q6k_elem(add_pack(row_base, q6_block * 210u), q6_e) * sqrt(f32(latent));

    // ── combine + write ple_input[il][t][k] ──
    let out = (proj_normed + gather) * (1.0 / sqrt(2.0));
    let out_idx = (il * params.n_tokens + t) * latent + k;
    ple_input[out_idx] = out;
}
