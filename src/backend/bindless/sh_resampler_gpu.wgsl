// ─────────────────────────────────────────────────────────────────────────────
// sh_resampler_gpu.wgsl  —  Perceiver Resampler cross-attention (GPU dispatch)
// Model: MiniCPM-V-2.6 mmproj resampler
//   n_queries=64, d_model=3584, kv_dim=1152, n_heads=16, head_dim=224
//
// Bind group:
//   @binding(0) vit_blob      [array<u32>]  raw mmproj GGUF bytes (read-only)
//   @binding(1) vit_features  [array<f32>]  [n_vit_tokens × kv_dim] ViT encoder output
//   @binding(2) query_state   [array<f32>]  [n_queries × d_model]   working query buffer
//   @binding(3) kv_state      [array<f32>]  [n_vit × d_model × 3]  three-slot scratch
//       kv_state[0          .. n_vit×D)   = slot 0: kv_lin/kv_ln intermediate (reused for Q proj temp)
//       kv_state[n_vit×D    .. 2×n_vit×D) = slot 1: K projected
//       kv_state[2×n_vit×D  .. 3×n_vit×D) = slot 2: V projected
//       (slot 0 also reused as final_proj output — caller reads kv_state[0..n_q*D])
//   @binding(4) offsets  ResamplerOffsets uniform
//   @binding(5) params   ResamplerParams  uniform
//
// Kernel sequence (one call per image, after ViT blocks complete):
//   1.  main_rsp_init_q   Copy query_embeds from vit_blob → query_state
//   2.  main_rsp_kv_lin   vit_features × kv_weight (F16) → kv_state slot 0
//   3.  main_rsp_ln_kv    LayerNorm kv_state slot 0 in-place (kv_lin → kv_ln)
//   4.  main_rsp_proj_k   (kv_ln + pos_embed_k) × attn_k_w + bias → kv_state slot 1
//   5.  main_rsp_proj_v   kv_ln × attn_v_w + bias → kv_state slot 2  [no race: reads slot 0, writes slot 2]
//   6.  main_rsp_ln_q     LayerNorm query_state in-place
//   7.  main_rsp_proj_q   query_state × attn_q_w + bias → kv_state slot 0 (safe temp)
//   7b. main_rsp_copy_q   kv_state slot 0 → query_state
//   8.  main_rsp_attn     Cross-attn: Q(query_state) K(slot 1) V(slot 2) → query_state
//   9.  main_rsp_out_proj attn_out_proj + residual(ln_q_stash) → query_state
//  10.  main_rsp_post_ln  LayerNorm query_state in-place
//  11.  main_rsp_final_proj query_state @ proj_w → kv_state slot 0  (x @ proj, column access)
//       (output: kv_state[0..n_q*d_model] — Rust caller reads this as 64 visual tokens)
//
// kv_state slot 0 sub-layout (all within 0..n_vit*D):
//   [0        .. n_q*D)   Q projection temp (kernels 7/7b)
//   [n_q*D    .. 2*n_q*D) ln_q stash: ln_q(query_embeds) saved by proj_q for out_proj residual
//   [2*n_q*D .. n_vit*D)  free
//   [0        .. n_q*D)   final_proj output (kernel 11 overwrites Q temp — both are n_q*D, safe)
// ─────────────────────────────────────────────────────────────────────────────

// Compile-time constant for Perceiver Resampler head dimension (3584 / 16 = 224)
const RSP_HEAD_DIM: u32 = 224u;

// ─── Uniform structs ──────────────────────────────────────────────────────────

/// Byte offsets for all resampler tensors (absolute from start of mmproj GGUF).
/// All F16 weight matrices; F32 biases, LN weights, query_embeds, pos_embed_k.
/// Size = 20 × 4 = 80 bytes (multiple of 16 ✓).
struct ResamplerOffsets {
    query_embeds: u32,    // [n_queries × d_model] F32
    kv_weight:    u32,    // [kv_dim × d_model]    F16
    ln_q_w:       u32,    // [d_model] F32
    ln_q_b:       u32,    // [d_model] F32
    ln_kv_w:      u32,    // [d_model] F32
    ln_kv_b:      u32,    // [d_model] F32
    attn_q_w:     u32,    // [d_model × d_model]   F16
    attn_q_b:     u32,    // [d_model] F32
    attn_k_w:     u32,    // [d_model × d_model]   F16
    attn_k_b:     u32,    // [d_model] F32
    attn_v_w:     u32,    // [d_model × d_model]   F16
    attn_v_b:     u32,    // [d_model] F32
    attn_out_w:   u32,    // [d_model × d_model]   F16
    attn_out_b:   u32,    // [d_model] F32
    pos_embed_k:  u32,    // [4900 × d_model]      F32  (we use first n_vit_tokens rows)
    ln_post_w:    u32,    // [d_model] F32
    ln_post_b:    u32,    // [d_model] F32
    proj_w:       u32,    // [d_model × d_model]   F16
    pad0:         u32,
    pad1:         u32,
}

/// Resampler dimensions.
/// Size = 8 × 4 = 32 bytes (multiple of 16 ✓).
struct ResamplerParams {
    n_queries:   u32,   // 64
    n_vit:       u32,   // 1024  (vit tokens fed into resampler)
    d_model:     u32,   // 3584
    kv_dim:      u32,   // 1152
    n_heads:     u32,   // 16
    head_dim:    u32,   // 224
    ln_eps:      f32,   // 1e-6
    pad0:        u32,
}

// ─── Bindings ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       vit_blob:     array<u32>;
@group(0) @binding(1) var<storage, read>       vit_features: array<f32>;
@group(0) @binding(2) var<storage, read_write> query_state:  array<f32>;
@group(0) @binding(3) var<storage, read_write> kv_state:     array<f32>;
@group(0) @binding(4) var<uniform>             offsets:      ResamplerOffsets;
@group(0) @binding(5) var<uniform>             params:       ResamplerParams;

// ─── Byte accessors ───────────────────────────────────────────────────────────

fn rsp_read_f32(byte_off: u32) -> f32 {
    return bitcast<f32>(vit_blob[byte_off >> 2u]);
}

fn rsp_read_f16(byte_off: u32) -> f32 {
    let word   = vit_blob[byte_off >> 2u];
    let shift  = (byte_off & 2u) << 3u;
    let bits   = (word >> shift) & 0xFFFFu;
    let sign   = (bits >> 15u) & 1u;
    let exp    = (bits >> 10u) & 0x1Fu;
    let mant   = bits & 0x3FFu;
    var f32bits: u32;
    if exp == 0u {
        if mant == 0u {
            f32bits = sign << 31u;
        } else {
            var m = mant;
            var e = 0u;
            loop {
                e += 1u; m <<= 1u;
                if (m & 0x400u) != 0u { break; }
            }
            f32bits = (sign << 31u) | ((127u - 15u + 1u - e) << 23u) | ((m & 0x3FFu) << 13u);
        }
    } else if exp == 31u {
        f32bits = (sign << 31u) | (0xFFu << 23u) | (mant << 13u);
    } else {
        f32bits = (sign << 31u) | ((exp + 127u - 15u) << 23u) | (mant << 13u);
    }
    return bitcast<f32>(f32bits);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: main_rsp_init_q
//   Copy learned query_embeds from vit_blob → query_state.
//   Dispatch: ((n_queries * d_model + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_init_q(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D = params.d_model;
    let g = gid.x;
    if g >= params.n_queries * D { return; }
    query_state[g] = rsp_read_f32(offsets.query_embeds + g * 4u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: main_rsp_kv_lin
//   Project vit_features [N_vit × kv_dim] through kv_weight [kv_dim × d_model]
//   → kv_state slot 0 [N_vit × d_model].
//   Dispatch: ((n_vit * d_model + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_kv_lin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D      = params.d_model;
    let KD     = params.kv_dim;
    let g      = gid.x;
    if g >= params.n_vit * D { return; }

    let tok   = g / D;
    let out_d = g % D;

    // kv_weight is [kv_dim × d_model] F16, column-major? or row-major?
    // In GGUF/ggml, weight matrices are stored as [out_features, in_features] row-major.
    // So kv_weight[out_d][in_d] = kv_weight byte at (out_d * kv_dim + in_d) * 2
    var dot = 0.0f;
    let row_byte = offsets.kv_weight + out_d * KD * 2u;
    let feat_base = tok * KD;
    for (var in_d = 0u; in_d < KD; in_d++) {
        dot += rsp_read_f16(row_byte + in_d * 2u) * vit_features[feat_base + in_d];
    }
    kv_state[tok * D + out_d] = dot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: main_rsp_ln_kv
//   LayerNorm kv_state slot 0 [N_vit × d_model] in-place.
//   Dispatch: ((n_vit + 255) / 256, 1, 1)  (one invocation per kv token)
// ─────────────────────────────────────────────────────────────────────────────

fn layernorm_kv_inplace(tok: u32) {
    let D    = params.d_model;
    let base = tok * D;

    var mean = 0.0f;
    for (var i = 0u; i < D; i++) { mean += kv_state[base + i]; }
    mean /= f32(D);

    var variance = 0.0f;
    for (var i = 0u; i < D; i++) {
        let d = kv_state[base + i] - mean;
        variance += d * d;
    }
    variance /= f32(D);
    let inv_std = 1.0f / sqrt(variance + params.ln_eps);

    for (var i = 0u; i < D; i++) {
        let x = kv_state[base + i];
        let w = rsp_read_f32(offsets.ln_kv_w + i * 4u);
        let b = rsp_read_f32(offsets.ln_kv_b + i * 4u);
        kv_state[base + i] = (x - mean) * inv_std * w + b;
    }
}

@compute @workgroup_size(256, 1, 1)
fn main_rsp_ln_kv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tok = gid.x;
    if tok >= params.n_vit { return; }
    layernorm_kv_inplace(tok);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: main_rsp_proj_k
//   Project K: (kv_ln + pos_embed_k) × attn_k_weight + bias → kv_state[slot 1]
//   (slot 1 = kv_state[n_vit × d_model .. 2 × n_vit × d_model])
//   Dispatch: ((n_vit * d_model + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_proj_k(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D      = params.d_model;
    let n_vit  = params.n_vit;
    let g      = gid.x;
    if g >= n_vit * D { return; }

    let tok   = g / D;
    let out_d = g % D;

    // K input = kv_ln (slot 0) + pos_embed_k[tok]
    var dot = 0.0f;
    let row_byte = offsets.attn_k_w + out_d * D * 2u;
    let kv_base  = tok * D;   // slot 0
    for (var in_d = 0u; in_d < D; in_d++) {
        // kv_state already holds ln_kv output; add pos_embed_k on the fly
        let kv_val     = kv_state[kv_base + in_d];
        let pos_val    = rsp_read_f32(offsets.pos_embed_k + (tok * D + in_d) * 4u);
        dot += rsp_read_f16(row_byte + in_d * 2u) * (kv_val + pos_val);
    }
    dot += rsp_read_f32(offsets.attn_k_b + out_d * 4u);

    // Write to slot 1
    kv_state[n_vit * D + tok * D + out_d] = dot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: main_rsp_proj_v
//   Project V: kv_ln × attn_v_weight + bias → kv_state slot 2.
//   Reads slot 0 (kv_ln, read-only at this point), writes slot 2 — no data race.
//   Dispatch: ((n_vit * d_model + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_proj_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D     = params.d_model;
    let n_vit = params.n_vit;
    let g     = gid.x;
    if g >= n_vit * D { return; }

    let tok   = g / D;
    let out_d = g % D;

    var dot = 0.0f;
    let row_byte = offsets.attn_v_w + out_d * D * 2u;
    let kv_base  = tok * D;   // slot 0 (kv_ln, read-only)
    for (var in_d = 0u; in_d < D; in_d++) {
        dot += rsp_read_f16(row_byte + in_d * 2u) * kv_state[kv_base + in_d];
    }
    dot += rsp_read_f32(offsets.attn_v_b + out_d * 4u);

    kv_state[2u * n_vit * D + tok * D + out_d] = dot;    // → slot 2
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6: main_rsp_ln_q
//   LayerNorm query_state [n_queries × d_model] in-place.
//   At this point query_state holds query_embeds (copied by main_rsp_init_q).
//   Dispatch: ((n_queries + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_ln_q(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if q >= params.n_queries { return; }

    let D    = params.d_model;
    let base = q * D;

    var mean = 0.0f;
    for (var i = 0u; i < D; i++) { mean += query_state[base + i]; }
    mean /= f32(D);

    var variance = 0.0f;
    for (var i = 0u; i < D; i++) {
        let d = query_state[base + i] - mean;
        variance += d * d;
    }
    variance /= f32(D);
    let inv_std = 1.0f / sqrt(variance + params.ln_eps);

    for (var i = 0u; i < D; i++) {
        let x = query_state[base + i];
        let w = rsp_read_f32(offsets.ln_q_w + i * 4u);
        let b = rsp_read_f32(offsets.ln_q_b + i * 4u);
        query_state[base + i] = (x - mean) * inv_std * w + b;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7: main_rsp_proj_q
//   Project Q: ln_q × attn_q_w + bias → kv_state slot 0 (safe temp).
//   Writing back to query_state in-place would be a data race: invocation (q, out_d=X)
//   writes query_state[q*D+X] while invocation (q, out_d=Y) reads query_state[q*D+X].
//   Fix: write to kv_state slot 0 (kv_ln no longer needed after proj_k/proj_v).
//   n_queries*D = 229376 << n_vit*D = 3670016, so slot 0 has plenty of space.
//   main_rsp_copy_q copies back to query_state after this kernel.
//   Simultaneously stashes the ln_q INPUT (query_state before projection) at
//   kv_state[n_q*D + q*D + out_d] so main_rsp_out_proj can use it as the
//   correct residual (ln_q(query_embeds)) rather than raw query_embeds.
//   Dispatch: ((n_queries * d_model + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_proj_q(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D = params.d_model;
    let g = gid.x;
    if g >= params.n_queries * D { return; }

    let q     = g / D;
    let out_d = g % D;
    let q_base = q * D;

    // Stash ln_q(query_embeds)[q, out_d] — needed as residual in out_proj.
    // Occupies kv_state[n_q*D .. 2*n_q*D), within slot 0 (0..n_vit*D), no overlap.
    kv_state[params.n_queries * D + q_base + out_d] = query_state[q_base + out_d];

    var dot = 0.0f;
    let row_byte = offsets.attn_q_w + out_d * D * 2u;
    for (var in_d = 0u; in_d < D; in_d++) {
        dot += rsp_read_f16(row_byte + in_d * 2u) * query_state[q_base + in_d];
    }
    dot += rsp_read_f32(offsets.attn_q_b + out_d * 4u);

    kv_state[q_base + out_d] = dot;    // → kv_state slot 0 Q projection temp
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7b: main_rsp_copy_q
//   Copy Q result from kv_state slot 0 → query_state.
//   Must run AFTER main_rsp_proj_q completes (separate compute pass).
//   Dispatch: ((n_queries * d_model + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_copy_q(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D = params.d_model;
    let g = gid.x;
    if g >= params.n_queries * D { return; }
    query_state[g] = kv_state[g];
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 8: main_rsp_attn
//   Cross-attention: 64 queries attend to n_vit kv tokens.
//   Q: from query_state (output of copy_q).
//   K: kv_state slot 1  (written by proj_k).
//   V: kv_state slot 2  (written by proj_v).
//   Output: query_state (safe in-place: each (q_idx,h) reads & writes its own head slice).
//   RSP_HEAD_DIM = 224 (compile-time const).
//   Dispatch: ((n_queries * n_heads + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_attn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D       = params.d_model;
    let n_vit   = params.n_vit;
    let n_q     = params.n_queries;
    let n_heads = params.n_heads;
    let hd      = RSP_HEAD_DIM;    // 224

    let g = gid.x;
    if g >= n_q * n_heads { return; }

    let q_idx = g / n_heads;
    let h     = g % n_heads;
    let scale = 1.0f / sqrt(f32(hd));

    let k_slot_base = n_vit * D;       // kv_state slot 1
    let v_slot_base = 2u * n_vit * D;  // kv_state slot 2

    // Load Q for this (q_idx, head) — reads own head slice of query_state, no conflict
    var q_vec: array<f32, 224>;
    let q_base = q_idx * D + h * hd;
    for (var d = 0u; d < hd; d++) {
        q_vec[d] = query_state[q_base + d];
    }

    // Online softmax + V accumulation over all n_vit kv tokens
    var out_vec: array<f32, 224>;
    for (var d = 0u; d < hd; d++) { out_vec[d] = 0.0f; }
    var running_max = -1.0e30f;
    var running_sum = 0.0f;

    for (var kv = 0u; kv < n_vit; kv++) {
        let k_base = k_slot_base + kv * D + h * hd;
        var qk = 0.0f;
        for (var d = 0u; d < hd; d++) {
            qk += q_vec[d] * kv_state[k_base + d];
        }
        qk *= scale;

        let new_max = max(running_max, qk);
        let exp_old = exp(running_max - new_max);
        let exp_new = exp(qk - new_max);
        running_sum = running_sum * exp_old + exp_new;

        let v_base = v_slot_base + kv * D + h * hd;
        for (var d = 0u; d < hd; d++) {
            out_vec[d] = out_vec[d] * exp_old + exp_new * kv_state[v_base + d];
        }
        running_max = new_max;
    }

    let out_base = q_idx * D + h * hd;
    for (var d = 0u; d < hd; d++) {
        query_state[out_base + d] = out_vec[d] / running_sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 9: main_rsp_out_proj
//   Output projection + residual.
//   Canonical: out = W_out @ attn_out + b_out + ln_q(query_embeds)
//   The ln_q stash written by main_rsp_proj_q at kv_state[n_q*D + q*D + out_d]
//   is the correct residual — NOT raw query_embeds (which lack the LayerNorm).
//   Dispatch: ((n_queries * d_model + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_out_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D = params.d_model;
    let g = gid.x;
    if g >= params.n_queries * D { return; }

    let q     = g / D;
    let out_d = g % D;
    let q_base = q * D;

    var dot = 0.0f;
    let row_byte = offsets.attn_out_w + out_d * D * 2u;
    for (var in_d = 0u; in_d < D; in_d++) {
        dot += rsp_read_f16(row_byte + in_d * 2u) * query_state[q_base + in_d];
    }
    dot += rsp_read_f32(offsets.attn_out_b + out_d * 4u);

    // Residual: ln_q(query_embeds) stashed by main_rsp_proj_q.
    let residual = kv_state[params.n_queries * D + q_base + out_d];
    query_state[q_base + out_d] = residual + dot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 10: main_rsp_post_ln
//   Post-LayerNorm in-place on query_state.
//   Dispatch: ((n_queries + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_post_ln(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if q >= params.n_queries { return; }

    let D    = params.d_model;
    let base = q * D;

    var mean = 0.0f;
    for (var i = 0u; i < D; i++) { mean += query_state[base + i]; }
    mean /= f32(D);

    var variance = 0.0f;
    for (var i = 0u; i < D; i++) {
        let d = query_state[base + i] - mean;
        variance += d * d;
    }
    variance /= f32(D);
    let inv_std = 1.0f / sqrt(variance + params.ln_eps);

    for (var i = 0u; i < D; i++) {
        let x = query_state[base + i];
        let w = rsp_read_f32(offsets.ln_post_w + i * 4u);
        let b = rsp_read_f32(offsets.ln_post_b + i * 4u);
        query_state[base + i] = (x - mean) * inv_std * w + b;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 11: main_rsp_final_proj
//   Final linear projection: out = post_ln_query @ proj_w → kv_state slot 0.
//
//   Canonical (modeling_minicpmv.py): out = x @ self.proj
//   → out[q, out_d] = ∑_in x[q, in] × proj[in, out_d]
//
//   proj_w in GGUF: stored as nn.Parameter [d_model × d_model] in PyTorch
//   row-major order (NOT transposed like nn.Linear.weight).
//   Element proj[in, out_d] at flat offset (in * D + out_d).
//   Access pattern: column out_d — stride D between successive in_d reads.
//
//   Reads from query_state, writes to kv_state slot 0 — no data race.
//   OUTPUT: kv_state[0 .. n_queries * d_model] = 64 visual tokens [64 × 3584].
//   Dispatch: ((n_queries * d_model + 255) / 256, 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_rsp_final_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let D = params.d_model;
    let g = gid.x;
    if g >= params.n_queries * D { return; }

    let q     = g / D;
    let out_d = g % D;
    let q_base = q * D;

    // Column out_d access: proj_w[in_d, out_d] at byte (in_d * D + out_d) * 2.
    // This computes out = x @ proj_w  (not proj_w @ x).
    var dot = 0.0f;
    for (var in_d = 0u; in_d < D; in_d++) {
        dot += rsp_read_f16(offsets.proj_w + (in_d * D + out_d) * 2u) * query_state[q_base + in_d];
    }

    kv_state[q * D + out_d] = dot;    // → kv_state slot 0 (final output)
}
