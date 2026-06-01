// ─────────────────────────────────────────────────────────────────────────────
// sh_vit_layer.wgsl  —  SigLIP ViT transformer block (GPU dispatch)
// Model: MiniCPM-V-2.6 vision tower (SigLIP-So400M, 27 blocks)
//   hidden_dim=1152, n_heads=16, head_dim=72, mlp_dim=4304, n_tokens=1024
//
// Bind group (shared across all 8 entry points):
//   @binding(0) vit_blob    [array<u32>]   raw mmproj GGUF file bytes (read-only)
//   @binding(1) activations [array<f32>]   [n_tokens × hidden_dim] residual (r/w)
//   @binding(2) temp_kqv    [array<f32>]   [n_tokens × hidden_dim × 4] scratch
//       slot 0 [0..n_t×H)        LayerNorm output  /  attention output
//       slot 1 [n_t×H..2n_t×H)   Q projections
//       slot 2 [2n_t×H..3n_t×H)  K projections
//       slot 3 [3n_t×H..4n_t×H)  V projections
//   @binding(3) temp_ffn    [array<f32>]   [n_tokens × mlp_dim] GELU output
//   @binding(4) offsets     VitBlockOffsets  per-block weight byte offsets
//   @binding(5) params      VitParams        model dimensions (constant per model)
//
// Kernel order per ViT block:
//   1. main_vit_ln1        LayerNorm (activations → temp_kqv slot 0)
//   2. main_vit_qkv        Project Q/K/V (slot 0 → slots 1/2/3)
//   3. main_vit_attn       Bidirectional MHA (slots 1/2/3 → slot 0), online softmax
//   4. main_vit_attn_proj  Output projection + residual (slot 0 → activations)
//   5. main_vit_ln2        LayerNorm (activations → temp_kqv slot 0)
//   6. main_vit_ffn_up     FFN up + GELU  (slot 0 → temp_ffn)
//   7. main_vit_ffn_down   FFN down + residual (temp_ffn → activations)
//   8. main_vit_post_ln    Post-block LayerNorm in-place on activations (final pass only)
//      Uses ln1_w / ln1_b byte offsets — caller sets these to post-LN weight offsets.
// ─────────────────────────────────────────────────────────────────────────────

// Compile-time constant for SigLIP-So400M head dimension (1152 / 16 heads = 72)
const HEAD_DIM: u32 = 72u;

// ─── Uniform structs ──────────────────────────────────────────────────────────

/// Per-block weight byte offsets (absolute from start of mmproj GGUF file).
/// All F16 weight matrices; all F32 bias vectors and LN weights.
/// Size = 16 × 4 = 64 bytes (multiple of 16 ✓).
struct VitBlockOffsets {
    ln1_w:    u32,   // [hidden_dim] F32  — LayerNorm 1 weight (also used by main_vit_post_ln)
    ln1_b:    u32,   // [hidden_dim] F32  — LayerNorm 1 bias
    attn_q_w: u32,   // [hidden_dim × hidden_dim] F16  — Q weight
    attn_q_b: u32,   // [hidden_dim] F32  — Q bias
    attn_k_w: u32,   // [hidden_dim × hidden_dim] F16  — K weight
    attn_k_b: u32,   // [hidden_dim] F32  — K bias
    attn_v_w: u32,   // [hidden_dim × hidden_dim] F16  — V weight
    attn_v_b: u32,   // [hidden_dim] F32  — V bias
    attn_o_w: u32,   // [hidden_dim × hidden_dim] F16  — out-projection weight
    attn_o_b: u32,   // [hidden_dim] F32  — out-projection bias
    ln2_w:    u32,   // [hidden_dim] F32  — LayerNorm 2 weight
    ln2_b:    u32,   // [hidden_dim] F32  — LayerNorm 2 bias
    ffn_up_w: u32,   // [hidden_dim × mlp_dim] F16  — FFN up weight
    ffn_up_b: u32,   // [mlp_dim] F32  — FFN up bias
    ffn_dn_w: u32,   // [mlp_dim × hidden_dim] F16  — FFN down weight
    ffn_dn_b: u32,   // [hidden_dim] F32  — FFN down bias
}

/// Model dimensions — same for every kernel invocation on this model.
/// Size = 8 × 4 = 32 bytes (multiple of 16 ✓).
struct VitParams {
    hidden_dim: u32,   // 1152
    n_heads:    u32,   // 16
    head_dim:   u32,   // 72  (redundant with HEAD_DIM const, but allows runtime checks)
    mlp_dim:    u32,   // 4304
    n_tokens:   u32,   // 1024 (patches per tile)
    ln_eps:     f32,   // 1e-6
    pad0:       u32,
    pad1:       u32,
}

// ─── Bindings ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       vit_blob:    array<u32>;
@group(0) @binding(1) var<storage, read_write> activations: array<f32>;
@group(0) @binding(2) var<storage, read_write> temp_kqv:    array<f32>;
@group(0) @binding(3) var<storage, read_write> temp_ffn:    array<f32>;
@group(0) @binding(4) var<uniform>             offsets:     VitBlockOffsets;
@group(0) @binding(5) var<uniform>             params:      VitParams;

// ─── Byte-level accessors (little-endian, same as gguf_blob in sh_layer_v1) ──

fn vit_read_byte(byte_off: u32) -> u32 {
    let word  = vit_blob[byte_off >> 2u];
    let shift = (byte_off & 3u) << 3u;
    return (word >> shift) & 0xFFu;
}

/// Read a 4-byte-aligned F32 from the GGUF blob.
fn vit_read_f32(byte_off: u32) -> f32 {
    return bitcast<f32>(vit_blob[byte_off >> 2u]);
}

/// Read an IEEE 754 F16 at byte_off and convert to F32.
/// Identical logic to dequant_f16_at() in sh_layer_v1.wgsl.
fn vit_read_f16(byte_off: u32) -> f32 {
    let word   = vit_blob[byte_off >> 2u];
    let shift  = (byte_off & 2u) << 3u;       // 0 or 16
    let bits   = (word >> shift) & 0xFFFFu;
    let sign   = (bits >> 15u) & 1u;
    let exp    = (bits >> 10u) & 0x1Fu;
    let mant   = bits & 0x3FFu;
    var f32bits: u32;
    if exp == 0u {
        if mant == 0u {
            f32bits = sign << 31u;                                        // ±zero
        } else {
            // Denormal: renormalize
            var m = mant;
            var e = 0u;
            loop {
                e += 1u;
                m <<= 1u;
                if (m & 0x400u) != 0u { break; }
            }
            f32bits = (sign << 31u) | ((127u - 15u + 1u - e) << 23u) | ((m & 0x3FFu) << 13u);
        }
    } else if exp == 31u {
        f32bits = (sign << 31u) | (0xFFu << 23u) | (mant << 13u);        // inf / nan
    } else {
        f32bits = (sign << 31u) | ((exp + 127u - 15u) << 23u) | (mant << 13u);
    }
    return bitcast<f32>(f32bits);
}

// ─── Helper: LayerNorm for one token, writing to temp_kqv slot 0 ─────────────

fn layernorm_one_token(tok: u32, ln_w_byte: u32, ln_b_byte: u32) {
    let H    = params.hidden_dim;
    let base = tok * H;

    // Pass 1 — mean
    var mean = 0.0f;
    for (var i = 0u; i < H; i++) {
        mean += activations[base + i];
    }
    mean /= f32(H);

    // Pass 2 — variance
    var variance = 0.0f;
    for (var i = 0u; i < H; i++) {
        let d = activations[base + i] - mean;
        variance += d * d;
    }
    variance /= f32(H);
    let inv_std = 1.0f / sqrt(variance + params.ln_eps);

    // Pass 3 — normalize + affine → slot 0
    for (var i = 0u; i < H; i++) {
        let x = activations[base + i];
        let w = vit_read_f32(ln_w_byte + i * 4u);
        let b = vit_read_f32(ln_b_byte + i * 4u);
        temp_kqv[base + i] = (x - mean) * inv_std * w + b;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: main_vit_ln1
//   Dispatch: ((n_tokens + 255) / 256, 1, 1)
//   One invocation per token.  Writes to temp_kqv slot 0.
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_vit_ln1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tok = gid.x;
    if tok >= params.n_tokens { return; }
    layernorm_one_token(tok, offsets.ln1_w, offsets.ln1_b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: main_vit_qkv
//   Dispatch: ((n_tokens * hidden_dim * 3 + 255) / 256, 1, 1)
//   One invocation per output element (proj_type, token, out_dim).
//   Reads from temp_kqv slot 0 (LN output); writes to slots 1/2/3 (Q/K/V).
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_vit_qkv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let H     = params.hidden_dim;
    let work  = params.n_tokens * H;  // elements per projection
    let g     = gid.x;
    if g >= work * 3u { return; }

    let proj  = g / work;        // 0 = Q, 1 = K, 2 = V
    let tok   = (g % work) / H;
    let out_d = g % H;

    // Select weight/bias byte offsets via branchless select
    let wkv = select(offsets.attn_v_w, offsets.attn_k_w, proj == 1u);
    let bkv = select(offsets.attn_v_b, offsets.attn_k_b, proj == 1u);
    let w_byte = select(wkv, offsets.attn_q_w, proj == 0u);
    let b_byte = select(bkv, offsets.attn_q_b, proj == 0u);

    // Dot product: weight row[out_d] (F16) × ln_out[tok] (F32)
    var dot = 0.0f;
    let row_byte = w_byte + out_d * H * 2u;   // F16 = 2 bytes / element
    let ln_base  = tok * H;                    // slot 0 base
    for (var in_d = 0u; in_d < H; in_d++) {
        dot += vit_read_f16(row_byte + in_d * 2u) * temp_kqv[ln_base + in_d];
    }
    dot += vit_read_f32(b_byte + out_d * 4u);

    // Write: Q → slot 1, K → slot 2, V → slot 3
    let out_slot_base = (proj + 1u) * work;
    temp_kqv[out_slot_base + tok * H + out_d] = dot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: main_vit_attn
//   Dispatch: ((n_tokens * n_heads + 255) / 256, 1, 1)
//   One invocation per (tok_q, head) pair.
//   Online softmax over all n_tokens key positions — no causal mask.
//   Reads Q from slot 1, K from slot 2, V from slot 3.
//   Writes normalized attention output to slot 0.
//
//   HEAD_DIM = 72 is a compile-time const.  Function-scope arrays are safe.
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_vit_attn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let H       = params.hidden_dim;
    let n_tok   = params.n_tokens;
    let n_heads = params.n_heads;
    let hd      = HEAD_DIM;          // 72 — compile-time constant

    let g = gid.x;
    if g >= n_tok * n_heads { return; }

    let tok_q  = g / n_heads;
    let h      = g % n_heads;
    let work   = n_tok * H;          // elements per KQV slot
    let scale  = 1.0f / sqrt(f32(hd));

    // Load Q vector for this (tok_q, head) pair — HEAD_DIM floats
    var q_vec: array<f32, 72>;
    let q_base = work + tok_q * H + h * hd;   // slot 1
    for (var d = 0u; d < hd; d++) {
        q_vec[d] = temp_kqv[q_base + d];
    }

    // Online softmax + weighted V accumulation — single pass over all kv tokens
    var out_vec: array<f32, 72>;
    for (var d = 0u; d < hd; d++) { out_vec[d] = 0.0f; }
    var running_max = -1.0e30f;
    var running_sum = 0.0f;

    let k_slot = 2u * work;    // slot 2 base
    let v_slot = 3u * work;    // slot 3 base

    for (var kv = 0u; kv < n_tok; kv++) {
        // Q · K for head h
        let k_base = k_slot + kv * H + h * hd;
        var qk = 0.0f;
        for (var d = 0u; d < hd; d++) {
            qk += q_vec[d] * temp_kqv[k_base + d];
        }
        qk *= scale;

        // Online softmax update (numerically stable)
        let new_max  = max(running_max, qk);
        let exp_old  = exp(running_max - new_max);
        let exp_new  = exp(qk - new_max);
        running_sum  = running_sum * exp_old + exp_new;

        // Accumulate V[kv][h] weighted by exp_new
        let v_base = v_slot + kv * H + h * hd;
        for (var d = 0u; d < hd; d++) {
            out_vec[d] = out_vec[d] * exp_old + exp_new * temp_kqv[v_base + d];
        }
        running_max = new_max;
    }

    // Normalize and write to slot 0
    let out_base = tok_q * H + h * hd;   // slot 0 base
    for (var d = 0u; d < hd; d++) {
        temp_kqv[out_base + d] = out_vec[d] / running_sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: main_vit_attn_proj
//   Dispatch: ((n_tokens * hidden_dim + 255) / 256, 1, 1)
//   One invocation per (token, out_dim).
//   Reads attn output from slot 0; writes (out_proj + residual) to activations.
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_vit_attn_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let H = params.hidden_dim;
    let g = gid.x;
    if g >= params.n_tokens * H { return; }

    let tok   = g / H;
    let out_d = g % H;

    var dot = 0.0f;
    let row_byte  = offsets.attn_o_w + out_d * H * 2u;    // F16 weight row
    let attn_base = tok * H;                               // slot 0 base
    for (var in_d = 0u; in_d < H; in_d++) {
        dot += vit_read_f16(row_byte + in_d * 2u) * temp_kqv[attn_base + in_d];
    }
    dot += vit_read_f32(offsets.attn_o_b + out_d * 4u);

    // Residual add
    activations[tok * H + out_d] += dot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: main_vit_ln2
//   Dispatch: ((n_tokens + 255) / 256, 1, 1)
//   Same as ln1 but uses ln2 weight/bias offsets.
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_vit_ln2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tok = gid.x;
    if tok >= params.n_tokens { return; }
    layernorm_one_token(tok, offsets.ln2_w, offsets.ln2_b);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6: main_vit_ffn_up
//   Dispatch: ((n_tokens * mlp_dim + 255) / 256, 1, 1)
//   One invocation per (token, mlp_dim).
//   Reads LN2 output from slot 0; applies up projection + GELU → temp_ffn.
//   SigLIP uses standard GELU (not SwiGLU — no separate gate weight).
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_vit_ffn_up(@builtin(global_invocation_id) gid: vec3<u32>) {
    let H = params.hidden_dim;
    let M = params.mlp_dim;
    let g = gid.x;
    if g >= params.n_tokens * M { return; }

    let tok   = g / M;
    let out_d = g % M;

    var dot = 0.0f;
    let row_byte = offsets.ffn_up_w + out_d * H * 2u;   // F16, row out_d
    let ln2_base = tok * H;
    for (var in_d = 0u; in_d < H; in_d++) {
        dot += vit_read_f16(row_byte + in_d * 2u) * temp_kqv[ln2_base + in_d];
    }
    dot += vit_read_f32(offsets.ffn_up_b + out_d * 4u);

    // GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let c0   = 0.7978845608028654f;    // sqrt(2/π)
    let c1   = 0.044715f;
    let gelu = dot * 0.5f * (1.0f + tanh(c0 * (dot + c1 * dot * dot * dot)));

    temp_ffn[tok * M + out_d] = gelu;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7: main_vit_ffn_down
//   Dispatch: ((n_tokens * hidden_dim + 255) / 256, 1, 1)
//   One invocation per (token, hidden_dim).
//   Reads GELU output from temp_ffn; applies down projection + residual add.
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_vit_ffn_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let H = params.hidden_dim;
    let M = params.mlp_dim;
    let g = gid.x;
    if g >= params.n_tokens * H { return; }

    let tok   = g / H;
    let out_d = g % H;

    var dot = 0.0f;
    let row_byte  = offsets.ffn_dn_w + out_d * M * 2u;   // F16, row out_d
    let gelu_base = tok * M;
    for (var in_d = 0u; in_d < M; in_d++) {
        dot += vit_read_f16(row_byte + in_d * 2u) * temp_ffn[gelu_base + in_d];
    }
    dot += vit_read_f32(offsets.ffn_dn_b + out_d * 4u);

    // Residual add
    activations[tok * H + out_d] += dot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 8: main_vit_post_ln
//   Dispatch: ((n_tokens + 255) / 256, 1, 1)
//   Applied once after all 27 blocks.  In-place on activations.
//   Caller sets offsets.ln1_w / offsets.ln1_b to the post-LN weight offsets.
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(256, 1, 1)
fn main_vit_post_ln(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tok = gid.x;
    if tok >= params.n_tokens { return; }

    let H    = params.hidden_dim;
    let base = tok * H;

    var mean = 0.0f;
    for (var i = 0u; i < H; i++) { mean += activations[base + i]; }
    mean /= f32(H);

    var variance = 0.0f;
    for (var i = 0u; i < H; i++) {
        let d = activations[base + i] - mean;
        variance += d * d;
    }
    variance /= f32(H);
    let inv_std = 1.0f / sqrt(variance + params.ln_eps);

    // Normalize + affine in-place (uses ln1_w / ln1_b — post-LN offsets set by caller)
    for (var i = 0u; i < H; i++) {
        let x = activations[base + i];
        let w = vit_read_f32(offsets.ln1_w + i * 4u);
        let b = vit_read_f32(offsets.ln1_b + i * 4u);
        activations[base + i] = (x - mean) * inv_std * w + b;
    }
}
