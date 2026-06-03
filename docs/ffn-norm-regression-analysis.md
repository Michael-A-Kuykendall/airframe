# FFN Norm Regression — Session Handoff

> **For next session:** Start here. This is the complete, accurate state. Do not re-read the chat.
> **Date:** 2026-06-02 | **Branch:** `feat/vision-multimodal` | **HEAD:** `be66694`

---

## 1. Current Status

| Metric | State |
|--------|-------|
| Build | ✅ `cargo build --release` — clean |
| Smoke test | ❌ 7/10 PASS — phi-2 WEAK, starcoder2 WEAK, gpt2 FAIL |
| Committed HEAD baseline | ✅ 10/10 PASS (`artifacts/model_smoke/smoke_20260531_155033.log`) |
| Uncommitted files | 14 files (`git diff --stat HEAD`) — **do not stash or discard** |

---

## 2. What Changed (all uncommitted)

All changes are in `src/backend/bindless/sh_layer_v1.wgsl` plus its Rust wiring.

### WGSL — new/modified kernels

**`var<workgroup> wg_act: array<f32, 256u>`** added at module scope (after binding declarations). Used by tiled kernels.

**`main_attn_proj`** — Q4_K path now uses tiled cooperative-load GEMM via `wg_act`. Non-Q4_K scalar fallback unchanged. ✅ All models pass this kernel.

**`main_ffn_down`** — Q4_K path now uses tiled cooperative-load GEMM via `wg_act`. Non-Q4_K scalar fallback unchanged. ✅ All models pass this kernel.

**`main_ffn_norm` (NEW KERNEL)** — dispatched `(1, batch_size, 1)`, 256 threads per workgroup. Does cooperative RMSNorm:
1. Each thread partial-sums `activation_in` squares over its stride-256 slice into `wg_act[lx]`
2. Tree reduction 256→1, result in `wg_act[0]`
3. Broadcasts `rms`, writes `activation_in[col] * rms * norm_bank[(layer_idx*4+1)*dim + col]` → `temp_state[temp_base + ffn_dim*2 + col]`

**`main_ffn_proj` (MODIFIED)** — **no longer computes RMSNorm inline**. Reads pre-normed activations from `temp_state[temp_base + ffn_dim*2 + col]` in every quant branch. This is the regression site.

### Rust wiring

`pipeline/mod.rs` — `layer_pipeline_ffn_norm: wgpu::ComputePipeline` added (line 130, constructed line 689).

`pipeline/inference.rs` — per-layer compute pass sequence is now:
```
FFNNorm  dispatch_workgroups(1, batch_size, 1)        ← new
FFNProj  dispatch_workgroups(wg_ffn, batch_size, 1)   ← reads FFNNorm stash
FFNDown  dispatch_workgroups(wg_dim, batch_size, 1)
```
Both passes are in the same `CommandEncoder` → **ordering is guaranteed, no missing barrier**.

---

## 3. The Regression

### Failing models
| Model | arch | dim | ffn_dim | quant | Symptom |
|-------|------|-----|---------|-------|---------|
| phi-2 | phi2 | 2560 | 10240 | Q4_K | `[PAD51199]` repeated |
| starcoder2-3b | starcoder2 | 3072 | 12288 | Q4_K | `sdbsdb...` repeated |
| gpt2 | gpt2 | 768 | 3072 | Q4_K | empty response |

### Passing models
| Model | arch | dim | ffn_dim | quant |
|-------|------|-----|---------|-------|
| TinyLlama-1.1B | llama | 2048 | 5632 | Q4_0 |
| Llama-3.2-1B | llama | 2048 | 8192 | Q4_K |
| Llama-3.2-3B | llama | 3072 | 8192 | Q4_K |

**The pattern: every failing model is non-LLaMA. Every passing model is `arch=llama`.**

---

## 4. Root Cause — What Was Ruled Out

- **`ffn_dim` not divisible by 256:** 10240, 12288, 3072 are all exact multiples of 256. Not the cause.
- **`temp_state` overflow for gpt2:** `ffn_norm_base = 6144`, stash range `6144..6911`, `temp_stride = max(3072, 6912) = 6912`. Fits exactly within bounds. Not the cause.
- **Missing GPU synchronization:** `main_ffn_norm` and `main_ffn_proj` are separate `begin_compute_pass` blocks in the same encoder. WebGPU guarantees sequential ordering within one encoder. Not the cause.
- **`wg_act` contamination between dispatches:** WebGPU zero-initializes `var<workgroup>` per dispatch. Not the cause.

## 4b. Root Cause — What Is Still Open

The honest answer: **exact mechanism not confirmed.** Two live hypotheses:

**Hypothesis A — phi2 parallel attention architecture.** phi-2 uses a parallel formulation where the same pre-norm residual feeds both attention AND FFN simultaneously. The `main_ffn_norm` kernel applies RMSNorm using `norm_bank slot 1`, but for phi-2 the FFN input should NOT be separately re-normed — it was already normed upstream for attention. Effect: the FFN receives double-normed input, producing garbage. This would explain WEAK (not total zero) output.

**Hypothesis B — numerical accumulation order.** The cooperative reduction in `main_ffn_norm` accumulates partial sums in a different order than the serial loop in baseline `main_ffn_proj`. FP32 is non-associative. For LLaMA the difference is sub-threshold; for these three models the architecture is more sensitive and the shifted `rms` value causes coherent decode failure.

Both hypotheses lead to the same fix.

---

## 5. The Fix — Option A (do this first)

**Restore `main_ffn_proj` to the committed baseline: inline RMSNorm, reads directly from `activation_in`.**

The `main_ffn_norm` kernel stays in the WGSL (Rust already has the pipeline), but `main_ffn_proj` ignores its output. The FFNNorm dispatch becomes a harmless no-op.

### Exact replacement

In `src/backend/bindless/sh_layer_v1.wgsl`, replace the entire `fn main_ffn_proj` body (currently at line 865) with this verbatim copy from committed HEAD:

```wgsl
@compute @workgroup_size(256, 1, 1)
fn main_ffn_proj(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ffn_dim = params.ffn_dim;
    let idx = global_id.x;
    let token_idx = global_id.y;

    if (idx >= ffn_dim * 2u || token_idx >= cache_params.batch_size) { return; }

    let act_base  = token_idx * params.dim;
    let temp_base = token_idx * params.temp_stride;

    // Inline RMSNorm — reads activation_in directly (no stash dependency).
    var sum_sq = 0.0;
    for (var i = 0u; i < params.dim; i++) {
        let val = activation_in[act_base + i];
        sum_sq += val * val;
    }
    let rms = inverseSqrt(sum_sq / f32(params.dim) + params.rms_eps);

    var weight_off: u32;
    var row_idx = idx;
    if (idx < ffn_dim) {
        weight_off = offsets.ffn_gate;
    } else {
        weight_off = offsets.ffn_up;
        row_idx = idx - ffn_dim;
    }

    var dot = 0.0;
    let norm_offset_base = (offsets.layer_idx * 4u + 1u) * params.dim;

    if ((params.quant_type & 0xFFu) == 14u) { // Q6_K
        let bpr = params.dim / 256u;
        let row_start = weight_off + (row_idx * bpr * 210u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 210u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q6k_elem(bb, e);
            }
        }
    } else if ((params.quant_type & 0xFFu) == 13u) { // Q5_K
        let bpr = params.dim / 256u;
        let row_start = weight_off + (row_idx * bpr * 176u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 176u;
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q5k_elem(bb, e);
            }
        }
    } else if ((params.quant_type & 0xFFu) == 12u) { // Q4_K
        let blocks_per_row_k = params.dim / 256u;
        let row_start_byte_k = weight_off + (row_idx * blocks_per_row_k * 144u);
        for (var b = 0u; b < blocks_per_row_k; b++) {
            let block_base_k = row_start_byte_k + (b * 144u);
            for (var e = 0u; e < 256u; e++) {
                let col = b * 256u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q4k_elem(block_base_k, e);
            }
        }
    } else if ((params.quant_type & 0xFFu) == 8u) { // Q8_0
        let bpr = params.dim / 32u;
        let row_start = weight_off + (row_idx * bpr * 34u);
        for (var b = 0u; b < bpr; b++) {
            let bb = row_start + b * 34u;
            for (var e = 0u; e < 32u; e++) {
                let col = b * 32u + e;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                dot += val_x * dequant_q8_0_elem(bb, e);
            }
        }
    } else if ((params.quant_type & 0xFFu) == 1u) { // F16
        for (var col = 0u; col < params.dim; col++) {
            let w_byte = weight_off + (row_idx * params.dim + col) * 2u;
            let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
            dot += val_x * dequant_f16_at(w_byte);
        }
    } else if ((params.quant_type & 0xFFu) == 0u) { // F32
        for (var col = 0u; col < params.dim; col++) {
            let w_idx = weight_off / 4u + row_idx * params.dim + col;
            let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
            dot += val_x * bitcast<f32>(gguf_blob[w_idx]);
        }
    } else { // Q4_0
        let blocks_per_row = params.dim / 32u;
        let row_start_byte = weight_off + (row_idx * blocks_per_row * 18u);
        for (var b = 0u; b < blocks_per_row; b++) {
            let block_base = row_start_byte + (b * 18u);
            let scale_idx = block_base / 4u;
            let scale_packed = extractBits(gguf_blob[scale_idx], (block_base % 4u) * 8u, 16u);
            let scale = unpack2x16float(scale_packed).x;
            let qs_byte_start = block_base + 2u;
            for (var i = 0u; i < 32u; i++) {
                let col = b * 32u + i;
                let val_x = activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col];
                let byte_idx = i % 16u;
                let qs_idx = qs_byte_start + byte_idx;
                let qs_word = gguf_blob[qs_idx / 4u];
                let qs_byte = extractBits(qs_word, (qs_idx % 4u) * 8u, 8u);
                let nib = select((qs_byte & 0x0Fu), (qs_byte >> 4u), i >= 16u);
                dot += val_x * (f32(nib) - 8.0) * scale;
            }
        }
    }

    if (idx < ffn_dim) {
        var activated: f32;
        if (params.attn_logit_softcap > 0.0) {
            activated = 0.5 * dot * (1.0 + tanh(0.7978845608f * (dot + 0.044715f * dot * dot * dot)));
        } else {
            activated = dot / (1.0 + exp(-dot));
        }
        temp_state[temp_base + idx] = activated;
    } else {
        temp_state[temp_base + idx] = dot;
    }
}
```

### Steps to execute
```
cargo build --release
powershell -ExecutionPolicy Bypass -File scripts/model_smoke_test.ps1
```
Expected: 10/10 PASS matching `smoke_20260531_155033.log`.

---

## 6. What NOT to Touch

| Kernel | Status | Action |
|--------|--------|--------|
| `main_attn_proj` | ✅ tiled Q4_K working | leave alone |
| `main_ffn_down` | ✅ tiled Q4_K working | leave alone |
| `main_ffn_norm` | stays in WGSL, dispatch stays in Rust | just stop reading its output in `main_ffn_proj` |
| Vision pipeline changes | unrelated | leave alone |

---

## 7. After Green — Option B (future, not now)

Keep `main_ffn_norm` as a real fast-path but gate its use:
- Add `use_precomputed_ffn_norm: u32` to `LayerParams` uniform struct
- Rust sets it `1` for `arch=llama`, `0` for all others
- `main_ffn_proj`: `if params.use_precomputed_ffn_norm == 1u` → read stash; else → inline norm

This preserves the cooperative norm speedup for LLaMA while keeping correctness everywhere.


