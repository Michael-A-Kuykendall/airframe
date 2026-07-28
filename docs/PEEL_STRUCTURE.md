# Full Inference Peel Structure (authoritative)

**Status:** FROZEN contract for deep stack observability  
**Product path:** `BindlessPipeline::run_full_model_with_cache_state` + `sh_layer_v1.wgsl`  
**Qwen3-4B reference sizes:** `dim=2560`, `n_head=32`, `n_kv=8`, `head_dim=128`, `ffn_dim=9728`, `dim_q=4096`, `dim_kv=1024`  
**Default sample position:** last prompt token (`token_idx = batch_size - 1`)  
**Default stats per intercept:** `{ rms, first8, nan_count, checksum? }` — full vector optional under flag  

Any dump that claims “full peel” **must** implement every **REQUIRED** intercept below with the **correct buffer+layout**. Proxies are forbidden for REQUIRED points.

---

## 0. Global buffers (product path)

| Buffer | WGSL / host | Layout | Notes |
|--------|-------------|--------|-------|
| `activation` | `activation_in` binding 1 | `[batch_size, dim]` f32 row-major | Residual stream |
| `temp` | `temp_state` binding 2 | `[batch_size, temp_stride]` f32 | Scratch; **overwritten every stage** |
| `kv_k` / `kv_v` | bindings 7/8 | `[max_seq, n_kv, head_dim]` f32 | Written by QKV; K may be QK-normed in place |
| `norm_bank` | binding 5 | `[n_layer * slots * dim]` | slots=4 or 6 if qk_norm |
| `rope_table` | binding 6 | `[max_dist, rope_dim/2, 2]` cos/sin | Relative RoPE at attn score |
| `blob_*` | 0/10/11 | packed GGUF | Weights |
| `logits` | host after lm_head | `[n_vocab]` f32 | Last token only (product) |

**Offsets (last token):**
```
act_base         = token_idx * dim
temp_base        = token_idx * temp_stride
last_token_byte  = (batch_size - 1) * dim * 4
temp_last_byte   = (batch_size - 1) * temp_stride * 4
```

**temp_stride** = `spec.temp_buffer_size` (must fit: max of `dim + dim_q + …`, `2*ffn_dim`, attn stash).

---

## 1. Outside the layer loop (once per prefill)

| ID | Name | When | Buffer / source | Shape (last pos) | REQUIRED |
|----|------|------|-----------------|------------------|----------|
| **P0** | config | load | ModelSpec + GGUF meta | scalars | YES |
| **P1** | tokens | encode | tokenizer | `[T]` ids + pieces | YES |
| **P2** | embd_row | after dequant each token | embd row | `[dim]` | YES last; optional all |
| **P2b** | embd_batch | after all embds packed | activation initial | `[T, dim]` | YES (at least last row stats) |
| **P3** | prefill_params | before layers | CacheParams | `current_pos, seq_len, batch_size, max_seq_len` | YES (logged) |

---

## 2. Per transformer layer `L = 0 .. n_layer-1`

Order matches `inference.rs` dispatch. After each **kernel**, one intercept.

### 2.1 Attention half

| ID | Kernel (`sh_layer_v1.wgsl`) | Writes | Correct intercept (last token) | Shape | REQUIRED | Current TRACE honesty |
|----|----------------------------|--------|--------------------------------|-------|----------|----------------------|
| **L.S0** | *(input residual)* | — | `activation[act_base .. +dim]` **before** AttnNorm | `[dim]` | YES | STAGE `input` OK if from activation |
| **L.S1** | `main_attn_norm` | `temp[temp_base + 0 .. dim)` | `temp[temp_base .. +dim]` | `[dim]` | YES | `attn_norm` OK |
| **L.S2a** | `main_qkv` (Q) | `temp[temp_base + dim .. dim+dim_q)` | **Q slice only** `temp_base+dim` | `[dim_q]` = `[n_head*head_dim]` | YES | **BROKEN today** — TRACE reads `temp[0:dim]` (=S1) |
| **L.S2b** | `main_qkv` (K) | `kv_k[pos, :, :]` | K for this pos: `pos = current_pos + token_idx` | `[n_kv*head_dim]` | YES | **not captured** |
| **L.S2c** | `main_qkv` (V) | `kv_v[pos, :, :]` | V for this pos | `[n_kv*head_dim]` | YES | **not captured** |
| **L.S3a** | `main_qk_norm` (Q) if `qk_norm` | Q in temp in-place | same as S2a **after** norm | `[dim_q]` | YES if qk_norm | **BROKEN** same as S2a |
| **L.S3b** | `main_qk_norm` (K) if `qk_norm` | K in kv_k in-place | same as S2b after norm | `[dim_kv]` | YES if qk_norm | **not captured** |
| **L.S4** | `main_attn_out` | `temp[temp_base + 0 .. attn_dim)` context | `temp[temp_base .. +attn_dim]` | `[attn_dim]` = `[n_head*head_dim]` | YES | TRACE reads dim-only; OK if attn_dim==dim (Qwen3: 4096≠2560) → **WRONG SIZE** for Qwen3 |
| **L.S5a** | `main_attn_proj` (pre-residual) | stash `temp[temp_base + dim + attn_dim + i]` | stash slice | `[dim]` | YES | not captured |
| **L.S5b** | `main_attn_proj` (post residual-add) | `activation` | `activation[act_base .. +dim]` | `[dim]` | YES | STAGE `attn_proj` OK |
| **L.S6** | `main_post_attn_norm` | activation if Gemma | activation | `[dim]` | if `post_norm` | no-op Qwen3 |

### 2.2 FFN half

| ID | Kernel | Writes | Correct intercept | Shape | REQUIRED | Current honesty |
|----|--------|--------|-------------------|-------|----------|-----------------|
| **L.S7** | `main_ffn_norm` (if `quant_ffn_down != Q4_K`) | normed into temp or used inline | post-attn residual or ffn-normed stream | `[dim]` | YES (gated path) | TRACE `ffn_norm` reads temp[0:dim] — may be **stale attn context** not ffn-norm |
| **L.S8a** | `main_ffn_proj` gate | `temp[temp_base + 0 .. ffn_dim)` SiLU/GELU | gate slice | `[ffn_dim]` | YES | TRACE reads dim only — **WRONG** |
| **L.S8b** | `main_ffn_proj` up | `temp[temp_base + ffn_dim .. 2*ffn_dim)` | up slice | `[ffn_dim]` | YES | **not captured** |
| **L.S9a** | `main_ffn_down` (pre-residual) | down proj only (in local) | would need stash | `[dim]` | optional | — |
| **L.S9b** | `main_ffn_down` (post residual) | `activation` | `activation[act_base .. +dim]` | `[dim]` | YES | STAGE `ffn_down` OK |
| **L.S10** | `main_post_ffw_norm` | activation if Gemma | activation | `[dim]` | if post_norm | no-op Qwen3 |
| **L.R** | layer residual out | = S9b (or S10) | same as S9b | `[dim]` | YES (= L3 residual) | stack_dump `StackLayerSnap` OK |

### 2.3 Weight tensors (GGUF, per layer — L5)

Read via dequant of **one row or full RMS of dequanted matrix** (full matrix optional / expensive).

| ID | Tensor name | Role | Shape (Qwen3-4B) | REQUIRED |
|----|-------------|------|------------------|----------|
| **L.W.attn_norm** | `blk.L.attn_norm.weight` | RMS γ | `[dim]` | YES (or norm_bank slot) |
| **L.W.q** | `blk.L.attn_q.weight` | Q proj | `[dim, dim_q]` stored | YES dequant gate already G1; peel = optional row probe |
| **L.W.k** | `blk.L.attn_k.weight` | K proj | `[dim, dim_kv]` | YES |
| **L.W.v** | `blk.L.attn_v.weight` | V proj | `[dim, dim_kv]` | YES |
| **L.W.o** | `blk.L.attn_output.weight` | O proj | `[dim_q, dim]` | YES |
| **L.W.q_norm** | `blk.L.attn_q_norm.weight` | QK-norm | `[head_dim]` | YES if qk_norm |
| **L.W.k_norm** | `blk.L.attn_k_norm.weight` | QK-norm | `[head_dim]` | YES if qk_norm |
| **L.W.ffn_norm** | `blk.L.ffn_norm.weight` | FFN RMS γ | `[dim]` | YES |
| **L.W.gate** | `blk.L.ffn_gate.weight` | gate | `[dim, ffn_dim]` | YES |
| **L.W.up** | `blk.L.ffn_up.weight` | up | `[dim, ffn_dim]` | YES |
| **L.W.down** | `blk.L.ffn_down.weight` | down | `[ffn_dim, dim]` | YES |

Activation peel (S*) is the **debug priority** for finite gibberish; weight peel is secondary after G1 quant_verify green.

---

## 3. After all layers (once)

| ID | Name | Source | Shape | REQUIRED |
|----|------|--------|-------|----------|
| **F0** | residual_pre_final_norm | activation last token post L-1 | `[dim]` | YES |
| **F1** | residual_post_final_norm | after `output_norm` | `[dim]` | YES |
| **F2** | logits | lm_head / tied embd | `[n_vocab]` | YES |
| **F3** | logits.top_k | host top-k | `k × {id,piece,logit}` | YES |
| **F4** | logits.argmax | host | one token | YES |

---

## 4. Decode step (each step `t`)

| ID | Name | Source | REQUIRED |
|----|------|--------|----------|
| **D0** | in_token | sampled / teacher | YES |
| **D1** | embd | dequant row | YES |
| **D2** | all **L.S\*** for layer loop with `batch_size=1` | same as §2 | YES (same structure) |
| **D3** | F0–F4 | same as §3 | YES |
| **D4** | out_token | sample | YES |

---

## 5. JSON shape under `layers[L]` (required for “full peel”)

```json
{
  "layer_idx": 0,
  "position": 4,
  "residual_in":  { "rms": 0, "first8": [], "nan_count": 0 },
  "stages": {
    "attn_norm":     { "rms": 0, "first8": [], "nan_count": 0, "sampled": "real", "buffer": "temp", "offset_elems": 0, "count": 2560 },
    "q":             { "rms": 0, "first8": [], "nan_count": 0, "sampled": "real", "buffer": "temp", "offset_elems": 2560, "count": 4096 },
    "k":             { "rms": 0, "first8": [], "nan_count": 0, "sampled": "real", "buffer": "kv_k", "pos": 4, "count": 1024 },
    "v":             { "rms": 0, "first8": [], "nan_count": 0, "sampled": "real", "buffer": "kv_v", "pos": 4, "count": 1024 },
    "q_norm":        { "...": "same as q after qk_norm", "count": 4096 },
    "k_norm":        { "...": "same as k after qk_norm", "count": 1024 },
    "attn_ctx":      { "buffer": "temp", "offset_elems": 0, "count": 4096 },
    "attn_proj_delta": { "buffer": "temp_stash", "count": 2560 },
    "attn_residual": { "buffer": "activation", "count": 2560 },
    "ffn_gate":      { "buffer": "temp", "offset_elems": 0, "count": 9728 },
    "ffn_up":        { "buffer": "temp", "offset_elems": 9728, "count": 9728 },
    "ffn_residual":  { "buffer": "activation", "count": 2560 }
  },
  "residual_out": { "rms": 0, "first8": [], "nan_count": 0 }
}
```

**Qwen3-4B counts are fixed above; other models substitute `dim`, `dim_q`, `dim_kv`, `ffn_dim` from config.**

---

## 6. What is NOT full peel (current gaps)

| Claim | Reality |
|-------|---------|
| stack_dump L3 residual only | Residual out only — **not** full stage peel |
| STAGE-TRACE `qkv`/`qk_norm` | Reads `temp[0:dim]` after QKV — **still attn_norm-sized residual, not Q** |
| STAGE-TRACE `attn_out` for Qwen3 | Uses `dim` not `attn_dim` — **truncates / wrong** |
| STAGE-TRACE `ffn_proj` | Uses `dim` not `2*ffn_dim` — **wrong** |
| ptensor product path | Empty / unsupported |

These gaps are **implementation debt**, not optional polish.

---

## 7. Implementation order (must match this structure)

1. Fix capture offsets/sizes for **S2a, S2b, S2c, S3a, S3b, S4, S8a, S8b** (honesty first)  
2. Emit full `stages` object for **every** layer (or layer0 first, then all)  
3. Wire product path only (`run_full_model_*`)  
4. Extend schema + `stack_dump_gpu`  
5. Compare tool: stage-level first diverge  

---

## 8. Acceptance for “peel complete”

- [x] Every **REQUIRED** ID in §§1–3 appears in `airframe.stack.json` with `sampled: "real"` and correct `count`  
- [x] Qwen3-4B: `q.count=4096`, `k.count=1024`, `attn_ctx.count=4096`, `ffn_gate.count=9728`  
- [x] No stage reuses another stage’s buffer without `sampled` ≠ `real`  
- [x] `force_yield` before every GPU readback  
- [ ] Schema validation passes  
- [ ] compare can name first diverge as e.g. `L0.S2a` not just `L6`  

---

## END
