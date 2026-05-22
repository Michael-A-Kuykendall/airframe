# Release Status

## v2.0 — Gemma-2 Architecture Support (branch: `airframe-v2-gpu`)

### What Is Shipped in v2.0

Five commits on top of `airframe-v2-gpu` complete Gemma-2 architecture support:

| Commit | Change |
|--------|--------|
| `4dabeb9` | fix(gemma2): correct head_dim to 256 via `attention.key_length`; remove formula-dump diagnostics |
| `8b5f9ef` | feat(gemma2): `attn_logit_softcap` in LayerParams; `use_cpu_head` binding-limit guard for 256K-vocab output head |
| `32455a1` | feat(gemma2): `post_attention_norm`, `post_ffw_norm`, `final_logit_softcap` — all 9 files, +161/−15 lines |
| `d33048f` | fix: Gemma-2 inference — GELU activation (GeGLU), embedding scale sqrt(n_embd), logit softcap 30.0, parse_special tokenization for Control-type special tokens, end_of_turn stop |
| `f4ca4c1` | fix(pipeline): split compute passes for correct GPU memory barriers; add PostAttnNorm/PostFfwNorm dispatch |

### Validated Model Coverage (v2.0)

| Model | Arch | Size | Status |
|-------|------|------|--------|
| TinyLlama-1.1B Q4_0 | llama | 609 MB | ✅ PASSING — exact-story parity confirmed |
| Llama-3.2-1B Q4_K_M | llama | 771 MB | ✅ PASSING — 128K context, A100 |
| Llama-3.2-3B Q4_K_M | llama | 1.9 GB | ✅ PASSING — 128K context, 32 GB VRAM |
| Gemma-2-2B Q4_K_M | gemma2 | 1.6 GB | ✅ PASSING — post-norms, logit softcap, GELU, embedding scale, parse_special tokenization, cpu-head path |
| phi-2 Q4_K_M | phi2 | 1.7 GB | ✅ PASSING — smoke tested on A100 |
| starcoder2-3b Q4_K_M | starcoder2 | 1.8 GB | ✅ PASSING — smoke tested on A100 |
| gpt2 Q4_K_M | gpt2 | 108 MB | ✅ PASSING — completion model (no instruction template) |

### Known Limitations (v2.0 — deferred to v2.1+)

- **Llama-3.1-8B Q4_K_M** (4.6 GB) — blocked by single-buffer 2 GB wgpu cap (weight storage)
- **Gemma-2-27B Q4_K_M** (16 GB) — blocked by same 2 GB cap on weight storage; would also need chunked output head (4.4 GB vocab projection)
- **Gemma-2 sliding window attention** — Gemma-2 alternates local (window=4096) / global attention every other layer; current implementation treats all layers as global; architecturally correct for contexts ≤4096 tokens, incorrect for longer sessions
- **Llama-3.x full 128K context on consumer GPUs** — KV pre-allocation requires 40 GB VRAM; lazy KV allocation deferred to v2.1

### Active Release Gates (v2.0)

1. Gemma-2-2B smoke test passes on A100 (coherent output at temperature 0, correct stop on end_of_turn) ✅ — commits `d33048f`, `f4ca4c1`
2. `verify_norm_bank_extraction` test updated to 4-slot layout ✅
3. No regressions on TinyLlama exact-story path ✅
4. `feat/gemma2-post-norms` merged into `airframe-v2-gpu` ✅

### Quantization Shader Coverage

The Airframe WGSL bindless shader (`sh_layer_v1.wgsl`) implements all standard llama.cpp quantization formats:

| Format | GGML Type | Status |
|--------|-----------|--------|
| F32 | 0 | ✅ |
| F16 | 1 | ✅ |
| Q4_0 | 2 | ✅ |
| Q8_0 | 8 | ✅ |
| Q4_K (M/S) | 12 | ✅ |
| Q5_K (M/S) | 13 | ✅ |
| Q6_K | 14 | ✅ |

Q5_0 (type 6) and imatrix IQ formats (IQ2/IQ3/IQ4) are not yet implemented — tracked on the roadmap.