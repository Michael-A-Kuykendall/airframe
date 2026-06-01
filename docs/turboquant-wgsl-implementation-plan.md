# TurboQuant WGSL — Full Implementation Plan

**Status:** PLANNING — do not start work until approved  
**Branch target:** `feat/turboquant-wgsl`  
**Estimate:** 13 points  
**Goal:** Fused int4 KV cache compression in WGSL, vendor-agnostic, running live inside Airframe's existing inference loop. End state: the largest models that currently cannot fit on an RTX 3060 12GB can run at 8K+ context with quality matching the F32 baseline.

---

## Architectural Context (Read This First)

### Why F32 is a gift, not a liability

The prior art (Open-TQ-Metal) had to prove that quantizing from F16 to INT4 didn't destroy quality. Airframe already rejected F16 KV (`kv_cache.rs` comment: "FP16 precision insufficient — 9.65e-4 error, 960x threshold"). The KV cache is **F32 all the way down**. Quantizing F32→INT4 gives more headroom than F16→INT4 because the per-vector scale computation has a clean, high-precision input. The correctness bar is easier to hit, not harder.

### YaRN and TurboQuant are orthogonal

YaRN lives in `preflight.rs` — it modifies the **frequency domain** of the precomputed RoPE cos/sin table. TurboQuant operates on the **amplitude domain** of stored K/V activations. They touch completely separate data at completely separate pipeline stages.

Crucially: Airframe stores K and V **RAW** — no RoPE is baked into the cache values (`sh_rope_shift.wgsl` comment: *"This codebase uses RELATIVE RoPE — K and V are stored RAW in the cache"*). RoPE is applied on-the-fly as `RoPE(query_pos - key_pos)` during the attention dot product in `main_attn_out`. The sequence is:

```
Token N arrives
  → main_qkv:    compute raw K vector (no RoPE) → [NEW] quantize inline → write packed to cache
  → main_attn_out: for each cached position p:
      dequantize K[p] inline (registers only) → apply RoPE(N-p) inline → dot with Q
```

RoPE rotation is isometric (length-preserving). Quantization error ε on K is rotated but not amplified. YaRN modifying the rotation frequencies does not interact with the amplitude of the stored values. **They stack without interference.** The interaction test (Step 26) will confirm this empirically.

### Buffer layout change

Current per-layer KV cache (F32):
- `kv_cache_k`: `[max_seq × n_head_kv × head_dim]` F32 = `max_seq × n_head_kv × head_dim × 4` bytes
- `kv_cache_v`: same

New per-layer KV cache (INT4 packed + scale):
- `kv_cache_k_packed`: `[max_seq × n_head_kv × (head_dim/8)]` U32 (8 int4 values per u32, 2 nibbles per byte)
- `kv_cache_k_scales`: `[max_seq × n_head_kv]` F32 (one scale per head-vector)
- `kv_cache_v_packed`: same shape as k_packed
- `kv_cache_v_scales`: same shape as k_scales

At 8K context, TinyLlama (22 layers, 4 KV heads, head_dim=64):
- F32 baseline per layer: 8192 × 4 × 64 × 4 = **8.0 MB**
- INT4+scale per layer: 8192 × 4 × 8 × 4 (packed) + 8192 × 4 × 4 (scales) = **1.125 MB**
- Compression: **7.1x per layer**. Total for 22 layers: 176 MB → 24.75 MB.

### Complete file change map

| File | Change type |
|------|-------------|
| `src/backend/bindless/kv_cache.rs` | New buffer fields: packed + scales |
| `src/backend/bindless/sh_layer_v1.wgsl` | Kernel 1 write path (inline quant) + Kernel 2 read path (inline dequant) |
| ~~`src/backend/bindless/sh_layer_q4k.wgsl`~~ | **DEAD CODE — never compiled, no changes needed** |
| `src/backend/bindless/sh_rope_shift.wgsl` | Shift packed U32 + F32 scale buffers |
| `src/backend/bindless/pipeline_shift.rs` | Updated bind group layout for shift |
| `src/backend/bindless/pipeline/mod.rs` | Add `layer_layout_int4`, compile `sh_layer_v1_int4.wgsl` as `BindlessPipelineInt4` |
| `src/backend/bindless/pipeline/layer.rs` | All 4 bind group creation sites (lines 64, 279, 546, 797) |
| `src/backend/bindless/pipeline/inference.rs` | 1 bind group site (line 310) + KV buffer allocation |
| NEW: `src/backend/bindless/sh_layer_v1_int4.wgsl` | Copy of sh_layer_v1.wgsl with INT4 KV read/write — compiled as the second pipeline |
| NEW: `src/backend/bindless/sh_quantize_kv.wgsl` | Standalone test shader (isolation validation only) |

---

## Phase 0 — Pre-flight and Branch

- [ ] **0.1** Check `git status` and `git stash list`. Confirm the dirty work in progress belongs to a different feature. Do NOT disturb it — stash if needed, or ensure the current branch is not `main`.
- [ ] **0.2** Create branch: `git checkout -b feat/turboquant-wgsl`. This branch is the exclusive working surface for all steps below.
- [ ] **0.3** Confirm clean build on the new branch before any edits: `cargo build --release --bin shimmy_server_gpu`. Note the binary size and build time as a baseline.
- [ ] **0.4** Run the short story SHA check to capture a before-hash: `cargo run --bin shimmy_server_gpu` (or the VS Code task) on a known prompt. Record the top-5 token IDs at the first sampled position. This is the correctness baseline.

---

## Phase 1 — KVCache Rust Struct (Storage Foundation)

*These steps add the new buffers without removing the old ones yet. Nothing breaks; nothing changes behavior.*

- [ ] **1.1** In `kv_cache.rs`, add four new `Vec<wgpu::Buffer>` fields to `KVCache`:
  - `k_packed_buffers`: one per layer, type `array<u32>`, size = `max_seq × n_head_kv × (head_dim/8) × 4` bytes
  - `k_scale_buffers`: one per layer, type `array<f32>`, size = `max_seq × n_head_kv × 4` bytes
  - `v_packed_buffers`: same shape as k_packed
  - `v_scale_buffers`: same shape as k_scale
- [ ] **1.2** In `KVCache::new()`, allocate these four buffer vecs alongside the existing `k_buffers` / `v_buffers`. Log their sizes in the existing `eprintln!` block. Do not remove the F32 buffers yet.
- [ ] **1.3** Add accessors: `get_k_packed_buffer(layer)`, `get_k_scale_buffer(layer)`, `get_v_packed_buffer(layer)`, `get_v_scale_buffer(layer)`.
- [ ] **1.4** In `KVCache`, add a `quant_enabled: bool` field. Initialize it from env var `SHIMMY_KV_QUANT=int4` at construction time. This is the feature flag — when `false`, all new buffers are allocated but the old code paths run unchanged.
- [ ] **1.5** Compile check: `cargo build`. No logic has changed; this must be a clean build.

---

## Phase 2 — Standalone Quantize Shader (Isolation Test)

*Write the math in isolation. Prove it correct before wiring it into the hot path.*

- [ ] **2.1** Write `src/backend/bindless/sh_quantize_kv.wgsl`. This shader takes:
  - `@binding(0)` `array<f32>` — input F32 vector buffer (one head-vector at a time)
  - `@binding(1)` `array<u32>` — output packed int4 (head_dim/8 u32s per vector)
  - `@binding(2)` `array<f32>` — output scale (1 f32 per vector)
  - `@uniform` — vector count, head_dim

  The kernel: one workgroup per head-vector. Load all `head_dim` floats → find `max_abs` → scale = `max_abs / 7.0` (signed 4-bit range [-8..7]) → for each element: `q = clamp(round(v / scale), -8.0, 7.0)` → pack pairs into u32 nibbles → write packed + write scale.

  INT4 packing convention (write this comment in the shader):
  ```
  // Each u32 holds 8 int4 values: bits [3:0]=elem0, [7:4]=elem1, ... [31:28]=elem7
  // Signed 4-bit: stored as unsigned nibble with bias=8 (i.e., store (q+8) & 0xF)
  // Dequant: val = f32(nibble) - 8.0; reconstructed = val * scale
  ```

- [ ] **2.2** Write a corresponding `unpack_int4_vec` WGSL helper function (will be copy-pasted into the layer shaders in Phase 3):
  ```wgsl
  fn unpack_int4(packed: u32, idx: u32) -> f32 {
      let nibble = (packed >> (idx * 4u)) & 0xFu;
      return f32(nibble) - 8.0;
  }
  ```

- [ ] **2.3** Write a Rust CPU-side test (in `src/backend/bindless/tests.rs` or a new test file) that:
  - Constructs a known F32 vector (e.g., `[1.0, -2.5, 0.0, 3.7, ...]`)
  - Runs `sh_quantize_kv.wgsl` on GPU
  - Reads back packed + scale buffers
  - CPU dequantizes: `reconstructed[i] = (unpack nibble i - 8.0) * scale`
  - Asserts max abs error < `max_abs / 7.0` (i.e., ≤ 1 quantization step)

- [ ] **2.4** Run the isolation test: `cargo test quantize_kv_isolation`. It must pass before proceeding. If the error bound is violated, the nibble packing formula is wrong — fix it here before touching the hot path.

---

## Phase 3 — Write Path: Inline Quantization in main_qkv

*Change how K and V are stored when a new token is computed. The read path still uses F32 (no behavior change for attention yet).*

- [ ] **3.1** In `sh_layer_v1.wgsl`, update shader bindings to add:
  - `@binding(10)` `array<u32>` — `kv_cache_k_packed` (read_write)
  - `@binding(11)` `array<f32>` — `kv_cache_k_scales` (read_write)
  - `@binding(12)` `array<u32>` — `kv_cache_v_packed` (read_write)
  - `@binding(13)` `array<f32>` — `kv_cache_v_scales` (read_write)

  Keep bindings 7 (kv_cache_k F32) and 8 (kv_cache_v F32) **unchanged** for now — the old read path is still live.

- [ ] **3.2** In `main_qkv` (Kernel 1) in `sh_layer_v1.wgsl`, after the existing write to `kv_cache_k[cache_idx]`, add a gated write:
  ```wgsl
  // [TurboQuant write path — activated when packed buffers are bound]
  // Quantize the full head-vector for this head+position inline.
  // Only the thread responsible for the first element of a head runs the scale pass.
  ```
  Because a workgroup dimension maps one thread per output dimension, the quantization must be coordinated across the head_dim threads writing one head-vector. Use workgroup shared memory for the max_abs reduction, then each thread packs its own nibble pair.

  **Exact approach for main_qkv inline quant (workgroup_size is confirmed `(256, 1, 1)`):**
  - For TinyLlama: K block = n_head_kv × head_dim = 4 × 64 = 256 = exactly one workgroup. All K writes for one token are synchronized by a single `workgroupBarrier()`. 
  - For LLaMA-3 8B: K block = 8 × 128 = 1024 = 4 workgroups. Each workgroup covers exactly 2 complete head-vectors (256 / 128 = 2). Head boundaries are always clean multiples — no partial-head workgroups on any supported model.
  - Each thread (when `target_stage == 1`) writes its F32 element to `kv_cache_k[cache_idx]` AND into `var<workgroup> wg_abs: array<f32, 256>` at index `local_invocation_id.x`.
  - After `workgroupBarrier()`, threads at `dim_in_head == 0` (i.e., every `head_dim`-th thread in the workgroup) scan their 64/128 elements, compute `max_abs`, write to `kv_cache_k_scales[pos * n_head_kv + head]`.
  - After a second `workgroupBarrier()`, all threads pack their nibble pair into `kv_cache_k_packed`.

  **Note:** The existing F32 write to kv_cache_k becomes the staging area for the scale pass. This means both buffers are written simultaneously — memory usage temporarily doubles per token's K/V head-vector, but since temp_state is already a scratchpad, this is fine. The F32 kv_cache buffers become logically deprecated once the packed path is validated; they will be removed in Phase 6.

- [ ] **3.3** ~~sh_layer_q4k.wgsl~~ — **no change needed, confirmed dead code.** All quant types route through `sh_layer_v1.wgsl` via the packed `quant_type` field in `LayerParams`.
- [ ] **3.4** In `pipeline/mod.rs`, create `layer_layout_int4` (bindings 0–13: the existing 0–9 plus 4 new packed+scale bindings). The existing `layer_layout` (bindings 0–9) is left untouched — it backs the F32 pipeline.
- [ ] **3.5** The 5 confirmed `create_bind_group` sites using `layer_layout` are at `pipeline/layer.rs` lines 64, 279, 546, 797 and `pipeline/inference.rs` line 310. The INT4 pipeline's bind group creation is a parallel set of 5 functions that supply the packed buffers. Do not modify the existing F32 bind group functions during Phase 3 — dual-buffer strategy means both sets coexist.
- [ ] **3.6** Compile check: `cargo build --release --bin shimmy_server_gpu`. This is the first end-to-end compile with the new layout.
- [ ] **3.7** Runtime smoke: run the server with `SHIMMY_KV_QUANT=int4` disabled (default). Confirm identical output to baseline (packed buffers allocated but not written to). Run the short story SHA check — SHA must match the baseline from Step 0.4.

---

## Phase 4 — Read Path: Inline Dequantization in main_attn_out

*This is the fused kernel. After this phase, with `SHIMMY_KV_QUANT=int4` enabled, the whole pipeline runs on packed storage.*

- [ ] **4.1** Copy the `unpack_int4` helper function from Phase 2.2 into `sh_layer_v1.wgsl` (at the top, before Kernel 2).

- [ ] **4.2** In `main_attn_out` (Kernel 2) in `sh_layer_v1.wgsl`, add a conditional dequant path for K reads. The current read is:
  ```wgsl
  let k_re  = kv_cache_k[k_base + doff];
  let k_im  = kv_cache_k[k_base + doff + 1u];
  ```
  Add a WGSL constant or param to switch between paths. For v1, use a `params`-driven flag (add `kv_quant_enabled: u32` to `LayerParams` struct):
  ```wgsl
  var k_re: f32;
  var k_im: f32;
  if (params.kv_quant_enabled == 1u) {
      let packed_base = pos * params.head_count_kv * (params.head_dim / 8u)
                      + kv_head_idx * (params.head_dim / 8u);
      let k_scale = kv_cache_k_scales[pos * params.head_count_kv + kv_head_idx];
      let word_re = kv_cache_k_packed[packed_base + p / 4u];
      let word_im = kv_cache_k_packed[packed_base + (p * 2u + 1u) / 8u]; // careful index
      k_re = unpack_int4(word_re, (p * 2u) % 8u) * k_scale;
      k_im = unpack_int4(word_im, (p * 2u + 1u) % 8u) * k_scale;
  } else {
      k_re = kv_cache_k[k_base + doff];
      k_im = kv_cache_k[k_base + doff + 1u];
  }
  ```
  **Index math note:** With 8 int4 values per u32, element `e` is at `packed[e/8]` nibble `e%8`. For RoPE pairs: k_re = element `p*2`, k_im = element `p*2+1`. Work through this carefully — a single off-by-one here gives silent wrong answers.

- [ ] **4.3** Add the same conditional dequant for V reads in `main_attn_out`.

- [ ] **4.4** **D3 is closed: two compiled pipelines.** The concrete mechanism: `pipeline/mod.rs` gains a second struct `BindlessPipelineInt4` compiled from `sh_layer_v1_int4.wgsl` (the INT4 variant created in this phase). Both structs share the same `layer_pipeline_*` entry points by name (`main_qkv`, `main_attn_out`, etc.) but the INT4 variants have no F32 KV bindings and no branches — they are statically fused for packed+scale storage. `server_inference.rs` reads `SHIMMY_KV_QUANT=int4` once at startup and holds either `&BindlessPipeline` or `&BindlessPipelineInt4` through the session. Zero per-element branching in either path.

- [ ] **4.5** ~~Apply same changes to sh_layer_q4k.wgsl~~ — **no action, confirmed dead code.**

- [ ] **4.6** Compile check: `cargo build --release --bin shimmy_server_gpu`.

---

## Phase 5 — Helical Shift: Update Rope Shift for Packed Buffers

*When the sliding window compacts, it must shift the packed buffers, not just the F32 ones.*

- [ ] **5.1** In `sh_rope_shift.wgsl`, add two new bindings:
  - `@binding(5)` `array<u32>` — `k_packed_src` (read)
  - `@binding(6)` `array<f32>` — `k_scale_src` (read)
  - `@binding(7)` `array<u32>` — `k_packed_dst` (read_write)
  - `@binding(8)` `array<f32>` — `k_scale_dst` (read_write)
  - (same for v: bindings 9–12)

  The thread dispatch already maps `(d=dimension, h=kv_head, z=seq_offset)`. For packed: the `d` dimension maps to `d/8` words. Only threads where `d % 8 == 0` need to write (or restructure to `d` = word index, max = `head_dim/8`). Simplest approach: run the shift on packed buffers with a separate dispatch using `head_dim/8` as the x dimension.

- [ ] **5.2** In `pipeline_shift.rs`, update `RopeShiftPipeline::execute()` to pass the four new packed+scale buffers when `quant_enabled`. Gated by `kv_cache.quant_enabled`.

- [ ] **5.3** Compile check.

---

## Phase 6 — Isolation Validation Gate

*Do not proceed to benchmarks until these pass.*

- [ ] **6.1** Enable `SHIMMY_KV_QUANT=int4`. Run the short story SHA check (VS Code task: *Validate Airframe Short SHA*). The SHA **must match** the F32 baseline from Step 0.4. If it doesn't, the index math in Step 4.2 is wrong. Debug before proceeding.

- [ ] **6.2** Run the long story check (*Validate Airframe Long Story*). This exercises longer context where quantization error can accumulate. SHA must still match.

- [ ] **6.3** If either SHA check fails: the diagnostic path is to add temporary `eprintln!` to compare K values loaded via the quantized path vs. the F32 path for the first few positions. The most likely bugs are:
  - Nibble packing off-by-one (wrong element extracted)
  - Scale computation using absolute max but forgetting the bias-8 encoding
  - Wrong stride computation in packed_base index

- [ ] **6.4** Run model smoke test (*Run Model Smoke Test* VS Code task) with `SHIMMY_KV_QUANT=int4`. Pass = proceed.

---

## Phase 7 — VRAM Accounting and F32 Buffer Removal

*Only after validation. Remove the now-redundant F32 kv buffers.*

- [ ] **7.1** Log the actual allocated buffer sizes with `SHIMMY_KV_QUANT=int4` at startup. Confirm the numbers match the theoretical values from the architectural context section above.

- [ ] **7.2** Remove `k_buffers` and `v_buffers` from `KVCache` (the old F32 buffers). Remove their allocations in `KVCache::new()`. Update all accessors.

- [ ] **7.3** Remove the F32 `kv_cache_k` / `kv_cache_v` bindings (7 and 8) from the layer shaders and the Rust bind group layouts. The packed buffers (10–13, now renumbered to 7–10) become the sole KV storage.

- [ ] **7.4** Remove the `else` branches in `main_attn_out` that read from the old F32 buffers. The `kv_quant_enabled` flag can be kept as a no-op safety check or removed entirely.

- [ ] **7.5** Compile + full smoke test pass. This is the first point where VRAM consumption measurably drops.

---

## Phase 8 — Benchmarks

- [ ] **8.1** Run *Needle Smoke (2K, default server)* VS Code task. Record tokens/sec and recall. Compare to pre-branch baseline.

- [ ] **8.2** Run *Needle Full Matrix (8K server)*. This requires the 8K context server task. Record:
  - KV cache buffer size (from startup log)
  - Tokens/sec at each context depth
  - Needle recall at 15/50/85% depth across 2K/4K/8K

- [ ] **8.3** Quality regression: run `scripts/battery_test.sh` — 4-question scored run across all supported GGUFs. `math_battery.py` is math-specific only; `battery_test.sh` is the correct cross-model quality signal. Pass = no question regressions vs. F32 baseline run.

- [ ] **8.4** Record results in `artifacts/turboquant_bench_results.json`. Format: same structure as existing `needle_bench_*.json` artifacts plus a `vram_kv_cache_mb_before` / `vram_kv_cache_mb_after` key.

---

## Phase 9 — YaRN + TurboQuant Combined Validation

- [ ] **9.1** Start server with both `SHIMMY_MAX_CTX=8192` and `SHIMMY_KV_QUANT=int4`. This combines YaRN extended-context RoPE scaling with packed KV storage.

- [ ] **9.2** Run the 8K needle bench with this combined config. Compare recall to:
  - F32 + native RoPE (2K baseline)
  - F32 + YaRN **[this IS the current production 8K baseline — already live in `server_inference.rs` lines 480–490, dynamic RoPE selection active on any request where total_seq > l_train]**
  - INT4 + YaRN (this step — the new path being validated)

- [ ] **9.3** Expected result: INT4+YaRN recall ≥ F32+YaRN recall (compression error is smaller than RoPE extension error; both are below perception threshold). If INT4+YaRN underperforms F32+native significantly, the interaction to investigate is whether the per-vector scale computation is sensitive to the higher-frequency activations in YaRN-extended contexts.

---

## Phase 10 — Large Model Stretch Goal

*This is the payoff: a model that currently cannot run on 12GB now can.*

- [ ] **10.1** Identify the largest available GGUF model that fits within 12GB VRAM with INT4 KV cache at 8K context. Candidate: Llama-3-8B-Instruct Q4_K_M (8B params). Calculate:
  - Weights: ~4.5 GB (Q4_K_M)
  - KV cache at 8K, 32 layers, 8 KV heads, head_dim=128: 8192 × 32 × 8 × (128/8) × 4 (packed) + scales = ~108 MB
  - Total: ~4.6 GB. Well within 12 GB. ✓

- [ ] **10.2** Download and validate the candidate model file with `scripts/scan_gguf_models.py`.

- [ ] **10.3** Run model smoke test on the large model with `SHIMMY_KV_QUANT=int4 SHIMMY_MAX_CTX=8192`. First, confirm it loads without OOM.

- [ ] **10.4** Run a needle bench at 4K and 8K context on the large model. Record tokens/sec, recall, KV cache size.

- [ ] **10.5** If 8B fits cleanly: attempt 13B or 14B (next Fibonacci up in model size). Calculate KV cache headroom first — do not attempt if calculation shows >11 GB total.

---

## Phase 11 — Cleanup and Merge Readiness

- [ ] **11.1** Remove all `eprintln!` debug statements added during this work (or gate them behind `SHIMMY_DEBUG_KV_QUANT=1`).

- [ ] **11.2** Update `kv_cache.rs` doc comments to describe the packed layout. Add a comment explaining the bias-8 nibble convention.

- [ ] **11.3** Update `docs/spike-wgsl-turboquant-port.md` status to COMPLETE and add link to bench results artifact.

- [ ] **11.4** Update `/memories/repo/airframe-rollout-state.md` with the new capability and confirmed model size ceiling.

- [ ] **11.5** Open PR against `main`. Title: `feat: INT4 fused KV cache (TurboQuant WGSL port)`. Include bench numbers in PR description.

---

## Decision Points

D1, D3 are closed by FSE reasoning (see `/memories/repo/fused-semantic-execution.md`). D2, D4, D5 stand as planned.

| # | Decision | Resolution | Reasoning |
|---|----------|------------|-----------|
| D1 | Inline quant in main_qkv vs. separate quantize pass | **CLOSED: Inline** | A separate dispatch is a second pass over data already touched. FSE: ∂cost/∂quant ≈ 0 when fused inline. The separate-pass option is removed from the plan. |
| D2 | Dual-buffer during validation or cold switch | **Dual-buffer (Phase 3–6), remove F32 in Phase 7** | Both buffers written in the same kernel pass — no second traversal. VRAM cost is temporary; runtime cost is zero. Rollback is trivial. |
| D3 | `kv_quant_enabled` as runtime params field vs. two compiled pipelines | **CLOSED: Two compiled pipelines** | A params field adds a branch inside the shader evaluated on every K/V access at every context position — O(seq_len × heads) per token. Two compiled pipelines pay the cost once at session start. FSE: compile-time selection, zero per-element branching. |
| D4 | Nibble packing: bias-8 or two's complement | **Bias-8** | KV activations, not weights. No GGUF interop needed. Simplest bit ops. |
| D5 | Scale granularity: per-head-vector or per-pair | **Per-head-vector for v1** | Scale is a broadcast value (FSE). One F32 serves head_dim dequant ops — 64x amortization. Revisit only if Phase 8 perplexity shows measurable degradation. |

---

## Success Criteria (from spike doc, restated)

| Metric | Target | Test |
|--------|--------|------|
| KV cache memory at 8K context | ≤ 25% of F32 baseline | Step 8.1 startup log |
| Top-1 token agreement with F32 | 100% on standard prompts | Steps 6.1, 6.2 SHA check |
| Tokens/sec at 8K | ≥ F32 baseline | Step 8.2 |
| Perplexity on wikitext-2 | ≤ 0.5 above F32 | Step 8.3 |
| Needle recall at 8K | ≥ F32 baseline | Step 8.2 |
| Largest model runnable on RTX 3060 12GB | ≥ 8B at 8K ctx | Phase 10 |

---

## Files Changed Summary

```
NEW  src/backend/bindless/sh_quantize_kv.wgsl          (Phase 2 — isolation test only)
NEW  src/backend/bindless/sh_layer_v1_int4.wgsl        (Phase 4 — fused INT4 pipeline, no F32 KV branches)
MOD  src/backend/bindless/kv_cache.rs                  (Phase 1)
MOD  src/backend/bindless/sh_layer_v1.wgsl             (Phase 3 — write path only; F32 read path unchanged)
MOD  src/backend/bindless/sh_rope_shift.wgsl           (Phase 5)
MOD  src/backend/bindless/pipeline_shift.rs            (Phase 5)
MOD  src/backend/bindless/pipeline/mod.rs              (Phase 3+4 — add BindlessPipelineInt4 + layer_layout_int4)
MOD  src/backend/bindless/pipeline/layer.rs            (Phase 3 — 4 confirmed bind group sites)
MOD  src/backend/bindless/pipeline/inference.rs        (Phase 3 — 1 confirmed bind group site)
MOD  src/bin/shimmy_server_gpu/server_inference.rs     (Phase 4 — pipeline selection at session start)

NOT TOUCHED: sh_layer_q4k.wgsl (dead code, never compiled)
```
