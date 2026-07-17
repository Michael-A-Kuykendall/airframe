# Spike: wgpu 2048 MB Storage Buffer Binding Limit — Root Cause & Resolution Plan

**Issue:** [shimmy#206](https://github.com/Michael-A-Kuykendall/shimmy/issues/206)
**Branch:** `spike/206-wgpu-2gb-buffer-cap` (off `sideline/v0.2.8-base`, no commits)
**Date:** 2026-07-17
**Author:** Agent (read-only audit, now written to repo)

---

## 1. User-Facing Bug (from #206)

**Error text:**
```
Failed to load model 'gemma-4-12b-it-q8-0':
Airframe GPU load failed: Model file (12083 MB) exceeds this GPU's
storage buffer binding limit (2048 MB).
Try a more quantized model or update your GPU drivers.
```

**Environment:** Shimmy v2.1.0 / airframe 0.2.8, Quadro RTX 4000 (16 GB, 13 GB free), Ubuntu, Rust 1.95.

**Confirmed reporters:** `longzou` (Quadro RTX 4000), `LinxiDev` (integrated-GPU misdetection), `legobyte` (Intel iGPU + dGPU), `updoo` (Mac M1). All see the same 2048 MB binding limit error or integrated-GPU misselection.

**Maintainer response** (Michael-A-Kuykendall, 2026-07-14):
- Bug 1: 2048 MB binding cap — "tracked engine limitation… on the roadmap for v2.1."
- Bug 2: Wrong GPU selected — confirmed bug in wgpu adapter enumeration.

---

## 2. Root Cause Analysis

### 2.1 Bug A — The 2048 MB Binding Cap

The airframe bindless engine stores the **entire GGUF file** in a single `wgpu::Buffer` called `gpu_buffer`:

```
src/backend/bindless/loader.rs:128
    let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GGUF Bindless Storage"),
        size,   // ← full GGUF file size (e.g. 12 083 MB)
    });
```

This single buffer is then exposed to WGSL shaders through **exactly three hardcoded sub-range bindings**, documented at:

```
src/backend/bindless/loader.rs:10-13
    pub const BLOB_CHUNK_BYTES: u64 = 2_000_000_000;  // ~1.86 GiB
```

```
src/backend/bindless/loader.rs:15-20
    // For models > 2 GB the buffer is exposed to shaders through three
    // sub-range bindings (blob_0 / blob_1 / blob_2) so that each individual
    // binding stays within max_storage_buffer_binding_size.
```

**The 3 bindings** (loader.rs:47-84):

| Binding | Offset | Size | Code location |
|---------|--------|------|---------------|
| `blob_0` | 0 | `min(BLOB_CHUNK_BYTES, size)` | loader.rs:47-54 |
| `blob_1` | 2 GB | `min(BLOB_CHUNK_BYTES, size - 2GB)` | loader.rs:58-68 |
| `blob_2` | **4 GB** | **`size - 2*BLOB_CHUNK_BYTES`** — THE ENTIRE REMAINDER | **loader.rs:73-83** |

**For a 12 083 MB model:**
- `blob_0` = 0..2 GB ✓
- `blob_1` = 2..4 GB ✓
- **`blob_2` = 4..12 GB = 8 GB** — FAILS wgpu validation on any GPU whose `max_storage_buffer_binding_size` is ≤ 2 GB.

wgpu's validation layer rejects binding an 8 GB sub-range onto a binding slot whose limit is 2 GB. The error surfaces as the "exceeds this GPU's storage buffer binding limit" message in #206.

A secondary latent bug with the **same symptom** exists in 7 test/bin files that incorrectly clamp `max_buffer_size` to the binding size:

```
tests.rs:49           limits.max_buffer_size = adapter_limits.max_storage_buffer_binding_size as u64;
test_int4_parity.rs:36  same pattern
attention_f6_f7_verify.rs:70  same
frontier_compare.rs:221  same
layer_dump_gpu.rs:104  same
debug_kv_cache_bos.rs:30  same
ffn_f8_verify.rs:72   same
```

These would fail for different reasons (buffer creation size, not binding size) but produce a similar user-facing failure for large models. The production path in `gpu.rs:150` uses `max_buffer_size = adapter.max_buffer_size` correctly (Pattern B).

### 2.2 Bug B — Discrete GPU Mis-selection

```
src/runtime/gpu.rs:96-103
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
```

On multi-GPU laptops (Intel iGPU + NVIDIA/AMD dGPU), `request_adapter` with `HighPerformance` still often returns the integrated adapter. wgpu's adapter enumeration order on Windows/D3D12 tends to list the iGPU first, and `request_adapter` returns the first compatible match, not necessarily the best. Confirmed by `legobyte`: "doesn't use dedicated GPU, shimmy tries to load model into Intel cpu's dedicated card."

---

## 3. Design Decision: Generalizing the Bindless Chunking

### 3.1 Constraints

- **WGSL** (as deployed in v0.2.8) requires every storage buffer to be a **fixed `var<storage, read>` declaration at a fixed `@binding(N)` slot**. Runtime-sized arrays of bindings (`binding_array`) are not used in the codebase, require experimental WGSL `enable` directives, and are not portable across D3D12/Vulkan/Metal.
- **The bind group layout** in `pipeline/mod.rs` currently defines 12 entries per `layer_layout` with blob_0 at binding 0 and blob_1/blob_2 at bindings 10 and 11 (the gap is intentional — other buffers live at 1–9).
- **The dummy-buffer pattern** already exists (`loader.rs:31`, `loader.rs:59-60`) for unused blob slots — a 4-byte zero STORAGE buffer bound when a chunk boundary is beyond the model size.
- **`max_storage_buffers_per_shader_stage` has a spec minimum of 8.** Many mobile / older-AMD / Intel-iGPU adapters clamp to this floor. Any blob-chunk plan must account for the *other* stage buffers (activation_in, temp_state, LayerOffsets, LayerParams, norm_bank, rope_table, kv_cache_k, kv_cache_v, cache_params ≈ 9 more), so the total per-stage storage-buffer count is `N + 9`. A hard `N=8` ⇒ 17 bindings ⇒ pipeline compile failure on low-limit devices. (See §8, external appraisal.)

### 3.2 Approach: Fixed-Max N with Generated WGSL

```
Fixed-max blob chunks in shader:     N = 8  (16 GB ceiling at 2 GB/chunk)
Code-generated WBGL read_blob:       N-branch if/else chain generated at build time
Rust side:                            model.blob_binding(i) for i in 0..num_chunks
Unused slots:                         4-byte dummy STORAGE buffer (existing pattern)
```

**Why N=8, but sized dynamically (external-appraisal correction):** A fixed `N=8` at a 2 GB floor consumes 8 storage-buffer bindings *just for weights*. With the other ~9 stage buffers (activation_in, temp_state, LayerOffsets, LayerParams, norm_bank, rope_table, kv_cache_k, kv_cache_v, cache_params), the shader needs **17 storage-buffer bindings**. The WebGPU spec minimum for `max_storage_buffers_per_shader_stage` is **8** (mobile / older AMD / Intel iGPU), so a hard N=8 pipeline will fail to compile on those platforms.

**Correction — dynamic chunk sizing:** compute the effective chunk size as the *larger* of `BLOB_CHUNK_BYTES` (2 GB floor) and the adapter's actual `max_storage_buffer_binding_size` (desktop dGPUs commonly allow 4–128 GB), then derive `N = ceil(file_size / effective_chunk)`. This keeps the binding count low on capable GPUs. For low-limit devices, add a preflight guard: if `N + other_stage_buffers > max_storage_buffers_per_shader_stage`, either scale execution down or emit a clear "device storage-buffer budget too low" error rather than a validation crash. The effective chunk size must additionally stay ≤ `max_buffer_size`.

**WGSL code-generation determinism (external-appraisal note):** the generated `read_blob` string must be byte-for-byte deterministic (exact spacing/ordering) so drivers can cache the compiled pipeline across model reloads.

**Repacking binding slots** (N=4 example, per external appraisal): current layout has blob bindings at 0, 10, 11; new layout packs all N blobs contiguously at 0..N-1 and shifts the other buffers down:

| Slot | Current Layout (pipeline/mod.rs) | New Proposed Layout (N=4) | Impact / Verification |
|------|----------------------------------|---------------------------|------------------------|
| 0 | `blob_0` | `blob_0` | Unchanged |
| 1 | `activation_in` | `blob_1` | Shifted — grep verify all .wgsl |
| 2 | `temp_state` | `blob_2` | Shifted — grep verify all .wgsl |
| 3 | `LayerOffsets` | `blob_3` | Shifted — grep verify all .wgsl |
| 4 | `LayerParams` | `activation_in` | Shifted down by N-1 slots |
| 5..N+2 | lower bindings | mid-tier bindings | Offsets updated in create_bind_group |
| 10 | `blob_1` | system parameters | Gap eliminated; packing density improved |
| 11 | `blob_2` | system parameters | Old trailing blob pointers wiped |

This eliminates the binding-index gap and simplifies `create_bind_group`. The 3 WGSL files using blob bindings update their `@binding` attributes; the other 11 (single `gguf_blob` at 0) are unchanged.

### 3.3 Hardening requirements (external appraisal #2)

A second independent review re-verified the root cause against `airframe` master / v0.2.x and confirmed it matches this spike exactly. It endorsed the approach and added concrete hardening items that must be satisfied before merge:

- **Alignment.** Every chunk offset and chunk size must be a multiple of `min_storage_buffer_offset_alignment` (almost always 256). The fixed `2_000_000_000` value is already 256-aligned, but the *dynamic* effective-chunk path must round the chunk size **down** to the alignment; offsets are then automatically aligned.
- **Word/byte indexing contract.** The current `read_blob` operates in **32-bit words** (`BLOB_SPLIT_0 = 500000000u` = 2 GB/4). The generated dispatcher must preserve this exactly: convert byte split points to word indices (`bytes / 4`) with no off-by-one, or dequant will corrupt.
- **Extreme-model ceiling.** With a 2 GB binding limit and a tight storage-buffer budget, models ≫ 16 GB are impractical under this scheme. Document the ceiling explicitly; acceptable for the current target range.

---

## 4. Files to Change — Exact Inventory

### 4.1 Module: BLOB chunking (`loader.rs`)

| What | Where | Change |
|------|-------|--------|
| Remove `blob_binding_0/1/2` | loader.rs:47-84 | Replace with `blob_binding(idx: usize) → BindingResource` |
| Add `num_blob_chunks()` | loader.rs (new) | `ceil(size / effective_chunk)` (effective_chunk from §4.5, not the raw constant) |
| Update `BindlessModel` | loader.rs:21-38 | Single shared `dummy_buffer` (4-byte) per §8.4; drop per-model dummy allocation |
| Update `load_from_disk` | loader.rs:97-181 | Per remaining slot use shared dummy; tail slice = `size - offset` |
| **Alignment** | loader.rs (new) | Round `effective_chunk` **down** to `min_storage_buffer_offset_alignment` (256); offsets then 256-aligned |
| **Word/byte contract** | loader.rs + WGSL gen | `read_blob` works in 32-bit words; convert byte splits to `bytes/4` with no off-by-one |
| **Binding-index audit gate** | loader.rs + mod.rs + inference.rs + 3 .wgsl | Exhaustive grep `@binding(` and `BindGroupEntry { binding:`; every entry must match new packed layout (§3.2 table) |

### 4.2 Module: Shader WGSL files (3 of 14)

| File | Lines | Change |
|------|-------|--------|
| `sh_layer_v1.wgsl` | 67-82 | Replace fixed `blob_0/1/2` declarations with generated N-branch `read_blob`; repack binding slots |
| `sh_head_blob.wgsl` | 23-28, 46-54 | Same pattern |
| `sh_rmsnorm.wgsl` | 13-28 | Same pattern |

The other 11 WGSL files (single `gguf_blob` at binding 0) — **no changes**.

### 4.3 Module: Bind group layouts (`pipeline/mod.rs`)

| Layout | Lines | Change |
|--------|-------|--------|
| `layer_layout` | 645-781 | Replace blob entries at 0,10,11 with contiguous N entries at 0..N-1; shift other bindings |
| `lm_head_blob_layout` | 821-886 | Same repack |
| `rmsnorm_layout` | 547-617 | Same repack |

### 4.4 Module: Bind group creation (`inference.rs`)

| Function | Lines | Change |
|----------|-------|--------|
| `make_bg` closure | 431-495 | Loop `0..model.num_blob_chunks()` → push entry per blob binding |
| LM head bind group | 789-822 | Same |
| RMSNorm bind group | 665-703 | Same |
| **Acceptance gate** | all 3 sites | Exhaustive grep `@binding(` (WGSL) + `BindGroupEntry { binding:` (Rust); every entry must match new packed layout (§3.2 table) |

### 4.5 Module: Runtime GPU initialization (`gpu.rs`)

| What | Lines | Change |
|------|-------|--------|
| Preflight comment | 108-109 | Update "up to 3" → "N = ceil(size / effective_chunk)" |
| Dynamic chunk size | 112-114 | Compute `effective_chunk = min(max_buffer_size, max(adapter.max_storage_buffer_binding_size, BLOB_CHUNK_BYTES))`; derive `N` from it |
| Add chunk-count guard | after 116 | `if N + STAGE_AUX_BUFFERS > max_storage_buffers_per_shader_stage` → error: **"device storage-buffer budget too low for this model size"** (STAGE_AUX_BUFFERS ≈ 9: activation_in, temp_state, LayerOffsets, LayerParams, norm_bank, rope_table, kv_cache_k, kv_cache_v, cache_params) |
| Preflight error alignment | 118-133 | Ensure error message matches shimmy's expected "exceeds storage buffer binding limit" wording (currently says "too small for bindless chunk size" — update for clarity) |
| WGSL gen determinism | build script / `include_str!` | Generate `read_blob` with byte-stable layout so drivers cache the pipeline across reloads |
| Adapter selection | 96-103 | Replace `request_adapter` with adapter enumeration + discrete preference + `SHIMMY_GPU` override (index **or** device-name substring, see §8.3) |

### 4.6 Module: Fix Pattern A limit bugs (7 files)

| File | Line | Change |
|------|------|--------|
| `tests.rs` | 49 | `max_buffer_size = adapter.max_storage_buffer_binding_size` → `adapter.max_buffer_size` |
| `test_int4_parity.rs` | 36 | Same |
| `attention_f6_f7_verify.rs` | 70 | Same |
| `frontier_compare.rs` | 221 | Same |
| `layer_dump_gpu.rs` | 104 | Same |
| `debug_kv_cache_bos.rs` | 30 | Same |
| `ffn_f8_verify.rs` | 72 | Same |

### 4.7 Module: Discrete GPU selection (new)

| File | Lines | Change |
|------|-------|--------|
| `gpu.rs` | 96-103 | Enumerate adapters; parse `SHIMMY_GPU` as integer index **or** case-insensitive substring of adapter name; else prefer `is_discrete()`; else first compatible. Fallback chain must be stable (see §8.3) |
| `shimmy_server_gpu.rs` | (parallel path) | Same pattern |
| `frontier_compare.rs` | (parallel path) | Same pattern |

### 4.8 Module: Validation (new test)

| File | Change |
|------|--------|
| `tests/bindless_chunking_verify.rs` (new) | Synthetic >2 GB GGUF fixture, assert `num_blob_chunks()`, assert all bindings resolve without panic, dequant roundtrip match. |

---

## 5. Implementation Branch Strategy

```
sideline/v0.2.8-base
  └── spike/206-wgpu-2gb-buffer-cap        ← this spike branch (audit only)
       ├── spike/206-t1-chunking-core       ← T1: loader.rs chunking core
       ├── spike/206-t2-pipeline-shaders    ← T2: WGSL + layouts + bind groups
       ├── spike/206-t3-limits-gpu-select   ← T3: limit unification + discrete GPU
       └── spike/206-t4-validation          ← T4: verification harness
```

Each sub-branch merges to the spike branch via rebase. The spike branch merges to `sideline/v0.2.8-base` when all sub-branches are green. Atomic commits per logical change (no mega-commits).

---

## 6. Testing Strategy

### 6.1 Unit-level
- `BindlessModel::num_blob_chunks()`: assert `ceil(size / 2GB)` for sizes in { 1 GB, 2 GB, 4 GB, 5 GB, 12 GB, 15.9 GB }.
- `BindlessModel::blob_binding(i)`: assert offset = i*2GB, size = min(2GB, remainder) for each i.

### 6.2 Integration-level
- **Existing GPU verify tests** (`tests/gpu_22layer_verify.rs`, `tests/attention_f6_f7_verify.rs`, `tests/ffn_f8_verify.rs`): must pass unchanged (these test small models fitting in 1 chunk — regression safety).
- **New `tests/bindless_chunking_verify.rs`**: create a synthetic >2 GB GGUF by concatenating a fixture header with large zero-filled tensors; load via `BindlessModel::load_from_disk`; assert 3+ chunks; run dequant roundtrip on a tensor straddling chunk 0/1 boundary.

### 6.3 End-to-end (via Shimmy harness)
- `cargo build -p shimmy` in `airframe-workspace/shimmy/` (picks up patched local airframe).
- Load a >2 GB model (e.g., `gemma-4-12b-it-q8-0` or `llama-3.2-8b`) and confirm `blob_binding_2` no longer exceeds binding limit.
- Confirm `SHIMMY_GPU=1` selects the discrete adapter on a multi-GPU system.

### 6.4 Regression safety
- The dummy-buffer fallback is the **same pattern already used** for `blob_1`/`blob_2` on small models — unchanged behavior for <2 GB models.
- WGSL files not using blob bindings (11 of 14) are untouched — zero risk.
- The repacked binding slots change indices of non-blob bindings; every `create_bind_group` call must match the new layout. Verified by exhaustive grep of `@binding` in WGSL and `wgpu::BindGroupEntry { binding: … }` in Rust.

---

## 7. Verification of Public Reproducibility

To independently verify this analysis:

1. Clone both repos at the versions specified in `AGENTS.md`.
2. Read `airframe/src/backend/bindless/loader.rs:47-84` — confirm the hardcoded 3-binding pattern.
3. Read `airframe/src/backend/bindless/loader.rs:77-78` — confirm `blob_binding_2` = `size - 2*CHUNK` (the entire remainder).
4. Read the 3 WGSL files (`sh_layer_v1.wgsl`, `sh_head_blob.wgsl`, `sh_rmsnorm.wgsl`) — confirm `blob_0`/`blob_1`/`blob_2` individual `var` declarations at fixed `@binding()`.
5. Read `pipeline/mod.rs` layouts + `inference.rs` bind group creation — confirm exact 3-entry blob pattern.
6. Read `gpu.rs:96-103` — confirm single `request_adapter` with `HighPerformance` (no discrete-GPU preference).
7. Build shimmy against a >4 GB model — observe the `blob_binding_2` validation failure.
8. Implement the fix per this plan — no new WGSL `enable` directives, no external dependencies.

The `gpu_buffer` is a single wgpu buffer of size = file size. Sub-range bindings reference slices within it. wgpu enforces that the *size* of each sub-range binding ≤ `max_storage_buffer_binding_size` (2048 MB on the reporter's Quadro RTX 4000). The fix dynamically partitions the single buffer into N sub-range bindings each ≤ that limit, rather than hardcoding 3 and dumping the remainder into the third.

---

## 8. External Appraisal (Second Opinion) — Integrated Findings

An independent technical review confirmed the root-cause analysis as "technically sound, incredibly precise" and endorsed the fixed-max-N + generated-WGSL approach as "the correct way to bypass WebGPU's static layout constraints without relying on non-portable extensions." It also surfaced **subtle wgpu traps the original plan under-addressed**, now folded into §3 and §4 above. Recorded here in full for the audit trail.

### 8.1 The `max_storage_buffers_per_shader_stage` ceiling trap
- The plan's chunk-count guard was correct in direction but **understated how tight the ceiling is**. On mobile / older AMD / Intel iGPU the limit is clamped to its spec minimum of **8**.
- With `N=8` blob chunks + ~9 auxiliary stage buffers = **17 bindings** ⇒ pipeline fails to compile on those platforms.
- **Resolution applied:** dynamic chunk sizing (§3.2) — raise the effective chunk up to the adapter's real `max_storage_buffer_binding_size` so fewer chunks are needed; add an explicit preflight guard that emits a clear "device storage-buffer budget too low" error instead of a validation crash when `N + 9 > max_storage_buffers_per_shader_stage`.

### 8.2 WGSL code-generation overhead
- Stitching WGSL strings in Rust works, but **non-deterministic output breaks driver pipeline caching** across model reloads.
- **Resolution applied:** the generator must emit a byte-for-byte deterministic layout (exact spacing/ordering) so the compiled pipeline object is cacheable (§4.5, §3.2).

### 8.3 Multi-GPU selection & `SHIMMY_GPU` override
- Adapter enumeration order is unstable across OS / driver revisions, so a hard index alone is fragile.
- **Resolution applied:** `SHIMMY_GPU` parses as either an integer index **or** a case-insensitive substring of the adapter name (e.g. `SHIMMY_GPU="RTX 4000"`), falling back to `is_discrete()`, then to the first compatible adapter (§4.7).

### 8.4 Recommended `loader.rs` architecture (single shared dummy)
To avoid OOM / validation failures when assigning dummy bindings for unused slots, the appraisal recommends a **single shared 4-byte dummy buffer** and a `ceil`-based chunk count:

```rust
// Proposed optimization for src/backend/bindless/loader.rs
pub struct BindlessModel {
    pub gpu_buffer: wgpu::Buffer,
    pub size: u64,
    pub dummy_buffer: wgpu::Buffer, // Single shared 4-byte dummy resource
}

impl BindlessModel {
    pub const BLOB_CHUNK_BYTES: u64 = 2_000_000_000;
    pub const MAX_CHUNKS: usize = 8;

    pub fn num_blob_chunks(&self) -> usize {
        ((self.size + Self::BLOB_CHUNK_BYTES - 1) / Self::BLOB_CHUNK_BYTES) as usize
    }

    pub fn blob_binding(&self, idx: usize) -> wgpu::BindingResource {
        let active_chunks = self.num_blob_chunks();

        if idx >= active_chunks {
            // Safe fallback to the shared dummy buffer mapping
            return self.dummy_buffer.as_entire_binding();
        }

        let offset = idx as u64 * Self::BLOB_CHUNK_BYTES;
        let size = if idx == active_chunks - 1 {
            self.size - offset // The remaining tail slice
        } else {
            Self::BLOB_CHUNK_BYTES
        };

        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.gpu_buffer,
            offset,
            size: std::num::NonZeroU64::new(size),
        })
    }
}
```

Note: `blob_binding(idx)` above always returns a 2 GB-sized sub-range for interior chunks; in the final design the *interior* chunk size must come from the dynamic `effective_chunk` (§4.5), not the fixed `BLOB_CHUNK_BYTES`, so the interior size is `effective_chunk` and only the final tail slice is `size - offset`. The single shared `dummy_buffer` replaces the per-model dummy allocation, eliminating the OOM risk the appraisal flags.

### 8.5 Before/after slot mapping (verification reference)
The N=4 table in §3.2 is the authoritative mapping. Acceptance gate: **every** `wgpu::BindGroupEntry { binding: N }` in `inference.rs` / `pipeline/mod.rs` and every `@binding(N)` in the 3 blob WGSL files must match the new packed layout; validate by exhaustive grep before merge.

### 8.6 Next-step offers from the appraisal (available on request)
1. Draft the exact WGSL string-builder for the 3 target `.wgsl` files.
2. Concrete wgpu adapter enumeration loop with substring `SHIMMY_GPU` overrides for `gpu.rs`.
3. The synthetic test harness for `tests/bindless_chunking_verify.rs` using mock headers to simulate oversized models without disk cost.

These are scoped as sub-tasks under T2 (item 1–2) and T4 (item 3).

---

## 9. External Appraisal #2 — Verification & Merge-Gate Refinements

A second independent reviewer (a) re-confirmed the root cause against `airframe` **master / v0.2.x** and found it matches this spike verbatim, and (b) rated the plan *"high-quality: accurate root-cause, realistic constraints, and a clean, portable fix path."* The consolidated refinements are now embedded in §3.3, §4.1, §4.4, §4.5, §6:

1. **Alignment** — dynamic chunk path must enforce `min_storage_buffer_offset_alignment` (256) by rounding `effective_chunk` down.
2. **Binding-index audit surface** — repacking moves every non-blob resource; exhaustive grep of both Rust and WGSL is non-negotiable (acceptance gate, §4.4).
3. **Storage-buffer budget edge cases** — clear preflight message *"device storage-buffer budget too low for this model size"* (§4.5), not a generic validation crash.
4. **Word vs byte indexing** — generated `read_blob` must preserve the 32-bit-word contract exactly (§3.3).
5. **Extreme models** — document the ≫16 GB ceiling as acceptable for current targets (§3.3).
6. **Test coverage** — synthetic >2 GB fixture + dequant-roundtrip-across-boundary; keep existing small-model GPU verify tests as regression anchors (§6.2/§6.4).

**Bottom line (reviewer):** proceed. Prioritize the **binding-index audit** and the **storage-buffer-count preflight** — those are the two places most likely to produce subtle breakages. The rest of the plan is solid.
