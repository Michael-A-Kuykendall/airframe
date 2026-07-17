# Airframe — Comprehensive Refactor Plan

> Goal: Ordered real architecture. Deprecated/stupid systems swept away.  
> Confidence target: **>95%** that every task can be completed without contextual slips.  
> Method: Slice into beads with full context per task. Execute in dependency order.

---

## Phase 0: Objective Testing Gates (PPT + Invariants) — DO FIRST

**Bead `airframe-i88`** (foundational, P0) establishes the quality-gate substrate
that every later bead depends on. Status: **implemented** (`src/invariant_ppt.rs`,
`tests/test_contracts.rs`, CI gate added). Do not start Phase 1+ until this is green.

**Why bake testing in before code:** the refactor is high-churn and partly AI-assisted.
Quality must be enforced by *objective, AI-independent* gates, not by reviewer judgment.
The PPT + Invariant framework (see `Downloads/ppt_invariant_guide.md` and
`shimmy/docs/ppt-invariant-testing.md`) provides that:

- Invariants (`assert_invariant`) are embedded in production engine logic.
- Every checked invariant is recorded in a static log.
- `contract_test` fails if a required invariant was *not actually exercised* — so a
  refactor or AI edit cannot silently drop a semantic guarantee.

**Per-bead acceptance rule (mandatory):** every Phase 1–4 bead MUST:
1. Embed at least one `airframe::invariant_ppt::airframe_invariants::*` check at the
   semantic boundary it touches (e.g. loader → `assert_buffer_within_limit` /
   `assert_alignment` / `assert_chunk_count_within_limit`; bind groups →
   `assert_word_index_in_range`).
2. Add a contract test under `tests/test_contracts.rs` (CPU-only) that exercises the
   invariant and asserts it via `contract_test`.
3. Keep `cargo test -p airframe --test test_contracts` green.

**Reusable invariants available now** (`src/invariant_ppt.rs::airframe_invariants`):
`assert_buffer_within_limit` (2 GB wgpu cap, issue #206), `assert_alignment` (256 B),
`assert_chunk_count_within_limit` (≤ `MAX_CHUNKS` = 8), `assert_word_index_in_range`.

**Phase 0 Completion Gate:**
- [x] `src/invariant_ppt.rs` present, clippy-clean, compiles.
- [x] `tests/test_contracts.rs` baseline + loader contract suite passes (CPU-only, `--test-threads=1`).
- [x] CI runs `cargo test -p airframe --test test_contracts -- --test-threads=1`.

---

## Executive Summary

**Current state:** Three inference entry paths, orphaned `generate_isf`, single-buffer load hack, legacy backend files, two fabrics with confusing names. Clean pieces: `control.rs` trait, `airframe_observe`, `core/dequant/*`.

**Target state:** One load strategy (multi-buffer), one inference facade (`GpuRuntime::generate_isf`), explicit fabric boundaries, legacy pruned, verification consolidated.

**Execution order:** Load → Orchestration → Product Integration → Cleanup. This respects dependencies and attacks accumulation before polishing.

---

## PPT Gate Matrix — objective per-bead invariant gates

Every refactor bead below MUST, when implemented, embed its gate from
`airframe::invariant_ppt::airframe_invariants` into production code and add the
named contract/property test to `tests/test_contracts.rs` (run with
`--test-threads=1`). These gates are the objective, AI-independent acceptance
criteria. Each `bd` issue also carries this spec in its notes.

| Bead (bd id) | Plan bead | Invariant gate(s) | Contract/property test |
|---|---|---|---|
| airframe-i88 | — | Foundation: `assert_invariant`/`contract_test`/`property_test` + 4 base gates | `framework_self_test` + baseline (DONE) |
| airframe-2wm | a1-load-multi-buffer-core | `compute_chunk_plan` (align + 2 GB limit + chunk count); `buffer_for_word` word range | `loader_chunk_plan_contract`, `loader_chunk_plan_property`, `loader_chunk_plan_overflow_violation`, `multi_buffer_word_resolution` |
| airframe-woh.1 | (T1 chunking core) | reuse `compute_chunk_plan` | `loader_chunk_plan_contract` |
| airframe-f9p | (probe adapter limits) | `compute_chunk_plan` via `adapter.limits().max_storage_buffer_binding_size` | `loader_chunk_plan_contract` |
| airframe-kjt | (multi-buffer loop) | per-buffer `assert_alignment` + `assert_buffer_within_limit`; `buffer_for_word` | `multi_buffer_word_resolution` |
| airframe-bwb | (replace blob_binding_*) | `buffer_for_word(word_idx, chunk_words, total_words)` + `assert_word_index_in_range` | `multi_buffer_word_resolution` |
| airframe-woh.2  | (T2 WGSL+bind groups) | combo of layout contiguity + read_blob mirror + binding audit | combination below |
| airframe-woh.2.1 | (T2a codegen) | `read_blob_chunk_offset` Rust mirror == WGSL `read_blob` | `wgsl_read_blob_mapping` |
| airframe-woh.2.2 | (T2b repack layouts) | `assert_bind_group_layout_contiguous(num_blobs)` | `bind_group_layout_contiguous` |
| airframe-woh.2.3 | (T2c bind sites + audit) | binding-index audit (WGSL `@binding` vs Rust) | `binding_index_audit` |
| airframe-jaa | a1-bind-group-repack | `assert_bind_group_layout_contiguous` for layer/lm_head/rmsnorm | `bind_group_layout_contiguous` |
| airframe-cyk | (layer/lm_head layouts) | `assert_bind_group_layout_contiguous` | `bind_group_layout_contiguous` |
| airframe-1my | (rmsnorm + sites) | `assert_bind_group_layout_contiguous` | `bind_group_layout_contiguous` |
| airframe-a6f | a1-wgsl-read-blob-uniform | `read_blob_chunk_offset` mirror == WGSL dynamic chunk via uniform | `wgsl_read_blob_mapping` |
| airframe-z9t | (binding audit gate) | exhaustive: every WGSL `@binding(N)` blob idx in `0..num_blobs`, uniform at 20, matches Rust | `binding_index_audit` |
| airframe-woh.3 | (T3 limits unification) | `compute_chunk_plan` unified; `effective_chunk` identical everywhere | `loader_chunk_plan_contract` |
| airframe-woh.4 | (T4 validation harness) | harness runs full PPT contract suite as gate | (CI `cargo test ... --test-threads=1`) |
| airframe-5s8 | (multi-buffer unit tests) | `buffer_for_word` + `compute_chunk_plan` property tests | `multi_buffer_word_resolution` |
| airframe-de3 | a2-wire-generate-isf | `assert_generation_valid(prompt, produced)` in `generate_isf` | `generation_contract` |
| airframe-0d9 | a2-retire-generate | swap `generate_isf` in as `generate`, delete old impl (no deprecation); `assert_generation_valid` enforced on canonical path | `generation_contract` |
| airframe-tvd | a3-fabric-boundary-doc | `docs/fabric-boundary.md` exists + linked | `doc_fabric_boundary` |
| airframe-br6 | a4-document-engine-role | `runtime/engine.rs` carries CPU-reference header | `engine_doc` |
| airframe-fwc | a4-retire-backend-top-level | `backend::pipeline`/`wgpu`/`tdr` removed + unreferenced | `legacy_backend_absent` |
| airframe-e2z / airframe-woh | epics | track completion of child beads' contract tests | — |

**Roadmap / deliberate limitation:** `MAX_CHUNKS = 8` (≈16 GB of weights) is an
*intentional* cap, not an oversight. It matches the commodity-model target
(≤13B-class Q4, which fits the RTX 3060's 12 GB VRAM); on that hardware VRAM is the
real wall, not the chunk count. Models needing more than 8 buffers are **rejected at
load time with a clear, roadmap-bearing message** that larger-model support is
deferred to a later release. Dynamic multi-buffer support (raising the cap + making
the WGSL `read_blob` loop over N chunks) is tracked future work, not part of this
plan. (Pin the target date — e.g. Q1 2027 — when confirmed.)

## Phase 1: Multi-Buffer Load Strategy (Highest Leverage)

### Rationale
Single `gpu_buffer` + sub-range bindings is the root of issue #206 and alignment fragility. Moving chunking to load time (N independent buffers) eliminates `max_buffer_size` risk and tail-alignment issues. Everything downstream becomes simpler.

### Beads

#### BEAD: `a1-load-multi-buffer-core`
**Title:** Implement multi-buffer loader core in `BindlessModel`

**Description:**  
Replace the single-giant-buffer design in `src/backend/bindless/loader.rs` with N independent `wgpu::Buffer`s. Each buffer holds one chunk of the GGUF file, sized ≤ `effective_chunk` where `effective_chunk = min(adapter.max_storage_buffer_binding_size, 2_000_000_000)` (floor 2GB, capped by adapter's real limit).

**Files to touch:**
- `src/backend/bindless/loader.rs` (primary)
- `src/backend/bindless/metadata.rs` (read-only, no changes needed)

**Changes:**
1. Change `pub gpu_buffer: wgpu::Buffer` → `pub gpu_buffers: Vec<wgpu::Buffer>`
2. Remove `pub dummy_buf: wgpu::Buffer` (no longer needed; each buffer is self-contained)
3. Add `pub effective_chunk: u64` field (stores the computed chunk size)
4. In `load_from_disk()`:
   - Probe adapter limits: `adapter_limits.max_storage_buffer_binding_size`
   - Compute `effective_chunk = adapter_limits.max_storage_buffer_binding_size.min(2_000_000_000)`. Ensure 256-byte aligned: `effective_chunk = (effective_chunk / 256) * 256`
   - Compute `num_chunks = (file_size + effective_chunk - 1) / effective_chunk`
   - Loop `i in 0..num_chunks`:
     - `offset = i * effective_chunk`
     - `chunk_size = (file_size - offset).min(effective_chunk)`
     - Create buffer from mmap slice: `device.create_buffer_init(&BufferInitDescriptor { contents: &mmap[offset..offset+chunk_size], ... })`
     - Push to `gpu_buffers`
5. Remove `blob_binding_0/1/2()` methods entirely
6. Add new method: `pub fn buffer_for_word(&self, word_idx: u32) -> (usize, u32)` returning `(buffer_index, word_offset_in_buffer)`. Implementation: `chunk_words = effective_chunk / 4; (word_idx / chunk_words, word_idx % chunk_words)`

**Part A — Objective CPU gate (acceptance bar):**
- `compute_chunk_plan(file_size, adapter_limit) -> ChunkPlan { effective_chunk, num_chunks }`
  in `loader.rs`, embedding `assert_alignment` + `assert_buffer_within_limit` +
  `assert_chunk_count_within_limit`. `load_from_disk` calls it and records
  `effective_chunk`/`num_chunks` on `BindlessModel`.
- `airframe_invariants::buffer_for_word` resolves an absolute word index to
  `(buffer_index, word_offset)` and embeds the word-range invariant.
- Contract/property tests (green): `loader_chunk_plan_contract`,
  `loader_chunk_plan_property`, `loader_chunk_plan_overflow_violation`,
  `multi_buffer_word_resolution`.

**Part B — GPU integration (adapter-validated) — pending:**
- `pub gpu_buffer` → `pub gpu_buffers: Vec<wgpu::Buffer>`; remove `dummy_buf`;
  create one `create_buffer_init` per chunk from the mmap slice.
- Replace `blob_binding_0/1/2()` with per-buffer bindings driven by `buffer_for_word`.
- Validated on the RTX 3060: synthetic >2 GB model → `gpu_buffers.len() == 2`, each
  ≤ 2 GB, total = file_size; small-model (<2 GB) path unchanged. Not part of CPU gate.

**Acceptance Criteria (CPU gate = Part A):** the four contract/property tests above
are green under `--test-threads=1`. (Part B verified on adapter.)

**Dependencies:** None (Phase 1, Task 1)

**Risk:** Low. Data model unchanged (absolute byte offsets still valid across multiple buffers via `buffer_for_word`). Only binding layer changes.

**PPT gate (done):** the chunk-plan math is extracted into `compute_chunk_plan(file_size, adapter_limit)` in `loader.rs` and gated by `airframe_invariants::{assert_alignment, assert_buffer_within_limit, assert_chunk_count_within_limit}`. `load_from_disk` calls it and records `effective_chunk` / `num_chunks` on `BindlessModel`. CPU contract + property tests: `loader_chunk_plan_contract`, `loader_chunk_plan_property`, `loader_chunk_plan_overflow_violation` in `tests/test_contracts.rs`. The remaining N-buffer creation + `buffer_for_word` + removal of `dummy_buf` is the GPU-coupled half (pairs with `a1-bind-group-repack`) and is verified on a real adapter.

---

#### BEAD: `a1-wgsl-read-blob-uniform`
**Title:** Update WGSL shaders to use dynamic chunk count via uniform

**Description:**  
The current `sh_layer_v1.wgsl`, `sh_head_blob.wgsl`, `sh_rmsnorm.wgsl` have hardcoded `BLOB_SPLIT_0 = 500000000u` and `BLOB_SPLIT_1 = 1000000000u`. Replace with a `BlobParams` uniform containing `chunk_words` and `num_chunks`, and update `read_blob()` to compute chunk index dynamically.

**Files to touch:**
- `src/backend/bindless/sh_layer_v1.wgsl`
- `src/backend/bindless/sh_head_blob.wgsl`
- `src/backend/bindless/sh_rmsnorm.wgsl`
- `src/backend/bindless/pipeline/mod.rs` (add uniform buffer creation)
- `src/backend/bindless/pipeline/inference.rs` (pass uniform to bind groups)

**Changes:**
1. In each shader, add:
   ```wgsl
   struct BlobParams { chunk_words: u32; num_chunks: u32; _pad: u32[1]; }
   @group(0) @binding(20) var<uniform> blob_params: BlobParams;
   ```
2. Replace `read_blob(word_idx)` old logic:
   ```wgsl
   // OLD: if word_idx < BLOB_SPLIT_0 { return blob_0[word_idx]; }
   //      else if word_idx < BLOB_SPLIT_1 { return blob_1[word_idx - BLOB_SPLIT_0]; }
   //      else { return blob_2[word_idx - BLOB_SPLIT_1]; }
   
   // NEW:
   let chunk = word_idx / blob_params.chunk_words;
   let offset = word_idx % blob_params.chunk_words;
   if chunk == 0u { return blob_0[offset]; }
   else if chunk == 1u { return blob_1[offset]; }
   else if chunk == 2u { return blob_2[offset]; }
   // ... up to MAX_CHUNKS (8)
   return 0u; // fallback
   ```
3. In `pipeline/mod.rs`, add uniform buffer creation for `BlobParams` in `BindlessPipeline::new()`. Store as `pub blob_params_buffer: wgpu::Buffer`.
4. In `inference.rs`, update each `create_bind_group` to include the uniform at binding 20, and set `blob_params.chunk_words = effective_chunk / 4`, `num_chunks = gpu_buffers.len()`.

**Part A — Objective CPU gate (acceptance bar):**
- Production source of truth for the word→(chunk, offset) mapping is the pure-Rust
  `airframe_invariants::read_blob_chunk_offset(word_idx, chunk_words)` (already in
  `src/invariant_ppt.rs`). The WGSL `read_blob` MUST compute the identical mapping
  (`chunk = word_idx / chunk_words`, `offset = word_idx % chunk_words`).
- Contract test `wgsl_read_blob_mapping` in `tests/test_contracts.rs` pins the
  mirror: it drives `read_blob_chunk_offset` across boundary cases (chunk edges,
  index 0, last index of a chunk, multi-chunk) against the exact integer arithmetic
  the WGSL uses, records the check via `assert_invariant`, and asserts it via
  `contract_test`. Gate must be green under `--test-threads=1`.

**Part B — GPU integration (adapter-validated):**
- Replace `BLOB_SPLIT_0/1` constants in `sh_layer_v1.wgsl`, `sh_head_blob.wgsl`,
  `sh_rmsnorm.wgsl` with the `BlobParams { chunk_words, num_chunks }` uniform and a
  dynamic `read_blob` loop matching `read_blob_chunk_offset`.
- Add uniform-buffer creation in `pipeline/mod.rs`; pass it at binding 20 in
  `inference.rs`. Set `chunk_words = effective_chunk / 4`, `num_chunks = gpu_buffers.len()`.
- Validated on the RTX 3060: shaders compile; small-model (chunk 0) and >2 GB model
  (chunks 0,1) produce correct output. Not part of the CPU gate.

**Acceptance Criteria (CPU gate = Part A):**
- `wgsl_read_blob_mapping` green; `read_blob_chunk_offset` is the single mapping source.
- (Part B) All 3 shaders compile; small + >2GB GPU tests correct on adapter.

**Dependencies:** `a1-load-multi-buffer-core` (needs `effective_chunk` value)

**Risk:** Medium. Shader logic change; must preserve 32-bit-word contract exactly.
Mitigated by pinning the mapping in Rust (Part A) so the WGSL cannot silently drift.

---

#### BEAD: `a1-bind-group-repack`
**Title:** Repack bind group layouts to N contiguous blob bindings

**Description:**  
Current layouts have blob bindings at slots 0, 10, 11 with gaps. New layout: blobs at 0..N-1 (contiguous), aux buffers shifted. For N=8 max, slots 0-7 are blobs, slot 8+ are aux.

**Files to touch:**
- `src/backend/bindless/pipeline/mod.rs` (layer_layout, lm_head_blob_layout, rmsnorm_layout)
- `src/backend/bindless/pipeline/inference.rs` (all `make_bg` closures)

**Changes:**
1. In `mod.rs`, update three `BindGroupLayoutEntry` arrays:
   - For `layer_layout`: entries 0-7 = `Storage { read_only: true }` (blobs), entry 8 = activation_in, entry 9 = temp_state, etc. Shift all non-blob entries up by N-1 positions.
   - Same for `lm_head_blob_layout` and `rmsnorm_layout`.
2. In `inference.rs`, update all `create_bind_group` calls:
   - Loop `i in 0..model.gpu_buffers.len()`: push entry `{ binding: i, resource: model.gpu_buffers[i].as_entire_binding() }`
   - Shift other entries to match new layout indices.

**Part A — Objective CPU gate (acceptance bar):**
- Embed `airframe_invariants::assert_bind_group_layout_contiguous(num_blobs, ctx)` at
  the layout-construction boundary so the blob-binding count is asserted to live in
  the contiguous slots `0..num_blobs-1` and within `[1, MAX_CHUNKS]`. Expose the
  layout-planning as a pure, GPU-free helper (returns the ordered binding indices)
  that calls the invariant, so it is unit-testable without a device.
- Contract test `bind_group_layout_contiguous` in `tests/test_contracts.rs` drives
  the helper for N = 1, 2, 4, 8 (legal) and N = 0, `MAX_CHUNKS+1` (must violate),
  and asserts via `contract_test`. Green under `--test-threads=1`.

**Part B — GPU integration (adapter-validated):**
- Update `layer_layout`, `lm_head_blob_layout`, `rmsnorm_layout` in `pipeline/mod.rs`
  to contiguous blob slots `0..N-1` with aux buffers shifted up.
- Update every `create_bind_group` site (`inference.rs`, `matmul.rs`, `layer.rs`,
  `dequant.rs`) to loop `i in 0..gpu_buffers.len()` pushing blob bindings, then aux.
- Validated on the RTX 3060: existing GPU verify tests pass; bind-group creation
  succeeds for N=1,2,4,8; exhaustive audit that every WGSL `@binding(N)` matches the
  Rust `binding: N`. Not part of the CPU gate.

**Acceptance Criteria (CPU gate = Part A):**
- `bind_group_layout_contiguous` green; layout planner asserts the contiguity invariant.
- (Part B) GPU verify tests pass on adapter; binding-index audit clean.

**Dependencies:** `a1-load-multi-buffer-core`, `a1-wgsl-read-blob-uniform`

**Risk:** Medium-High. Many binding indices shift; easy to miss one. Mitigated by the
Part A contiguity invariant + the Part B exhaustive `@binding` audit on the adapter.

---

### Phase 1 Completion Gate
CPU gate (Part A — objective, required before moving on):
- [ ] `a1-load-multi-buffer-core` Part A: `loader_chunk_plan_contract`,
       `loader_chunk_plan_property`, `loader_chunk_plan_overflow_violation`,
       `multi_buffer_word_resolution` (contract test, pending implementation).
- [ ] `a1-wgsl-read-blob-uniform` Part A: `wgsl_read_blob_mapping` green.
- [ ] `a1-bind-group-repack` Part A: `bind_group_layout_contiguous` green.

GPU integration (Part B — adapter-validated on RTX 3060, not gated by CPU CI):
- [ ] `a1-load-multi-buffer-core` Part B: `Vec<wgpu::Buffer>` + `buffer_for_word` bindings
- [ ] `a1-wgsl-read-blob-uniform` Part B: `BlobParams` uniform wired
- [ ] `a1-bind-group-repack` Part B: contiguous layouts + all bind sites
- [ ] All existing GPU verify tests green
- [ ] Synthetic >2GB test passes on adapter

---

## Phase 2: Unify Inference Entry Points

### Rationale
Currently three live paths: `generate()` (imperative), `generate_isf()` (reactive, ORPHANED), and server calling `pipeline.*` directly. Collapse to one: `generate_isf` as the single source of truth.

### Beads

#### BEAD: `a2-wire-generate-isf`
**Title:** Wire `generate_isf()` as the production inference path

**Description:**  
`generate_isf()` exists but has zero callers (Q1 confirmed). Make it the primary path called by `shimmy_server_gpu`. This fulfills the public changelog promise that ISF is production.

**Files to touch:**
- `src/bin/shimmy_server_gpu/server_inference.rs` (primary)
- `src/runtime/gpu.rs` (verify `generate_isf` signature)

**Changes:**
1. In `server_inference.rs`, locate the direct `pipeline.run_full_model_prefill_chunked_with_cache_state()` calls (lines ~935, ~984).
2. Replace with call to `GpuRuntime::generate_isf(prompt, params, on_token_callback)`.
3. Pass control hooks (if any) via `generate_isf`'s `InferenceControl` parameter (check signature).
4. Remove or comment out the old direct-pipeline code paths.

**Acceptance Criteria:**
- Shimmy server builds and starts.
- Generate request returns tokens (smoke test).
- ISF log file `/tmp/shimmy_isf_run.log` is created (ISF logging is enabled in `generate_isf`).

**Dependencies:** Phase 1 completion (multi-buffer loader must be stable)

**Risk:** Medium. Server logic change; must ensure ISF path produces same output as old direct path.

---

#### BEAD: `a2-retire-generate`
**Title:** Swap `generate_isf` into place as `generate` (delete old imperative impl) after parity proof

**Description:**  
Once `generate_isf` is wired and parity is proven, make `generate` the single public facade that *is* the ISF path — either repoint `generate`'s body at `generate_isf` or rename `generate_isf` to `generate` — and **delete the old imperative implementation in the same bead**. No `#[deprecated]` window; one canonical `generate` remains. Internal callers (e.g. `runtime/engine.rs` tests) already call `generate`, so they pick up the ISF path automatically.

**Files to touch:**
- `src/runtime/gpu.rs`

**Changes:**
1. After parity proof, repoint `generate` at the ISF implementation (or rename `generate_isf` -> `generate`).
2. Delete the old imperative `generate` body.
3. Keep `assert_generation_valid` enforced on the canonical path.

**Acceptance Criteria:**
- Only one `generate` exists and it runs the ISF path.
- Old imperative implementation is deleted (no dead code left behind).
- Tests pass; `assert_generation_valid` still enforced on the canonical path.

**Dependencies:** `a2-wire-generate-isf`

**Risk:** Low. Single-bead swap + delete; no deprecation window, no lingering dead symbol.

---

### Phase 2 Completion Gate
CPU gate (Part A — objective, done):
- [ ] `a2-wire-generate-isf` Part A: `generation_contract` (contract test, pending implementation).
- [ ] `a2-retire-generate` Part A: `generation_contract` (contract test, pending implementation).

GPU integration (Part B — adapter-validated, pending):
- [ ] `a2-wire-generate-isf` Part B: `generate_isf` wired into `shimmy_server_gpu`.
- [ ] `a2-retire-generate` Part B: `generate_isf` swapped in as `generate`, old impl deleted.
- [ ] Shimmy server smoke test passes
- [ ] `generate_isf` ISF logs show activity

---

## Phase 3: Fabric Boundary Contract

### Rationale
Two fabrics with confusingly similar names (`libfse` vs `airframe_observe::isf`) cause mental overhead. Write a one-page contract clarifying their roles. No code move needed yet.

### Beads

#### BEAD: `a3-fabric-boundary-doc`
**Title:** Write fabric boundary contract document

**Description:**  
Create `docs/fabric-boundary.md` explaining the difference between `libfse` and `airframe_observe::isf`, their intended uses, and why they both exist. This prevents future confusion.

**Files to touch:**
- `docs/fabric-boundary.md` (new file)

**Content outline:**
```
# Fabric Boundary Contract

## libfse (crates/libfse/)
- What: Low-level FSE trie / DFA scanner. Aho-Corasick-like multi-pattern matching.
- Use cases: Text scanning (fse_control.rs), server-side rule enforcement, metrics.
- API: `FseMap::compile(rules)`, `ScanCursor`, `scan_with_cursor()`.
- NOT for: Inference orchestration, reactive graphs.

## airframe_observe::isf (crates/airframe_observe/src/isf.rs)
- What: Inference Saturation Fabric. Reactive forward/backward graph for inference orchestration.
- Use cases: `generate_isf()` inference loop, fact-driven token generation, saturation-based fixpoint.
- API: `InferenceSaturationFabric`, `ISFState`, `InferenceFact`, `run_to_fixpoint()`.
- NOT for: Text pattern matching, rule scanning.

## When to use which
- Need to scan text for keywords/patterns? → libfse.
- Need to orchestrate inference steps reactively? → airframe_observe::isf.
- Both? → Use libfse inside an ISF rule (e.g., fse_control implements InferenceControl using libfse).

## Why two fabrics?
Different domains. libfse = string processing DFA. ISF = inference control flow graph. They complement; don't merge.
```

**Acceptance Criteria:**
- Document reviewed and approved (by you).
- Added to repo docs folder.
- Link referenced in `README.md` or `AGENTS.md`.

**Dependencies:** None (can be done anytime; recommended early to prevent confusion)

**Risk:** None. Pure documentation.

---

## Phase 4: Prune Legacy Surface

### Rationale
Top-level `backend/pipeline.rs`, `backend/wgpu.rs`, `backend/tdr.rs` are legacy/test utilities (Q2 confirmed). Retire them to reduce surface area. Keep `runtime/engine.rs` (CPU reference, Q3 confirmed active).

### Beads

#### BEAD: `a4-retire-backend-top-level`
**Title:** Remove legacy top-level backend files

**Description:**  
Delete `src/backend/pipeline.rs`, `src/backend/wgpu.rs`, `src/backend/tdr.rs`, `src/backend/tests.rs` (or fold useful parts into `backend/bindless/`). Confirm no production callers beyond `backend/tests.rs`.

**Files to touch:**
- `src/backend/pipeline.rs` (delete)
- `src/backend/wgpu.rs` (delete)
- `src/backend/tdr.rs` (delete)
- `src/backend/mod.rs` (remove module declarations)
- `src/backend/tests.rs` (delete or move relevant helpers to `backend/bindless/tests.rs`)

**Pre-check:** Run `grep -rn "backend::pipeline\|backend::wgpu\|backend::tdr" src --include=*.rs` and confirm only `backend/tests.rs` references them.

**Changes:**
1. Delete the four files.
2. In `src/backend/mod.rs`, remove lines like `pub mod pipeline;`, `pub mod wgpu;`, `pub mod tdr;`, `pub mod tests;`.
3. If `tests.rs` had useful helpers, copy them to `backend/bindless/tests.rs`.

**Acceptance Criteria:**
- Code compiles without errors.
- All tests pass (bindless tests still run).
- No broken imports remain.

**Dependencies:** Phase 1 + Phase 2 complete (ensure bindless path is stable before deleting anything)

**Risk:** Low-Medium. Deletions are irreversible; verify no callers first.

---

#### BEAD: `a4-document-engine-role`
**Title:** Document `runtime/engine.rs` role clearly

**Description:**  
Add a header comment to `runtime/engine.rs` explaining its purpose: CPU reference implementation for parity testing against GPU. Not the primary inference path.

**Files to touch:**
- `src/runtime/engine.rs`

**Changes:**
1. At top of file, add:
   ```rust
   /// CPU Reference Engine
   /// 
   /// Purpose: Provide a CPU-only inference implementation for parity testing
   /// against the GPU engine (GpuRuntime). Used by shimmy_eval.rs benchmarks.
   /// 
   /// DO NOT use for production inference. Use GpuRuntime::generate_isf() instead.
   ```

**Acceptance Criteria:**
- Comment present and clear.
- No functional changes.

**Dependencies:** None

**Risk:** None. Documentation only.

---

### Phase 4 Completion Gate
- [ ] `a4-document-engine-role` (`engine_doc` contract test, pending implementation).
- [ ] `a4-retire-backend-top-level` (depends on Phase 1+2 Part B / GPU stable).
- [ ] Build clean, tests pass

---

## Phase 5+: Deferred (Optional Future Work)

These are explicitly deferred until Phases 1-4 complete. Do not start yet.

- **Phase 5:** Consolidate `validation/*` + `conformance/*` if overlap proven.
- **Phase 6:** Thin the 9+ diagnostic binaries (`src/bin/*`).
- **Phase 7:** Create dedicated "load-strategy" module for GGUF reading policies (emerges naturally after multi-buffer loader is stable).

---

## Execution Checklist

Mark each bead as you complete it:

### Phase 1: Multi-Buffer Load  (Part A = CPU gate, Part B = adapter)
- [ ] `a1-load-multi-buffer-core` Part A  ·  [ ] Part B (GPU)
- [ ] `a1-wgsl-read-blob-uniform` Part A  ·  [ ] Part B (GPU)
- [ ] `a1-bind-group-repack` Part A  ·  [ ] Part B (GPU)

### Phase 2: Unify Inference
- [ ] `a2-wire-generate-isf` Part A: `generation_contract` (`assert_generation_valid`
       embedded on `generate_isf` return path)  ·  [ ] Part B (wire into `shimmy_server_gpu`, GPU)
- [ ] `a2-retire-generate` Part A: shares `generation_contract` (canonical-path invariant)
       ·  [ ] Part B (swap `generate_isf` → `generate`, delete old impl, GPU parity proof)

### Phase 3: Fabric Contract
- [ ] `a3-fabric-boundary-doc`: `docs/fabric-boundary.md` to be created + linked in README;
       `doc_fabric_boundary` contract test. (No GPU part.)

### Phase 4: Prune Legacy
- [ ] `a4-document-engine-role`: CPU-reference header on `runtime/engine.rs`; `engine_doc`
       contract test. (No GPU part.)
- [ ] `a4-retire-backend-top-level`: destructive deletion of
       `backend/{pipeline,wgpu,tdr}.rs`; dependency is Phase 1 + Phase 2 **Part B**
      (GPU path stable) per the bead. Part A gate `legacy_backend_absent` is authored
      to assert file+reference absence and will be added *with* the deletion so the
      CPU gate stays green (an absence test cannot pass while the files still exist).

---

## Risk Mitigation

- **Every phase has a completion gate.** Do not proceed until gates pass.
- **Keep working branches atomic.** One bead per branch; PR each separately.
- **Test coverage:** Maintain all existing tests; add new ones for multi-buffer path.
- **Rollback plan:** Each bead is reversible (git revert) if something breaks.

---

## Confidence Level

**Current baseline:** 75-80% (due to unknown adapter limits, orphaned ISF path).

**After Phase 1:** 85-90% (multi-buffer removes biggest unknown).

**After Phase 2:** 95%+ (single inference path, ISF proven).

**Final target:** >95% confidence across entire plan.

---

*End of Refactor Plan*
