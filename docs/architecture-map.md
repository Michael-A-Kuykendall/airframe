# Airframe — Architectural Map (as-built)

> Goal: see the actual physical machine before simplifying it. "What connects to what"
> at the file/module level, where the complexity piled up, and where it can be cut.
> Source: `airframe` workspace clone at `sideline/v0.2.8-base`, branch `spike/206-…`.
> This is a MAP only — no changes yet.

---

## 1. Crate / module layering

```
┌──────────────────────────────────────────────────────────────────────────┐
│ BINARIES  src/bin/*                                                          │
│  shimmy_server_gpu (+server_inference) · shimmy_eval · layer_dump_gpu ·     │
│  quant_verify · frontier_compare · cert_check · probe_tokens ·              │
│  vault_generator · vault_seed · window_boundary_probe                       │
└───────────────┬───────────────────────────────┬───────────────────────────┘
                │ server drives pipeline DIRECT   │ other bins call GpuRuntime
                ▼                                 ▼
┌────────────────────────────┐        ┌──────────────────────────────────────┐
│ backend/bindless/pipeline/* │◄───────│ runtime/gpu.rs  (GpuRuntime)          │
│  dequant · inference · layer│ uses   │  load() · generate() · generate_isf() │
│  · matmul · mod(layouts)   │ pipeline│ runtime/engine.rs (Engine facade)    │
│  run_full_model_…_with_state│        └───────────────┬──────────────────────┘
└───────────────┬────────────┘                        │ (generate_isf only)
                │ binds                                ▼
┌────────────────────────────┐        ┌──────────────────────────────────────┐
│ BindlessModel (loader.rs)   │        │ airframe_observe crate                │
│  ONE gpu_buffer + N sub-    │        │  isf  = Inference Saturation Fabric   │
│  range blob bindings        │        │  facts · observer · output · plan ·   │
│  + BindlessMetadata         │        │  session · internal/tdr               │
└───────────────┬────────────┘        └──────────────────────────────────────┘
                │ reads GGUF
                ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐
│ core/model.rs   │  │ core/dequant/*   │  │ core/spec.rs │  │ core/ggml_   │
│ GgufTensorInfo  │  │ q4_0 q4_k q5_0   │  │ ModelSpec    │  │ types.rs     │
│ core/weight_id  │  │ q5_k q6_k q8_0   │  │             │  │ core/tensor  │
└─────────────────┘  └──────────────────┘  └──────────────┘  └──────────────┘

CONTROL PLANE (per-token intervention) — control.rs defines the trait:
   InferenceControl trait + InferenceEvent + ControlDecision
   ├─ fse_control.rs        → libfse Aho-Corasick scan      (uses libfse crate)
   ├─ math_bypass_control.rs→ arithmetic bypass
   └─ schoolmarm_control.rs → grammar constraint           (uses schoolmarm)

TWO FABRICS (different things — easy to conflate):
   • libfse crate           = low-level FSE trie / DFA scanner (TEXT rules)
   • airframe_observe::isf  = Inference Saturation Fabric (reactive inference graph)

TWO VERIFICATION SUBSYSTEMS:
   • validation/*  (artifacts · errors · evidence · projection · slice_validator)
   • conformance/* (diff · fixtures)

OBSERVE ("rank the numbers"): airframe_observe — facts/observer/output/plan/session
```

---

## 2. One inference — physical data flow

```
GGUF file on disk
   │ BindlessMetadata::new()            reads tensor offsets / data_start
   ▼
BindlessModel::load_from_disk()
   │ device.create_buffer(ONE gpu_buffer = full file size)
   │ + N sub-range blob bindings (blob_0..blob_N)   ← THE 206 BUG LIVES HERE
   ▼
GpuRuntime::load()
   │ builds KVCache, dequants output head → output_head_f32, tokenizer, ModelSpec
   │
   ├─ generate()                       [imperative loop]
   │     embed dequant → prefill chunked → decode loop → sampling
   │       └─ each step: InferenceControl.intervene(event)   ← fse / math / schoolmarm
   │
   └─ generate_isf()                   [reactive]
         ISFState + InferenceFacts → airframe_observe::isf::InferenceSaturationFabric
         .run_to_fixpoint()   (wraps the SAME pipeline calls as generate)

SHIMMY SERVER (the live product path) BYPASSES GpuRuntime entirely:
   server_inference.rs → pipeline.run_full_model_prefill_chunked_with_cache_state() DIRECT
                        + libfse metrics / text scan
   (no GpuRuntime, no generate_isf — it drives the bindless pipeline straight)
```

---

## 3. The Rube Goldberg inventory — "two of everything"

| # | Duplication / pile-up | Where | Note |
|---|----------------------|-------|------|
| 1 | **Two inference orchestrators in the facade** | `gpu.rs::generate` (imperative) vs `gpu.rs::generate_isf` (reactive) | same pipeline, two control styles stacked |
| 2 | **Server bypasses the facade** | `server_inference.rs` calls `pipeline.*` directly, not `GpuRuntime` | 3rd live entry path (facade / engine / direct) |
| 3 | **Two "fabric" libraries** | `libfse` (text DFA) vs `airframe_observe::isf` (inference graph) | confusable names; different jobs, but both called "fabric" |
| 4 | **Two verification subsystems** | `validation/*` vs `conformance/*` | overlapping intent, separate trees |
| 5 | **backend top-level vs backend/bindless** | `backend/pipeline.rs`, `backend/wgpu.rs`, `backend/tdr.rs` sit beside `backend/bindless/` | likely legacy/alt path beside the real bindless engine — **needs confirmation** |
| 6 | **Four control modules** | `control.rs` (trait) + 3 impls | trait is clean; spread across 4 files |
| 7 | **Many diagnostic binaries** | 9+ `src/bin/*` tools | overlapping verification intent (layer_dump, quant_verify, frontier_compare, cert_check…) |
| 8 | **The blob hack (206)** | `loader.rs` 3 hardcoded sub-range bindings, remainder dumped in blob_2 | the symptom this whole spike started from |

---

## 4. What is actually clean (keep it)

- **`control.rs` InferenceControl trait** — a genuine, small control-plane abstraction.
  Plug-in `InferenceControl` implementors (`fse_control`, `math_bypass`, `schoolmarm`)
  intervene per token. This is the "deliberate control plane" idea, realized as a trait.
- **`airframe_observe`** — the "rank the numbers from the GGUF" crate: `facts` + `observer`
  + `isf` separation is coherent; it is the verification/observability layer done right.
- **`core/dequant/*`** — one module per quant type, no cross-talk. Clean.
- **Single data model**: tensors are absolute byte offsets in ONE buffer; chunking is a
  *binding-layer* concern only, so metadata/spec never move. (This is why the multi-buffer
  strategy from the spike discussion is low-risk to the data model.)

---

## 5. Simplification seams (where complexity can be cut)

1. **Unify the inference entry point.** Make `shimmy_server_gpu` go through `GpuRuntime`
   (or make `GpuRuntime` the *only* facade) instead of driving `pipeline.*` directly.
   Kills duplication #2 and collapses #1 toward one path. Biggest single win.

2. **Disambiguate the two fabrics by role, not name.**
   - `libfse` = text/rule scanning DFA → keep for `fse_control` + server text scan.
   - `airframe_observe::isf` = inference orchestration graph → keep for `generate_isf`.
   Document the boundary so future work stops conflating them. (No code move needed yet —
   just a Written-Down contract.)

3. **Resolve the blob design at the load layer** (the earlier "multiple real buffers"
   strategy): split the model into N independent `wgpu::Buffer`s at `load_from_disk`,
   each ≤ `effective_chunk`. Removes the single-buffer `max_buffer_size` risk AND the
   tail-alignment fragility. Supersedes the sub-range hack cleanly.

4. **Confirm & prune `backend/*` top-level vs `backend/bindless/*`.** If the top-level
   `pipeline.rs`/`wgpu.rs`/`tdr.rs` are dead/legacy, retire them so there is ONE engine.

5. **Consolidate `validation/*` + `conformance/*`** if their intent overlaps (verify first).
   One verification tree, not two.

6. **The "control plane for reading GGUFs" you remember may never have been built.**
   What exists is `BindlessMetadata`/`loader.rs` (read+upload) — no higher-level
   *strategy* layer for load/chunk decisions. If you want a deliberate load-strategy
   module (probe adapter → decide chunking → load), that is a NEW small module, not
   a refactor of what's there. The multi-buffer strategy (#3) is the first cut of it.

---

## 6. Open questions — RESOLVED (via targeted greps)

**Q1: Is `generate_isf` wired?**  
Answer: **NO.** Zero callers found. `generate_isf()` is defined in `gpu.rs` but never invoked by any bin or server. It is currently ORPHANED code despite being declared "production path" in public changelog. This is a critical gap.

**Q2: Are top-level `backend/pipeline.rs`, `backend/wgpu.rs`, `backend/tdr.rs` live?**  
Answer: **LEGACY/TEST ONLY.** Only referenced in `backend/tests.rs` (`LogitMaskPipeline`, `WgpuContext`). Not used by production bindless path. Safe to retire or fold into test utilities.

**Q3: What is `runtime/engine.rs` role?**  
Answer: **CPU inference engine facade.** Used by `shimmy_eval.rs` (benchmarks evals against CPU vs GPU). Contains `Engine::new(model)` + tests. Separate from `GpuRuntime`. Role: CPU reference implementation for parity testing. Keep, but don't promote as primary facade.

**Q4: Do `validation/*` and `conformance/*` overlap?**  
Answer: **Both active, different purposes.** `validation/*` used by `core/error.rs`, `runtime/multi_token_engine.rs`, `validation/artifacts.rs`, `slice_validator.rs`. `conformance/*` used by `diff.rs`, `fixtures.rs`. Validation = error/evidence/projection; Conformance = diff/fixtures/parity. They complement rather than duplicate. Defer consolidation.

---

## 6b. External appraisal — prioritized remediation sequence

A second independent review confirms the map and adds a dependency-ordered execution plan:

1. **Fix the load layer first (the 206 root)**  
   Make multi-buffer strategy real at `BindlessModel::load_from_disk`. Split into N independent `wgpu::Buffer`s, each ≤ effective chunk size. Removes single-buffer `max_buffer_size` risk + tail-alignment fragility. This is the highest-leverage structural change.

2. **Unify inference entry points (kill the three-path Rube Goldberg)**  
   Force product path through the facade:
   - Make `GpuRuntime` the **only** public way to run inference.
   - Change `server_inference.rs` to go through `generate_isf` (not direct pipeline calls).
   - Since `generate_isf` is currently ORPHANED (Q1), wire it up first, prove parity, then retire imperative `generate`.

3. **Disambiguate the two fabrics permanently**  
   Write a one-page boundary contract doc:
   - `libfse` = low-level text/rule DFA scanner (keep for `fse_control` + server text scan).
   - `airframe_observe::isf` = Inference Saturation Fabric (reactive graph). Keep for `generate_isf`.
   No code move needed yet — just stop the name collision.

4. **Confirm and prune obvious dead/legacy surface**  
   - Retire top-level `backend/pipeline.rs`, `backend/wgpu.rs`, `backend/tdr.rs` (Q2 confirmed legacy).
   - Leave `runtime/engine.rs` alone (Q3 confirmed: CPU reference, keep for parity testing).
   - **DO NOT touch** `validation/*` + `conformance/*` yet (Q4: they complement; deleting verification early creates ghosts).
   - Leave diagnostic binary zoo alone until core paths unified.

5. **Only after the above**  
   - Consolidate verification trees if overlap is proven.
   - Thin the 9+ diagnostic binaries.
   - The "control plane for reading GGUFs" can emerge as a small load-strategy module on top of the now-clean multi-buffer loader.

**Why this order:** Load → orchestration → product integration → cleanup respects the dependency graph. You attack accumulation (three entry points + blob hack) before polishing clean parts. Resulting shape: clean data model + one load strategy + one inference facade + explicit control plane + explicit observe fabric. Lowest long-term debt surface.

---

## 7. TL;DR for the "egg rolling down the hill"

The machine is not uniformly garbage — the **control trait**, the **observe crate**, and
the **dequant tree** are deliberate and clean. The Rube Goldberg-ness is *accumulation*:
three inference entry paths, two fabrics with confusingly similar names, two verification
subtrees, a likely-legacy backend layer, and a 3-binding blob hack bolted onto a single
giant buffer. The highest-leverage simplifications are (1) one inference entry point and
(2) moving the chunk decision up into the load layer as N real buffers. Both shrink the
surface area without touching the clean parts.
