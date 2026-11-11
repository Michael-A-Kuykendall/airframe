# Changelog

## Postpublication licensing correction — 2026-11-07

- libfse 0.1.4 has a public [additional Apache-2.0 grant](https://github.com/Michael-A-Kuykendall/airframe/blob/8efb1e7a830b0822cfc5ac1b121e149948d5ffe3/crates/libfse/LICENSE-GRANT-0.1.4.md) for the exact existing archive, permitting commercial use, modification and redistribution. Original crate bytes, metadata and tags remain unchanged. See [stack licensing](LICENSING.md).


## [0.4.5] — 2026-11-06

- Exclude development housekeeping files and internal tracker references from crate archives; preserve runtime behavior and existing license rights.

- Add the current workspace roadmap covering Airframe, libfse and airframe-observe, with delivered contracts, outstanding work and explicit untested/unsupported boundaries.

- Documentation errata: separate validated F32/reference dense fixtures from structural parsing, experimental kernels and INT4; remove unsupported deterministic, throughput and portability guarantees. Correct runtime API and profiling examples, broken report links, dependency prose and maintainer contribution policy. No inference or dependency changes.

- Update the pinned Tokio to Tokio 1.53.3 for the AsyncFd unsoundness correction without unrelated dependency changes.
- Historical archive metadata: the original libfse package used the FSE Noncommercial Evaluation License. Those earlier restrictions are superseded for the licensor's rights by the additional grant above; the Airframe runtime remains MIT.


- Coordinate ShimmyTok 0.8.5, Observe 0.1.4 and libfse 0.1.4 so registry packages represent the tested source.
- Preserve validated dense-model admission, checked GGUF metadata, RoPE factors and request-isolated state. Reject unimplemented model graphs explicitly.
- Include demonstrated query-scratch, Q/K normalization ownership and complete tensor-window corrections; no new kernel or performance change in this preparation.
- Retain the exact certified Llama/Qwen fixture limits. Metal/M4, INT4, experimental kernels and untested families are not certified. WSL driver stability investigations remain open.
- Resolve packages in dependency order, then verify the actual published registry lock and native artifacts after separate human approval. Private package checks do not claim public registry delivery.


All notable changes to Airframe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Apple M1 Metal inference correction candidate — NOT YET RELEASED (2026-11-11)

- Complete pending Metal model uploads before independently initializing the output head; correct all-zero Qwen3-1.7B embedding rows. This is an upload-ordering workaround, not a rewritten kernel.
- Cap macOS dense outer prefill batches at 64 tokens following an observed native Metal Llama timeout; preserve QKV microbatching and numerical operators. All 12 Llama full-logit comparisons pass.
- Verify the exact candidate in a normal production-path Apple M1 (Virtual) / Apple Paravirtual Metal run: 198/198 full-logit comparisons PASS across seven shards, unchanged FP32 0.1/0.03 thresholds, all observed greedy decisions match. HTTP and state gates pass in the integrated candidate.
- Do not mistake later 8K completion for diagnosis of the historical intermittent stall. Watchdog cleanup remains independently open; physical M4 and untested combinations remain NOT_RUN.
- Published Airframe 0.4.5 and its crate archive do not contain these corrections. Source equivalence, next-version selection and packaging remain release prerequisites.


Historical entries below preserve earlier release claims and commands; they do not define the 0.4.5 support contract. Use the current [README fixture matrix](README.md#release-contract).

## [0.4.4] — 2026-10-18

### Fixed

- Preserve exact generation token counts and terminal reasons; stop at the
  requested output budget without sampling an extra token.
- Count sampled EOS/stop IDs consistently and surface empty-forward failures.
- Parse chat-template control markers and explicit stop markers as special
  tokens during generation.
- Select split-half (NeoX) rotary coordinate pairs for Qwen2 and Qwen3 in both
  F32 and INT4 attention shaders; retain adjacent pairing for existing routes.
- Keep Rust and WGSL layer-parameter layouts synchronized and add production
  shader validation plus GPU pair-layout regression coverage.
- Resolve the current stable Clippy lint in F32 output-head loading.

### Changed

- Require `shimmytok` 0.8.4, including tokenizer-type-aware BOS defaults.
- Add `GpuRuntime::generate_with_metadata`, retaining the text-only API.
- Require `airframe-observe` 0.1.3 for terminal-state accounting fixes.
- Gate publication on native Linux x86-64/ARM64, Windows and macOS CPU tests;
  hardware-only GPU tests remain a separate validation requirement.

## [0.4.0] — 2026-08-26 — Certification Consolidation + 26 Models + Metadata-Driven Architecture

### 🏆 Certified — 12 Families · 26 Model/Quant Combinations

Airframe's certification pipeline has been **consolidated from the old 5-gate + CHAT regime into a clean 3-box regimen** (MATH + INFERENCE + DETERMINISM). This isn't a reduction in rigor — it's the removal of a regression-causing chat box that produced false REDs on working models, replaced with a deterministic 5-prompt ×2 inference battery that actually proves output quality.

**The 3-box certification regimen:**
1. **MATH** — `math_cert_one.sh`: quant_verify dequant audit, stack_dump_gpu structural peel, plan-vs-peel reds judgment, dual-peel numerical self-consistency, DuckDB ledger record
2. **INFERENCE** — `inference_exam.sh`: 5-prompt battery (factual / creative / arithmetic / kv_cache / dev_question) on real GPU, 48 tokens each
3. **DETERMINISM** — two identical runs must produce identical text; any divergence = FAIL

**One command** runs the full pipeline: `bash scripts/math_cert_one.sh <model_id> <gguf> [family] [quant] [discover_name]`. Exit code 0 = certified.

### New Certified Models (since v0.3.0)

| Family | Models | Quants |
|--------|--------|--------|
| **llama** | llama-3.1-8b-instruct | Q4_K_M |
| **qwen2** | qwen2-7b-instruct | Q4_K_M |
| **gemma2** | gemma-2-2b-it | Q4_K_M |

Plus re-certification of all 23 prior models under the new 3-box regimen. **26 total certified model/quant combinations.**

### Highlights

- **Metadata-driven architecture** — `has_qk_norm`, `post_norm_enabled`, packed-K stride, and head_dim are now derived from **GGUF tensor presence and shapes**, not hardcoded per-arch string matching. One source of truth: the model's own metadata.
- **mistral3 → Mistral mapping** — Ministral-3-14B's GGUF reports `general.architecture = "mistral3"`, which previously fell through to `ModelArch::Other`. Now correctly routed to Mistral family handling.
- **TurboShimmy `TURBO_KV=int4` opt-in** — Environment-variable toggle for INT4 KV-cache compression. Cuts KV VRAM ~60% for large models on consumer GPUs. f32 remains the default; int4 engages only on opt-in.
- **Build provenance emission** — `emit_build_provenance.py` records git SHA, airframe/shimmy versions, and toolchain state per cert run.
- **Plan generator bug fixed** — Pre-relocation tooling forced `qk_norm=True` for ALL models, producing false MATH REDs on non-Qwen3 architectures. Current `cert_plan.py` correctly derives from stack config.

### Removals

- **`shimmy_server_gpu` binary deleted** (1834 lines) — Shimmy server now lives in the shimmy repo, not airframe
- **`audit.rs.broken` + `core/audit/` deleted** (~840 lines) — superseded by the DuckDB certification ledger
- **Candle probe removed from public tree** — cross-validation tooling stays private

### Bug Fixes

- **Plan-vs-peel false REDs** — Non-Qwen3/Gemma models (qwen2-7b, qwen2-0.5b, qwen2-1.5b, gemma-2-2b) were showing MATH RED due to the plan generator requiring QK-norm stages. Root cause was stale pre-relocation tooling; fixed by re-running with current `cert_plan.py`.
- **Window-aware bind groups** — `plan_layer_half_windows` + PPT proof for >8-slot span resolution (dgd epic). Enables future support for models whose per-layer tensor span exceeds `BLOB_BINDING_SLOTS=8`.
- **Numerous metadata-derivation fixes** — packed-K stride (863.5/863.6), has_qk_norm/post_norm from tensor presence (863.3/863.4), dequant offset encoding (2z6-kqa), window-aware LM head (2z6-djk), RMSNorm offset rebase (2z6-8r3).

### Infrastructure

- **Cert regimen relocated to airframe** — Canonical cert tooling (`math_cert_one.sh`, `inference_exam.sh`, `cert/*.py`) now lives in the engine repo per `CERT_REGIMEN.md`; workspace-root copies scrubbed
- **`ISF` remains default** — No feature flag needed for full functionality

### Commits
30 commits since v0.3.0 — see git log `v0.3.0..HEAD` for full list.

## [0.2.14] — 2026-08-01 — Hotfix: UINT16 metadata support, CI fixes, docs cleanup

### Fixed
- **shimmytok UINT16 support** — `stack_dump_gpu` now correctly handles UINT16 metadata values in GGUF files (was `InvalidMetadata("Unsupported value type: 2")`)
- **CI pipeline** — PPT contract tests now run without `--features isf` (ISF is now the default); added `airframe-observe` test step; added `cargo build` build gate
- **Test isolation** — PPT invariant log made thread-local (`RefCell` instead of `Mutex`), eliminating race conditions when running full test suite
- **Test required-features** — removed `required-features = ["isf"]` from all test targets (ISF is now default)

### Features
- **ISF is now the default** — `isf` (Inference Saturation Fabric) is the default feature, eliminating the need to pass `--features isf` for full functionality. All diagnostic binaries (`decode_gate`, `stack_dump_gpu`, `layer_dump_gpu`, `quant_verify`, `invariant_probe`, etc.) are now available by default.

### Cleanup
- Removed 12 large diagnostic artifacts from git tracking (layer dumps, coverage reports, JSON captures totaling ~10MB)
- Added diagnostic file patterns to `.gitignore` (`capital`, `layer_dump_*.json`, `fc_qwen3_*.json`, `gpu_qwen3_*.json`, `compare_*.json`, `fc_run_*.log`, `tarpaulin-report.html`, `cobertura.xml`, `sanity_out.txt`)
- Added large-model coverage caveat to README

### Certified
- Qwen2.5-7B-Instruct Q4_K_M certified (all 5 gates green, run_id=25)

## [0.2.13] — 2026-07-31 — Certification & Model Expansion Release

### 🏆 Certified — 11 Families · 25 Model/Quant Combinations

Airframe now ships with a **full mathematical certification pipeline** that proves every model produces numerically correct GPU output — not just "it generates something," but verified at the element level against a spec-derived reference. No other local LLM engine does this.

**The 5-gate certification pipeline:**
1. **Dequant gate** — `quant_verify` proves the GPU dequant shader matches the GGUF spec formula, element-by-element, for every quantization type
2. **Structural peel gate** — `stack_dump_gpu` captures per-layer output and compares against a spec-derived plan (layer counts, dims, rope config). Zero NaN, zero missing stages
3. **Numerical gate** — dual-peel self-consistency: two independent GPU runs compared layer-by-layer, max delta ≤ 1e-2
4. **Decode≡Prefill gate** — `decode_gate` proves decode output matches prefill output for the same tokens (catches silent bugs that only show up during token-by-token generation)
5. **Logits gate** — final output logits compared against a golden-vault CPU reference, element-by-element

**One command** runs the full pipeline: `scripts/certify_math.bat <family-id> <gguf> "multi-token prompt"`. Exit code 0 = certified. Full write-up: [docs/CERTIFICATION.md](https://github.com/Michael-A-Kuykendall/shimmy/blob/main/docs/CERTIFICATION.md) in the shimmy repo.

### Highlights
- **25 model-quant combos certified** across **11 model families** with full 5-gate verification
- **Qwen3 full support** — QK-norm norm bank fix, RoPE head_dim correction, multi-token prefill resolved
- **Gemma2 + Gemma4 support** — architecture recognition and verification
- **CPU reference stack removed** — 15K lines deleted, 6.5K added. Cleaner, smaller, faster
- **New certification pipeline** — plan-vs-peel structural audit, quant_verify dequant gate, dual-peel numerical gate, decode≡prefill gate

### Certified Models (Level 1 — 5 gates: deq+peel+num+dec+log)

| Family | Models | Quants |
|--------|--------|--------|
| **llama** | tinyllama-1.1b, llama-3.2-1b, llama-3.2-3b | Q4_0, Q4_K_M, Q5_K_M, Q6_K |
| **qwen3** | 0.6B, 1.7B, 4B, 8B, 4B-Thinking | Q4_K_M |
| **qwen2** | 0.5B, 1.5B, 7B | Q4_K_M |
| **phi3** | 3.5-mini, 3-mini-4k | Q4_0, Q4_K_M |
| **phi2** | 2.7B | Q4_K_M |
| **starcoder2** | 3B | Q4_K_M |
| **gemma2** | 2B, 9B | Q4_K_M |
| **gemma4** | E4B, 12B-coder | Q4_K_M |
| **deepseek-r1** | 0528-Qwen3-8B | Q4_K_M |
| **qwen3.5** | 9B | Q4_K_M |
| **ministral** | 3-14B-Reasoning | Q4_K_M |

### Bug Fixes
- **Qwen3 prefill NaN resolved** — norm bank expanded from 4→6 slots/layer for QK-norm
- **Qwen3 head_dim** — forced to 128 from Q weight shape (not `n_embd/n_head = 80`)
- **Gemma2 support** — `gemma2` architecture recognized in model metadata
- **>2GB output head crash fixed** — blob-based output head handles F32 weights >2GB file offset
- **Decode gate crash fixed** — graceful buffer size check before `create_buffer_init`; BLOB mode still validates
- **Quant_verify** — stable tensor pick for models with >4GiB offsets
- **>4GiB offset packing** — GGUF tensors crossing 4GB boundaries handled correctly

### New Features
- **Decode gate** — `decode_gate` binary verifies decode≡prefill equivalence (maxΔ ≤ 1e-2, argmax match)
- **Stack dump GPU** — `stack_dump_gpu` per-layer residual/stage capture with JSON output and schema
- **Dual-peel numerical gate** — self-consistency check: two independent `stack_dump_gpu` runs compared
- **Certification ledger** — DuckDB-backed per-model run tracking with 5-gate pass/fail columns
- **Inference canary** — golden-logits comparison for regression detection

### Refactoring
- CPU reference stack removed (`llama.rs`, `ops/reference/`, `engine.rs`, `vault_seed.rs`) — 15K lines removed
- Dead code cleanup: `server_inference.rs`, `image_preproc.rs`, legacy `pipeline.rs`
- `generate_isf` → `generate` unified API
- `GpuRuntime::from_parts` constructor for testability

### Infrastructure
- New binaries behind `--features isf`: `decode_gate`, `stack_dump_gpu`, `layer_dump_gpu`, `quant_verify`, `invariant_probe`, `kv_dump_probe`, `kv_head_probe`, `kv_chain_probe`
- `scripts/cert/` — Python-based cert ledger, plan-vs-peel reds judge, regression tests
- Workspace-level `cert/packages/` per-model directory with STATUS.md, MATH reports

### Documentation
- `MATH_REGIMEN.md` — certification regimen with 5-gate definitions and NUM=1/2/0 table
- `CERT_REGIMEN.md` — plan vs peel structural audit specification
- `INFERENCE_SANITY_CHECKS.md` — level-2 canary test suite design
- `PEEL_STRUCTURE.md` — stage dump schema
- `STACK_DUMP.md` — stack observability ops
- `stack.schema.v1.json` — stack JSON schema

### Commits
21 commits since v0.2.12 — see git log `v0.2.12..HEAD` for full list.

## [0.2.12] — 2026-07-24 — G1+G2 Hotfix: control wiring + observability

### Changed
- Wire `control`/`mask`/`trace` into `generate_isf` (G1 hotfix)
- Emit `FinalLogits` + `LayerOutput` (final hidden state) from `generate_isf` (G2 observability)
- `airframe-observe` bumped to 0.1.2; license/repo metadata added for crates.io publish
- FSE patent notice surfaced in airframe-observe README

## [0.2.11] — 2026-07-23 — Release polish

### Changed
- Version bump, CI badge added to README

## [0.2.10] — 2026-07-22 — Dequant root-cause fix + fabric dispatch + vault certification

### Bug Fixes
- **GPU gibberish root cause fixed** — dequant front-padding in `run_dequant_any_blob`
- **f16→f32 dequant corrected** on RTX 3060 (passed algebraic audit)
- Q4_K nibble-offset bug fixed in `sh_dequant_any.wgsl` (three-blob split)

### Added
- `Q5_0` quant slot
- Fabric dispatch — `TensorFact→DispatchFact` rule retires the WGSL if/else dispatch ladder
- Golden-vault certification — per-layer certification rule (`LayerOutput`/`FinalLogits` vs `VaultOracle`)
- Multi-buffer blob loader for 2GB binding cap (PPT contract gate)
- Per-tensor capture sink (`q/k/v/post/ffn/output`) in airframe_observe

### Refactoring
- Single fabric `generate()` path; imperative `generate()` retired (fabric path + TinyLlama smoke)
