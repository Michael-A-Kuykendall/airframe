# Changelog

All notable changes to Airframe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

**One command** runs the full pipeline: `scripts/certify_math.bat <family-id> <gguf> "multi-token prompt"`. Exit code 0 = certified. Full write-up: [docs/CERTIFICATION.md](docs/CERTIFICATION.md) in the shimmy repo.

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
- `.beads/` — bead tracking system for certified models

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