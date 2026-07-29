# Changelog

## [0.3.0] — 2026-07-29 — Certification & Model Expansion Release

### Highlights
- **21 models certified** across **10 model families** with full 5-gate verification
- **Qwen3 full support** — QK-norm norm bank fix, RoPE head_dim correction, multi-token prefill resolved
- **Gemma2 support** — architecture recognition and verification (2B and 9B)
- **CPU reference stack removed** — 15K lines deleted, 6.5K added. Cleaner, smaller, faster
- **New certification pipeline** — plan-vs-peel structural audit, quant_verify dequant gate, dual-peel numerical gate, decode≡prefill gate

### Certified Models (Level 1 — 5 gates: deq+peel+num+dec+log)

| Family | Models | Quants |
|--------|--------|--------|
| **llama** | tinyllama-1.1b, llama-3.2-1b, llama-3.2-3b | Q4_0, Q4_K_M, Q6_K |
| **qwen3** | 0.6B, 1.7B, 4B, 8B, 4B-Thinking | Q4_K_M |
| **qwen2** | 0.5B, 1.5B, 7B | Q4_K_M |
| **phi3** | 3.5-mini | Q4_K_M |
| **phi2** | 2.7B | Q4_K_M |
| **starcoder2** | 3B | Q4_K_M |
| **gemma2** | 2B, 9B | Q4_K_M |
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
18 commits since v0.2.12 — see git log `v0.2.12..HEAD` for full list.