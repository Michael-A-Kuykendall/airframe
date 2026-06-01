# Airframe — Copilot Instructions

## Project overview

Airframe is a pure-Rust WebGPU inference engine for GGUF models. No C++, no llama.cpp. Runs on NVIDIA/AMD/Intel/Apple Silicon via wgpu.

- **Crate:** `airframe` v0.2.1 — published to crates.io
- **Public repo:** https://github.com/Michael-A-Kuykendall/airframe (master branch)
- **Sub-crate:** `libfse` v0.1.3 — published to crates.io (at `crates/libfse/`)
- **Workspace member:** `crates/libfse` only. Console CLI is deferred.

## Active branches

| Branch | Purpose | Status |
|--------|---------|--------|
| `master` | Stable release — airframe 0.2.1 | Green CI |
| `feat/math-pack-turboquant-cas` | Math-pack research checkpoint | Planning only — do NOT merge until CAS work arrives |

## Math-pack / CAS branch — `feat/math-pack-turboquant-cas`

This branch holds planning and research docs for the math parsing pipeline, which will eventually couple with CAS (Computer Algebra System) work being built separately.

**What's in it:**
- `docs/turboquant-wgsl-implementation-plan.md` — Full implementation plan for TurboQuant INT4 KV cache compression in WGSL. Status: PLANNING. Do not start implementation until approved. Target branch: `feat/turboquant-wgsl`.
- `docs/spike-turboquant.md` — Research spike on KV cache quantization (TurboQuant, KVQuant, Open-TQ-Metal)
- `docs/spike-wgsl-turboquant-port.md` — WGSL port analysis
- `docs/spike-lcm-rlm.md` — Lossless Context Management + Recursive Language Models (arXiv:2605.04050)
- `docs/spike-pageindex.md` — PageIndex vectorless RAG via document tree structure
- `scripts/math_entropy_probe.py` — Entropy probe for math token distributions
- `artifacts/battery_int4_*.txt` — INT4 KV cache battery test artifacts from 2026-05-30

**Architecture coupling:**
- Math intent detection lives in `tests/math_pack_detection.rs` on `master` (Math Pack v0.1.0)
- Detection pipeline: `split_clauses()` → `normalize()` → `FseScanner` → `passes_integer_guard()`
- When CAS work arrives here, look for this branch and wire it in

## Key operational notes

- GPU server readiness: `GET http://127.0.0.1:8080/api/repro/queue` (not /health or /v1/models)
- Never start a second server on port 8080 — silent crash
- Large prompt submissions on Windows Git Bash: use temp files, not inline args
- INT4 KV cache: already shipped in v2.1.0 of shimmy. WGSL kernels: `sh_kv_pack_int4.wgsl` / `sh_kv_unpack_int4.wgsl`

## Deferred / do not touch

- `crates/console/` — scaffolded, not implemented, keep isolated
- Vision code — all on dedicated archived branches in shimmy-private, not in airframe
