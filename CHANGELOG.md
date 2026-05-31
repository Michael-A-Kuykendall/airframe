# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.0] — 2026-05-31

### Added
- **TurboShimmy INT4 KV cache** (`SHIMMY_KV_QUANT=int4`): per-head-vector bias-8 nibble
  encoding compresses KV cache by ~7× (e.g. 512 MB → ~72 MB for Llama-3.2-3B at ctx=2048)
  with zero measured retrieval degradation at ctx≤256.
- New WGSL compute shaders: `quantize_kv_int4.wgsl`, `decode_kv_int4.wgsl`.
- `KVCache::new_int4` constructor; `KVCache::is_int4()` query method.
- `BindlessPipeline::run_layer_with_cache_int4` — 3-encoder per-layer decode path.
- `BindlessPipeline::requantize_all_kv_int4` — bulk F32→INT4 requantize after prefill.
- `SHIMMY_KV_QUANT` env var on `shimmy_server_gpu` (default: `f32`).
- `SHIMMY_PREFILL_CHUNK` env var: chunked prefill batch size (default: `64`).
- `SHIMMY_VRAM_LIMIT_MB` env var: VRAM budget warning threshold (default: `10500`).
- `/v1/models` response now includes `"kv_mode": "f32"|"int4"` field.
- `docs/turboshimmy.md`: feature spec, needle bench results, deployment notes.
- `docs/turboshimmy-release-checklist.md`: production readiness gate tracking.
- `scripts/needle_bench.py`: needle-in-a-haystack evaluation script against live server.
- README: Memory Optimization section documenting all server env vars.

### Fixed
- **Windows TDR crashes during prefill**: GPU command encoder is now submitted and polled
  once per layer in the F32 prefill loop (`run_full_model_prefill_chunked_with_cache_state`),
  preventing any single GPU dispatch from exceeding the ~2 s Windows TDR watchdog limit.
  Previously crashed at seq_len ≥ 144; now verified stable at seq_len ≈ 224 (ctx=256).
- `scripts/needle_bench.py`: fixed HTTP 400 from float seed (now cast to `int`); fixed
  connection-reset on stop-token emission; added fuzzy `check_pass`.

### Changed
- Version bump: `0.1.1` → `0.2.0`.

---

## [0.1.1] — 2026-05-15

Initial public release with Bindless WebGPU pipeline, FSE subsystem, and OpenAI-compatible
HTTP server. Supports Llama, Gemma, Phi, Qwen2, Qwen3 architectures via GGUF Q4_K_M and
related quantization types.
