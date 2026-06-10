# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.2] — 2026-06-09

### Fixed

- **Multi-architecture GGUF loading** — `model_spec_from_metadata` was hardcoded to
  `llama.*` key prefix. Now uses suffix-based matching, correctly loading Qwen3, Qwen2,
  Gemma, Phi3, and all other non-Llama architectures. This was a silent failure for
  every non-Llama model.

- **Tied embeddings support** — Models where `output.weight` is absent (Qwen3 uses
  tied embeddings sharing `token_embd.weight`) now load correctly. Previously crashed
  with `WeightMissing: OutputProj`.

- **Context cap safety** — Added default context cap (4096) during oracle generation
  to prevent 28+ GB KV cache allocation on large-context models (Qwen2-7B has
  `n_ctx=32768`). GPU inference is unaffected; this only applies to CPU oracle runs.

### Added

- **Golden Reference Vault** — DuckDB-backed certification database populated with
  oracle traces for 22 models across 6 architectures (Llama, Qwen2, Qwen3, Gemma,
  Phi, DeepSeek). Internal tool; not published.

- **airframe-observe crate** — FSE-based inference observation layer powered by
  d0-engine reactive fact engine. Selector-first, single-pass capture of layer
  outputs and logits. Internal; not published.

## [Unreleased] — v0.2.3

### Added

- **>2 GB GGUF blob split** — splits GGUF files across multiple ≤2 GB GPU storage
  buffers, bypassing the WebGPU per-buffer size cap. Enables 7B+ parameter models
  (deepseek-7b, qwen2-7b, MiniCPM-V 2.6 text path) on hardware that previously
  hit the 2 GB ceiling. Also resolves the Gemma-2 output-head limitation
  (output.weight = 2.19 GB) without any special-case code.

- **GPU blob-based LM head matmul** — eliminates the CPU fallback logit path for
  large models. All vocab projection now runs on-device regardless of model size.
  Weight tensor reads use word offsets (`byte_offset / 4`) to prevent `u32` overflow
  on tensor byte offsets > 4 GB.

- **QKV bias + Qwen2 chat template** — adds additive QKV bias support required by
  Qwen2-series models; wires the Qwen2-Instruct chat template for correct
  `<|im_start|>` / `<|im_end|>` framing.

### Planned (next)

- **INT4 KV auto-suggestion** — at server startup, estimate peak VRAM
  (`model_weights + ctx × layers × kv_heads × head_dim × 4 bytes`) and compare
  against `adapter_limits.max_buffer_size`. When the model + F32 KV cache would
  exceed safe limits, print a clear startup advisory:
  ```
  [Airframe] VRAM advisory: estimated peak 13.2 GB exceeds adapter limit 12.0 GB.
  Consider setting SHIMMY_KV_QUANT=int4 to reduce KV cache by ~7x.
  ```
  No automatic mode switch — the user retains explicit control. Targeted at
  7B+ models on ≤12 GB VRAM cards running with large context windows.

---

## [0.2.1] — 2026-06-02

### Fixed

- **wgpu 27 staging-buffer panic on non-Gemma models** (shimmy#205).  
  Two root causes:
  1. `GpuRuntime::load()` set `max_buffer_size` to `max_storage_buffer_binding_size`.
     On some older GPU/driver stacks (e.g. GTX 1050 Ti) these limits differ, causing
     a wgpu validation error when the 608 MB model buffer was created. Now uses
     `adapter_limits.max_buffer_size` instead.
  2. A pre-flight size guard now returns a clear `Err` if the model file exceeds the
     GPU's storage-buffer binding limit, instead of letting wgpu defer the error to a
     later `map_async` call (where it appeared as a cryptic "Staging Buffer is invalid"
     panic in wgpu 27).
- **Spurious WARNING logs for standard Llama/Mistral/Phi/Qwen models**.  
  `post_attention_norm` and `post_ffw_norm` tensors only exist in Gemma / Gemma2
  architectures. Absence warnings are now suppressed for all other model families.
- Added `device.on_uncaptured_error` handler so any future wgpu validation errors
  produce a descriptive `[Airframe] GPU error:` message instead of a fatal panic.

---

## [0.2.0] — 2026-05-31

### The headline: TurboShimmy INT4 KV cache

This release ships **TurboShimmy** — a per-head-vector INT4 KV cache compression system
that cuts KV VRAM usage by ~7× with no measured retrieval loss at the context lengths that
matter for most chat workloads. Enable it with a single environment variable:

```bash
SHIMMY_KV_QUANT=int4 cargo run --bin shimmy_server_gpu --release
```

KV cache example (Llama-3.2-3B, `ctx=2048`): **512 MB → ~72 MB**.

This was designed top-down for [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy),
the OpenAI-compatible server powered by Airframe. Shimmy 2.1.0 will surface
`SHIMMY_KV_QUANT` as a first-class user config option — enabling memory-constrained
deployments (8 GB VRAM cards, cloud spot instances, consumer laptops) to run 7B+ models
with a context window that would otherwise OOM. See the
[Shimmy integration roadmap](#shimmy-21-integration-targets) below.

---

### Validation results

All 10 smoke-tested model families pass (TinyLlama through Qwen2-7B). Needle-in-a-haystack
retrieval at ctx=256 is **identical between F32 and INT4** — zero degradation measured on
Llama-3.2-3B across 10%, 50%, and 90% depth probes.

| Model | ctx=256 needle (INT4) | smoke test |
|---|---|---|
| Llama-3.2-3B | 2/3 pass (matches F32) | ✅ |
| deepseek-llm-7b | — | ✅ |
| deepseek-coder-6.7b | — | ✅ |
| qwen2-7b | — | ✅ |
| TinyLlama, Llama-1B, phi-2, starcoder2, gpt2, Qwen3-0.6B | — | ✅ |

Full artifact: `artifacts/model_smoke/smoke_20260531_155033.csv`.

---

### Shimmy 2.1 integration targets

The following work lands in Shimmy after airframe 0.2.0 is published:

- `airframe = "0.2"` dep bump in the Shimmy private repo
- `SHIMMY_KV_QUANT` exposed as a top-level config key (`shimmy.toml`) and CLI flag
  (`--kv-quant int4`)
- `/v1/models` response already carries `"kv_mode"` field — Shimmy's dashboard can
  surface this without code changes
- Shimmy 2.1.0 release note: *"Memory optimization mode (experimental): set
  `kv_quant = "int4"` in shimmy.toml to reduce KV cache VRAM by ~7×"*
- Homebrew formula update for shimmy 2.1.0

---

### Added

- **TurboShimmy INT4 KV cache** — WGSL shaders `quantize_kv_int4.wgsl` and
  `decode_kv_int4.wgsl`; per-head-vector bias-8 nibble encoding; ~7× KV compression.
- `KVCache::new_int4` constructor; `KVCache::is_int4()` query method.
- `BindlessPipeline::run_layer_with_cache_int4` — 3-encoder per-layer decode path with
  per-encoder GPU polls to stay within the Windows TDR watchdog on each INT4 decode step.
- `BindlessPipeline::requantize_all_kv_int4` — bulk F32→INT4 requantize after prefill,
  followed by an explicit `device.poll(wait_indefinitely())` before decode begins.
- `SHIMMY_KV_QUANT` env var (`f32` | `int4`, default `f32`).
- `SHIMMY_PREFILL_CHUNK` env var: chunked prefill batch size (default `64`); reduce on
  lower-VRAM cards or if TDR crashes appear on very long prompts.
- `SHIMMY_VRAM_LIMIT_MB` env var: VRAM budget warning threshold (default `10500`).
- `/v1/models` response includes `"kv_mode": "f32"|"int4"` — downstream clients
  (including Shimmy's dashboard) can query the active KV mode without log parsing.
- Server startup log: `[GPU Server] KV cache mode: INT4 (TurboQuant)` when INT4 is active.
- `docs/turboshimmy.md`: full feature spec, implementation notes, needle bench results,
  deployment guidance, and Windows TDR context.
- `scripts/needle_bench.py`: needle-in-a-haystack evaluation script against a live server.
  Supports arbitrary `--ctx`, `--depths`, `--runs`, and `--out` for artifact collection.
- README: *Memory Optimization* section with VRAM savings table and full env var reference.
- `shimmyjinja` dependency updated to `0.5.0` (published alongside this release): adds
  `try_render_chat_template_with_context` returning `Result<String, RenderError>`, seals
  internal modules as `pub(crate)`, and adds CHANGELOG + doc-tests.

### Fixed

- **Windows TDR crashes during prefill** — GPU command encoder is now submitted and polled
  once per layer in `run_full_model_prefill_chunked_with_cache_state`, preventing any
  single dispatch from exceeding the ~2 s Windows TDR watchdog. Previously crashed at
  seq_len ≥ 144; now verified stable at seq_len ≈ 224 (ctx=256) on RTX 3060 Windows.
  The fix is structural — no user-tunable knob required for typical SHIMMY_PREFILL_CHUNK
  values (≤64).
- `scripts/needle_bench.py`: HTTP 400 from float seed (now cast to `int`); connection-reset
  on stop-token emission (caught and treated as end-of-stream); fuzzy `check_pass` to
  tolerate minor whitespace variation in model responses.

### Changed

- Version: `0.1.1` → `0.2.0`.
- `shimmyjinja` dependency: `0.2.1` → `0.5.0`.

---

## [0.1.1] — 2026-05-15

Initial public release: Bindless WebGPU pipeline, FSE subsystem, OpenAI-compatible HTTP
server (`shimmy_server_gpu`). Supports Llama, Gemma, Phi, Qwen2, Qwen3 architectures via
GGUF Q4_K_M and related quantization types. Ships `shimmyjinja` (Jinja2 chat_template
renderer) and `shimmytok` (GGUF-native tokenizer) as zero-dependency companion crates.
