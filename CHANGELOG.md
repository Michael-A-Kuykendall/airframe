# Changelog

All notable changes to Airframe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.9] - 2026-07-20

### Fixed
- **batch_count: 0 -> batch_count: 1** in `server_inference.rs` — was killing all QKV threads
- **Rust/WGSL struct layout mismatch** — per-field quant types aligned between Rust `LayerParams` and WGSL
- **GPU adapter selection** — now prefers discrete GPU over integrated (multi-GPU laptops)

### Added
- **Grammar control hooks** — schoolmarm+grammar module for structured generation
- **PPT invariant cage (B1-B3)** — golden-vault-based regression detection with per-layer RMS/checksum verification (12 models, CPU-only, single-thread)
- **Shimmy API**: `tokenizer_arc()`, `eos_token()`, `im_end_token()`, `fse_control_from_patterns()`, `trace_callback()`

### Infrastructure
- 357 library tests pass
- Releases backfilled for all versions back to v0.2.0

## [0.2.8] - 2026-07-18

### Fixed
- ISF/TDR/duckdb hotfix release
- DuckDB now optional (shimmy builds cleanly without it)

## [0.2.7] - 2026-06-12

### Added
- Inference Saturation Fabric (ISF) refit
- TDR (Timeout Detection and Recovery) transport
- Encoder pools operational
- DuckDB support

### Fixed
- 10/10 smoke tests pass across all architectures (llama/phi2/gemma2/qwen2/qwen3/starcoder2)
- Qwen3 attention.scale tensor mapping

## [0.2.6] - 2026-06-02

### Added
- LM head tile dispatch for large-vocab models (Gemma-2 256K, Qwen3, Llama-3.2-3B)

## [0.2.5] - 2026-05-28

### Fixed
- 7 NaN/stall bugs for multi-arch models (Qwen3, Gemma-2, StarCoder2, Llama-3.2)
- NaN in FFN gate/up projections — wrong quant type
- Buffer alignment in bindless tests (354/354 passing)

## [0.2.4] - 2026-05-24

### Added
- Vault-driven fixes infrastructure

## [0.2.3] - 2026-05-20

### Added
- Complete vault with 48 commodity models

## [0.2.1] - 2026-05-15

### Added
- **TurboShimmy INT4 KV Cache** — ~7× less KV VRAM with one env var
- Quality validation: needle-in-a-haystack benchmarks show zero retrieval degradation vs F32 at ctx≤2048
- `device.on_uncaptured_error` handler surfaces GPU validation errors as clean HTTP 500 responses

## [0.2.0] - 2026-05-10

Initial public release as the GPU inference core for Shimmy v2.0.

### Supported Architectures
- Llama (Llama 3.2, Llama 3, Llama 2, DeepSeek)
- Mistral (Mistral 7B, Mixtral dense layers)
- Phi (Phi-3.5, Phi-3, Phi-2)
- Qwen2 (0.5B–7B)
- Qwen3 (0.6B–8B)
- Gemma (Gemma-2 2B, 9B)
- StarCoder2 (3B)
- GPT-2

### Supported Quantization
F32, F16, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0

### Features
- Bindless WebGPU pipeline — all weight tensors uploaded once, addressed by index
- YaRN RoPE scaling for extended context
- Deterministic sampling — identical output across restarts and GPU vendors
- GGUF-native model spec auto-derivation
