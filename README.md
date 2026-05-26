# Airframe

**Pure-Rust WebGPU inference engine for Llama-family GGUF models.**

Airframe is the GPU inference core powering [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy). It runs transformer inference directly on the GPU via WGSL compute shaders — no C++ toolchain, no Python, no llama.cpp.

[![Crates.io](https://img.shields.io/crates/v/airframe.svg)](https://crates.io/crates/airframe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it is

Airframe is a FP32-first transformer runtime built on [wgpu](https://github.com/gfx-rs/wgpu). It implements full attention, dequantization, and matmul pipelines in WGSL compute shaders that run on any WebGPU-capable GPU — NVIDIA, AMD, Intel, and Apple Silicon via Metal.

Core design properties:
- **Single-pass fused execution** — Fused Semantic Execution (FSE) architecture minimizes GPU round-trips
- **Deterministic** — same model, same seed, same output every run
- **GGUF native** — reads quantization metadata directly from model files, no conversion needed
- **Zero runtime deps** — ships as a single Rust crate, no dynamic libraries required

## Supported architectures

| Architecture | Models |
|---|---|
| Llama | Llama 3, Llama 3.2, Llama 2 |
| Mistral | Mistral 7B, Mixtral (dense layers) |
| Phi | Phi-2, Phi-3 |
| Qwen2 | Qwen2 7B |
| Falcon | Falcon 7B |
| GPT-NeoX | StableLM |
| Gemma | Gemma 2B |

## Supported quantization types

`F32`, `F16`, `Q4_0`, `Q4_K_M`, `Q8_0` — all implemented in both GPU shader and CPU reference, with parity validation.

## Usage

Airframe is used directly by Shimmy as its GPU backend. For end users, the easiest path is to download a [Shimmy release binary](https://github.com/Michael-A-Kuykendall/shimmy/releases/latest) — Airframe is compiled in.

To use Airframe as a library:

```toml
[dependencies]
airframe = "0.0.1"
```

```rust
use airframe::runtime::gpu::{GpuRuntime, SamplingParams};

let runtime = GpuRuntime::load("/path/to/model.gguf").await?;
let output = runtime.generate("Hello, world!", SamplingParams::default(), None).await?;
```

## Architecture

See [`docs/architecture-map.md`](docs/architecture-map.md) for a full breakdown of the bindless pipeline, FSE execution model, and KV cache design.

The FSE execution architecture is described in [`fused_semantic_execution_full_markdown_reconstruction.md`](fused_semantic_execution_full_markdown_reconstruction.md).

## License

MIT — see [LICENSE](LICENSE).

## Related

- [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy) — the OpenAI-compatible inference server powered by Airframe
