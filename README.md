<div align="center">
  <img src="assets/airframe-logo.png" alt="Airframe" width="480" height="auto" />

  ### Pure-Rust WebGPU Inference Engine for GGUF Models

  [![Crates.io](https://img.shields.io/crates/v/airframe.svg)](https://crates.io/crates/airframe)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://rustup.rs/)
  [![GitHub Stars](https://img.shields.io/github/stars/Michael-A-Kuykendall/airframe?style=social)](https://github.com/Michael-A-Kuykendall/airframe/stargazers)
  [![Powered by Shimmy](https://img.shields.io/badge/powers-Shimmy-blueviolet)](https://github.com/Michael-A-Kuykendall/shimmy)

  **No C++ toolchain. No Python. No llama.cpp. Just Rust and your GPU.**
</div>

---

Airframe is the GPU inference core powering [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy). It runs full transformer inference directly on the GPU via WGSL compute shaders — works on NVIDIA, AMD, Intel, and Apple Silicon.

```toml
[dependencies]
airframe = "0.1"
```

> **Patent Notice**: The Fused Semantic Execution (FSE) subsystem (`crates/libfse`) is covered by a pending US patent. The WebGPU inference runtime (attention, GGUF loader, quantization) is unencumbered MIT. See [license section](#license) for full terms.

---

## Why Airframe?

Most Rust LLM inference libraries are thin wrappers around llama.cpp — they require a C++ toolchain, link against native libraries, and make cross-compilation painful. Airframe is different:

| | Airframe | llama.cpp bindings |
|---|---|---|
| Build toolchain | `cargo build` | C++ compiler required |
| GPU backend | WebGPU (wgpu) — any GPU | CUDA / Metal / Vulkan |
| Cross-compilation | Native Rust | Complex |
| Determinism | Guaranteed | Platform-dependent |
| Dependency count | Minimal | Large C++ dep tree |
| `cargo publish` friendly | ✅ | ❌ |

---

## Quick Start

```rust
use airframe::runtime::gpu::{GpuRuntime, SamplingParams};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = GpuRuntime::load("path/to/model.gguf").await?;
    let output = runtime
        .generate("The capital of France is", SamplingParams::default(), None)
        .await?;
    println!("{}", output);
    Ok(())
}
```

Or run the included example with any GGUF model:

```bash
LIBSHIMMY_MODEL_PATH=/path/to/model.gguf cargo run --example simple_flight -- "Hello, world!"
```

---

## Supported Architectures

| Architecture | Models |
|---|---|
| **Llama** | Llama 3.2, Llama 3, Llama 2 |
| **Mistral** | Mistral 7B, Mixtral (dense layers) |
| **Phi** | Phi-3, Phi-2 |
| **Qwen2** | Qwen2 7B |
| **Falcon** | Falcon 7B |
| **GPT-NeoX** | StableLM |
| **Gemma** | Gemma 2B |

## Supported Quantization

`F32` · `F16` · `Q4_0` · `Q4_K_M` · `Q8_0`

All quantization types are implemented in both GPU shader and CPU reference paths, with parity validation — the same model produces bit-identical output on CPU and GPU.

---

## Architecture

Airframe is built around three principles:

### 1. Bindless WebGPU Pipeline

The GPU backend uses a bindless resource model — all weight tensors are uploaded once to GPU memory and addressed by index in the shader, eliminating per-layer bind group churn. This gives near-linear throughput scaling with context length.

### 2. Fused Semantic Execution (FSE)

The policy enforcement layer (`crates/libfse`) compiles multiple independent semantic rules into a single fused DFA evaluated during token generation. Rule evaluation cost is **O(1) in rule count** for shared selectors — a property that is not an optimization but an architectural inversion.

```
Input stream → Compiled DFA → Fused opcode table → Fail-closed decision
                                (single pass)
```

See [`fused_semantic_execution_full_markdown_reconstruction.md`](fused_semantic_execution_full_markdown_reconstruction.md) for the full technical specification and patent drawings.

### 3. Deterministic Sampling

Given the same model file, seed, and sampling parameters, Airframe produces identical output on every run — across restarts, machines, and GPU vendors. This makes it suitable for reproducible evaluation pipelines.

---

## Design Diagrams

```
┌─────────────────────────────────────────────┐
│                airframe crate               │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   core/  │  │ family/  │  │  ops/    │  │
│  │ GGUF load│  │  Llama   │  │ attn/FFN │  │
│  │ tensors  │  │  forward │  │ RoPE/RMS │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └─────────────┼─────────────┘         │
│                     ▼                        │
│  ┌──────────────────────────────────────┐    │
│  │           runtime/                   │    │
│  │   engine · KV cache · sampler        │    │
│  └─────────────────┬────────────────────┘    │
│                    ▼                         │
│  ┌──────────────────────────────────────┐    │
│  │       backend/bindless/ (WebGPU)     │    │
│  │   14 WGSL compute shaders            │    │
│  │   dequant · matmul · RoPE · attn     │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │   crates/libfse  (FSE policy engine) │    │
│  │   Patent Pending — see LICENSE note  │    │
│  └──────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
         ▲ used by
┌────────┴──────────────────┐
│  Shimmy GPU Server binary  │
│  shimmy_server_gpu         │
│  HTTP · job queue · eval   │
└───────────────────────────┘
```

Full architecture reference: [`docs/architecture-map.md`](docs/architecture-map.md)

---

## Benchmarks

Airframe has been validated on standard LLM evaluation benchmarks. Results are tracked in [`artifacts/`](artifacts/).

The FSE policy layer benchmarks 27% faster than raw `aho-corasick` iterator on 7KB payloads (see `crates/libfse/AUDIT_INFO.md` for methodology).

To run performance baselines:

```bash
cargo bench
# or with a model:
LIBSHIMMY_MODEL_PATH=/path/to/model.gguf cargo run --bin shimmy_server_gpu --release
```

---

## Development

```bash
git clone https://github.com/Michael-A-Kuykendall/airframe
cd airframe
cargo build
cargo test
cargo run --example simple_flight  # requires LIBSHIMMY_MODEL_PATH
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Ecosystem

| Project | Description |
|---|---|
| [**Shimmy**](https://github.com/Michael-A-Kuykendall/shimmy) | OpenAI-compatible inference server — powered by Airframe |
| [**libfse**](https://crates.io/crates/libfse) | Fused Semantic Execution policy engine — ships as part of this repo |
| [**shimmytok**](https://crates.io/crates/shimmytok) | GGUF-native tokenizer used by both Airframe and Shimmy |

---

## License

MIT — see [LICENSE](LICENSE).

**Inference runtime** (attention kernels, GGUF loader, quantization, WebGPU backend): unencumbered MIT.

**FSE subsystem** (`crates/libfse`): MIT for non-commercial use. The Fail-Closed Policy Fusion and Execution Kernel methods are covered by a pending US patent. Commercial embedding requires a separate license — contact michaelallenkuykendall@gmail.com.
