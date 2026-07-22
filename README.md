<div align="center">
  <img src="https://raw.githubusercontent.com/Michael-A-Kuykendall/airframe/master/assets/airframe-logo.png" alt="Airframe" width="480" height="auto" />

  ### Pure-Rust WebGPU Inference Engine for GGUF Models

  [![Crates.io](https://img.shields.io/crates/v/airframe.svg)](https://crates.io/crates/airframe)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://rustup.rs/)
  [![GitHub Stars](https://img.shields.io/github/stars/Michael-A-Kuykendall/airframe?style=social)](https://github.com/Michael-A-Kuykendall/airframe/stargazers)
  [![Powered by Shimmy](https://img.shields.io/badge/powers-Shimmy-blueviolet)](https://github.com/Michael-A-Kuykendall/shimmy)

  **No C++ toolchain. No Python. No llama.cpp. Just Rust and your GPU.**
</div>

---

### 💝 Support Airframe

🚀 **If Airframe helps you, consider [sponsoring](https://github.com/sponsors/Michael-A-Kuykendall) — 100% of support goes to keeping it free forever.**

- **$5/month**: Coffee Hero ☕ — Eternal gratitude + name in [SPONSORS.md](SPONSORS.md)
- **$25/month**: Developer Supporter 🐛 — Priority bug response + roadmap influence
- **$100/month**: Corporate Backer 🏢 — Logo in README + release-note recognition
- **$500/month**: Enterprise Partner 🚀 — Prominent logo + monthly office hours + roadmap input

[**🎯 Become a Sponsor**](https://github.com/sponsors/Michael-A-Kuykendall) | See our amazing [sponsors](SPONSORS.md) 🙏

**Thank you to our sponsors:** [ZephyrCloudIO](https://github.com/ZephyrCloudIO) (Corporate Backer) · alistairheath (Coffee Hero)

---

Airframe is the GPU inference core powering [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy). It runs full transformer inference directly on the GPU via WGSL compute shaders — works on NVIDIA, AMD, Intel, and Apple Silicon.

**⚡ v0.2.10**: GPU gibberish root cause fixed (dequant front-padding in `run_dequant_any_blob`), f16→f32 dequant corrected on RTX 3060, `Q5_0` quant slot added, WGSL if/else dispatch ladder retired for a fabric `TensorFact→DispatchFact` rule, and per-layer golden-vault certification (10/10 models certified).

**⚡ v0.2.9**: batch_count fix for QKV shader (was killing all threads), GPU adapter selection now prefers discrete GPU over integrated, grammar control hooks integrated, PPT invariant cage (B1-B3) for regression detection. **357 tests pass.**

**⚡ v0.2.6**: LM head tile dispatch for large-vocab models (Gemma-2 256K, Qwen3, Llama-3.2-3B).

**⚡ v0.2.7**: Inference Saturation Fabric (ISF) refit complete. TDR transport, encoder pools, DuckDB optional.

**⚡ v0.2.1**: [TurboShimmy INT4 KV Cache](#-turboshimmy-int4-kv-cache) — ~7× less KV VRAM with one env var. Run Llama-3.2-3B on 4 GB GPUs.

```toml
[dependencies]
airframe = "0.2"
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

| Architecture | Models | Status |
|---|---|---|
| **Llama** | Llama 3.2, Llama 3, Llama 2, DeepSeek | ✅ Verified |
| **Mistral** | Mistral 7B, Mixtral (dense layers) | ✅ Verified |
| **Phi** | Phi-3.5, Phi-3, Phi-2 | ✅ Verified |
| **Qwen2** | Qwen2 0.5B–7B | ✅ Verified (fixed in v0.2.2) |
| **Qwen3** | Qwen3 0.6B–8B | ✅ Verified (GPU forward pass certified layer-by-layer vs golden vault) |
| **Gemma** | Gemma-2 2B, 9B | ✅ Verified (fixed in v0.2.2) |
| **StarCoder2** | StarCoder2 3B | ✅ Verified |
| **GPT-2** | GPT-2 | ✅ Verified |

## Supported Quantization

`F32` · `F16` · `Q4_0` · `Q4_K_M` · `Q5_0` · `Q5_K_M` · `Q6_K` · `Q8_0`

All quantization types are implemented in both GPU shader and CPU reference paths, validated by `quant_verify` (GPU/CPU dequant consistency) and per-layer golden-vault certification — the same model produces numerically consistent output on CPU and GPU, within numerical tolerance.

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

## ⚡ TurboShimmy INT4 KV Cache

TurboShimmy is Airframe's on-GPU INT4 KV-cache compression system, shipping in v0.2.1. It squeezes the KV cache from 32-bit floats down to per-head-vector 4-bit integers — entirely in WGSL compute shaders with no CPU roundtrips — delivering ~7× less KV VRAM with no measurable quality loss at normal context lengths.

**One env var. ~7× less KV VRAM. Same output quality. Pure Rust, pure GPU.**

```bash
# Enable TurboShimmy
SHIMMY_KV_QUANT=int4 LIBSHIMMY_MODEL_PATH=/path/to/model.gguf \
  cargo run --bin shimmy_server_gpu --release

# Or with the prefill-chunk flag (prevents Windows TDR resets on long prompts)
SHIMMY_KV_QUANT=int4 SHIMMY_PREFILL_CHUNK=8 LIBSHIMMY_MODEL_PATH=/path/to/model.gguf \
  cargo run --bin shimmy_server_gpu --release
```

**Why it matters** — TurboShimmy changes what fits on consumer GPUs:

| GPU VRAM | Without TurboShimmy | With TurboShimmy |
|---|---|---|
| 3 GB | Llama-3.2-1B only | **Llama-3.2-3B fits ✅** |
| 4 GB | Llama-3.2-3B, ctx=2048 (tight) | **Llama-3.2-3B at ctx=8192 ✅** |
| 6 GB | 3B models, short context | **7B models with reasonable context ✅** |

**VRAM savings (ctx=2048):**

| Model | F32 KV | INT4 KV | Savings |
|---|---|---|---|
| TinyLlama 1.1B (Q4_0) | 88 MB | ~13 MB | **~7× less** |
| Llama-3.2-1B (Q4_K_M) | ~128 MB | ~18 MB | **~7× less** |
| Llama-3.2-3B (Q4_K_M) | ~512 MB | ~72 MB | **~7× less** |

**How it works:** Each K/V head vector is independently quantized to 4-bit integers with a per-vector F32 scale factor (`max_abs / 7.0`), packed into U32s (8 nibbles each) by `sh_kv_pack_int4.wgsl`. Dequantization via `sh_kv_unpack_int4.wgsl` happens on-the-fly before each attention computation. The helical context-shift operates directly on the packed INT4 representation — no decompression needed. Zero CPU roundtrips throughout.

**Quality validation:** Needle-in-a-haystack benchmarks on Llama-3.2-3B show zero retrieval degradation vs F32 at ctx≤2048 across all tested insertion depths (15%, 50%, 85%). See [`docs/turboshimmy.md`](docs/turboshimmy.md) and the [Shimmy wiki TurboShimmy page](https://github.com/Michael-A-Kuykendall/shimmy/wiki/TurboShimmy) for full benchmark data and setup guide.

**Server environment variables**:

| Variable | Default | Description |
|---|---|---|
| `LIBSHIMMY_MODEL_PATH` | *(required)* | Path to `.gguf` model file |
| `SHIMMY_PORT` | `8080` | HTTP listener port |
| `SHIMMY_MAX_CTX` | `2048` | Maximum context window (tokens) |
| `SHIMMY_PREFILL_CHUNK` | `64` | Prefill batch size; reduce to `8` if you see TDR crashes on Windows |
| `SHIMMY_KV_QUANT` | `f32` | KV cache mode: `f32` or `int4` (TurboShimmy) |
| `SHIMMY_VRAM_LIMIT_MB` | `10500` | VRAM budget warning threshold (MB); tune for your GPU |

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## Ecosystem

| Project | Description |
|---|---|
| [**Shimmy**](https://github.com/Michael-A-Kuykendall/shimmy) | OpenAI-compatible inference server — powered by Airframe |
| [**libfse**](https://crates.io/crates/libfse) | Fused Semantic Execution policy engine — ships as part of this repo |
| [**shimmytok**](https://crates.io/crates/shimmytok) | GGUF-native tokenizer used by both Airframe and Shimmy |
| [**shimmyjinja**](https://github.com/Michael-A-Kuykendall/shimmyjinja) | Pure-Rust Jinja2 engine for HuggingFace `chat_template` strings — **live in v0.1.1**, powers the prompt rendering pipeline |

---

## License

MIT — see [LICENSE](LICENSE).

**Inference runtime** (attention kernels, GGUF loader, quantization, WebGPU backend): unencumbered MIT.

**FSE subsystem** (`crates/libfse`): MIT for non-commercial use. The Fail-Closed Policy Fusion and Execution Kernel methods are covered by a pending US patent. Commercial embedding requires a separate license — contact michaelallenkuykendall@gmail.com.
