# Spike: TurboQuant — KV Cache Quantization

**Date:** 2026-05-30  
**Status:** Research  
**Claim source:** User-cited, attributed to Google Research / ICLR 2026

---

## What Is It

TurboQuant is a KV cache quantization technique. KV cache is the memory that an LLM keeps around during generation — it stores the key and value tensors computed for each token it has already processed. At long context lengths (4K, 8K, 32K tokens), this cache dominates GPU memory consumption.

TurboQuant compresses that cache from fp16 (16 bits per value) down to approximately 3 bits per value. The claimed results:

- **6x+ memory reduction** on KV cache
- **Up to 8x attention speedup** (attention is computed over the cache)
- **Zero accuracy loss** (no perplexity degradation)

It uses two sub-techniques:
- **PolarQuant** — quantizes vectors in polar (angular) space rather than Cartesian, exploiting the directional structure of attention key/value representations
- **Lloyd-Max quantization** — an optimal scalar quantizer that minimizes distortion for a given number of bits by adapting to the data distribution

---

## What I Found

The TurboQuant name does not have a single authoritative arxiv paper under that exact title. However, the ideas are real and well-documented in adjacent work:

**KVQuant** (arXiv:2401.18079, NeurIPS 2024) — the most directly comparable verified paper:
- Sub-4-bit KV cache quantization on LLaMA / Mistral
- <0.1 perplexity degradation at 3-bit
- 1.7x attention speedup via custom CUDA kernels
- Enables LLaMA-7B with 1M context on a single A100-80GB

**Open-TQ-Metal** (arXiv:2604.16957, April 2026) — implements TurboQuant-style fused attention on Apple Silicon:
- int4 KV cache quantization, attention computed directly on compressed representation (no dequantize step)
- **48x attention speedup** at 128K context on Metal GPU
- 3.2x KV cache memory reduction (40 GB → 12.5 GB for Llama 3.1 70B)
- References PolarQuant explicitly
- **Critical finding:** PolarQuant fails on some architectures. Specifically, Gemma 4 uses `attn_scale=1.0` instead of the standard `1/sqrt(d)`. This amplifies angular quantization error 25–100x. PolarQuant only works reliably when the model's attention scale is close to standard.

**PolyKV** (arXiv:2604.24971, April 2026) — references TurboQuant explicitly as prior work, extends it for multi-agent shared KV pools.

---

## Current State of the Technique

**The core idea is verified and working.** Multiple groups have implemented it. The speedup numbers are real — 48x at 128K context in Open-TQ-Metal. The memory reduction is real. The accuracy-loss claim at 3 bits is supported by KVQuant's perplexity numbers.

**The caveats are real too:**
- PolarQuant is architecture-sensitive. It breaks on non-standard attention scale factors.
- Implementations in vLLM/Triton are emerging but not yet mainstream — most production stacks still use fp16 or naive int8.
- The gains are largest at long context. At 2K context the benefit is much smaller.

---

## Relevance to Airframe / Shimmy

Airframe uses a custom WGSL attention pipeline. The KV cache is in GPU memory managed by wgpu buffers.

**Direct applicability: Medium.** Here is what matters:

| Question | Answer |
|---|---|
| Can we use TurboQuant directly? | No — it requires CUDA (KVQuant) or Metal (Open-TQ-Metal) kernels. Airframe is WebGPU/WGSL. |
| Can we implement the concept? | Yes, in WGSL compute shaders. It is not trivial but it is bounded work. |
| Who benefits most? | Long context users — 4K+ tokens. At 2K (airframe's current default) the gain is smaller. |
| Is PolarQuant safe for our models? | Only for standard-scale models. TinyLlama, Llama 3.x use standard 1/sqrt(d). Safe. Not safe for Gemma 4. |
| What's the practical unlock? | Our wgpu 2GB buffer cap is a known blocker. KV cache compression would let us run larger models or longer contexts within that cap. |

**The most immediately useful learning is architectural:** fused compressed-domain attention (compute attention directly on quantized cache, no dequantize step) is how you get the 8–48x speedup. Just quantizing and then dequantizing before attention only helps memory, not compute.

---

## Recommended Action

1. **Track, do not implement now.** The wgpu 2GB buffer cap is deferred to v2.1. KV cache compression is a v2.1 concern.
2. **When we hit v2.1:** implement a WGSL compute shader that quantizes KV cache to int4 on the fly and computes attention in compressed space. Study Open-TQ-Metal's kernel design (Apache 2.0 license, code is public).
3. **Do not use PolarQuant for Gemma 4 support.** Use scalar per-channel quantization (KVQuant-style) instead.
4. **Benchmark target:** 3.2–6x memory reduction, 4–8x attention speedup at 8K+ context.

---

## References

- KVQuant: https://arxiv.org/abs/2401.18079
- Open-TQ-Metal: https://arxiv.org/abs/2604.16957
- PolyKV: https://arxiv.org/abs/2604.24971
- TurboESM (protein LM variant): https://arxiv.org/abs/2603.26110
