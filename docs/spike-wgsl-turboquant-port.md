# Spike: WGSL TurboQuant Port — Fused Compressed-Domain Attention

**Date:** 2026-05-30  
**Status:** UNDERWAY  
**Owner:** TBD  
**Blocked by:** Nothing — this is parallel work to CAS/math-pack

---

## Goal

Port TurboQuant-style fused compressed-domain KV cache quantization to WGSL.

No one has done this yet. The CUDA (KVQuant) and Metal (Open-TQ-Metal) implementations exist and are public. Airframe runs on WebGPU/WGSL — vendor-agnostic, works on NVIDIA, AMD, Intel, Apple. A WGSL port would be the first cross-platform implementation of this technique.

**The payoff:** the wgpu 2GB buffer cap stops being a blocker for long-context inference. A model needing 8GB of KV cache at 8K context would need ~1.3GB instead (6x compression). That is the difference between impossible and fits-with-headroom.

---

## Background: What TurboQuant Actually Does

During inference, the model generates one token at a time. For each token, it computes Key and Value tensors and stores them. Every subsequent token must attend over all previously stored K/V pairs. This is the KV cache.

At 8K context, the KV cache for a 7B model is roughly 8GB at fp16. That's why long context is expensive.

TurboQuant compresses each K/V vector from fp16 (16 bits) down to ~3–4 bits using two ideas:

**1. Per-vector quantization**  
Each vector is quantized independently, adapting to its own distribution. This is better than per-tensor quantization, which uses one scale factor for everything and loses precision on outliers.

**2. Fused attention kernel**  
Instead of: store fp16 → dequantize → run attention  
Do: store int4 → run attention directly on int4  

The fused kernel is where the speedup comes from. Dequantizing back to fp16 before every attention pass wastes memory bandwidth. Operating directly in compressed space eliminates that step entirely.

---

## Reference Implementations

| Implementation | Language | License | Notes |
|---|---|---|---|
| KVQuant | CUDA | Apache 2.0 | The original. Per-channel + per-vector quantization. NeurIPS 2024. |
| Open-TQ-Metal | Metal (Apple) | Apache 2.0 | Fused sdpa_int4 kernel. 48x speedup at 128K context. Best reference for kernel design. |

**Start with Open-TQ-Metal.** It is the most recent, most clearly documented, and the speedup numbers are the most dramatic. Code: https://github.com/svv232/gemma4metal and https://github.com/svv232/turboquant-llama3.170B

---

## Where to Start in Airframe

The existing WGSL attention pipeline is the insertion point. The relevant files:

- **`src/bin/shimmy_server_gpu/`** — server inference loop, where KV cache is read/written per token
- **WGSL shaders** — wherever the attention compute shader is defined (the `@compute` stage that computes Q·Kᵀ·V)

The port has two phases:

### Phase 1: KV Cache Quantization Pass (storage side)

Add a WGSL compute shader that runs after each K/V pair is computed and before it is stored:

```wgsl
// Pseudocode — quantize a fp16 vector to int4
@compute @workgroup_size(64)
fn quantize_kv(@builtin(global_invocation_id) id: vec3<u32>) {
    // Load fp16 K or V vector
    // Compute per-vector scale = max(abs(vec)) / 7.0  (4-bit signed range)
    // Quantize: q = round(vec / scale)  → stored as i4 packed into u32
    // Store scale separately (fp16, one per vector)
    // Store packed int4 vector
}
```

**Storage layout:** for each K/V vector of dimension D:
- `D/2` bytes for packed int4 values (2 values per byte)
- `2` bytes for the fp16 scale factor
- Total: `D/2 + 2` bytes vs `D*2` bytes at fp16 → **~4x reduction** at minimum

### Phase 2: Fused Attention Kernel (compute side)

Modify the attention compute shader to operate on the packed int4 representation:

```wgsl
// Pseudocode — attention over quantized KV
fn dequantize_vec(packed: array<u32>, scale: f16) -> array<f32> {
    // Unpack int4 values from u32
    // Multiply by scale → reconstruct approximate fp16/fp32 vector
}

@compute @workgroup_size(64)
fn attention_fused_int4(...) {
    // Load Q (fp16, not quantized — queries stay fp16)
    // For each K in KV cache:
    //   dequantize K inline (no separate dequant pass)
    //   compute dot(Q, K)
    // Softmax over scores
    // For each V in KV cache:
    //   dequantize V inline
    //   accumulate weighted V
    // Output: fp16 attention result
}
```

The "fused" part means dequantization happens inside the attention kernel, on registers, not as a separate memory pass. This is the speedup.

---

## Implementation Plan

### Step 1 — Audit the current attention shader
Find the existing WGSL compute shader(s) for attention. Map:
- Where are K/V tensors written?
- Where are K/V tensors read during attention?
- What buffer layout are they in?

### Step 2 — Implement quantize_kv shader
Write a standalone WGSL compute shader that takes an fp16 vector buffer and outputs a packed int4 buffer + scale buffer. Test it in isolation: quantize a known vector, dequantize it, measure error.

### Step 3 — Wire quantize_kv into the token generation loop
After computing K and V for a new token, run `quantize_kv` before writing to the KV cache buffer. The cache buffer type changes from `array<f16>` to `array<u32>` (packed) + `array<f16>` (scales).

### Step 4 — Modify attention to dequantize inline
Update the attention compute shader to unpack int4 + scale on the fly instead of reading fp16 directly. Verify correctness: outputs should be numerically close to fp16 attention.

### Step 5 — Benchmark
Run the needle bench at 2K, 4K, 8K context:
- Memory: compare KV cache buffer size before and after
- Speed: tokens/sec, time-to-first-token
- Accuracy: perplexity on wikitext-2, needle-in-haystack recall

---

## Realistic First-Pass Targets

The initial port should be evaluated against conservative expectations before chasing peak numbers:

- **Memory win comes first**: 3–4x KV cache reduction is achievable in the first working pass without workgroup tuning. The 6x headline number requires coalesced loads and shared-memory dequant (optimization pass, not v1).
- **Decode speedup is secondary**: The fused kernel eliminates a memory pass, but the wall-clock speedup at 2K–8K context may be modest on the first pass. Memory savings alone justify shipping v1.
- **Correctness gate**: Top-1 token agreement (100% on standard prompts) is non-negotiable before benchmarking speed. If you see divergence, per-vector scale computation is the first thing to audit.

---

## Known Risks

**PolarQuant architecture sensitivity**  
Open-TQ-Metal found that PolarQuant (angular quantization) fails on Gemma 4 because its `attn_scale=1.0` instead of `1/sqrt(d)`. **Do not use PolarQuant.** Use scalar per-vector quantization (max-based scale). It is simpler, safer across architectures, and still gives 4x compression.

**wgpu int4 packing**  
WGSL does not have a native int4 type. Pack two int4 values into one u8, two u8s into one u16, etc. The unpack arithmetic is a few lines of bit manipulation but must be correct.

**Workgroup memory pressure**  
Long-context attention is memory-bound. The fused kernel trades a memory pass for register pressure. At very long contexts (32K+), workgroup shared memory limits may require tiling the attention computation. Start with 2K–8K where this is not an issue.

**Correctness bar**  
The goal is zero top-1 token prediction changes on standard prompts. The Open-TQ-Metal paper verified this at int4. If you see divergence, first check the scale computation — per-vector scaling is critical.

---

## v2 Extensions (Deferred)

Do not block v1 on these. File them and move on.

- **QJL residual correction**: After per-vector int4, a residual quantization pass (Quantized Johnson-Lindenstrauss) can recover precision on outlier dimensions. Useful at 2-bit or for models with heavy activation outliers. Adds ~10% overhead; measure before committing.
- **Outlier handling**: Identify a small set of "heavy" channels (top ~1% by activation magnitude) and keep those in fp16 while quantizing the rest. KVQuant uses this; it closes the perplexity gap vs full fp16 at the cost of a more complex buffer layout.
- **Fast Walsh-Hadamard rotation**: If PolarQuant is revisited for future architectures (not Gemma 4), the Walsh-Hadamard transform (O(d log d), implementable via WGSL bit manipulation or precomputed tables) is the right rotation primitive. Skip for v1 scalar quant.
- **Parametric compressor**: Make the quantization path a compile-time or runtime flag (`QUANT_MODE: scalar_int4 | polar | qjl`) so architecture-specific quirks can be handled without code forks.

---

## Success Criteria

| Metric | Target |
|---|---|
| KV cache memory at 8K context | ≤ 25% of fp16 baseline |
| Tokens/sec at 8K context | ≥ fp16 baseline (ideally 2–4x faster) |
| Top-1 token agreement with fp16 | 100% on standard prompts |
| Perplexity on wikitext-2 | ≤ 0.5 degradation vs fp16 |
| Needle-in-haystack recall at 8K | ≥ fp16 baseline |

---

## What This Unlocks

Once this works:

- The 2GB wgpu buffer cap is no longer a hard wall for context length
- 8K context on consumer GPUs (RTX 3060 12GB) becomes practical
- Larger models fit in the same memory budget
- This is the first cross-platform (WGSL) implementation — potential open-source contribution back to the community

---

## References

- Open-TQ-Metal paper: https://arxiv.org/abs/2604.16957
- Open-TQ-Metal code: https://github.com/svv232/gemma4metal
- turboquant-llama3 (170B variant): https://github.com/svv232/turboquant-llama3.170B
- KVQuant paper: https://arxiv.org/abs/2401.18079
- wgpu-llm (Rust WebGPU LLM, WGSL compute patterns reference): https://github.com/officialcjunior/wgpu-llm
- TurboQuant spike research doc: docs/spike-turboquant.md
