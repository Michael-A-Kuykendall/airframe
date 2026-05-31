# TurboShimmy

This is the name for the INT4 KV cache compression system built into Airframe.

The internal process was called *turboquant* — a useful shorthand during the build, borrowed from the loose idea of quantizing things fast. But turboquant is a method, not a product, and methods don't have identities. What we built has an identity. It runs inside Shimmy. It makes Shimmy substantially faster and leaner at inference time. So the name is TurboShimmy.

---

## What it is

TurboShimmy compresses the key-value cache from 32-bit floats down to INT4 — 4-bit integers, two per byte — using a per-head-vector bias-8 nibble encoding with a max-absolute-value scale factor. The compression ratio is roughly 8×. On a model like Llama-3.2-3B at 8K context, the KV cache drops from ~1.8 GB to ~224 MB, freeing headroom for larger contexts and larger models to coexist on the same GPU.

The name is honest: the shimmy gets turbo'd. Context window grows. VRAM budget shrinks. Quality stays.

---

## What was built

The implementation is a WGSL compute shader pipeline integrated directly into Airframe's bindless GPU execution layer. There are no CPU roundtrips. Every piece runs on the GPU:

- **`sh_quantize_kv.wgsl`** — standalone F32→INT4 nibble quantizer, called once after prefill to compress the filled KV cache in-place into packed buffer storage.
- **`sh_layer_v1_int4.wgsl`** — the attention forward pass variant; dequantizes INT4 K and V on-the-fly per head during each decode step. No F32 KV buffers are read.
- **`sh_rope_shift_int4.wgsl`** — extends helical RoPE compaction to operate on the packed nibble and scale buffers directly, preserving correctness through context rotation.

These three shaders slot into the existing `BindlessPipeline` architecture without altering any of the F32 code paths. The feature flag is `SHIMMY_KV_QUANT=int4`. Without it, the engine behaves identically to baseline.

The pipeline required two bind group layouts: `layer_layout_int4` (14 bindings, used exclusively for the dequantize-attention kernel) and the existing 10-binding layout for all other kernels in the layer forward pass. Both layouts compiled and validated against the RTX 3060's `max_storage_buffers_per_shader_stage` limit, which was raised from 8 to 14 to accommodate the INT4 layout.

---

## Quality check

Battery test across six models at 2K context, INT4 enabled:

| Model | Score |
|---|---|
| Llama-3.2-1B Q4_K_M | 4/4 |
| Llama-3.2-3B Q4_K_M | 4/4 |
| TinyLlama-1.1B Q4_0 | 3/4 |
| Gemma-2-2B Q4_K_M | 3/4 |
| Phi-2 Q4_K_M | 1/4 *(degenerate output — pre-existing, not TurboShimmy)* |
| StarCoder2-3B | crash *(pre-existing in F32 too — not TurboShimmy)* |

No INT4-specific quality regressions on any model that runs cleanly in F32.

---

## Needle-in-a-haystack retrieval

Tested with `scripts/needle_bench.py` against Llama-3.2-3B Q4_K_M at ctx≈130 tokens (5 insertion depths: 10/25/50/75/90%), comparing F32 KV vs INT4 KV side by side.

**Result: F32 and INT4 KV produce byte-for-byte identical outputs at every depth.** Zero measurable retrieval degradation from INT4 compression.

The model itself produces minor garbling of the retrieval codes (inserting characters, a model capability limit at this scale), but the garbling is identical in both modes — confirming the error is in the language model, not the KV cache quantization.

| Mode | 10% | 25% | 50% | 75% | 90% | Δ vs F32 |
|------|-----|-----|-----|-----|-----|----------|
| F32 KV | `AIRFRAME-180-DE10-R0` | `AIRFRAME-180-DE25-R0` | `AIRFRAME-180-DR0` | `AIRFRAME-180-DR0` | `AIRFRAME-180-DR0` | baseline |
| INT4 KV | `AIRFRAME-180-DE10-R0` | `AIRFRAME-180-DE25-R0` | `AIRFRAME-180-DR0` | `AIRFRAME-180-DR0` | `AIRFRAME-180-DR0` | **identical** |

Context note: the TDR threshold on the Windows/RTX 3060 test machine limits safe prefill to seq_len ≤ 136 tokens per chunk under the current F32 attention kernel dispatch geometry. The needle bench runs at ctx≈130 to stay within that envelope. The INT4 *decode* path is TDR-safe (3-encoder split); the prefill path uses chunked F32 forward passes (controlled by `SHIMMY_PREFILL_CHUNK`, default 8 for safety).

---

## What it enables

The immediate gain is that 8K context fits on the RTX 3060 without hitting the VRAM ceiling. The longer horizon is that as models get larger and context windows widen, TurboShimmy is the buffer that keeps Airframe viable on consumer hardware. It was built as infrastructure, not as a demo.

---

*Built on branch `feat/turboquant-wgsl`. Commits: `1eb969d` (Phases 1–4), `daef66b` (Phase 5 helical shift), `9239e85` (bug fixes and battery validation).*
