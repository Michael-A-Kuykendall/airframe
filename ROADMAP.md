# Airframe workspace roadmap

Current release scope: **Airframe 0.4.5**, **libfse 0.1.4** and
**airframe-observe 0.1.4**. Updated 2026-11-06. This roadmap describes delivered
contracts and outstanding work; it is not a promise that a backend or model
family has passed certification.

## airframe 0.4.5

**Delivered:** checked shared GGUF metadata and tensor validation, validated
dense descriptors and layer execution plans, explicit capability admission,
Llama RoPE frequency factors, request-isolated state, and the demonstrated
query-scratch, Q/K ownership and complete tensor-span window corrections.
F32 KV and reference kernels define the validated execution path.
See the [release contract](README.md#release-contract) and
[changes](CHANGELOG.md).

The measured fixtures are Llama 3.2 3B Instruct Q4_K_M at context 512,
Qwen3-0.6B Q4_K_M in selected 512/2048/4096/8192 modes, and Qwen3-1.7B Q4_K_M
for the original 2032-token request at context 2048. Independent
dequantized-FP32 full-logit comparisons use maximum absolute 0.1 and RMS 0.03
limits plus matching greedy decisions. Exact model hashes and mode limits
are in [Shimmy's fixture matrix](https://github.com/Michael-A-Kuykendall/shimmy/blob/main/docs/SUPPORTED_MODELS.md).
Repeated positions and modes are not independent model certifications.

**Outstanding:** resolve the separate WSL device-destruction and
active-generation device-loss investigations; expand numerical coverage only
with exact reference fixtures and explicit admission contracts. Gemma2
local/global attention groundwork remains isolated and is not an enabled
executor. Future MoE, MLA, hybrid recurrent/Gated DeltaNet, state-space and
multimodal execution require independent implementation and certification.

**Boundaries:** GPU evidence comes from NVIDIA through WSL Dozen/Vulkan;
numerical agreement does not certify driver stability. **NOT_RUN:** Metal/M4,
native Windows/macOS/Linux GPU correctness, extended-context Llama, larger
unlisted Qwen fixtures, distinct alternate prefill chunks, INT4 KV and
experimental kernels. INT4/TurboShimmy and experimental configurations are
excluded from the validated release path. Parsing metadata, quantization or
tokenizers is not proof that a complete model executes correctly. Performance
experiments remain deferred; no throughput, latency or universal portability
claim is made.

## libfse 0.1.4

**Delivered:** byte-pattern/action compilation with FseMap::compile and
incremental scanning with FseMap::scan_with_cursor, returning scan summaries
or violations. See the [scanner README](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/libfse/README.md)
and [changes](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/libfse/CHANGELOG.md).

**Outstanding:** continue source-grounded scanner regression coverage and
caller integration checks. Any future responsibility-overlap assessment with
airframe-observe must establish actual duplicated behavior before proposing
consolidation; no consolidation or new inference capability is delivered here.

**Boundaries:** callers must invoke the scanner and honor violations before
acting on rejected data. Byte-pattern matching is not general semantic
understanding, inference certification or a universal security guarantee.
libfse 0.1.4 is covered by the [additional Apache-2.0 grant](https://github.com/Michael-A-Kuykendall/airframe/blob/8efb1e7a830b0822cfc5ac1b121e149948d5ffe3/crates/libfse/LICENSE-GRANT-0.1.4.md), including commercial use, modification and redistribution without separate permission. The original archive remains unchanged. See [stack licensing](LICENSING.md).

## airframe-observe 0.1.4

**Delivered:** inference facts, observer registration/saturation and host-side
coordination using Dzero. ObservationSession consumes emitted layer, logit,
text and dispatch-timing facts; ISF coordinates caller-provided embedding,
prefill, decode and control callbacks. The Dzero requirement is 0.1 and this
release resolves 0.1.0. See the
[observer README](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/airframe_observe/README.md)
and [changes](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/airframe_observe/CHANGELOG.md).

**Outstanding:** extend fact/observer and callback coverage alongside actual
engine producers; validate timing availability per backend before making
platform-specific observation claims. New observer coverage does not
automatically certify new model executors.

**Boundaries:** observations depend on the facts producers actually emit.
These contracts do not implement GPU arithmetic, prove logit correctness,
repair driver failures or guarantee performance. GPU timestamp availability
and unexecuted platform behavior remain **NOT_RUN** where no execution
evidence exists. This crate is MIT with its separate patent notice, whereas
libfse has different licensing; see the
[license and patent notice](https://github.com/Michael-A-Kuykendall/airframe/blob/main/crates/airframe_observe/README.md#patent-notice).
