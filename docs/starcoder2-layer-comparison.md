# StarCoder2 Layer Comparison (Airframe vs llama.cpp)

Date: 2026-06-05
Branch context: feat/starcoder-triage

## Scope

This document compares the StarCoder2 decode path layer-by-layer between Airframe and llama.cpp, using local code plus trace/formula artifacts.

The goal is to identify the first likely mathematical divergence point, not to re-check routing or prompt rendering.

## Evidence Inputs

- Airframe route + policy derivation:
  - src/core/routing.rs
- Airframe bindless layer execution + dispatch order:
  - src/backend/bindless/pipeline/inference.rs
  - src/backend/bindless/sh_layer_v1.wgsl
  - src/backend/bindless/metadata.rs
- Existing StarCoder2 traces/formula diffs:
  - artifacts/debug/starcoder2_probe_now_chat/raw_vs_chat_formula.json
  - artifacts/debug/starcoder2_delib_1_chat/raw_vs_chat_formula.json
  - artifacts/debug/starcoder2_probe_now_chat/alignment_report.json
- llama.cpp StarCoder2 architecture references (external):
  - src/models/starcoder2.cpp

## Quick Findings

1. Prompt/template is not the primary failure surface for StarCoder2.
   - In probe_now artifacts, raw-vs-chat formula diff is exactly zero at all compared points.
   - shared_layer_points=120, mean_score=0.0.

2. A deeper decode-path mismatch is still present in other StarCoder2 runs.
   - In delib_1 artifacts, mean_score=0.215 and max_score=0.864.
   - Largest deltas cluster in mid/late layers (for example layer 21 and layer 28) and include large kv_mean_gap shifts.

3. The strongest code-level suspect is non-gated FFN normalization source/indexing in WGSL.
   - main_ffn_norm returns early for layer_norm_enabled or non-gated models.
   - main_ffn_proj non-gated path multiplies by norm_bank[norm_offset_base + col] while norm_offset_base remains 0 in that branch.
   - That implies a layer-agnostic norm-bank read pattern in the non-gated path, which can corrupt StarCoder2/GPT2/Phi-family FFN inputs.

## Fix Attempt 1 (Implemented)

### Change

In src/backend/bindless/sh_layer_v1.wgsl, the non-gated FFN path was changed to consume the staged per-layer FFN-normalized vector (written by main_ffn_norm) instead of recomputing from activation_in with a zero-based norm-bank index.

Also, main_ffn_norm no longer early-returns for non-gated/layer-norm models, so the staged FFN norm is computed for StarCoder2.

### Validation

- Build: cargo build --release --bin shimmy_server_gpu passes.
- Runtime probe (same prompt):
  - script: scripts/prompt_mode_formula_probe.sh
  - profile: starcoder2_probe_fix1
  - model: D:/shimmy-test-models/gguf_collection/starcoder2-3b-Q4_K_M.gguf

Post-fix raw-vs-chat formula metrics:

- mean_score: 0.008566
- max_score: 0.023548

This confirms raw/chat behavior is now nearly identical in this probe and that the patch materially changed internals.

Pre-fix vs post-fix raw trace comparison:

- mean_score: 0.758505
- max_score: 1.295225

So the patch is not a no-op; it strongly shifts decode math.

### Remaining issue

Despite the internal shift, output quality is still weak for the probe prompt (completion remains garbage-like and hits max_tokens). This means at least one additional core-path mismatch remains after non-gated FFN normalization was corrected.

## Layer-by-Layer Comparison Table

| Phase / layer step | Airframe path | llama.cpp StarCoder2 path | Current trace evidence | Risk level |
|---|---|---|---|---|
| Prompt render and tokenization | prompt_renderer_family=Completion, fallback_arch_other in probe traces | Standard completion/chat template flow; architecture is not prompt-special | probe_now raw/chat formula is identical (mean_score=0.0) | Low |
| Attention pre-norm | main_attn_norm computes norm into temp_state; supports LayerNorm and RMSNorm modes | build_norm(attn_norm, attn_norm_b, LLM_NORM) before QKV | No direct mismatch signal in probe_now raw/chat; downstream mismatch appears in other runs | Medium |
| QKV matmul + RoPE | main_qkv handles separate/fused QKV via compiled metadata offsets, then RoPE on Q/K | build_qkv + ggml_rope_ext on Q and K | No isolated early-layer explosion in probe_now pair | Medium |
| Attention score/value accumulation | main_attn_out computes streaming softmax and context into temp_state | build_attn with scaling 1/sqrt(head_dim) | Not dominant in top delib_1 diffs relative to FFN-heavy metrics | Medium |
| Attention projection + residual | main_attn_proj projects context via attn_out weight and adds residual | build_attn output then residual add to inpSA | Some output_energy/post_attn_energy shifts appear in delib_1 | Medium |
| FFN normalization staging | main_ffn_norm currently returns early when layer_norm_enabled!=0 or non-gated | build_norm(ffn_norm, ffn_norm_b, LLM_NORM) before FFN | This is a likely divergence point for StarCoder2 (non-gated + layer-norm architecture) | High |
| FFN up/gate projection | main_ffn_proj uses non-gated special path and GELU activation | build_ffn(..., LLM_FFN_GELU, LLM_FFN_SEQ) | delib_1 top diffs include ffn_energy and output_to_ffn_absmax_ratio shifts | High |
| FFN down + residual | main_ffn_down multiplies gate/up (up slot forced to 1.0 in non-gated mode) and adds residual | FFN down projection then residual add | Mid/late-layer divergence persists across decode steps | High |
| Final norm / lm_head | final norm and logits computed after layer loop | result_norm then output matmul | Divergence likely already present before head stage | Medium |

## Concrete Code Anchors for the Main Suspect

- Non-gated detection helper:
  - src/backend/bindless/sh_layer_v1.wgsl:84
- FFN norm early return condition:
  - src/backend/bindless/sh_layer_v1.wgsl:887
- FFN proj non-gated branch + norm_offset_base usage:
  - src/backend/bindless/sh_layer_v1.wgsl:969
  - src/backend/bindless/sh_layer_v1.wgsl:980
  - src/backend/bindless/sh_layer_v1.wgsl:1034
- Layer execution order showing FFNNorm then FFNProj:
  - src/backend/bindless/pipeline/inference.rs:677
  - src/backend/bindless/pipeline/inference.rs:688

## Immediate Next Verification

1. Instrument one decode step for StarCoder2 and dump the exact FFN input vector used by main_ffn_proj in non-gated mode.
2. Compare that vector against a CPU-side reference LayerNorm(ffn_input, ffn_norm_w, ffn_norm_b) for the same layer/token.
3. If mismatch is confirmed, patch non-gated FFN source to consume correctly normalized per-layer FFN input (not layer-0 norm-bank fallback).

## Why this is the current best candidate

- It explains why prompt-path changes can show little or inconsistent effect.
- It is architecture-selective (non-gated/layer-norm families) and lines up with historical weak set composition.
- It directly impacts FFN energy and residual evolution, which matches the dominant formula-diff metrics.
