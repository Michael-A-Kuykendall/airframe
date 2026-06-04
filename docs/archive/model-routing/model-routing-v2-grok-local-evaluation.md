# Model Routing v2 Grok Draft - Local Context Evaluation

Date: 2026-06-04
Scope: evaluate docs/model-routing-v2-grok-proposal.md against current Airframe code and branch constraints.

## Verdict

The Grok draft is directionally strong and mostly aligned with the intended architecture. It is not implementation-ready as written because several field names, strict-mode semantics, and route assumptions do not match current source reality.

## Findings (ordered by severity)

### 1) Pseudocode references ModelSpec fields that do not exist (compile blockers)

Severity: high

The draft pseudocode uses fields like:

- spec.architecture
- spec.name
- spec.layer_norm_rms_epsilon
- spec.rope_dimension_count
- spec.embedding_length
- spec.head_count
- spec.rope_freq_base
- spec.attn_logit_softcapping
- spec.final_logit_softcapping
- spec.attention_key_length
- spec.chat_template

Current ModelSpec exposes different names and shapes:

- arch (enum) and arch_string()
- model_name
- rms_eps
- rope_dim, n_embd, n_head, rope_base
- attn_logit_softcap, final_logit_softcap
- has_qk_norm

Source anchors:
- src/core/spec.rs:78
- src/core/spec.rs:98
- src/core/spec.rs:100
- src/core/spec.rs:102
- src/core/spec.rs:110
- src/core/spec.rs:112
- src/core/spec.rs:303
- src/core/spec.rs:360

Impact: direct copy/paste from the draft will fail to compile and mislead implementation planning.

### 2) Strict mode logic in pseudocode is incorrect

Severity: high

Draft logic:

- strict_pass = hard_errors.is_empty() && env var is absent
- then fail only when strict_pass is false and env var exists

This makes strict_pass false whenever strict env is enabled even with no errors.

Current implementation semantics:

- strict mode is enabled when SHIMMY_ROUTE_CHECK_STRICT is true-like
- startup fails only when strict mode is enabled and warnings exist

Source anchor:
- src/bin/shimmy_server_gpu.rs:1468

Impact: draft strict-mode pseudocode will produce incorrect pass/fail behavior.

### 3) post_norm routing assumption overreaches current behavior

Severity: high

Draft suggests qwen2/qwen3/gemma2 -> post_norm_enabled=true.

Current runtime sets post_norm only for gemma2 in layer params population.

Source anchor:
- src/backend/bindless/pipeline/inference.rs:207

Impact: enabling post_norm for Qwen paths without shader and model-validation evidence risks regressions.

### 4) "explicit enum policy, not inferred at runtime" claim is internally inconsistent

Severity: medium

The draft says qkv_layout/ffn_kind become explicit policy, but still derives both from tensor presence at build time.

Current code already does the same derivation in route builder and shader offsets.

Source anchors:
- src/core/routing.rs:67
- src/core/routing.rs:132
- src/backend/bindless/sh_layer_v1.wgsl:957
- src/backend/bindless/sh_layer_v1.wgsl:965

Impact: language should be tightened to: inferred once at load-time into explicit typed policy, then consumed downstream without re-inference.

### 5) Prompt unification step ignores current ownership boundaries

Severity: medium

Current prompt routing is fully implemented in shimmy_server_gpu via PromptRenderer and make_prompt_renderer, then surfaced in route check as selected mode/source.

Source anchors:
- src/bin/shimmy_server_gpu.rs:275
- src/bin/shimmy_server_gpu.rs:423
- src/bin/shimmy_server_gpu.rs:1435

Impact: v2 plan should explicitly include a migration seam from PromptRenderer to core routing artifacts, not assume prompt metadata is directly in ModelSpec.

### 6) LayerParams evolution is underspecified versus current shader behavior

Severity: medium

Current LayerParams already includes:

- post_norm_enabled
- qk_norm_enabled
- layer_norm_enabled

And WGSL still performs non-gated decisions from offsets.ffn_gate.

Source anchors:
- src/backend/bindless/pipeline/mod.rs:75
- src/backend/bindless/pipeline/mod.rs:86
- src/backend/bindless/pipeline/mod.rs:87
- src/backend/bindless/pipeline/mod.rs:88
- src/backend/bindless/sh_layer_v1.wgsl:957
- src/backend/bindless/sh_layer_v1.wgsl:985
- src/backend/bindless/sh_layer_v1.wgsl:1078

Impact: plan should add explicit ffn_kind field to LayerParams first, then gate removal of offsets-based branches.

## What Grok got right (keep these)

1. Single canonical load-time route artifact is the right direction.
2. Option A incremental rollout is the correct migration strategy.
3. Trait-based matrix is superior to model-name-only gating.
4. Need to remove shader-side implicit heuristics.
5. Need auditable reasons/warnings and route digest.

## Local-context gaps the draft could not know

1. Current ModelSpec intentionally normalizes arch and derived traits in compute_derived(); many raw GGUF key names are not preserved directly.
2. Prompt renderer currently handles Jinja panic fallback and reasoning-policy toggles in server-local logic.
3. Current strict check fails on warnings, not only hard errors.
4. Current post_norm runtime path is Gemma-focused, despite broader families in future matrix plans.

## Strengthened v2 adjustments

### A) Keep schema intent, rename fields to match existing ModelSpec now

Recommended substitutions for immediate compatibility:

- architecture -> arch_string()
- name -> model_name
- layer_norm_rms_epsilon -> rms_eps
- rope_dimension_count -> rope_dim
- embedding_length -> n_embd
- head_count -> n_head
- rope_freq_base -> rope_base
- attn_logit_softcapping -> attn_logit_softcap
- final_logit_softcapping -> final_logit_softcap

### B) Add a prompt-source input seam to route builder

Do not read chat_template from ModelSpec. Instead pass prompt decision inputs from server:

- prompt mode
- prompt family
- template source

Then embed these into ModelRoutePlan for unified audit output.

### C) Correct strict-mode contract

Use this contract:

- strict off: log reasons/warnings/errors, continue
- strict on: fail on hard_errors; optionally fail on warnings via a second toggle

Suggested env split:

- SHIMMY_ROUTE_CHECK_STRICT=1 (fail on hard errors)
- SHIMMY_ROUTE_CHECK_FAIL_ON_WARN=1 (optional stronger gate)

### D) Make FFN/QKV de-heuristic work explicitly staged

Stage 1:

- add ffn_kind and qkv_layout fields to LayerParams
- populate from ModelRoutePlan

Stage 2:

- shader branches switch to params.ffn_kind / params.qkv_layout
- keep offsets-based fallback only under compatibility flag

Stage 3:

- remove offsets-based control branches

### E) Keep post_norm conservative until validated

Default route matrix should preserve current behavior:

- Gemma: post_norm_enabled=true
- others: false unless explicit model-family validation lands

## Recommended immediate patch plan

1. Evolve src/core/routing.rs in-place to add v2 fields while retaining current constructor behavior.
2. Extend RouteCheckReport in src/bin/shimmy_server_gpu.rs to emit route_version and digest.
3. Add LayerParams enum-like integer fields for ffn_kind and qkv_layout in src/backend/bindless/pipeline/mod.rs.
4. Populate new fields from route plan in src/backend/bindless/pipeline/inference.rs under feature flag.
5. Add strict-gate tests against known model matrix before enabling shader branch removal.

## Bottom line

The Grok draft is a strong architecture blueprint. With the naming, strict-mode, post_norm, and prompt-source corrections above, it becomes executable against the current Airframe codebase without hidden mismatch debt.
