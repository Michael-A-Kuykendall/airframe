# Model Routing Cloud Preload Pack

Purpose: provide a cloud model with all routing variables, known facts, llama.cpp decision logic patterns, Airframe current-state facts, and concrete architecture options so we can converge on one strong design without ad hoc iteration.

## 1) Original Objective

Build model routing that is:

1. Comparable to llama.cpp for correctness and determinism.
2. Compatible with Airframe's FSE execution principles (selector-first, extract-once, broadcast-many).
3. Low-tech-debt and modular enough to lift/shift across new model families.
4. Explicit and auditable (why route X was chosen for model Y).

## 2) Inputs The Router Must Consume

These are the authoritative input classes. Everything downstream should derive from them once.

### A. GGUF metadata keys

Required core keys:

1. general.architecture
2. general.name
3. general.file_type
4. <arch>.embedding_length
5. <arch>.block_count
6. <arch>.feed_forward_length
7. <arch>.attention.head_count
8. <arch>.attention.head_count_kv (optional fallback to head_count)
9. <arch>.context_length
10. <arch>.attention.layer_norm_rms_epsilon or layer_norm_epsilon
11. <arch>.rope.freq_base
12. <arch>.rope.dimension_count

Optional but important keys:

1. <arch>.attn_logit_softcapping
2. <arch>.final_logit_softcapping
3. <arch>.hidden_activation
4. <arch>.attention.key_length
5. tokenizer.chat_template

### B. Tensor manifest facts

Layer-0 probes (extend per-layer for full validation):

1. blk.0.attn_q.weight
2. blk.0.attn_k.weight
3. blk.0.attn_v.weight
4. blk.0.attn_qkv.weight
5. blk.0.ffn_gate.weight
6. blk.0.ffn_up.weight
7. blk.0.ffn_down.weight
8. blk.0.ffn_norm.weight
9. blk.0.attn_norm.weight
10. blk.0.attn_q_norm.weight
11. blk.0.attn_k_norm.weight
12. output_norm.weight

Tensor type and shape facts:

1. quant types per tensor
2. dims for split logic (fused qkv splitting, GQA geometry)
3. offset availability per layer

### C. Runtime policy inputs

1. context override (example SHIMMY_MAX_CTX)
2. rope scaling override
3. kv mode (f32/int4)
4. strict routing mode toggle

## 3) Current Airframe Facts (As Implemented)

From codebase at time of writing:

1. ModelSpec is GGUF-driven and parsed once from metadata in src/core/spec.rs.
2. Route scaffold exists in src/core/routing.rs as ModelRoutePlan.
3. Startup minimum route check exists and logs ROUTE_CHECK JSON in src/bin/shimmy_server_gpu.rs.
4. Per-layer compiled tensor lookup exists in src/backend/bindless/metadata.rs.
5. Prompt routing currently supports embedded Jinja template plus family fallback.
6. Layer params currently include norm/qk/post flags but not full explicit route policy fields.
7. Some routing still relies on heuristics or local branch assumptions rather than one canonical route plan.

Known architecture-smell still present:

1. Split control planes (prompt/math decisions not fully unified under one route contract).
2. Some low-level behavior inferred from missing tensors/offsets instead of explicit policy.
3. F32 and INT4 paths are not yet guaranteed to consume identical route semantics.

## 4) Llama.cpp Decisioning Logic Facts (For Reference)

These are the routing-style facts to mirror conceptually.

1. Parse GGUF once into hparams and architecture enum.
2. Use architecture enum to select model implementation and graph behavior.
3. Use explicit key-driven config for:
   - norm kind
   - rope type and scaling
   - qkv layout
   - ffn op/activation family
4. Determine rope family with architecture-level mapping (none/norm/neox/mrope variants).
5. Determine FFN op from metadata hidden_activation mapping where present, with controlled fallback.
6. Build graph from already-decided policy, not ad hoc hot-loop branch inference.

Important llama.cpp-style patterns to preserve:

1. policy decision at load time
2. typed architecture dispatch
3. explicit fallbacks
4. local invariants and fail-fast for unsupported combinations

## 5) Route Variable Catalog (Canonical)

The cloud model should treat this as the desired ModelRoutePlan schema.

### Identity and source

1. arch_family
2. model_name
3. gguf_file_type
4. route_version

### Norm policy

1. norm_kind (layer_norm or rms_norm)
2. norm_eps_source_key
3. qk_norm_enabled
4. post_norm_enabled
5. output_norm_kind

### Attention policy

1. qkv_layout (separate or fused)
2. fused_qkv_split_policy
3. head_count
4. head_count_kv
5. head_dim
6. gqa_ratio

### Positional policy

1. rope_kind (none, norm, neox, mrope, imrope)
2. rope_dim
3. rope_base
4. rope_scale
5. rope_scaling_mode
6. context_train
7. context_runtime

### FFN policy

1. ffn_kind (gated or non_gated)
2. ffn_activation_kind
3. ffn_norm_source
4. ffn_bias_policy

### Quant policy

1. packed_quant_policy
2. per-tensor quant map
3. kv_quant_mode

### Prompt policy

1. prompt_renderer_mode (jinja or family)
2. prompt_template_source (embedded or fallback)
3. prompt_family
4. bos_policy
5. eos_policy

### Validation outputs

1. reasons
2. warnings
3. hard_errors
4. strict_mode_pass

## 6) Non-Negotiable Routing Invariants

1. No hot-path free-form string contains checks for core math routing.
2. No shader branch should infer FFN behavior from missing ffn_gate alone.
3. Norm source indexing must be explicit and layer-local for all families.
4. F32 and INT4 consume the same route policy contract.
5. Unsupported route combinations must fail at startup in strict mode.
6. Prompt and math routing must share one route plan artifact.

## 7) Known Failure Class We Must Prevent

Failure class:

1. Metadata chooses correct high-level route.
2. Low-level branch applies implicit fallback.
3. Result: model appears routed correctly in logs but math diverges.

Required prevention:

1. route_plan_digest attached to inference trace start
2. per-kernel policy fields derived from route plan only
3. route-check strict mode for CI and debug workflows

## 8) Architecture Options For Cloud Evaluation

### Option A: Incremental hardening (recommended)

1. Expand ModelRoutePlan to full schema.
2. Route prompt and layer params from this object.
3. Add missing explicit kernel policy fields.
4. Keep compatibility mode temporarily.
5. Remove legacy branches after validation.

Pros:

1. low migration risk
2. fits current code factoring
3. easiest rollback

Cons:

1. temporary dual-path complexity

### Option B: Big-bang route engine swap

1. Replace all routing call sites in one pass.
2. Remove legacy logic immediately.

Pros:

1. cleaner end-state quickly

Cons:

1. high regression risk
2. hard debug surface

Recommendation: choose Option A.

## 9) Suggested Module Boundaries

1. src/core/routing.rs
   - route plan types
   - builder
   - validation
2. src/core/spec.rs
   - metadata parsing only
3. src/backend/bindless/metadata.rs
   - tensor manifest compilation only
4. src/bin/shimmy_server_gpu.rs
   - consume route plan, do not re-decide
5. src/backend/bindless/pipeline/*
   - execute policy fields only

## 10) Minimal Test Matrix (Trait-Based)

Do not test only by model names. Test trait combinations.

1. non_gated + layer_norm + separate_qkv
2. non_gated + layer_norm + fused_qkv
3. non_gated + rms_norm + separate_qkv
4. gated + rms_norm + separate_qkv
5. qk_norm enabled and disabled
6. post_norm enabled and disabled
7. rope none and rope enabled families
8. f32 kv and int4 kv

Acceptance:

1. route check JSON explains every decision
2. strict mode catches intentionally broken manifests
3. no unexplained warning deltas in baseline set

## 11) Data Needed From Cloud Model Back To Repo

Ask cloud output to provide:

1. final canonical ModelRoutePlan schema
2. decision tree pseudocode
3. per-variable source mapping (metadata key, tensor probe, derived)
4. fallback policy table (safe fallback vs fail-fast)
5. migration order and cutover conditions
6. CI test matrix and pass/fail criteria
7. risk register with mitigation gates

## 12) Prompt Template For Cloud Run

Use this prompt exactly or adapt slightly:

"You are designing a GGUF routing control plane for a Rust WebGPU inference engine. You must produce:

1) canonical typed route schema
2) deterministic decision tree from metadata + tensor manifest
3) strict invariants to prevent implicit kernel routing
4) migration plan from partial route system to unified route engine
5) trait-based test matrix

Constraints:

- must be llama.cpp-comparable in decisioning quality
- must preserve selector-first extract-once broadcast-many performance philosophy
- must support prompt routing and math routing from one route artifact
- must avoid ad hoc string heuristics in hot path
- must preserve rollback-friendly phased rollout

Given this input pack, produce architecture spec, pseudocode, and implementation checklist."

## 13) My Recommendations (Operator Guidance)

1. Keep current incremental direction, but centralize all routing into ModelRoutePlan before more shader surgery.
2. Do not add new model-family hacks in server or shaders until route schema is complete.
3. Upgrade route check output to include full route plan digest and source provenance per field.
4. Introduce explicit FFN policy fields into LayerParams before changing FFN math paths.
5. Keep strict routing mode available and wire it into CI for route regressions.

## 14) Immediate Next Internal Step After Cloud Return

1. Compare cloud schema against current src/core/routing.rs.
2. Merge schema differences first.
3. Wire server and prompt routing to new schema.
4. Wire pipeline params to schema.
5. Run trait-matrix route checks before full model sweeps.

---

This document is intentionally dense so an external model can reason with full context and return a better architecture proposal in one pass.
