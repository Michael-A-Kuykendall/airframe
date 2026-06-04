# Model Routing Behavior Spike

## Why This Spike Exists

The current behavior is partially metadata-driven and partially implicit in shader branches. That split is why a model can be routed correctly at a high level and still produce wrong math in a low-level path.

This spike proposes one explicit routing contract built from GGUF header metadata and tensor presence, then compiled once into runtime policy structures consumed consistently by prompt rendering, CPU control logic, and GPU kernels.

## What Llama.cpp Does (Reference Pattern)

Llama.cpp follows a strong control-plane pattern:

1. Parse GGUF metadata into hparams once during load.
2. Map general.architecture to a concrete model architecture enum.
3. Build architecture-specific graph/model behavior from that enum and hparams.
4. Use explicit metadata keys for norm style, RoPE family, activation, and tensor layout.
5. Keep hot-path compute kernels consuming already-decided policy, not re-deciding behavior from ad hoc fallbacks.

Practical implication: model behavior is primarily decided once from metadata and structured traits, then executed.

## How Airframe Routes Today

Airframe already does several good things:

1. Reads GGUF metadata once at startup and builds ModelSpec.
2. Detects architecture from general.architecture and computes derived dimensions.
3. Compiles per-layer tensor offset/quant lookup table at model load.
4. Routes prompt rendering from embedded template first, then family fallback.
5. Exposes norm/qk/post-norm flags in trace and layer params.

Current design gaps:

1. Some routing is string heuristic instead of typed policy (for example contains(gpt2), contains(starcoder), contains(falcon)).
2. Some behavior is still inferred from missing tensors in shader branches.
3. F32 and INT4 FFN behavior is not guaranteed to obey one shared policy contract.
4. Prompt routing and math routing are decided in different places with no single route plan artifact.

## Root Design Problem

The system has mixed control planes:

1. High-level behavior is metadata-driven.
2. Low-level behavior still has implicit fallbacks.

That is a design smell because metadata says what the model is, but some shader code still decides what to do by local absence/presence rules instead of an explicit model policy.

## Prescriptive Target: GGUF-Header-First Route Plan

Build one immutable ModelRoutePlan at load time and make all runtime behavior depend on it.

### Route Plan Inputs

1. GGUF key-values (general.architecture and relevant architecture-prefixed keys)
2. Tensor manifest (presence/absence, shapes, quant types)
3. Preflight constraints (RoPE resources, KV mode, context limits)

### Route Plan Outputs

1. Prompt policy
2. Norm policy
3. RoPE policy
4. FFN policy
5. Attention layout policy
6. Quant behavior policy
7. Validation status and warnings

### Minimal Route Plan Schema

- arch_family
- norm_kind (layernorm or rmsnorm)
- qk_norm_enabled
- post_norm_enabled
- rope_kind (none, norm, neox, mrope family)
- rope_dim
- rope_base
- ffn_kind (gated or non_gated)
- ffn_activation_kind
- qkv_layout (separate or fused)
- tensor_presence_mask
- quant_pack_policy
- prompt_template_policy

## Command Structure (Load To Inference)

### Phase 1: Model Load

1. Parse GGUF header and tensor index.
2. Build ModelSpec from required keys.
3. Build ModelRoutePlan from ModelSpec plus tensor manifest.
4. Validate route plan invariants (hard fail for unsupported required combos).
5. Compile per-layer GPU tables from route plan.

### Phase 2: Inference Setup

1. Prompt renderer consumes prompt_template_policy only.
2. Layer params consume norm/ffn/rope/quant policy only.
3. Shader paths consume explicit policy flags, never infer behavior from missing offsets.

### Phase 3: Hot Loop

1. Execute kernels using precomputed flags and offsets.
2. Trace emits route plan summary once plus per-step compact deltas.

## Non-Negotiable Invariants

1. No branch should infer model family from free-form string contains checks in hot path.
2. No branch should infer FFN behavior from missing ffn_gate alone.
3. Norm source indexing must be explicit and layer-local for every policy mode.
4. F32 and INT4 paths must share one policy contract.
5. Unsupported combinations fail at load time, not silently degrade at runtime.

## FSE Alignment (Do Not Betray The Fast Path)

This proposal matches the existing FSE design language already documented in the repository:

1. selector-first
2. extract-once
3. broadcast-many
4. bounded working state

Applied here:

1. Select model behavior once from metadata and tensor facts.
2. Extract route decisions into one immutable plan.
3. Broadcast that plan to all subsystems.
4. Keep kernels branch-light and policy-explicit.

This avoids stapling hacks on top of elegant throughput architecture and keeps policy derivation cheap and deterministic.

## Compare With Other Engines

### Llama.cpp and Derivatives

1. Llama.cpp is explicitly metadata/architecture-driven.
2. Llamafile and many wrappers inherit the same route discipline through llama.cpp.
3. Behavior is generally decided up front in structured model classes and graph builders.

### Non-GGUF Stacks

1. Engines like vLLM or TensorRT-LLM typically use HF config/model definitions rather than GGUF header routing.
2. The equivalent good practice is still the same: decide architecture policy once from structured metadata, then execute.

## Concrete Airframe Refactor Plan

### Step A: Introduce Typed Route Plan

1. Add a new route module producing ModelRoutePlan from metadata plus tensor manifest.
2. Move current arch heuristics into typed trait mapping with explicit fallback levels.

### Step B: Make GPU Params Plan-Driven

1. Add explicit FFN mode and norm source fields in LayerParams.
2. Remove implicit non-gated behavior that depends on uninitialized or default norm slots.
3. Mirror this contract in both F32 and INT4 shaders.

### Step C: Unify Prompt And Math Routing Sources

1. Prompt renderer consumes the same route plan object used by math path.
2. Trace route plan digest at request start for auditability.

### Step D: Add Trait-Matrix Tests

1. Build tests by architecture traits, not model names only.
2. Required rows include:
   - non_gated + layernorm
   - non_gated + rmsnorm
   - gated + rmsnorm
   - fused_qkv + separate_qkv
   - qk_norm on and off
   - rope none and rope active

### Step E: Add Load-Time Validation

1. If route plan requires a tensor that is missing, fail with precise error.
2. If fallback is used, emit explicit warning with fallback reason.

## Risk And Effort

Effort: medium for architecture cleanup, low-to-medium per code change.

Primary risk: accidental behavior shifts for previously passing models.

Mitigation:

1. Introduce route plan first with compatibility mode.
2. Keep old path behind temporary guard.
3. Run trait-matrix smoke and formula-diff checks before flipping default.

## Acceptance Criteria

1. One model route artifact exists and is logged at startup.
2. All prompt and kernel routing derives from that artifact.
3. No shader branch uses implicit family inference from missing tensors.
4. StarCoder2, Phi-2, GPT2, and baseline Llama set pass trait-matrix checks.
5. Formula divergence improves or stays stable for all previously passing models.

## Immediate Next Action

Implement only the control-plane scaffold first:

1. Create ModelRoutePlan and builder from existing metadata plus tensor map.
2. Wire route plan into trace/log output.
3. Do not change math yet until route plan visibility is complete.

Then do one surgical math fix using the new explicit FFN mode fields and validate on StarCoder2 first, followed by baseline regression sweep.
