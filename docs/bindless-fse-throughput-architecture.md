# Bindless FSE Throughput Architecture

## Purpose

This document defines the implementation-facing architecture for applying selector-first, extract-once, broadcast-many execution principles to the current Airframe bindless GPU path.

The immediate target is the current 1B TinyLlama path. The goal is not to redesign the engine abstractly. The goal is to remove repeated per-consumer work, keep long-context behavior stable, and create a specification that can be refined before code changes are committed to the hot path.

This document is private working architecture. It is not a public product statement.

## Status

- Current status: draft architecture specification
- Primary target: `src/backend/bindless` decode and prefill path
- Model target: TinyLlama 1.1B first
- Design bias: minimal new surface, measurable wins, rollback-friendly sequencing

## Problem Statement

The current bindless layer path is still too consumer-first in the GPU sense.

In the active layer shader path:

1. attention-side normalization work is recomputed inside many projection consumers
2. FFN-side normalization work is recomputed inside many projection consumers
3. split passes create repeated launch overhead and repeated reads of the same activation stream
4. positional extension work is only partially represented as a first-class resource contract

This creates a structural mismatch with the desired execution model:

- shared extraction should happen once
- extracted values should be broadcast to all dependent consumers
- per-token working state should be bounded and explicit
- positional policy should be selected by metadata and preflight resources, not by ad hoc hot-loop branching

## Design Goals

### Primary Goals

1. Remove repeated per-consumer RMS extraction work from the bindless hot path.
2. Convert Q, K, V and FFN gate/up into explicit broadcast consumers of shared normalized token state.
3. Define one runtime contract for standard RoPE, linear scaling, YaRN, and future private experimental positional variants.
4. Preserve deterministic execution and current numerical discipline.
5. Keep the first implementation narrow enough that it can be benchmarked and reverted cleanly.

### Secondary Goals

1. Reduce effective dispatch cost per token without forcing a full architectural rewrite in the first pass.
2. Make the prefill and decode paths share the same conceptual provider-consumer model.
3. Leave room for later model-family expansion after the 1B path is proven.

### Non-Goals

1. This document does not define a universal architecture for every model family now.
2. This document does not require immediate support for all GGUF RoPE modes.
3. This document does not assume arbitrary experimental positional policies are quality-safe.
4. This document does not mandate immediate full kernel fusion in one step.
5. This document does not expose proprietary FSE implementation details beyond the architectural pattern needed for internal engine work.

## Core Thesis

The bindless engine should treat per-token activation processing the same way FSE treats shared selector extraction:

1. compute shared token state once
2. materialize it into a bounded provider buffer
3. broadcast that provider buffer to all dependent consumers
4. update only the downstream states that actually depend on it

For the current transformer path, the first shared provider states are:

1. attention-normalized token activations
2. FFN-normalized token activations

The same principle extends to positional policy resources:

1. construct positional resources in preflight
2. bind a single positional contract at runtime
3. let hot kernels consume the selected resource without recomputing policy semantics in the inner loop

## Architectural Principles

### 1. Extract Once, Broadcast Many

Any token-level value reused by multiple projection families must be computed once and reused.

### 2. Provider Buffers Over Implicit Recompute

Shared values must be represented as explicit provider buffers or provider regions, not rediscovered independently by each consumer kernel.

### 3. Fixed Runtime Contract

Positional behavior must be controlled by a fixed, typed runtime contract rather than scattered shader assumptions.

### 4. Bounded Working State

Each stage must have explicit working-state bounds. No stage should hide unbounded per-rule or per-consumer growth.

### 5. Rollout In Small Auditable Steps

The architecture must be introduced in checkpoints that can be benchmarked, diffed, and reverted independently.

## Current Stack Facts This Spec Assumes

1. Whole GGUF payload is resident in bindless GPU-accessible storage.
2. Transformer layer weights are primarily quantized and decoded in shader.
3. KV cache is F32 and must remain precision-safe.
4. Output head is already dequantized to F32 once and reused.
5. Prefill is currently chunked.
6. Decode currently loops layer-by-layer with split dispatches.
7. RoPE resources already exist in preflight, but policy coverage is incomplete.

## Required Logical Components

The first architecture revision must expose seven logical components:

1. `Activation Provider`
2. `Attention Provider`
3. `Attention Consumers`
4. `FFN Provider`
5. `FFN Consumers`
6. `Positional Policy Provider`
7. `Runtime Policy Contract`

### 1. Activation Provider

The activation provider owns the current token or batch activation stream entering a layer.

Responsibilities:

- expose the current residual input
- provide bounded staging space for normalized provider outputs
- avoid hidden duplication of the same activation vector across consumer families

### 2. Attention Provider

The attention provider computes the shared attention-normalized activation state once per token or per token row.

Responsibilities:

- compute RMS statistics once
- apply attn norm weights once
- materialize the normalized activation vector into a provider buffer
- expose this buffer to Q, K, and V consumers

### 3. Attention Consumers

The attention consumers are the Q, K, V projections and their downstream attention logic.

Responsibilities:

- consume the shared attention provider buffer
- never recompute the provider extraction internally
- write Q temporary state and K/V cache outputs

### 4. FFN Provider

The FFN provider computes the shared FFN-normalized activation state once after the attention projection residual update.

Responsibilities:

- compute FFN RMS statistics once
- apply FFN norm weights once
- materialize the normalized activation vector into a provider buffer
- expose this buffer to FFN gate and up consumers

### 5. FFN Consumers

The FFN consumers are the gate and up projections followed by the FFN down projection.

Responsibilities:

- consume the shared FFN provider buffer
- avoid recomputing normalization state in each projection row
- preserve current residual semantics

### 6. Positional Policy Provider

The positional policy provider owns the precomputed policy resources needed for RoPE-family behavior.

Responsibilities:

- build standard RoPE resources
- build linear-scaled RoPE resources
- build YaRN-compatible resources when configured
- optionally host additional private experimental variants behind explicit internal policy IDs

### 7. Runtime Policy Contract

The runtime contract binds one selected positional policy to execution.

Responsibilities:

- define the active policy kind
- define the active policy parameters
- bind the associated preflight buffer resources
- prevent shader-side ambiguity about which policy is active

## Required Buffer and Resource Model

The first implementation should assume these logical buffers per active token batch:

1. `activation_in`
2. `attn_normed`
3. `ffn_normed`
4. `temp_q`
5. `temp_attn`
6. `temp_ffn_gate_up`
7. `kv_cache_k`
8. `kv_cache_v`
9. `positional_policy_resource`

The exact packing can change, but the contract must preserve the logical distinction between provider buffers and consumer scratch.

## Required Runtime Contract

The runtime positional contract must be structurally equivalent to:

```json
{
  "policy_kind": "standard | linear | yarn | custom_private",
  "rope_dim": 64,
  "rope_base": 10000.0,
  "max_ctx_or_dist": 2048,
  "freq_scale": 1.0,
  "n_ctx_orig": 2048,
  "ext_factor": 0.0,
  "attn_factor": 1.0,
  "beta_fast": 32.0,
  "beta_slow": 1.0,
  "resource_id": "selected preflight positional buffer"
}
```

The host-side representation can use Rust structs and enums, but the contract must remain fixed-shape and auditable.

## Pass Graph Requirements

The first implementation revision should target this logical pass graph for one layer:

1. `AttnNormProvider`
2. `QKVConsumer`
3. `AttentionOut`
4. `AttentionProj`
5. `FfnNormProvider`
6. `FfnGateUpConsumer`
7. `FfnDown`

This is still a split-pipeline architecture, but it removes the major repeated extraction error.

Later fusion can reduce pass count further. The first milestone is not maximum fusion. The first milestone is correct provider-consumer separation.

## Mandatory Invariants

### Normalization Invariants

1. Attention RMSNorm is computed exactly once per token row per layer.
2. FFN RMSNorm is computed exactly once per token row per layer.
3. Q, K, V must consume the same attention-normalized source values.
4. Gate and up must consume the same FFN-normalized source values.

### Positional Invariants

1. Positional policy is selected only through the runtime contract.
2. Positional tables or policy resources are created in preflight, not inside the hot inner loop.
3. Policy selection must not require ad hoc shader edits for each supported mode.

### Numerical Invariants

1. Determinism must not regress.
2. F32 KV cache must remain unchanged unless a separate precision study proves otherwise.
3. Any approximation introduced for performance must be separately benchmarked and justified.

### Operational Invariants

1. The architecture must support direct A/B comparison against the current path.
2. The rollout must permit rollback by commit boundary.
3. Validation must separate throughput wins from behavioral regressions.

## Positional Policy Construction Requirements

### Standard RoPE

Must support precomputed cos/sin resources derived from:

- `rope_base`
- `rope_dim`
- active position or relative distance

### Linear-Scaled RoPE

Must support precomputed resources derived from:

- standard RoPE ladder
- explicit `freq_scale`

### YaRN

Must support a policy-compatible construction equivalent in semantics to the chosen reference implementation.

The contract must explicitly include:

- `n_ctx_orig`
- `freq_scale`
- `ext_factor`
- `attn_factor`
- `beta_fast`
- `beta_slow`

The implementation must compute any required correction ramp or correction dimensions outside the hot consumer loop unless measurement proves the alternative is superior.

### Hybrid or Experimental Variants

Any nonstandard positional variant must:

1. be identified by an explicit internal policy ID
2. define its angle transform and any magnitude scaling in a written policy note
3. use the same fixed runtime contract surface where possible
4. never silently replace standard, linear, or YaRN semantics

## Rollout Plan

### Phase 1: Specification and Contract Lock

1. lock the provider-consumer vocabulary
2. lock the runtime positional contract
3. lock the first-pass validation matrix

### Phase 2: Attention Provider Extraction

1. add attention-normalized provider staging
2. remove repeated attention-side normalization work from QKV consumers
3. benchmark decode and prefill impact

### Phase 3: FFN Provider Extraction

1. add FFN-normalized provider staging
2. remove repeated FFN-side normalization work from gate/up consumers
3. benchmark decode and prefill impact

### Phase 4: Positional Policy Contract

1. replace ad hoc RoPE assumptions with fixed policy selection
2. support standard and linear policies first
3. add YaRN only after policy semantics are pinned and validated

### Phase 5: Dispatch Reduction Follow-Up

1. inspect whether provider extraction creates obvious fusion opportunities
2. collapse passes only where correctness and measurement justify it

## Validation Requirements

The first implementation is not complete unless it passes all of the following:

### Correctness Validation

1. layer-level CPU versus GPU comparisons at fixed seeds
2. prefill and decode parity checks on existing probes
3. no new context-cliff behavior at current supported window

### Performance Validation

1. single-token decode latency before and after attention provider extraction
2. single-token decode latency before and after FFN provider extraction
3. chunked prefill wall-clock comparison before and after provider extraction
4. dispatch-count accounting per token

### Stability Validation

1. deterministic run-hash comparison on the current proof path
2. no new helical-shift regressions
3. no new out-of-bounds behavior at context edge conditions

## Open Questions

1. Should provider buffers be separate named buffers or carved from the existing temp buffer with explicit regions?
2. Is a two-provider architecture enough for the first pass, or is a combined norm-plus-broadcast kernel already justified?
3. Should YaRN be table-backed, parameter-backed, or mixed depending on the final relative-position design?
4. Does the current relative-position attention path want full policy tables, compact factors, or both?
5. Which performance counter will serve as the primary gate for declaring the first pass successful?

## Acceptance Bar For The First Code Change

Do not start the first hot-path refactor until all of the following are true:

1. the provider and consumer stages are named and agreed
2. the positional runtime contract fields are agreed
3. the benchmark protocol is agreed
4. the rollback checkpoint exists as a clean commit boundary

## Current Working Summary

The immediate architectural bet is not "more clever math in the hot loop."

The immediate bet is:

1. shared token extraction once
2. bounded broadcast to many consumers
3. fixed positional policy contract
4. narrow rollout with measurable checkpoints

If this architecture is correct, TinyLlama 1.1B becomes the proving ground for:

1. provider-consumer bindless execution
2. long-context-safe positional policy selection
3. later expansion to larger quants and larger same-family models
