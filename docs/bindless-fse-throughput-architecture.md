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

## Concrete First-Pass Buffer Contract

The first implementation should not introduce unnecessary new GPU allocations per layer dispatch. The default assumption should be reuse of the existing temporary storage with explicit logical regions.

The first-pass logical layout should be treated as:

1. `activation_in`
2. `attn_normed_region`
3. `q_region`
4. `attn_out_region`
5. `ffn_normed_region`
6. `ffn_gate_region`
7. `ffn_up_region`

The exact byte offsets may remain implementation-defined, but the following must be true:

1. `attn_normed_region` must be immutable for all QKV consumers within a layer pass.
2. `ffn_normed_region` must be immutable for all FFN gate/up consumers within a layer pass.
3. provider regions must not alias consumer output regions unless the aliasing is proven safe and documented.
4. the same layout logic must work for both batch prefill and single-token decode.

The first implementation should prefer region reuse over new buffer proliferation unless measurement proves separate buffers materially improve performance or correctness.

### Derived Region Sizes

For a model with:

- `dim = n_embd`
- `head_dim = n_embd / n_head`
- `q_dim = n_head * head_dim`
- `ffn_dim = feed_forward_length`

the first-pass logical region sizes are:

1. `attn_normed_region = dim`
2. `q_region = q_dim`
3. `attn_context_region = dim`
4. `ffn_normed_region = dim`
5. `ffn_gate_region = ffn_dim`
6. `ffn_up_region = ffn_dim`

Peak simultaneous pressure by phase is:

1. attention provider plus Q staging: `dim + q_dim`
2. FFN provider plus gate/up staging: `dim + 2 * ffn_dim`

For the current TinyLlama 1.1B path:

- `dim = 2048`
- `n_head = 32`
- `head_dim = 64`
- `q_dim = 2048`
- `ffn_dim = 5632`

So the concrete region sizes are:

1. `attn_normed_region = 2048`
2. `q_region = 2048`
3. `attn_context_region = 2048`
4. `ffn_normed_region = 2048`
5. `ffn_gate_region = 5632`
6. `ffn_up_region = 5632`

And the concrete peak working-set sizes are:

1. attention phase peak: `2048 + 2048 = 4096`
2. FFN phase peak: `2048 + 5632 + 5632 = 13312`

This means the current TinyLlama temp stride can support the first-pass provider architecture by region reuse without introducing a larger scratch allocation, as long as the phase schedule is explicit.

### Required TinyLlama First-Pass Region Reuse Schedule

For the current TinyLlama path, the intended logical reuse schedule should be:

#### Phase A: Attention Provider And Consumers

1. `[0, dim)` -> `attn_normed_region`
2. `[dim, dim + q_dim)` -> `q_region`

Execution contract:

1. `AttnNormProvider` writes `[0, dim)`
2. `QKVConsumer` reads `[0, dim)` and writes Q to `[dim, dim + q_dim)` while writing K/V to cache
3. `AttentionOut` reads Q from `[dim, dim + q_dim)` and writes context to `[0, dim)` by reusing the former attention-provider region
4. `AttentionProj` reads context from `[0, dim)` and then both regions become available for FFN use

#### Phase B: FFN Provider And Consumers

1. `[0, dim)` -> `ffn_normed_region`
2. `[dim, dim + ffn_dim)` -> `ffn_gate_region`
3. `[dim + ffn_dim, dim + 2 * ffn_dim)` -> `ffn_up_region`

Execution contract:

1. `FfnNormProvider` writes `[0, dim)`
2. `FfnGateUpConsumer` reads `[0, dim)` and writes gate/up outputs to the two FFN regions
3. `FfnDown` reads both FFN regions and writes the residual update to `activation_in`

The implementation may choose different exact offsets, but if it does, the replacement layout must still satisfy the same peak-memory and non-aliasing rules.

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

## Required Host-Side API Shape

The host-side code should expose the new architecture through explicit structs rather than implicit parameter packing spread across multiple call sites.

At minimum, the implementation should converge toward the following concepts:

1. `LayerExecutionPlan`
2. `ProviderLayout`
3. `PositionalPolicyConfig`
4. `ExecutionMode`

### LayerExecutionPlan

Must describe:

1. which provider stages are active
2. which consumer passes read which provider regions
3. which positional resource is bound
4. whether execution is prefill or decode

### ProviderLayout

Must describe:

1. the logical provider and scratch regions
2. the element counts and offsets for each region
3. whether the layout is valid for the current batch shape

### PositionalPolicyConfig

Must describe:

1. policy kind
2. all scalar parameters required by that policy
3. the preflight resource identifier or buffer handle

### ExecutionMode

Must distinguish:

1. chunked prefill
2. single-token decode
3. debug or probe execution if retained

This is required so the implementation does not silently fork behavior across ad hoc call paths.

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
4. provider extraction must not change the mathematical order of operations without an explicit comparison against the current reference behavior.

### Operational Invariants

1. The architecture must support direct A/B comparison against the current path.
2. The rollout must permit rollback by commit boundary.
3. Validation must separate throughput wins from behavioral regressions.
4. the first production candidate must preserve a truthful public launch envelope even if the internal architecture is capable of more.

## Failure Modes And Required Mitigations

The architecture is not production-credible unless it explicitly names the main ways it can fail.

### 1. Provider Region Corruption

Failure mode:

- one consumer overwrites provider data needed by another consumer in the same layer execution

Mitigation:

1. explicit region layout contract
2. debug assertions on region bounds and overlap assumptions
3. one probe path that stages provider regions back to host for verification

### 2. Prefill And Decode Semantic Drift

Failure mode:

- prefill and decode use nominally the same architecture words but materially different math or buffer semantics

Mitigation:

1. shared host-side execution-plan surface
2. shared positional-policy contract
3. parity probes at layer and full-model boundaries

### 3. Positional Policy Drift

Failure mode:

- standard, linear, and YaRN paths diverge because parameters are packed differently or interpreted differently across host and shader

Mitigation:

1. one typed policy config on the host
2. one shader-side contract layout
3. explicit reference tests for standard and linear first, then YaRN

### 4. Dispatch Count Improvement With Hidden Numerical Regression

Failure mode:

- throughput improves while output quality, determinism, or long-context behavior regresses

Mitigation:

1. throughput checks never run alone
2. every performance comparison is paired with correctness and determinism checks
3. long-context checks remain a release gate, not a later cleanup item

### 5. Helical Shift Interaction Regression

Failure mode:

- provider refactor is locally correct but changes the behavior around shift or compaction boundaries

Mitigation:

1. retain helical shift validation as a required gate
2. include multi-boundary stress runs before declaring production readiness

## Observability Requirements

The architecture must be observable enough that a regression can be localized without guesswork.

The first production-oriented implementation must expose:

1. dispatch counts per token for decode and per chunk for prefill
2. provider-stage timings
3. consumer-stage timings
4. positional-policy selection for each run
5. shift and compaction events during long decode

The system does not need a permanent heavyweight tracing stack immediately, but it must have at least one repeatable way to capture these values during validation.

## Benchmark Protocol Requirements

The benchmark protocol must be fixed before implementation claims are accepted.

The first-pass benchmark matrix should include:

1. single-token decode latency at short context
2. single-token decode latency near the active context limit
3. chunked prefill wall-clock time for a representative prompt length
4. throughput stability across repeated seeded runs
5. memory footprint before and after provider extraction

Every benchmark row must record:

1. commit SHA
2. model path and quant
3. prompt or fixture identifier
4. active positional policy
5. batch or chunk size
6. stop reason if generation is involved

### Required First-Pass Fixture Set

The first-pass benchmark and validation cycle should use a small fixed fixture set drawn from the checked-in repo state.

Required fixtures:

1. short sanity fixture: `artifacts/story_seed7777_128tok_request.json`
2. exact-story long fixture: `artifacts/story_4k_exact_request_nostream.json`
3. helical stress fixture: `artifacts/helical_multi_boundary_request.json`
4. short SHA helper: `scripts/short_story_sha_check.ps1`
5. long-story helper: `scripts/long_story_check.ps1`
6. rope ladder helper: `scripts/rope_ladder_test.ps1`

Optional coding-benchmark fixtures such as the existing HumanEval runner may still be useful for evaluation work, but they are not required gates for the first provider-consumer refactor.

## Production Availability Gates

The following gates define whether this architecture is ready to be treated as production-capable for the codebase segment it touches.

### Gate 1: Contract Clarity

Required:

1. provider layout is explicit
2. positional contract is explicit
3. prefill and decode execution modes are explicit

### Gate 2: Mathematical Preservation

Required:

1. no unexplained CPU/GPU parity regressions
2. no unexplained decode drift increase versus current tip
3. deterministic proof path remains intact

### Gate 3: Throughput Win

Required:

1. measurable decode or prefill improvement on TinyLlama 1.1B
2. no hidden regression large enough to erase that win at realistic context lengths

### Gate 4: Long-Context Safety

Required:

1. no newly introduced context cliff inside the currently supported envelope
2. helical-shift edge-case validation remains acceptable
3. positional policy selection does not silently degrade long-context behavior

### Gate 5: Operational Safety

Required:

1. rollback is one commit or a small known series of commits
2. release envelope remains honest
3. debug and validation hooks remain available until the architecture is proven stable

## Current Blockers To Calling This Production-Ready

As of this draft, the architecture is not yet production-ready for the full codebase. The current blockers are explicit:

1. provider-consumer refactor is specified but not yet implemented
2. positional-policy contract is not yet fully wired end to end
3. YaRN semantics are not yet validated in this engine
4. long-context and helical-shift validation are still active work, not closed work
5. current public launch envelope remains narrower than the long-context internal ambition

This section is intentionally blunt. Confidence should rise only when blockers move from this list into recorded validation results.

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

### Production Validation

1. no hidden dependency on one-off local artifacts or generated indexes
2. required runbooks and benchmark scripts exist and still execute from the checked-in repo state
3. the architecture remains understandable from the docs and code without chat archaeology

## Open Questions

1. Should provider buffers be separate named buffers or carved from the existing temp buffer with explicit regions?
2. Is a two-provider architecture enough for the first pass, or is a combined norm-plus-broadcast kernel already justified?
3. Should YaRN be table-backed, parameter-backed, or mixed depending on the final relative-position design?
4. Does the current relative-position attention path want full policy tables, compact factors, or both?
5. Which performance counter will serve as the primary gate for declaring the first pass successful?
6. Which exact prompt fixtures should be the permanent decode and prefill benchmark set?
7. Which parts of the current runtime should be lifted out of binary-oriented paths before calling the system production-capable?

## Acceptance Bar For The First Code Change

Do not start the first hot-path refactor until all of the following are true:

1. the provider and consumer stages are named and agreed
2. the positional runtime contract fields are agreed
3. the benchmark protocol is agreed
4. the rollback checkpoint exists as a clean commit boundary

## Acceptance Bar For Production Availability

Do not describe this architecture as production-available until all of the following are true:

1. the first-pass provider refactor is merged and benchmarked
2. standard and linear positional policies are fully wired through the fixed runtime contract
3. the selected YaRN path is either implemented and validated or explicitly excluded from the release envelope
4. long-context and helical-shift behavior have recorded validation artifacts
5. the runtime and docs are coherent enough that another engineer can operate the system without relying on unreproducible chat context

## Production Readiness Matrix

Before calling the work production-capable, every row below must be marked with a concrete result, not an intention.

| Area | Required state | Evidence |
|---|---|---|
| Provider extraction | implemented and benchmarked | before/after latency and parity records |
| Positional contract | wired end to end | host config, shader binding, and policy probe output |
| Standard RoPE | working | reference comparison output |
| Linear scaling | working | reference comparison output |
| YaRN | validated or explicitly deferred | policy note plus validation record or release exclusion |
| Determinism | preserved | run-hash proof |
| Helical shift | acceptable | recorded long-run outputs and classification |
| Rollback | simple | checkpoint commit chain |
| Runbooks | current | checked-in docs and scripts |
| Public envelope | honest | release docs consistent with tested behavior |

No row may be marked complete based on chat memory alone.

## Required Sign-Off Artifacts

The architecture should not be considered ready for production use until the repo contains or references all of the following artifacts:

1. one benchmark summary for decode and prefill
2. one parity summary for layer or full-model comparisons
3. one determinism summary for the chosen proof path
4. one helical-shift validation summary
5. one release-envelope statement consistent with the actual validated behavior
6. one rollback note identifying the commit or small commit range to revert to if the rollout fails

These do not need to be polished reports. They do need to exist in a form another engineer can inspect without reconstructing the work from conversation history.

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

## Confidence Rule

This document should be revised until confidence comes from explicit contracts, measured benchmarks, and recorded validation outputs rather than intuition.

The standard for confidence is not:

1. the architecture sounds elegant
2. the hot loop looks cleaner
3. the design matches an intended philosophy

The standard for confidence is:

1. the contracts are explicit
2. the implementation matches the contracts
3. the measurements are favorable
4. the failure modes are named
5. the rollback path is simple
