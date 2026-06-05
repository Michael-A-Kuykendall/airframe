# Model Routing Unified Plan

Date: 2026-06-04
Branch: feat/model-routing-unified-control-plane
Status: authoritative implementation plan for unified prompt + math routing.

## Purpose

Create one load-time route artifact that controls both prompt rendering and math execution paths, with explicit policy semantics and strict validation gates.

This plan replaces fragmented routing-plan docs and incorporates validated local-code constraints.

## Problem Statement

Current routing is split across multiple control surfaces:

1. Prompt routing is selected in server logic.
2. Math routing is partially encoded via typed fields and partially inferred in shader branches.
3. Route checks exist, but schema/provenance is not yet complete.

Resulting risk:

1. Metadata can select a high-level route correctly while hot-path branches infer behavior differently.
2. F32 and INT4 can drift if they do not consume identical policy contracts.

## Current Ground Truth (Code-Verified)

1. Model metadata parsing is centralized in src/core/spec.rs.
2. Route scaffold exists in src/core/routing.rs.
3. Startup route check exists in src/bin/shimmy_server_gpu.rs and can fail in strict mode.
4. LayerParams already carries post_norm_enabled, qk_norm_enabled, layer_norm_enabled.
5. WGSL hot path still infers FFN behavior from offsets.ffn_gate in key branches.

## Target Architecture

## 1. Single Route Artifact

Define ModelRoutePlan v2 as the source of truth for:

1. Identity: arch family, model name, file type, route version.
2. Norm policy: norm kind, epsilon, qk_norm, post_norm.
3. Attention policy: qkv layout, head geometry, softcaps.
4. Positional policy: rope kind/dim/base/scale and context train/runtime.
5. FFN policy: gated vs non-gated, activation family.
6. Prompt policy: renderer mode/family/template source.
7. Validation: reasons, warnings, hard errors, strict pass/fail.
8. Audit: stable route digest.

## 2. Fused Semantic Execution Contract

All runtime consumers must read explicit route-plan fields only:

1. Prompt renderer selection.
2. LayerParams population.
3. Shader policy behavior for FFN/QKV/norm branches.

No consumer may re-infer model policy from missing tensors in hot kernels after cutover.

## 3. Strictness Model

Use two gates:

1. SHIMMY_ROUTE_CHECK_STRICT=1: fail on hard errors.
2. SHIMMY_ROUTE_CHECK_FAIL_ON_WARN=1: optional warning-fail mode for CI hardening.

## Hot-Path Breakpoints To Eliminate

Primary breakpoints:

1. offsets.ffn_gate-driven behavior in F32 shader path.
2. mirrored implicit behavior in INT4 path.
3. non-plan-owned post_norm toggles outside route contract.

Cutover condition:

1. policy branches consume params.ffn_kind / params.qkv_layout (or equivalent explicit fields).
2. offsets-presence fallback only allowed in compatibility mode, then removed.

## Implementation Plan

## Phase 0: Consolidate and Instrument

1. Keep existing behavior.
2. Expand route-plan schema with compatibility mapping to current ModelSpec field names.
3. Emit route digest and full route snapshot in ROUTE_CHECK.

Exit criteria:

1. Route report includes full policy fields.
2. No behavior changes yet.

## Phase 1: Dual Population

1. Populate both legacy flags and v2 route fields.
2. Add explicit LayerParams fields for ffn_kind and qkv_layout behind feature flag route_v2_layer_params.

Exit criteria:

1. Route v2 and legacy outputs match expected model family decisions.
2. Strict checks pass on baseline models.

## Phase 2: Consumer Wiring (Data Path)

1. Drive LayerParams from ModelRoutePlan v2.
2. Ensure F32 and INT4 consume identical route semantics.

Exit criteria:

1. No regression in baseline text matrix.
2. Phi harness and formula-diff gates remain green.

## Phase 3: Hot-Path De-Heuristic

1. Replace offsets.ffn_gate inference branches with explicit policy fields.
2. Keep compatibility fallback under temporary flag route_v2_compat.

Exit criteria:

1. Semantic protocol checks pass for both F32 and INT4.
2. No unexplained divergence increase at formula checkpoints.

## Phase 4: Prompt-Math Unification Finalization

1. Make prompt renderer decisions route-plan-owned.
2. Remove legacy split control path.

Exit criteria:

1. One artifact governs prompt + math routing.
2. Legacy route code deleted.

## Phase 5: Default-On and Cleanup

1. Flip route_v2 defaults on.
2. Remove compatibility branch and dead fields.
3. Bump route_version.

Exit criteria:

1. CI strict route gate green.
2. Baseline + Gemma gate + INT4 sweep green or classified by hardware limits.

## Semantic Execution Protocols (Pass/Fail)

Protocol A: Route Determinism

1. Same model manifest -> same route digest across runs.
2. Any route mismatch is explainable by explicit runtime override.

Protocol B: Consumer Consistency

1. Prompt path and math path report same route version and digest.
2. F32 and INT4 report same policy decisions for shared fields.

Protocol C: Hot-Path Purity

1. No architecture behavior inferred from free-form string matching in kernels.
2. No FFN policy inferred solely from missing ffn_gate in final mode.

Protocol D: Regression Safety

1. Phi smoke formula path green.
2. Baseline text matrix green.
3. Gemma blob gate green.
4. INT4 regression pass green for baseline + Gemma.

## Confidence-Weighted Risks

1. Highest risk: shader de-heuristic step (mitigation: feature-gated staged cutover).
2. Medium risk: broadening post_norm semantics without model-specific validation (mitigation: conservative default).
3. Medium risk: prompt unification seam regressions (mitigation: dual emit + digest checks).

## Ownership and File Map

1. src/core/spec.rs: metadata parsing and derived traits only.
2. src/core/routing.rs: route schema, builder, validation, digest.
3. src/backend/bindless/metadata.rs: tensor manifest compilation only.
4. src/backend/bindless/pipeline/mod.rs and inference.rs: route-plan-driven params.
5. src/backend/bindless/sh_layer_v1.wgsl and sh_layer_v1_int4.wgsl: explicit policy consumption.
6. src/bin/shimmy_server_gpu.rs: startup route check, prompt seam integration.

## Immediate Execution Checklist

1. Finalize v2 route schema names aligned to current ModelSpec symbols.
2. Add route digest + strict-mode split gates.
3. Add explicit ffn_kind/qkv_layout execution fields behind feature flag.
4. Wire params from route plan.
5. Run semantic protocol matrix and smoke gates per phase.
6. Remove compatibility path after two consecutive clean sweeps.

## Execution Status (2026-06-04)

1. Control-plane route selection matrix completed with current release binary.
2. Artifact: `artifacts/route_check/route_check_20260604T233346Z.csv`.
3. Coverage: TinyLlama, Llama-3.2-1B/3B, phi-2, starcoder2-3b, gpt2, Qwen3-0.6B, gemma-2-2b.
4. Result: PASS=8, FAIL=0, SKIP=0 for route-plan selection and startup route-check invariants.
5. Scope note: this validates routing selection correctness, not output-quality pass/fail semantics.
6. Contract manifest added: `fixtures/control_plane_route_manifest.json` with expected per-model routing fields.
7. Manifest validator added: `scripts/validate_route_manifest.py`.
8. Contract validation result: PASS against `artifacts/route_check/route_check_20260605T005239Z.csv`.

## Document Governance

This document is the single routing-plan authority.

Superseded planning docs are archived under docs/archive/model-routing/.
