# Three-Model Regression Workstream

Date: 2026-06-04
Branch: feat/model-routing-unified-control-plane

## Scope Boundary

This workstream is separate from control-plane completion.

Control-plane status is complete for selection correctness:
- Strict route matrix pass: 8/8
- Manifest contract pass: 8/8

These remaining regressions are output-quality issues, not control-plane selection gating issues.

## Current Affected Models

- phi-2.Q4_K_M.gguf
- starcoder2-3b-Q4_K_M.gguf
- Qwen3-0.6B-Q4_K_M.gguf

## Latest Known Snapshot

Latest smoke artifact:
- artifacts/model_smoke/smoke_20260604T230830Z.csv

Result split:
- PASS: 5
- WEAK: 3
- FAIL: 0

Weak signatures:
- phi-2: repetitive "To To ..." pattern
- starcoder2: repetitive "azi..." pattern
- Qwen3: multilingual garbage/token-noise output

## Historical Pattern (Smoke Artifacts)

Observed timeline includes repeated state flips:
- PASS intervals exist for all three
- Multiple WEAK clusters
- Multiple FAIL clusters (startup/process exits in earlier runs)

Per-model aggregate from available smoke CSVs:
- phi-2: FAIL=5, PASS=4, WEAK=13
- starcoder2: FAIL=5, PASS=2, WEAK=14
- Qwen3: FAIL=5, PASS=2, WEAK=14

Interpretation:
- Regression is not monotonic and not fully locked to one mode
- A deterministic commit-window isolation pass is required

## Next Execution Steps (Separate Track)

1. Commit-window bisect around first stable PASS -> first sustained WEAK transition.
2. Keep control-plane flags fixed during triage:
   - SHIMMY_ROUTE_CHECK_STRICT=1
   - SHIMMY_ROUTE_CHECK_FAIL_ON_WARN=1
   - SHIMMY_ROUTE_V2_LAYER_PARAMS=1
3. Run per-model A/B trace pairs (raw vs chat) with identical decode parameters.
4. Compare formula-lens divergence by model and isolate first failing phase/layer.
5. Apply one minimal math-path change at a time, with immediate per-model rerun.
6. Reconfirm control-plane manifest contract after each candidate fix to avoid drift.

## Non-Goals For This Track

- Do not reopen control-plane selection design.
- Do not block control-plane completion status on these three model outputs.
- Do not merge prompt and shader hypotheses in the same edit batch.
