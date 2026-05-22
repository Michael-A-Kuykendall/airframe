# Airframe

Airframe is the private inference engine/runtime currently being prepared as the engine behind Shimmy.

## Current Status

- No substantive blocker is currently confirmed for the exact TinyLlama story repro path.
- Long-run exact-story parity is currently being revalidated; the short-prefix `7777` check remains the stable confirmed proof point.
- Context window is model-native: the server reads `n_ctx` directly from the GGUF header (`spec.n_ctx`). No hardcoded limit — it is whatever the loaded model reports.
- Shimmy remains the intended public product surface; Airframe remains the internal engine.

## Right Now

The repo is in a proof-and-cleanup phase, not an engine-crisis phase.

Current priorities are:

- keep the launch envelope honest
- validate helical shift behavior under deliberate long-run tests
- reduce repository clutter so release work is easier to reason about

## Key Docs

- `docs/shimmy-airframe-launch-envelope.md`
- `docs/shimmy-airframe-release-strategy.md`
- `docs/shimmy-airframe-integration-checklist.md`
- `docs/airframe-current-stack-audit.md`
- `docs/helical-shift-validation-plan.md`
- `RELEASE_STATUS.md`
