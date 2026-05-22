# Release Status

## Summary

Airframe is not currently blocked by a confirmed engine failure on the TinyLlama exact-story repro path.

What is established right now:

- current tip matches the historical `7777` short-prefix baseline
- long-run exact-story status is currently under revalidation, and the short-prefix proof remains the only fully re-confirmed parity point in this session
- the truthful preview claim remains `2048` tokens of public context

The repo is in cleanup, proof, and contract-hardening mode.

## No-Known-Blocker Statement

At this time, no substantive current repro blocker has been confirmed for the exact TinyLlama story path that was under dispute.

That means:

- no confirmed early-token math divergence
- long-run content parity is not yet closed for the current session
- provider proofing is no longer part of the active release gate for this pass

## Active Release Gates

1. Keep the public context claim honest: context window is model-native (read from GGUF `n_ctx`). Do not advertise a specific number — it varies by model.
2. Characterize helical-shift behavior with explicit long-run edge-case tests.
3. Reduce repository clutter and generated noise.
4. Defer OpenClaw provider rollout until a real 16K-capable model path exists.

## What Is Not Yet Claimed

- no claim of native `4096` or `8192` user-visible context for this preview
- no claim that OpenClaw should ship on the current 2048-token launch envelope
- no claim that helical shift has been exhaustively validated across long-run edge cases

## Immediate Next Proofs

1. Execute the helical validation plan in `docs/helical-shift-validation-plan.md`.
2. Keep any new launch-facing docs aligned with `docs/shimmy-airframe-launch-envelope.md`.
3. Do not resume OpenClaw release work until a 16K-capable model/runtime path exists.