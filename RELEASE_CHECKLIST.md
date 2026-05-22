# Airframe -> Shimmy Release Checklist

This document tracks the concrete steps required to ship the current `airframe` backend as the default provider for `shimmy_integration`. 

## 1. Context Window — Model-Native (not a fixed number)
- [x] **Confirmed**: Server reads `spec.n_ctx` directly from GGUF metadata. No hardcoded context limit.
- [x] **`SHIMMY_MAX_CTX` override**: Available to cap or extend beyond the model default.
- [x] **Practical VRAM bound**: KV cache allocation scales with context × layers × heads — large context on large models requires adequate VRAM.
- [x] **Per-model verified context**: TinyLlama 2048, Llama-3.2-1B/3B 131072, Gemma-2-2B 8192, StarCoder2-3B 16384.

## 2. Helical Shift Validation
- [x] **Short Sanity Check**: Run current tip with default settings and confirm the short SHA matches `f82a1ad...`. (Use `scripts/short_story_sha_check.ps1`).
- [x] **Long Default Run**: Run the full exact-story request on current tip. Saved final text output and confirmed a substantive divergence from the historical extracted story file starting near the beginning of the text.
- [x] **Long Helical-Off Run**: Run the same full story with helical shift disabled. It now terminates cleanly at `context_limit` instead of crashing, and it diverges from the historical baseline at the same early point as the default run.
- [x] **Stress Cases**: 
  - [x] Cross the compaction boundary multiple times. A fixed-seed raw decode ran to `5200` generated tokens with stop reason `max_tokens`, which necessarily crossed at least two helical compaction thresholds on the current `4096`-slot KV cache.
  - [x] Repeated-seed runs to verify determinism under long decode. Two runs of `artifacts/helical_multi_boundary_request.json` produced identical `20979`-character outputs with matching SHA `f7c88f33787b3715cd5f9f49e2acb8c5e5574fc71e94173cda141fc4ca3de4e5`.
  - [x] Session reuse vs. fresh session for the same prompt. Two fresh runs and the first run of a new `session_id` all returned identical empty outputs (`eos`, `0` tokens), while the second reused-session call diverged to a non-empty `96`-token response, confirming that session replay materially changes the prompt path.
- [x] **Review & Document**: Current long-run divergence is not helical-specific, helical-off now stops gracefully at the 2048 context boundary, long decode remains deterministic across repeated multi-boundary runs, and session reuse is confirmed to alter behavior relative to a fresh prompt.

## 3. Deferred: OpenClaw / 16K Support
- [ ] **Do not treat OpenClaw as a current release gate**: Provider proofing is deferred until Airframe has a real 16K-capable model path rather than the current 2048-token launch envelope.
- [ ] **Revisit provider integration only after 16K support exists**: When a 16K-capable model/runtime path is real, restore provider smoke validation and launch-facing OpenClaw work.

## 4. Repository Cleanup & Commit
- [ ] **Stash/Remove Archaeology**: Delete or stash the `artifacts/` folder, `chat-3-22.md`, and any other temporary reconstructions generated during the branch archaeology phase.
- [ ] **Finalize Documentation**: Commit `RELEASE_STATUS.md`, the Launch Envelope, and this checklist.
- [ ] **Review `git status`**: Ensure only clean, engine-relevant modifications to `.rs` files and `shimmy_integration` exist.
- [ ] **Commit Changes**: Commit the final release readiness changes to the `openclaw-local-provider` branch.

---
*Note: Do not push `shimmy_integration` to public origin. If pushed, target the `private` remote as per the repository push policy.*