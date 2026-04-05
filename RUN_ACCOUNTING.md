# Run Accounting

## Verified Current Session Runs

| Lane | Request file / source | Seed | Result | SHA / Match status |
|---|---|---:|---|---|
| Short sanity check | `artifacts/story_seed7777_128tok_request.json` | 7777 | Completed, 128 tokens | SHA `f82a1ad07e5f74415a3121821e580998eecda4edd30b43efc9b294aa591c7974`, matches expected |
| Long default exact-story request | `artifacts/story_4k_exact_request_nostream.json` | 55555 | Completed, saved to `artifacts/story_4k_exact_current_status_new_text.txt` | SHA `6cf24939e3afcc889332d0daed2aa735b2271a7d2f996dbd17db6cb25d660636`, does **not** match historical reconstructed artifact |
| Long helical-off exact-story request | `artifacts/story_4k_exact_request_nostream.json` via `8081` with `SHIMMY_DISABLE_HELICAL_SHIFT=1` | 55555 | Completed after server fix, saved to `artifacts/story_4k_exact_helical_off_text.txt` | SHA `6da85c62d12eb6ee49251a894f454f061b076247f65ae46bb09faa3e6f43645f`, stop reason `context_limit`, does **not** match historical reconstructed artifact |
| Multi-boundary long decode run 1 | `artifacts/helical_multi_boundary_request.json` | 7777 | Completed, `5200` generated tokens, saved to `artifacts/longctx/helical_multi_boundary_run1.txt` | SHA `f7c88f33787b3715cd5f9f49e2acb8c5e5574fc71e94173cda141fc4ca3de4e5` |
| Multi-boundary long decode run 2 | `artifacts/helical_multi_boundary_request.json` | 7777 | Completed, `5200` generated tokens, saved to `artifacts/longctx/helical_multi_boundary_run2.txt` | SHA `f7c88f33787b3715cd5f9f49e2acb8c5e5574fc71e94173cda141fc4ca3de4e5`, exact byte match to run 1 |
| Provider smoke test | `scripts/openclaw_provider_smoke_test.ps1` | n/a | Passed | Not a story hash test |

## Verified Current Session Stress Checks

| Check | Input | Result |
|---|---|---|
| Multi-boundary decode | `artifacts/helical_multi_boundary_request.json` | Reached `5200` generated tokens with stop reason `max_tokens`, which is enough to cross at least two helical compaction thresholds on the current `4096`-slot KV cache |
| Repeated-seed long decode determinism | Same request, same seed `7777`, two runs | Exact byte-identical output; both runs hashed to `f7c88f33787b3715cd5f9f49e2acb8c5e5574fc71e94173cda141fc4ca3de4e5` |
| Session reuse vs fresh | Raw prompt `Write exactly one Rust function named add_two...`, seed `4242` | `fresh1`, `fresh2`, and first new-session call all returned identical empty `eos` outputs; the second reused-session call diverged to a non-empty `96`-token output with SHA `88b59079f3f3b002d12604df53313761217236696253991314ac226a09093e70` |

## Historical / Archived Artifacts Present In Workspace

| Artifact | Implied lane | Seed | SHA / Status |
|---|---|---:|---|
| `artifacts/historical_seed7777_reconstructed.txt` | Historical reconstructed story artifact | likely 7777 | SHA `c683570c1d06e9252267c1e0bb977a993b217b31de66417d841e7fbee1136eeb` |
| `artifacts/story_4k_exact_tip_rerun_final.txt` | Prior claimed clean rerun artifact | unclear from file alone | SHA `8a09a2f8a23e4565b6ab13594e86a3dffce687fb3aff4339e53ce3325e1fb082` |
| `artifacts/story_4k_exact_current_status.json` | Older exact-story status capture | 55555 | Completed with `265` tokens, Olivia Clark variant, not comparable as a match proof |
| `artifacts/story_4k_exact_compare.json` | Older compare summary | unknown compare pair | Records mismatch: current hash `621ea706...`, old hash `b4b352ad...` |

## Important Reconciliation

1. The short sanity proof is a `7777` / 128-token front-of-run check.
2. The long exact-story request currently checked in is **not** seed `7777`; it is seed `55555`.
3. The workspace contains documentation claiming a prior clean full rerun matched history, but the presently visible artifacts do not collapse to one single self-consistent SHA chain.
4. The current session's long default run therefore does **not** disprove the short `7777` sanity proof.
5. It **does** show that the current full exact-story lane using the checked-in `55555` request diverges from `historical_seed7777_reconstructed.txt`.
6. The helical-off comparison shows the same early divergence as the default run, so the currently observed long-run mismatch is not explained solely by helical shift.
7. A fixed-seed long decode now shows exact determinism even after crossing multiple helical compaction boundaries.
8. Session replay is a real behavioral variable on tip: the same prompt with the same seed diverges once prior session state is reused.

## Archived Historical Claim References

The workspace also contains archived text claiming an earlier multi-run determinism proof in libshimmy history:
- `7777` original + repeat identical
- `9999` control + repeat identical
- same seed = same hash, different seed = different hash

Those claims are documented in `artifacts/libshimmy_test_search.txt`, but the underlying JSON result files for those runs are not present here, so they should be treated as documented historical claims rather than directly re-verified evidence in this workspace.
