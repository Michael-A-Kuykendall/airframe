# Current Work Directive — Stabilization + Gentek Local Dev (2026-06-12)

## Roles (Strict)
- **Developer (Grok / this agent)**: Primary implementer. Creates branches (feature, fix/*, hotfix/*), does the coding, runs falsifiable experiments, prepares evidence, commits on proper branches.
- **PR Approving Development Manager (user)**: Owns approval, direction, final sign-off on PRs / merges. Defines priorities (e.g. "stabilize primary hotfix for all models").
- **Local KIRO Agent (user's secondary)**: Follow-up scanning audit helper. Scans branches, todos, diffs, artifacts, steering for compliance with ethos, past lessons, and this directive. Used for review before user approval.

You (developer) may create as many branches as useful for parallel sub-work, but **all work funnels through the primary hotfix stabilization branch**.

## Primary Goal Right Now
Stabilize the **primary hot fix branch `fix/v0.2.5-all-fixes`** so that **every sub-2GB model** passes a simple smoke:
`shimmy generate <model> --prompt "2+2=" --max-tokens 5` (or equivalent via `scripts/test_model.ps1`).

See `docs/internal/MASTER_HOTFIX_WORKSTREAM.md` for the live status table (GROUP A: TDR/device-lost on 7 models; GROUP B: empty/silent output head on Q6_K models; GROUP C: arch routing panics on fused-QKV like Phi-2/GPT-2).

Target: gentek local development platform fully operational as **primary development venue** (move all primary dev here). See `docs/internal/local_dev_platform_spike.md` + remaining tasks. Fast iteration = reliable env-var model paths, one-command tests, vault-driven verification, shimmy (console/chat) + airframe engine loop without constant D: hardcodes or manual eyeballing.

## Compliance Constraints (Things That Have Happened Before — Do Not Repeat)
Sourced from:
- This repo's `.kiro/steering/` (inference-testing.md, deploy-hotfix.md, terminal-discipline.md, **fse-d0-lens.md**) and global `C:\Users\micha\.kiro/steering/skills/`
- `docs/internal/kiro-handoff-2026-06-06.md` (Session Ethos)
- `docs/internal/MASTER_HOTFIX_WORKSTREAM.md`, NAN_BUG_ANALYSIS, previous session handoffs
- Immediate prior context: the Golden Reference Vault was built (vault_seed, DuckDB with 500+ layer_oracles rows for 11+ models, import/certify scripts) but repeatedly ignored in favor of manual `shimmy generate` + eyeballing or layer_dump_gpu. gpu_22layer_verify remained CSV-hardcoded + ignored. **Never again.**
- **FSE + D0 Lens is now mandatory perspective** (see `.kiro/steering/fse-d0-lens.md`): every challenge must be reframed as facts emitted from data streams, rules deriving semantics, consequents driving actions — in single fused/reactive passes with deduplication and broadcast. Use airframe_observe + d0-engine as the vehicle. 

## FSE + D0 Lens (Permanent Required Filter)
All work in this workspace (TDR stabilization, vault/golden traces, model verification, observation, console, etc.) **shall be viewed and solved through the FSE + D0 lens**:
- Data streams emit structural facts (Tier 1).
- Rules (alpha/beta, selectors) derive semantic facts (Tier 2).
- Consequents drive immediate actions/mutations (Tier 3).
- Single-pass, deduplicated selectors, value broadcast, early exit, saturation to empty.
- `∂runtime / ∂rules ≈ 0` for shared selectors.
- Arena for structural sharing where applicable.
- Implemented via ObservationSession / ReactiveGraph<InferenceFact> (extend as needed).
- This is the "increasingly fast shit" architecture. Do not re-implement imperative repeated traversals.

See full enshrinement and TDR application in `.kiro/steering/fse-d0-lens.md`.

### 1. KIRO Handoff Ethos (Non-Negotiable)
- One hypothesis at a time.
- **No code edits without a pre-declared falsifiable test** + single go/no-go metric (e.g. "first bad layer from vault oracle", "RMS delta < 1e-4", "PASS on test_model.ps1 for this model").
- Single metric set per run. If change does not improve the declared metric on the target, reject immediately.
- No broad branch surgery while diagnosing target models.
- Prepare clear artifacts for KIRO auditor scan (this todo list, evidence tables per model/group, branch state, minimal diffs, stdout logs, vault query outputs).

### 2. Hotfix & Branch Discipline (deploy-hotfix.md + inference-testing.md)
- Primary stabilization happens on `fix/v0.2.5-all-fixes` (or hotfix/** variants that trigger CI).
- Always branch hotfixes from the *latest release base*, not old clean branches.
- Hotfix content: the minimal fix(es) + version bump (if rolling) + CHANGELOG entry only. Cherry-pick style.
- CI: push to private first, wait green, then merge to release/* and public.
- Never leave dirty trees on main release branches for long; land or stash.
- Sub-branches off primary hotfix are encouraged for focused work (e.g. fix/v0.2.5-q4k-tdr).

Current primary: `fix/v0.2.5-all-fixes` (local + remotes/private).

### 3. Vault / Mechanized Testing First (Critical Lesson from This Session)
- The DuckDB `vault/vault.duckdb` + `vault/seeds/*.json` (populated by `cargo run --bin vault_seed`) + Python import/certify is the golden source for per-layer `expected_rms`, NaN/Inf, checksums for seeded models (TinyLlama, Qwen* series, Llama-3.2, etc.).
- **For ANY NaN, divergence, empty output, or model failure investigation: start here.**
  - Query vault/seeds for the model (or use mechanized test).
  - Use `frontier_compare` (with --dump-layers) or the (to-be-universal) vault-driven gpu_22layer_verify equivalent to get exact first failing layer + delta vs oracle.
  - Only after vault evidence: consider `shimmy generate` or layer_dump.
- Update tests (gpu_22layer_verify.rs etc.) and workflows to be vault/seed-driven, not hardcoded CSV + D: paths + ignored.
- "hi" prompt (or "2+2=") for smokes per inference-testing.md. One model at a time.

### 4. Testing & Terminal Discipline (inference-testing.md + terminal-discipline.md + model-testing skill)
- One-command: `powershell ... scripts/test_model.ps1 -ModelPath "..."` (or equivalent). One model, wait for finish. Never parallel.
- Use "hi" or short no-quote prompts.
- Kill stale shimmy before tests.
- `layer_dump_gpu` is **diagnostic only** and often differs from production GpuRuntime path — prefer frontier_compare + vault for truth.
- Max 2 background processes. Always `list_processes` (or equivalent via tools), stop finished ones immediately.
- For vault sweeps: follow the exact checklist in terminal-discipline.md.
- Rebuild protocol after changes: cargo build (airframe) then shimmy rebuild if needed, then cert with test script.

### 5. Gentek / Local Dev Platform Target
- Move primary development here (airframe as venue).
- Eliminate hardcoded `D:/shimmy-test-models/...` — centralize on env vars: `SHIMMY_BASE_GGUF`, `SHIMMY_MODEL_PATHS`, `SHIMMY_TEST_MODELS`.
- Fast loop: edit airframe → cargo build --release --bin ... → test via script or `shimmy generate` (from built shimmy that uses the airframe dep or workspace path) → vault compare → evidence.
- Console/chat in shimmy + airframe engine should become the daily driver (see local_dev_platform spikes).
- All scripts, docs, steering must support reproducible local runs on this machine (Windows pwsh).

### 6. Artifacts & Auditability (for KIRO + user PR review)
- Use `artifacts/` for traces, smokes, formula diffs.
- Maintain this todo list (update status real-time).
- When ready for review/audit: provide branch, `git diff` or cherry range, test output table (per model), vault query evidence, "before/after" metric, hypothesis statement.
- Update relevant steering (inference-testing, this file) and MASTER_HOTFIX as truth moves.

## Immediate Next After Setup
1. Baseline current state on the primary hotfix using test scripts + vault/frontier_compare (one model/group at a time).
2. Break MASTER groups into falsifiable sub-tasks (TDR root kernel diagnosis for Q4K path, Q6K head_blob fix, fused QKV metadata routing for Phi/GPT2 + similar).
3. Implement minimal changes on focused branches off `fix/v0.2.5-all-fixes`.
4. Re-certify with mechanized tools + script.
5. Land only what moves the metric.

This directive supersedes older handoffs for the duration of stabilization. Revisit after gentek local dev is the daily norm and all listed models PASS the smoke.

---
**Status:** Active. Owner: Developer (Grok) under PR Manager direction. Auditor: KIRO.
