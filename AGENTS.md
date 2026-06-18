# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

---

## Workspace State (2026-06-18)

**Repos:** `C:\Users\micha\repos\airframe` and `C:\Users\micha\repos\shimmy`
**Shell:** PowerShell 7+ (NOT bash/Cygwin). Use PowerShell syntax everywhere.
**GPU:** NVIDIA GeForce RTX 3060 (4GB VRAM), driver 32.0.15.9649
**Toolchain:** Rust 1.89.0, Cargo 1.89.0

### Public vs Private Split

| Project | Public Remote | Private Remote | Purpose |
|---------|-------------|----------------|---------|
| **airframe** | `public` → `airframe.git` | `private` → `airframe-private.git` | GPU inference library (crates.io) |
| **shimmy** | `origin` → `shimmy.git` | `private` → `shimmy-private.git` | Main inference server (popular OSS project) |
| **console/vision** | — | `shimmy-console` → `shimmy-console.git` | Private products (unreleased, archive) |

**Rules:**
- Push to `private` only. Never push to `public` without explicit approval.
- Console and Vision are PRIVATE products under development in `shimmy-console` remote.
- `shimmy-console` remote is an abandoned spec repo (Sept 2025, 2 commits). Keep for reference only.

### Active Branches
| Repo | Branch | HEAD |
|------|--------|------|
| airframe | `feat/phase4-pingpong-activation` | 85ccfb1 (modified: vault/vault.duckdb) |
| shimmy | `fix/template-apply-raw-prompt` | cc8ee88c (ahead of origin by 1 commit — airframe-e0b fix) |

### Branch Cleanup Status (2026-06-18)
- **Airframe:** 28 merged local branches deleted. 2 kept (`feat/control-plane-release-package`, `feat/vision-multimodal` — still ahead of private remote). 2 stashes dropped.
- **Shimmy:** 3 merged local branches deleted. 2 stashes dropped. Remotes consolidated: `private` now points to `shimmy-private.git`, duplicate `public` and `airframe` (local path) remotes removed.
- **P1 items (airframe-0h5, airframe-e0b):** Both closed 2026-06-18.

### Build & Test

```powershell
# Airframe
cargo check                        # Build check (passes with 1 dead_code warning)
cargo build --release              # Full release build
cargo test                         # CPU-only tests (GPU tests are #[ignore])
cargo test -- --ignored            # GPU-dependent tests (requires GPU + model)
cargo clippy -- -D warnings        # Lint gate

# Shimmy (in C:\Users\micha\repos\shimmy)
cargo check --no-default-features --features fast  # CI-safe build (no GPU)
cargo build --release                               # Full build with airframe GPU
```

### Model Paths & Env Vars

| Variable | Value |
|----------|-------|
| `SHIMMY_MODEL_PATHS` | `D:\shimmy-test-models\gguf_collection` |
| `OLLAMA_MODELS` | `D:\shimmy-test-models\gguf_collection` |
| `SHIMMY_DEV_LICENSE` | `dev-key-michael-2024-shimmy-console` |

Available test models (all under `D:\shimmy-test-models\gguf_collection`):
- `Phi-3.5-mini-instruct.Q4_K_M.gguf` (2.4 GB) — ChatML, good for smoke tests
- `Llama-3.2-1B-Instruct-Q4_K_M.gguf` (808 MB) — Llama3 template
- `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (2 GB) — Llama3 template, current TDR focus
- `Qwen3-0.6B-Q4_K_M.gguf` (397 MB) — Qwen3 QK-norm testing
- `Qwen3-1.7B-Q4_K_M.gguf` (1.28 GB) — Qwen3 TDR
- `Qwen2-1_5b-instruct-q4_k_m.gguf` (986 MB) — Qwen2 TDR
- `Gemma-2-2B-Q4_K_M.gguf` (1.7 GB) — Gemma-2 TDR
- `TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf` (638 MB) — TinyLlama template
- `deepseek-coder-6.7b-instruct.Q4_K_M.gguf` (4 GB) — DeepSeek Coder template
- `starcoder2-3b-Q4_K_M.gguf` (1.8 GB) — Fused QKV arch panic
- `phi-2.Q4_K_M.gguf` (1.8 GB), `qwen2-7b-instruct-q4_k_m.gguf` (4.7 GB), etc.

For quick smoke test (once shimmy is built):
```powershell
cd C:\Users\micha\repos\shimmy
$env:SHIMMY_MODEL_PATHS = "D:\shimmy-test-models\gguf_collection"
cargo run --release -- generate --name "Phi-3.5-mini-instruct" --prompt "Hello" --max-tokens 20
```

### Critical Architecture Facts
- `sh_layer_q4k.wgsl` was **DELETED** on 2026-06-17. Do not recreate it.
- `sh_layer_v1.wgsl` is now the **only** transformer layer shader. It handles Q4_0, Q4_K, Q5_K, Q6_K, F16, F32 via quant_type branch checks.
- `use_q4k_pipeline` conditionals are gone from `inference.rs` and `layer.rs`.
- All `layer_pipeline_q4k_*` fields are gone from `BindlessPipeline`.

### Open Issues (bd ready)

#### airframe repo
 1. **airframe-6ex** [P2] — `[DIAG]`/`[ISF-TDR]` stderr noise. Grep `eprintln!` in `src/runtime/gpu.rs` and `crates/airframe_observe/src/isf.rs`. Gate behind `AIRFRAME_LOG_TDR_POLLS=1` env var.
 2. **airframe-mbc** [P3] — `frontier_compare` layer 2+ NaN (debug path only). Check zero-valued `LayerParams` fields guarding V1 shader early-returns.
 3. **airframe-dna** [P2] — Qwen3-0.6B QK-norm path broken (NaN with V1 shader).
 4. **airframe-pz9** [P2] — Stabilize Qwen3-1.7B-Q4_K_M (TDR).
 5. **airframe-guf** [P2] — Stabilize Llama-3.2-3B-Q4_K_M (TDR).
 6. **airframe-dv0** [P2] — Stabilize Qwen2-1.5B-Q4_K_M (TDR).
 7. **airframe-b41** [P2] — Stabilize Gemma-2-2B-Q4_K_M (TDR).
  8. **airframe-o9e** [P2] — StarCoder2-3B fused QKV arch panic.
 9. **airframe-6jg** [P2] — TDR Transport: Shader dispatch splitting (base_offset push constants for head_blob, matmul_f32).
10. **airframe-mbt** [P2] — TDR Transport: GPU timestamp query pool (replace CPU wall timing).
11. **airframe-eri** [P2] — TDR Transport: Encoder pool design (bounded submit+pipeline without blocking).
12. **airframe-dar** [P2] — TDR Transport: ISF integration spec (fact schema, rules).
13. **airframe-68s** [P2] — TDR Transport: Calibration tooling (standalone per-workgroup timing benchmark).
14. **airframe-q5d** [P2] — TDR Transport: Migration & rollout plan (staged replacement of tdr.rs).
15. **airframe-zuy** [P3] — TDR Transport: Cross-platform policy (Windows/Mac/Linux budgets).

#### Closed this session
- **airframe-0h5** [P1] — committed + pushed (b3edf65)
- **airframe-e0b** [P1] — shimmy generate template wrapping fix committed + pushed (cc8ee88c)
- **airframe-2fq** [P1] — Vault infrastructure cleanup complete: import_seeds.py auto-heal, dedup, idempotent; vault_verify.py written; inference_formulas + formula_comparisons populated
- **airframe-uty** [P2] — TinyLlama Q6_K blob-head verified correct (MAE=0.0 vs F32 matmul). Bug was already fixed by prior V1/batch_count changes. Closed 2026-06-18.

#### Created this session
- **airframe-6jg** [P2] — Shader dispatch splitting (base_offset push constants)
- **airframe-mbt** [P2] — GPU timestamp query pool
- **airframe-eri** [P2] — Encoder pool design
- **airframe-dar** [P2] — ISF integration spec
- **airframe-68s** [P2] — Calibration tooling
- **airframe-q5d** [P2] — Migration & rollout plan
- **airframe-zuy** [P3] — Cross-platform policy
- All 7 created 2026-06-18 as part of TDR Transport Layer architecture breakdown.

#### Vault Status (as of 2026-06-18)
- **22 models** in vault DB, **322 oracle rows**, **0 duplicates**, **26/26 seeds import clean**
- `import_seeds.py` now auto-heals seeds (quant, rms_sum, oracle count) — no more manual fix scripts needed
- `vault_verify.py` built — runs frontier_compare traces against vault oracles, populates `inference_formulas` + `formula_comparisons`
- **132 formula rows** (3 models: TinyLlama Q4_0, Llama 3.2 1B, Qwen3 1.7B) — all FAIL (old buggy traces from June 17)
- All 3 failing traces show GPU Q/K all-zeros from layer 2+ (likely pre-`batch_count:1` fix traces)
- **TinyLlama Q6_K confirmed working:** fresh frontier_compare trace passes vault_verify at layer level (Mean log2-fold 0.0003, GvO_l2 < 0.21). Blob-head matches F32 matmul head perfectly (MAE=0.000000). Q6_K dequant formula identical across CPU Rust, sh_layer_v1.wgsl (layer), and sh_head_blob.wgsl (head).
- **7 models need fresh frontier_compare traces:** Llama 3.2 1B/3B, TinyLlama Q4_0, Qwen3 0.6B/1.7B/8B, DeepSeek Coder (with current code to verify bug status)
- Seed files are gitignored (regeneratable via `vault_seed`); vault DB is tracked, `vault_verify.py` is tracked
- Key models with oracles: TinyLlama q4_0 (23), TinyLlama q6_k (23), Llama-3.2 1B q4_k_m (17), Llama-3.2 3B q4_k_m (29), Qwen3-1.7B q4_k_m (29), Qwen3-8B q4_k_m (37), deepseek-coder q4_k_m (33), deepseek-llm q4_k_m (31), qwen2 family (79 total)
- TODO: Spike on Vault + Saturation Fabric integration

### Shimmy Template System
- `TemplateFamily` enum in `src/templates.rs:6-15` — ChatML, Llama3, OpenChat, TinyLlama, DeepSeekCoder
- `render(system, messages, input)` — hand-coded per variant (no Jinja)
- `spec.template` is a `Option<String>` populated during model registration
- `api.rs` has 6+ copies of `match spec.template.as_deref()` — the Generate fix should follow `api.rs:129-137` pattern
- No `from_spec()` method exists; inline the string-to-TemplateFamily match

### Do Not Break
- `batch_count: 1` in frontier_compare LayerParams (was 0, caused V1 QKV no-op)
- `weights_offset / 4` in rmsnorm params (word index, not byte offset)
- `run_dequant_any_hot` in `dequantize_embeddings` (not `run_dequant_request`)
- `quant_type` derived from metadata in frontier_compare (not hardcoded 0)

### Available Skills (`.opencode/skills/`)
| Skill | What it covers |
|-------|---------------|
| `inference-testing` | One-liner frontier_compare smoke tests, shimmy generate tests, build commands, pass thresholds |
| `vault-usage` | Using vault/vault.duckdb as ground truth for verifying correctness |
| `shimmy-generate` | End-to-end `shimmy generate` test for template wrapping verification |

## Session Handoff Protocol

**Context:** Sessions are NOT resumable. The user launches opencode via Ollama, then switches to a free provider. Each session is stateless — all context must be serialized to disk before sign-off.

### Every session MUST end with:

1. **Update AGENTS.md** — Reflect current branch, HEAD, modified files, open issues, and any new context learned.

2. **Commit and push ALL changes** — Work is NOT complete until `git push` succeeds. Both repos (airframe + shimmy) if touched.

3. **Update beads** — Run `bd sync`, close completed issues, mark in-progress items.

4. **Write explicit sign-off** — The last message of the session should contain a summary of:
   - What was accomplished
   - What remains (with issue IDs)
   - Current git state (branch, last commit, any dirty files)
   - Any decisions or dead ends for the next agent

### Rules
- NEVER assume a previous session's context survives — always re-read AGENTS.md first thing.
- Never say "ready to push when you are" — the agent MUST push before the session ends.
- If a push fails, resolve and retry. Do not end the session with uncommitted work.

## Session Cleanup Results (2026-06-18)

### Git Infrastructure
| Repo | Remotes | Local branches | Stashes | Status |
|------|---------|---------------|---------|--------|
| **airframe** | `public` + `private` (clean) | 19 (down from 29) | 0 (dropped 2) | Modified: AGENTS.md, skills, vault/vault.duckdb |
| **shimmy** | `origin` + `private` + `shimmy-console` | 54 (mostly old pre-v2 branches) | 0 (dropped 2) | Clean |

### Remote Fix (shimmy)
- `private` was pointing to `shimmy-console.git` (abandoned spec repo) → **fixed** to `shimmy-private.git`
- `public` (duplicate of `origin`) and `airframe` (local path) **removed**
- `shimmy-console` kept — worktrees reference it; abandoned Sept 2025, only 2 commits

### Airframe Branches Kept (not pruned)
- `feat/control-plane-release-package` (ahead of private, unpushed)
- `feat/vision-multimodal` (ahead of private, unpushed)
- `agents/product-launch-preparations-v20` (worktree branch)

### Shimmy Branches Left (not pruned)
~50 old branches remain. None are merged into current `fix/template-apply-raw-prompt` or into `origin/main`. Most are pre-v2 issue fix branches. Left in place to avoid data loss — prune only after verifying each one against origin/main history.

### Secrets Scan
All public branches in **both repos** scanned for: `ghp_/gho_/ghu_/ghs_/ghr_`, `sk-` keys, AWS `AKIA`, private keys. **Zero secrets found.** Only hits were the secret-scanning regex patterns in `.github/workflows/secret-hygiene.yml` (expected).

### Build Status
- `airframe` cargo check: **passes** (1 dead_code warning, pre-existing)
- `shimmy` cargo check (fast): **passes** (clean)

### Relevant Files
- `vault/scripts/vault_verify.py` — vault-driven frontier_compare verification: computes formula signatures, compares GPU vs vault oracles, populates inference_formulas + formula_comparisons
- `vault/scripts/import_seeds.py` — auto-heals seeds (quant, rms_sum, oracle count), idempotent upsert, case-insensitive dedup
- `vault/vault.duckdb` — clean state: 22 models, 322 oracles, 132 formula rows (3 models), 3 comparisons (all FAIL — old buggy traces)
- `artifacts/tinyllama_fc.json` etc. — old frontier_compare traces from June 17 (pre-batch_count:1 fix, Q/K all-zeros from layer 2+)
- `docs/internal/tdr-transport-layer-analysis.md` — Architectural analysis of TDR problem and proposed TdrTransport/TdrPipeline layer.
- `docs/internal/tdr-transport-layer-assessment-2026-06-18.md` — External review of the analysis, identifying 7 implementation gaps now tracked as beads.
- `docs/internal/code-export-2026-06-18.md` — Full source dump (2.6MB) of airframe + shimmy for cloud/AI review.

### Current Issues (next session)
- **TDR Transport Layer** — 7 new beads created (airframe-6jg, mbt, eri, dar, 68s, q5d, zuy) covering: shader splitting, GPU timestamp queries, encoder pool, ISF integration, calibration tooling, migration plan, cross-platform policy
- Original TDR beads (pz9, guf, dv0, b41) updated with BLOCKED-BY cross-refs to transport layer
- All 3 vault_verify comparisons FAIL (old traces) — need fresh frontier_compare traces with current code
- 7 models need vault-driven debugging (fresh frontier_compare traces): Llama 3.2 1B/3B, TinyLlama Q4_0, Qwen3 0.6B/1.7B/8B, DeepSeek Coder
- 10 models have metadata only (no oracles) — vault_seed CPU forward pass fails for non-Llama architectures
- Non-Llama models (StarCoder2, Gemma 2, DeepSeek Coder V2 MoE, GPT2) need CPU path fixes for vault_seed

### Full context
See `docs/internal/opencode-handoff-2026-06-18.md` for complete session history and decisions.
