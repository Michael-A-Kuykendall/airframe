# Agent Instructions
This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash

bd ready          # Find available work

bd show <id>      # View issue details

bd update <id> --status in_progress  # Claim work

bd close <id>     # Complete work

bd sync           # Sync with git

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

**ZERO TOLERANCE POLICY:**

- This is a solo project. **Every warning, error, and lint violation MUST be fixed immediately.**

- Never silence, suppress, or disable warnings/errors. Fix the root cause.

- `cargo check` must produce **zero warnings**.

- `cargo clippy -- -D warnings` must **pass clean**.

- CI (format + clippy + build + test) must pass on every push.

---

## Workspace State (2026-06-19)

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

| airframe | `feat/phase4-pingpong-activation` | 907a6bc (modified: inference.rs, matmul.rs, frontier_compare.rs, gpu.rs, ci.yml, CHANGELOG.md, AGENTS.md, Cargo.toml) |

| shimmy | `fix/template-apply-raw-prompt` | cc8ee88c (ahead of origin by 1 commit — airframe-e0b fix) |

### Branch Cleanup Status (2026-06-19)

- **Airframe:** 28 merged local branches deleted. 2 kept (`feat/control-plane-release-package`, `feat/vision-multimodal` — still ahead of private remote). 2 stashes dropped.

- **Shimmy:** 3 merged local branches deleted. 2 stashes dropped. Remotes consolidated: `private` now points to `shimmy-private.git`, duplicate `public` and `airframe` (local path) remotes removed.

- **P1 items (airframe-0h5, airframe-e0b):** Both closed 2026-06-18.

### Build & Test

```powershell

# Airframe

cargo check                           # Build check (passes with 1 dead_code warning)

cargo build --release                 # Full release build

cargo test                            # CPU-only tests (GPU tests are #[ignore])

cargo test -- --ignored              # GPU-dependent tests (requires GPU + model)

cargo clippy -- -D warnings           # Lint gate
# Shimmy (in C:\Users\micha\repos\shimmy)

cargo check --no-default-features --features fast  # CI-safe build (no GPU)

cargo build --release                              # Full build with airframe GPU

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
 1. **airframe-6ex** [P2] — `[DIAG]`/[ISF-TDR] stderr noise. Grep `eprintln!` in `src/runtime/gpu.rs` and `crates/airframe_observe/src/isf.rs`. Gate behind `AIRFRAME_LOG_TDR_POLLS=1` env var.
 2. **airframe-dna** [P2] — Qwen3-0.6B QK-norm path broken (NaN with V1 shader).
 3. **airframe-pz9** [P2] — Stabilize Qwen3-1.7B-Q4_K_M (TDR). BLOCKED-BY transport layer.
 4. **airframe-guf** [P2] — Stabilize Llama-3.2-3B-Q4_K_M (TDR). BLOCKED-BY transport layer.
 5. **airframe-b41** [P2] — Stabilize Gemma-2-2B-Q4_K_M (TDR). BLOCKED-BY transport layer.
 6. **airframe-6jg** [P2] — DONE: Shader dispatch splitting (base_offset push constants for head_blob). Tiled hot path wired and validated.

7. **airframe-mbt** [P2] — TDR Transport: GPU timestamp query pool (replace CPU wall timing).
8. **airframe-eri** [P2] — TDR Transport: Encoder pool design (bounded submit+pipeline without blocking).
9. **airframe-dar** [P2] — TDR Transport: ISF integration spec (fact schema, rules).
10. **airframe-68s** [P2] — TDR Transport: Calibration tooling. Cache scaffolding done, needs timestamp queries for sweep.


#### Vault Status (as of 2026-06-18)

- **22 models** in vault DB, **322 oracle rows**, **0 duplicates**, **26/26 seeds import clean**

- `import_seeds.py` now auto-heals seeds (quant, rms_sum, oracle count) — no more manual fix scripts needed

- `vault_verify.py` built — runs frontier_compare traces against vault oracles, populates `inference_formulas` + `formula_comparisons`

- **132 formula rows** (3 models: TinyLlama Q4_0, Llama 3.2 1B, Qwen3 1.7B) — all FAIL (old buggy traces from June 17)

- All 3 failing traces show GPU Q/K all-zeros from layer 2+ (likely pre-`batch_count:1` fix traces)

- **TinyLlama Q6_K confirmed working:** fresh frontier_compare trace passes vault_verify at layer level (Mean log2-fold 0.0003, GvO_l2 < 0.21). Blob-head matches F32 matmul head perfectly (MAE=0.000000). Q6_K dequant formula identical across CPU Rust, sh_layer_v1.wgsl (layer), and sh_head_blob.wgsl (head).

- **7 models need fresh frontier_compare traces:** Llama 3.2 1B/3B, TinyLlama Q4_0, Qwen3 0.6B/1.7B, DeepSeek Coder (with current code to verify bug status)

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

### Tooling Upgrade (dzero-cas Phase 1) ✅ COMPLETE

All AST-aware and terminal superpower tools installed & tested in bash environment:

| Tool | Purpose | Path |
|------|---------|------|
| `ast-grep` | AST-aware pattern matching | `/c/Users/micha/.cargo/bin/ast-grep.exe` |
| `fd` | Fast file finder (replaces find) | `/c/Users/micha/scoop/shims/fd` |
| `bat` | Syntax-highlighted cat | `/c/Users/micha/.cargo/bin/bat.exe` |
| `eza` | Git-aware ls replacement | `/c/Users/micha/.cargo/bin/eza.exe` |
| `fzf` | Fuzzy terminal navigation | `/c/ProgramData/chocolatey/bin/fzf` |
| `zoxide` | Smart directory jumping | `/c/Users/micha/.cargo/bin/zoxide.exe` |

**Configured in opencode.json:**
- Line 119: `ast-grep` alias (`ag`) for Rust pattern matching
- Line 124: `terminal-triage` command (`fd .rs \| fzf \| xargs bat`)

**Tested & Working:** All tools verified in bash environment (see `docs/opencode-tooling-test-results.md`).

### Available Skills (`.opencode/skills/`) - Use these for specialized tasks:

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

## Session Hotfix Release (2026-06-19)

### Changes since v0.2.5

- **TDR-safe LM head dispatch** — `sh_head_blob.wgsl` + `HeadBlobParams`: added `base_row` field for tile offset

- **`run_lm_head_blob_tiled()`** in matmul.rs — dispatches head in tiles of `max_safe_wgs` workgroups, each writing correct output region

- **Production hot path** (inference.rs): single `dispatch_workgroups(wg_head_blob, 1, 1)` replaced with tiled loop using `tdr_calibration::ensure_calibrated()`

- **Calibration cache** (`tdr_calibration.rs`): per-(pipeline, n_embd) safe WG limits at `%LOCALAPPDATA%/Airframe/tdr-calibration.json`, conservative 512-WG default

- **Validated**: TinyLlama Q6_K tiled vs unsplit MAE=0.0 PASS; frontier_compare layers all <0.001

- **`.github/workflows/ci.yml`** — format + clippy + build + test (all must pass clean)

### Version

- `Cargo.toml`: `0.2.5` → `0.2.6`

- `CHANGELOG.md`: updated with all changes, deduplicated section ordering

### Dirty files (to commit)

- `src/backend/bindless/pipeline/inference.rs` — tiled hot path, div_ceil fix

- `src/backend/bindless/pipeline/matmul.rs` — run_lm_head_blob_tiled()

- `src/bin/frontier_compare.rs` — --validate-head-tile flag, collapsible_if fix

- `src/runtime/gpu.rs` — removed dead layer_params/norm_params fields

- `.github/workflows/ci.yml` — added format check

- `CHANGELOG.md`, `AGENTS.md`, `Cargo.toml`

### Build Status

- `cargo check`: **passes** (zero warnings)

- `cargo clippy -- -D warnings`: **passes clean**

- `frontier_compare smoke test (TinyLlama Q6_K)`: **PASS** — MAE <0.001 per-layer, head tile MAE=0.0

### Relevant Files

- `src/backend/tdr_calibration.rs` — cache helpers (save/load/clear), ensure_calibrated() API

- `src/backend/bindless/sh_head_blob.wgsl` — `base_row` param, output indexing `base_row + global_id.x`

- `src/backend/bindless/pipeline/mod.rs` — HeadBlobParams with `base_row: u32`

- `src/backend/bindless/pipeline/matmul.rs` — run_lm_head_blob_tiled() utility

- `src/backend/bindless/pipeline/inference.rs` — tiled hot path in run_full_model_with_cache_state

- `src/bin/frontier_compare.rs` — --validate-head-tile flag

- `docs/internal/tdr-transport-layer-analysis.md` — architectural analysis (gitignored)

- `docs/internal/tdr-calibration-strategy-2026-06-18.md` — calibration design refinements (gitignored)

- `docs/internal/tdr-transport-layer-assessment-2026-06-18.md` — external review identifying 7 gap beads (gitignored)

### Open Beads

- **airframe-6jg** [P2] — DONE (shader splitting + hot path)   

- **airframe-mbt** [P2] — GPU timestamp query pool (next step)

- **airframe-68s** [P2] — Calibration sweep (blocked by mbt)

- **airframe-eri** [P2] — Encoder pool design

- **airframe-dar** [P2] — ISF integration spec

- **airframe-q5d** [P2] — Migration & rollout plan

- **airframe-zuy** [P3] — Cross-platform policy

- Original TDR beads (pz9, guf, dv0, b41) — BLOCKED-BY transport layer

### Full context

See `docs/internal/opencode-handoff-2026-06-18.md` for prior session history.
