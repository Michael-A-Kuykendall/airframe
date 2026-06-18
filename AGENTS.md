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

## Workspace State (2026-06-17)

**Repos:** `C:\Users\micha\repos\airframe` and `C:\Users\micha\repos\shimmy`  
**Shell:** Always bash (Cygwin). Paths: `/c/Users/micha/...`. Never PowerShell syntax.  
**Remote:** Push to `private` remote only. Never push to `public` without explicit approval.

### Active Branches
| Repo | Branch | HEAD |
|------|--------|------|
| airframe | `feat/phase4-pingpong-activation` | c7a39b1 |
| shimmy | `fix/template-apply-raw-prompt` | clean |

### Critical Architecture Facts
- `sh_layer_q4k.wgsl` was **DELETED** on 2026-06-17. Do not recreate it.
- `sh_layer_v1.wgsl` is now the **only** transformer layer shader. It handles Q4_0, Q4_K, Q5_K, Q6_K, F16, F32 via quant_type branch checks.
- `use_q4k_pipeline` conditionals are gone from `inference.rs` and `layer.rs`.
- All `layer_pipeline_q4k_*` fields are gone from `BindlessPipeline`.

### Open Issues (in priority order)
1. **shimmy_server_gpu.rs template fix — UNCOMMITTED** (22 lines, `git diff` shows it). Commit: `git add src/bin/shimmy_server_gpu.rs && git commit -m "fix: wire Llama3 model-name patterns into classify_template()"` then push to private.
2. **shimmy generate command doesn't apply chat template** — `src/main.rs` ~line 548, `Command::Generate` passes raw prompt to `loaded.generate()`. Needs `TemplateFamily::from_spec(&spec).render(...)` wrapper.
3. **[DIAG]/[ISF-TDR] stderr noise** — grep and gate/remove eprintln! in `src/runtime/gpu.rs` and `crates/airframe_observe/src/isf.rs`.
4. **frontier_compare layer 2+ NaN** — debug-path only, not production. Check for zero-valued fields in `LayerParams` that guard V1 kernels (similar to `batch_count: 0` bug already fixed).

### Do Not Break
- `batch_count: 1` in frontier_compare LayerParams (was 0, caused V1 QKV no-op)
- `weights_offset / 4` in rmsnorm params (word index, not byte offset)
- `run_dequant_any_hot` in `dequantize_embeddings` (not `run_dequant_request`)
- `quant_type` derived from metadata in frontier_compare (not hardcoded 0)

### Full context
See `docs/internal/opencode-handoff-2026-06-17.md` for complete session history and decisions.
