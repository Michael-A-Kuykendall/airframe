# Agent Instructions

## ⛔ Repository Push Policy — READ FIRST

This workspace contains two Shimmy-related remotes. **Getting this wrong leaks proprietary code.**

| Repo | Visibility | Remote name (in shimmy_integration submodule) | Push? |
|---|---|---|---|
| `Michael-A-Kuykendall/shimmy` | **PUBLIC** | `origin` | **NEVER push here. NEVER.** |
| `Michael-A-Kuykendall/shimmy-private` | **PRIVATE** | `private` | ✅ All shimmy pushes go here |
| `Michael-A-Kuykendall/airframe` | **PRIVATE** | `origin` (parent repo) | ✅ OK to push |

**Rules:**
- When pushing `shimmy_integration` changes: `git push private <branch>` — NEVER `git push origin`.
- The `origin` remote in shimmy_integration points to the **public** shimmy repo. Do NOT push to it.
- Only the repo owner decides when code goes public. Agents do NOT make that call.
- If you are uncertain which remote to use, run `git remote -v` and **stop to confirm** before pushing.
- Violations of this policy expose proprietary Airframe integration code to the public internet.

---

This project uses **bd** (beads) for issue tracking. We use it strictly over conventional `-plan.md` Markdown files to preserve working memory across long horizons.

## Initialization / Awakening
1. Run `bd ready` (or `bd ready --json`) to list your immediate unwrapped stack context.
2. Claim an item with `bd update <id> --status in_progress`.
3. Read its details via `bd show <id>`.

## Discovery Protocol
If you identify "broken windows", technical debt, or necessary architecture pivots during a session:
**DO NOT** fix them inline if it clutters the current task context.
**DO NOT** write a `TODO: fix earlier` in the current PR.
**DO** run `bd create --title "<issue>" --dep discovered-from=<current_id>` to log the work permanently.

## Quick Reference

```bash
bd ready              # Find available work with 0 blockers
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd dep add <child> blocks <parent>    # Modify dependency graph
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY (respect push policy above!):
   ```bash
   # Parent airframe repo — origin is private, safe to push
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"

   # shimmy_integration submodule — MUST use 'private' remote
   cd shimmy_integration
   git push private <branch>   # NEVER 'git push origin'
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

