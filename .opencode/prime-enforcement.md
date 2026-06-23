# Enforcement Rules — Session Start Checklist

## ⚠️ NEGATIVE STATE — DO NOT DO THESE:

### Tool Usage Violations
- ❌ Using plain `ls`, `find`, `cat`, or `grep` instead of `ls()`, `find()`, `cat()`, `ag()`
- ❌ Not using `bd ready --json` before starting work sessions  
- ❌ Not visualizing dependencies with `bd graph --all` when issues are complex
- ❌ Forgetting to kill agents after completing each issue (leads to disavowal)
- ❌ Ignoring discovered problems instead of filing beads issues immediately
- ❌ Not using `bd prime` or `bd prime --export` for AI context before session restarts
- ❌ Skipping `bd sync` before git push operations
- ❌ Not checking `bd status` to understand database state
- ❌ Forgetting configuration fixes from `bd doctor` warnings

### Tool Mapping Reference
| Plain Command | Function-Style Alternative | What It Replaces |
|---------------|---------------------------|------------------|
| `ls dir/` | `ls()` | eza with git status colors |
| `find . -name "*.rs"` | `find() .rs` | fd (3x faster, ignores .git/) |
| `cat file.rs` | `cat()` file.rs | bat with syntax highlighting + line numbers |
| `grep -r "pattern"` | `ag() -p "pattern" .` | ast-grep (AST-aware Rust pattern matching) |

## ✅ POSITIVE STATE — MUST DO THESE:

### Session Start Workflow
1. **Run `bd ready --json`** — Get definitive list of unblocked work
2. **Run `bd graph --all`** — Visualize dependency graph with execution order (for complex issues)
3. **Run `bd prime --export`** — Inject workflow context before starting work
4. **Run `bd doctor`** — Check database health and fix all warnings

### During Session
5. **File beads issue immediately** when discovering problems (don't wait for end of session)
6. **Check `bd status`** periodically to understand database state

### Before Git Push
7. **Run `bd sync`** before git push operations (MANDATORY per AGENTS.md)
8. **Fix all `bd doctor` warnings** before proceeding with significant work

## Quality Gates — Run Before Committing Changes

```bash
cargo check                           # Build check (must produce zero warnings)
cargo clippy -- -D warnings           # Lint gate (must pass clean)
git status                            # Verify working tree is clean except intended changes
bd sync                               # Sync beads with git before push
```

## Beads Workflow — Daily Checklist

- [ ] `bd ready --json` at session start
- [ ] `bd graph --all` for complex issues (>3 dependencies)
- [ ] File bead issue immediately for discovered problems
- [ ] `bd prime --export` before session restarts
- [ ] `bd status` to understand database state
- [ ] `bd doctor` and fix all warnings
- [ ] `bd sync` before git push (MANDATORY)

## Available Skills

| Skill | Purpose |
|-------|---------|
| `inference-testing` | One-liner frontier_compare smoke tests, shimmy generate tests, build commands, pass thresholds |
| `vault-usage` | Using vault/vault.duckdb as ground truth for verifying correctness |
| `shimmy-generate` | End-to-end `shimmy generate` test for template wrapping verification |

## Available Agents

| Agent | Purpose |
|-------|---------|
| `guardrail` | Reviews tool usage and beads compliance before allowing work to proceed |
| `build` | Default build agent |
| `plan` | Planning mode (edit: deny) |
| `general` | General purpose tasks |
| `explore` | Codebase exploration |

## Available Tools

- `ls()` — List directory with git status colors
- `find()` — Fast file search (ignores .git/)
- `cat()` — Read file with syntax highlighting + line numbers
- `ag()` — AST-aware pattern matching (ast-grep)
- `read()` — Read files from local filesystem
- `write()` — Write files to local filesystem  
- `edit()` — Edit files with proper diff tracking
- `glob()` — Fast file pattern matching
- `task()` — Launch subagents for complex tasks
- `skill()` — Load specialized skills when task matches

## Critical Rules (AGENTS.md)

1. **Work is NOT complete until `git push` succeeds**
2. **NEVER stop before pushing** — leaves work stranded locally
3. **NEVER say "ready to push when you are"** — YOU must push
4. **If push fails, resolve and retry until it succeeds**
5. **Every warning/error/lint violation MUST be fixed immediately**
6. **Never silence, suppress, or disable warnings/errors**

## Session Completion (Landing the Plane)

When ending work session, MUST complete ALL steps:

1. **File issues for remaining work** — Create issues for anything needing follow-up
2. **Run quality gates** — Tests, linters, builds (if code changed)
3. **Update issue status** — Close finished work, update in-progress items
4. **PUSH TO REMOTE** — `git pull --rebase && bd sync && git push`
5. **Clean up** — Clear stashes, prune remote branches
6. **Verify** — All changes committed AND pushed
7. **Hand off** — Provide context for next session
