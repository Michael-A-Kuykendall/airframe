---
name: tool-enforcement
description: Enforces AGENTS.md tool usage rules: ls() not ls, find() not find, cat() not cat, ag() not grep. Must run bd ready --json first, bd graph for complex issues, bd prime before session restarts.
---

# Tool Enforcement Rules

## NEGATIVE STATE — DO NOT DO THESE:

**❌ NEVER use plain commands:**
- `ls dir/` → use `ls()` instead
- `find . -name "*.rs"` → use `find() .rs` instead  
- `cat file.rs` → use `cat()` file.rs instead
- `grep -r "pattern"` → use `ag() -p "pattern" .` instead

**❌ NEVER skip beads commands:**
- Always run `bd ready --json` before starting work
- Use `bd graph --all` when issues are complex (multiple dependencies)
- File beads issue immediately for discovered problems
- Use `bd prime --export` at session restarts for context injection
- Check `bd status` to understand database state

**❌ NEVER skip quality gates:**
- Run `bd doctor` and fix all warnings before proceeding
- Skip `bd sync` before git push operations (AGENTS.md rule)

## POSITIVE STATE — MUST DO THESE:

**✅ ALWAYS use function-style tools:**
- `ls()` — eza with git status colors
- `find()` — fd (3x faster, ignores .git/)
- `cat()` — bat with syntax highlighting + line numbers
- `ag()` — ast-grep (AST-aware Rust pattern matching)

**✅ ALWAYS run beads workflow:**
1. `bd ready --json` — get unblocked work list
2. `bd graph --all` — visualize dependencies for complex issues
3. File issue immediately when discovering problems
4. `bd prime --export` — inject context before session restarts
5. `bd status` — understand database state

**✅ ALWAYS check beads health:**
- Run `bd doctor` and fix all warnings
- Never skip `bd sync` before git push (AGENTS.md MANDATORY)

## Enforcement Priority

If you see yourself about to use a plain command or skip beads:
1. STOP
2. Read this skill's rules
3. Use the function-style tool instead
4. Run the beads command if required

This is not optional. AGENTS.md enforcement rules are mandatory.
