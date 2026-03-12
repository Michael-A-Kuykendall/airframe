# AI Agent Primer

## ⛔ Repository Push Policy — MANDATORY

**shimmy_integration** has TWO remotes. Pushing to the wrong one leaks proprietary code publicly.

- `origin` → `shimmy` (PUBLIC) — **NEVER push here.**
- `private` → `shimmy-private` (PRIVATE) — **All pushes go here.**
- `airframe` parent repo is private — `origin` push is fine.

When pushing shimmy_integration: **`git push private <branch>`** — never `git push origin`.
Only the repo owner decides when code goes public.

---

# The Beads Memory System

## Rationale
We use the **Beads (`bd`) Memory System** instead of standalone Markdown planning files. Standalone Markdown files suffer from context amnesia during long sessions (agents start expanding phases recursively, losing sight of the main goal, or dropping side-tasks because of token exhaustion). Beads acts as a centralized `.beads/beads.db` issue tracker that completely nullifies work disavowal, memory resets, and lost work scenarios. It structures work like an actual operational DAG instead of chaotic unstructured text.

**Why Beads > Markdown Plans:**
- **No Amnesia:** State perfectly persists between sessions. Run `bd ready` when you wake up.
- **Dependency Graphs:** You don't have to parse text like "blocked by X". Beads uses explicit relations (`blocks`, `duplicates`, `children`). You can query `bd ready --json` for purely actionable items.
- **Spontaneous Discovery:** Don't inline "TODOs" in the code when you see unrelated issues (like a broken test in another module or missing type defs). Run `bd create --title "Fix broken auth pipeline tests" --dep discovered-from=bd-current-task`.
- **Session Throwaways:** You can stop at any time once an issue is successfully updated to `closed`. Subsequent agents handle the next item in `bd ready`.

---

## Core Command Cheatsheet

### 1. Daily Developer Loop
When you are initialized or asked "what's next?":
- Run `bd ready` (or `bd ready --json`) to list actionable tasks with zero blockers.
- Pick an issue: `bd update <id> --status in_progress`
- Work on it. When complete: `bd close <id>`
- (Optional) Need to set it aside? `bd defer <id>`

### 2. Information Operations
- `bd show <id>` - Inspect description, relations, bounds.
- `bd status` - View the high-level DB stats.
- `bd orphans` - See things closed via commits but untracked in the DB.
- `bd sync` - Export/Import state natively as JSONL (auto-runs with hooks if configured).

### 3. Work Graph Engineering
- `bd create --title "<issue>" --desc "Description"` - Create an issue.
- `bd dep add <child_id> blocks <parent_id>` - Or just `--parent` when creating.
- Types of relationships: `blocks` (blocks work), `parent/child` (epic composition), `discovered-from` (spontaneous side quests).

### 4. Advanced:
- **Recipes & Flows**: Consider checking formula tools, `bd mol` or `bd cook` if working in higher orchestrations.
- **Audit**: `bd audit record` traces arbitrary agent interactions to the appendix JSONL.

---

## Directives for GitHub Copilot / Agents
1. **Never create a `-plan.md` file.** If a complex plan arises, turn it into `bd create --type epic` and file `bd create --type task --parent <epic_id>` for the phases. 
2. **Handle Discoveries Gracefully:** If you are halfway through `bd-a320f` and realize you need to rewrite a core library that takes time—stop, run `bd create --title "Rewrite lib" --type refactor`, set it to block the main task, switch to it, and move the original task status to `blocked` or `open`. 
3. **Commit often, update Beads always.** When `git push` is performed, run `bd sync` if your commits didn't automatically trigger it.
4. **Use JSON flags in pipelines:** If extracting lists, use `bd list --json` so you do not have to regex-parse CLI text. 

When beginning integration tasks (such as merging Shimmy CLI tools into Airframe), your first priority is translating the top-level Markdown integration plan into a chain of `bd` issues, and closing them out iteratively. 
