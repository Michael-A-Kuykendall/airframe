# AI Agent Primer

Keep this file limited to current operating rules. Do not carry forward historical branch notes, old repro archaeology, or expired session context unless the active task explicitly asks for them.

## Deployment Model (Read This First)

```
airframe/                   ← PRIVATE repo — GPU engine, WGSL shaders, inference
└── shimmy_integration/     ← submodule → shimmy-private (LOCAL DEV INVERSION only)
```

- **Shimmy is the product.** Airframe is the private engine dependency.
- `shimmy_integration/` is a local dev convenience so airframe can build shimmy against itself. It does NOT reflect the production relationship.
- Production/CI: shimmy is the root; `AIRFRAME_ACCESS_TOKEN` clones airframe to `../airframe` before `cargo build --features airframe`.
- `cargo build` (default, no flags) works for any public shimmy clone — airframe not required.
- Cloning shimmy does **not** expose airframe source. The path dep simply fails to resolve if `../airframe` is absent.

## Repository Push Policy

- `airframe` remote `origin` → `https://github.com/Michael-A-Kuykendall/airframe.git` (PRIVATE). Also has `lambda` remote (A100 dev server).
- `shimmy_integration` has **ONE** remote: `private` → `https://github.com/Michael-A-Kuykendall/shimmy-private.git`. **There is NO `origin` remote. Do not add one.**
- If a push to `shimmy_integration` is explicitly requested, use `git push private <branch>`.
- Do not push any repo unless the user explicitly asks for it.

## Test Failures

**Zero tolerance. No exceptions.**

`cargo test` must finish with 0 failures before any task is considered done.
There is no such thing as a "pre-existing" failure that can be left alone.
If a test was already broken before your change, you still own it — fix it before moving on.
Do not declare work complete, summarize results, or ask what is next while any test is red.

## Session Focus

- Prefer current repo state over historical side worktrees.
- Treat release-readiness, cleanup, provider behavior, and deterministic validation as the default focus areas.
- Re-introduce old repro branches or rollback branches only when the active task specifically depends on them.

## Work Tracking

- Use `bd` instead of ad hoc plan markdown when the work needs to persist across sessions.
- Do not create `-plan.md` files.
- If unrelated follow-up work is discovered, log it in `bd` instead of expanding scope inline.

## Terminal Discipline

- Use named terminals with `terminal-tools_sendCommand`.
- Keep long-running processes isolated from inspection commands.
- Use separate terminals for builds, benchmarks, git operations, and file inspection.
- If a command fails because it was sent to the wrong shell, correct the shell choice immediately instead of continuing analysis.
- Never use `captureOutput: true` with `terminal-tools_sendCommand`; in this workspace it can kill the invocation and stall the session.

## Release Workflow Lockdown

- For Airframe and Shimmy bring-up, prefer `run_task` over ad hoc terminals whenever a task exists.
- Treat the `.vscode/tasks.json` `U:` tasks as the canonical entry points for validation and provider smoke flows.
- Do not spin up duplicate server or provider terminals on the same port if a task already exists for that role.
- For this stack, do not create extra named terminals beyond one read-only inspection shell unless the user explicitly asks for manual terminal work.
- Do not hand-roll polling loops when a checked-in script or task already performs the wait and readiness checks.

## Context Hygiene

- Keep instructions short enough to stay relevant.
- Remove stale operational details once they stop serving the active mission.
- Do not add narrative rationale where a short rule is enough.

## Current Launch Separation Rules

- Tuesday launch scope is architecture/runtime path only; console feature work is deferred.
- Keep console work on dedicated branches and out of release-candidate merges.
- Keep vision continuation on dedicated vision branches; merge only curated runtime commits post-launch.
- When both public and private Shimmy checkouts exist, prefer clean private checkout for planning docs and branch parking.
- If public checkout is dirty, do not perform broad rewrite/refactor passes there without explicit user direction.
