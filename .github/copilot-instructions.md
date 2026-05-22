# AI Agent Primer

Keep this file limited to current operating rules. Do not carry forward historical branch notes, old repro archaeology, or expired session context unless the active task explicitly asks for them.

## Repository Push Policy

`shimmy_integration` has two remotes:

- `origin` points to the public Shimmy repo. Never push there.
- `private` points to the private Shimmy repo. If a push is explicitly requested for `shimmy_integration`, use `git push private <branch>`.
- The parent `airframe` repo is private.

If a push is not explicitly requested, do not push.

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
