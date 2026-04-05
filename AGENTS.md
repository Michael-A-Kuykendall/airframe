# Agent Instructions

## Repository Push Policy

This workspace contains a private `airframe` repo and a `shimmy_integration` submodule whose `origin` remote is public.

- Never push `shimmy_integration` to `origin`.
- If a push to `shimmy_integration` is explicitly requested, use `git push private <branch>`.
- Do not push any repo unless the user explicitly asks for it.

## Work Tracking

- Use `bd` for multi-session task tracking.
- Do not create `-plan.md` files.
- Log discovered side work in `bd` instead of expanding scope inline.

## Scope Control

- Keep the active task narrow.
- Treat user workflow constraints as operating rules.
- Do not preserve stale session detail in instruction files once it stops helping the current mission.

## Terminal Isolation

- Use named terminals.
- Keep long-running processes isolated from quick inspection commands.
- Use separate terminals for benchmarks, builds, git work, and filesystem inspection.
- If a terminal is running a live workload, do not send unrelated commands into it.
- Never use `captureOutput: true` with `terminal-tools_sendCommand`; in this workspace it can kill the invocation and stall the session.

## Release Workflow Lockdown

- For Airframe release validation and Shimmy provider bring-up, use the `.vscode/tasks.json` `U:` tasks and existing validation tasks first.
- Do not create duplicate server or provider terminals on the same port when a task-owned process should be used.
- Use at most one read-only inspection terminal during these flows unless the user explicitly asks for manual terminal orchestration.
- Prefer checked-in scripts over ad hoc polling commands for readiness and result capture.

