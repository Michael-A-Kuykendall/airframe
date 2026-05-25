# Agent Instructions

## Two-Repo Architecture and Deployment Model

```
airframe/                     ← PRIVATE repo (GPU engine, WGSL shaders, inference)
└── shimmy_integration/       ← submodule → shimmy-private repo (local dev inversion)
      Cargo.toml              ← airframe = { path = "../", optional = true }
```

**Shimmy is the product. Airframe is the private engine.**

- `shimmy-private` (`https://github.com/Michael-A-Kuykendall/shimmy-private.git`) is the public-facing CLI/server product.
- `airframe` (`https://github.com/Michael-A-Kuykendall/airframe.git`) is the private GPU engine. It is an OPTIONAL Cargo dependency of shimmy.
- `shimmy_integration/` inside this workspace is a LOCAL DEV INVERSION — a git submodule pointing to shimmy-private so airframe (the parent) can build shimmy against itself during development. This does NOT reflect the production relationship.
- In production and CI: shimmy is the root, airframe is cloned to `../airframe` before the GPU build.

**Feature flags in shimmy's Cargo.toml:**
```toml
default = ["huggingface"]   # crates.io-safe; airframe NOT included by default
airframe = ["dep:airframe"] # optional; requires ../airframe path dep
```
- `cargo build` (default) works for anyone cloning shimmy — airframe is never touched.
- `cargo build --features airframe` fails unless `../airframe` is present (private path dep).
- CI uses `AIRFRAME_ACCESS_TOKEN` to clone private airframe before GPU builds. Source never in public artifacts.

**Cloning shimmy does NOT expose airframe source.** This is by design.

---

## Repository Push Policy

- `airframe` has remote `origin` → `https://github.com/Michael-A-Kuykendall/airframe.git` (PRIVATE). Also has `lambda` remote (A100 dev server).
- `shimmy_integration` has ONE remote: `private` → `https://github.com/Michael-A-Kuykendall/shimmy-private.git`. **NO `origin` remote exists. Do not add one.**
- If a push to `shimmy_integration` is explicitly requested, use `git push private <branch>`.
- Do not push any repo unless the user explicitly asks for it.

## Work Tracking

- Use `bd` for multi-session task tracking.
- Do not create `-plan.md` files.
- Log discovered side work in `bd` instead of expanding scope inline.

## Test Failures

**Zero tolerance. No exceptions.**

`cargo test` must finish with 0 failures before any task is considered done.
There is no such thing as a "pre-existing" failure that can be left alone.
If a test was already broken before your change, you still own it — fix it before moving on.
Do not declare work complete, summarize results, or ask what is next while any test is red.

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

## Cross-Repo Coordination (Shimmy)

- `C:/Users/micha/repos/shimmy` is potentially dirty/recovery-heavy — treat as untrusted unless verified clean.
- `C:/Users/micha/repos/shimmy-private` is the standalone private checkout; `origin/main` @ `961cbf8` is the last pushed state (as of 2026-05-25). `shimmy_integration/` submodule HEAD is `e70ed39` (4 commits ahead — unpushed).
- Do not land release-critical changes in the dirty public checkout when a clean private checkout is available.
- Keep console feature work (`crates/console/`) isolated from launch-critical runtime changes.
- Keep vision work isolated on dedicated branches; do not blend into architecture release branches.
- Use `C:/Users/micha/repos/shimmy-private` for parking internal audit docs and branch decisions.
- `shimmy_integration` has NO `origin` remote — only `private`. Do NOT add an `origin` remote.

## Current Branch State (as of 2026-05-25)

**airframe:**
- `feat/fse-compiled-layers` @ `cd422f8` — HEAD, most advanced, release-ready line
- `airframe-v2-gpu` @ `8a1b1c3` — 15 commits behind; pure fast-forward target
- `master` / `agents/product-launch-preparations-v20` @ `419c0d8` — stale

**shimmy_integration (shimmy-private):**
- `main` @ `e70ed39` — HEAD, 4 commits ahead of `private/main` (unpushed)
- `private/main` @ `961cbf8` — last pushed state

**Pending consolidation steps (execute in order, with user approval):**
1. `git tag backup/airframe-v2-gpu-pre-consolidation-2026-05-25 airframe-v2-gpu` → push tag
2. `git branch -f airframe-v2-gpu feat/fse-compiled-layers` → fast-forward (safe, no history loss)
3. `git push origin airframe-v2-gpu` → push consolidated branch
4. `cd shimmy_integration && git push private main` → push 4 unpushed shimmy commits
5. Verify `cargo test` passes in shimmy-private with `--features airframe`
6. Human-gated: public shimmy merge (requires explicit approval)

