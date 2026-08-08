> **ENVIRONMENT (read this first):** This repo is developed in **MSYS2 / Git-Bash on Windows** (`bash` shell, case-insensitive FS). Windows CLI tools (`taskkill`, `cmd`, `netsh`, etc.) take single-slash flags like `/f`, `/pid`, `/im`, but **MSYS rewrites a single leading `/` into a path** (garbage like `F:/`, `Invalid argument/option`). **Escape Windows flags with a DOUBLE slash: `//f`, `//pid`, `//im`** — or wrap in `cmd /c "..."`. This applies to every Windows command, not just `taskkill`.

# Agent Instructions — Airframe (current release line: v0.3.0)

Airframe is Shimmy's GPU engine library (crates.io: `airframe`). The combined
workspace `AGENTS.md` at `C:\Users\micha\repos\airframe-workspace\AGENTS.md` is the
canonical process document: it owns session startup, Serena/Openstate usage,
specification spikes, the local `[patch.crates-io]` link, the PPT invariant gate,
and push/remote policy. Read it before navigating or editing this repository.

## Branch model

- **Single live branch: `main`.** There is no `master` branch on any remote.
- All work merges into `main` locally; push main + tag to `origin` (public) and
  `private` (working copy). No cloud PRs, no cloud merges.

## Release Process

Load the `deploy` skill before cutting a release.
Releases are coordinated with Shimmy via `scripts/deploy.sh` in the
workspace root (see workspace AGENTS.md for the full deploy process).
One command handles version bumps, commits, tags, crates.io publish,
and GitHub Releases for both repos. Never bump versions or tag manually.
