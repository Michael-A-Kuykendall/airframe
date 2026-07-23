> **ENVIRONMENT (read this first):** This repo is developed in **MSYS2 / Git-Bash on Windows** (`bash` shell, case-insensitive FS). Windows CLI tools (`taskkill`, `cmd`, `netsh`, etc.) take single-slash flags like `/f`, `/pid`, `/im`, but **MSYS rewrites a single leading `/` into a path** (garbage like `F:/`, `Invalid argument/option`). **Escape Windows flags with a DOUBLE slash: `//f`, `//pid`, `//im`** — or wrap in `cmd /c "..."`. This applies to every Windows command, not just `taskkill`.

# Agent Instructions — Airframe (current release line: v0.2.10)

Airframe is Shimmy's GPU engine library (crates.io: `airframe`). See the combined
workspace `AGENTS.md` at `C:\Users\micha\repos\airframe-workspace\AGENTS.md` for the
full development workflow, the local `[patch.crates-io]` link, the PPT invariant
gate, and the push/remote policy.

## Release Process

Load the `release` skill (`.opencode/skills/release/SKILL.md`) before cutting a release.
Releases are coordinated with Shimmy via `scripts/release-coordinated.sh` in the
workspace root. One command handles version bumps, commits, tags, crates.io publish,
and GitHub Releases for both repos. Never bump versions or tag manually.
