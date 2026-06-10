# Shimmy Console — Canonical Location

**Consolidated:** 2026-06-10  
**Branch:** `release/v0.2.2-clean`  
**Status:** Compiles clean, zero warnings, zero errors.

---

## THIS IS THE ONE

`airframe/crates/console/` is the **canonical, authoritative implementation** of shimmy-console.

Do not look elsewhere. Do not start a new implementation. Everything is here.

---

## What's Built

- `src/main.rs` — CLI entrypoint: `shimmy-console chat|analyze|edit|config|tool`
- `src/commands/chat.rs` — Full agentic REPL loop with tool execution
- `src/adapters/` — Four adapters: `LocalInferenceAdapter` (airframe direct, no server), `HttpInferenceAdapter`, `ShimmyServerAdapter`, `WsInferenceAdapter`
- `src/tools/` — 12 tools: file_ops, git, analysis, command, system, docs, image, loader
- `src/discovery/` — Shimmy instance discovery (port scanning + health check)
- `src/config.rs` — Config from env + file (`~/.shimmy/config.toml`)
- `src/session_store.rs`, `src/history.rs` — Session and history persistence
- `src/license/` — License validation scaffolding (dev backdoor active, replace before prod)

---

## Architecture: Single-Command Launch

The goal is `shimmy-console chat` (or `shimmy console` once integrated) with:
1. No second terminal — `LocalInferenceAdapter` calls airframe engine directly inline
2. Model discovery at launch — scans standard paths + configured dirs
3. First-run model chooser if no default configured
4. Theme layer (Phase 2): corporate splash → theme chooser → "arcade" as default theme
5. Config file: `~/.shimmy/config.toml`

```toml
[shimmy]
default_theme = "arcade"
default_model = "tinyllama-1.1b-chat"
model_dirs = [
    "D:/shimmy-test-models/gguf_collection"
]
```

---

## Theme Naming

- Default theme: **`arcade`** (approved 2026-06-10)
- "amiga" was never committed anywhere — no rename needed
- Theme spec format: `~/.shimmy/themes/{name}/theme.toml` + assets
- User themes: drop folder in `~/.shimmy/themes/`, auto-discovered

---

## What's Next (Step 2+)

1. Wire `LocalInferenceAdapter` into `chat.rs` based on config/flag
2. Add model discovery + chooser at launch
3. Build theme layer: splash, chooser, arcade theme
4. Integrate into shimmy CLI as `shimmy console [theme]`

---

## Prior Implementations (Dead — Do Not Resurrect)

| Location | Status |
|---|---|
| `shimmy/console/` (feature/console branch) | Dead skeleton — kept for git history only |
| `shimmy/src/` console feature stubs | Intentionally preserved — integration point for when console goes public |
| `airframe-v2-gpu` branch console code | Identical to current branch — superseded |

---

## Patent Notice

This software implements Fused Semantic Execution (FSE).  
Patent pending by Michael A. Kuykendall. All rights reserved.
