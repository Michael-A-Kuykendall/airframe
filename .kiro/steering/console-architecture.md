# Console — Two Different Products, Same Name

**Corrected:** 2026-06-10  
**Authoritative source:** `shimmy/docs/internal/CONSOLE_WORKSTREAM_2026-06-10.md`

---

## CRITICAL: There are TWO things called "console". Do not confuse them.

### 1. `shimmy console` — THE PRODUCT being shipped (lives in shimmy repo)
A **browser-based themed UI**. `shimmy console [theme]` spawns shimmy serve, opens
the arcade web frontend in the browser, chats over WebSocket to the local airframe engine.
This is the consumer product. It does NOT live in this (airframe) repo.

### 2. `airframe/crates/console/` — a developer terminal tool (lives HERE)
A terminal/CLI agentic console: chat REPL, 12 tools, adapters. This is a SEPARATE
developer utility. It is NOT `shimmy console`. It is NOT the product being shipped.

**DO NOT touch `airframe/crates/console/` for shimmy console work.**  
**DO NOT treat it as the canonical shimmy console.**  
It is a standalone dev tool that happens to share the word "console."

---

## Where shimmy console actually lives

| Piece | Location |
|---|---|
| CLI command `shimmy console [theme]` | `shimmy/src/cli.rs` (to be wired) |
| WebSocket handler `/ws/console` | `shimmy/src/api.rs` (committed) |
| Arcade theme frontend | `C:\Users\micha\repos\arcade` (GitHub: `Michael-A-Kuykendall/arcade`) |
| Embedded server spawn helpers | `shimmy/console/embedded_server.rs` |

---

## This airframe crate (`crates/console`)

It compiles clean (51 tests). Keep it as a standalone developer terminal tool if useful.
It is NOT part of the shimmy console product line. Leave it alone unless explicitly
working on the terminal dev tool itself.

---

## Theme

Default theme: **`arcade`**. The repo `amiga-ai-interface` was renamed to `arcade`.
No instance of "amiga" remains in that codebase.

---

## Patent Notice

This software implements Fused Semantic Execution (FSE).  
Patent pending by Michael A. Kuykendall. All rights reserved.
