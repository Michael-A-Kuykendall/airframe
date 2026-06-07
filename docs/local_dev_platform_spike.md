# Local Development Platform - Spike Document

**Purpose:** Document the current state and requirements for setting up a local AI-assisted development platform using Shimmy/Airframe stack.

**Generated:** 2026-06-07

---

## 1. Current Stack Overview

### Core Repositories (Public)

| Repo | Local Path | Purpose |
|------|------------|---------|
| **airframe** | `~/repos/airframe` | WebGPU inference engine |
| **shimmy** | `~/repos/shimmy` | Ollama-compatible LLM server |
| **shimmytok** | `~/repos/shimmytok` | Tokenizer |
| **shimmyjinja** | `~/repos/shimmyjinja` | Chat templates |
| **schoolmarm** | `~/repos/schoolmarm` | Grammar-constrained decoding |

### Internal/Not Going Public

| Repo | Local Path | Status |
|------|------------|--------|
| **shimmy-workspace** | `~/repos/shimmy-workspace` | Vision features |
| **shimmy-console** | `~/repos/shimmy-console` | Console UI |
| **libshimmy** | `~/repos/libshimmy` | Private |

---

## 2. What We Have Today

### CLI Tools (in shimmy)

```
shimmy serve     - HTTP server
shimmy list      - List models
shimmy discover  - Auto-discover models  
shimmy probe     - Test model loading
shimmy bench     - Throughput benchmark
shimmy generate  - One-off generation
shimmy gpu-info  - GPU capabilities
```

### Shimmy-Console Libraries (in shimmy-console repo)

- **shimmy-context** - Token budgeter, context building
- **shimmy-session-store** - Session storage

### Current shimmy Features

- OpenAI-compatible API at `/v1/chat/completions`
- Model backends: HuggingFace, llama, MLX
- GPU acceleration via Airframe
- WebSocket support (for streaming)

---

## 3. Target: Local AI Development Platform

### Aspirational Spec (from Claude Local Tool Spec)

Based on the reference spec, we want:

1. **Chat Interface** - Interactive conversation with local AI
2. **Tool Execution** - Run commands, edit files, search code
3. **History Retention** - Persist session context
4. **Local Context** - Project-aware context delivery

### Key Components Needed

| Component | Source | Status |
|-----------|--------|--------|
| Chat UI | shimmy-console | Needs implementation |
| WebSocket Handler | Spec in shimmy-console | Needs implementation |
| Tool Registry | Spec in shimmy-console | Needs implementation |
| License Validation | In shimmy-console | Existing spec |
| Context Builder | shimmy-context lib | Existing (in shimmy-console) |

---

## 4. Gap Analysis

### What's Implemented

- [x] shimmy server (OpenAI-compatible API)
- [x] shimmy-context library (token budgeting)
- [x] shimmy-session-store library
- [x] Feature flag architecture (`--features=console`)
- [x] WebSocket endpoint spec (`/ws/console`)

### What's Missing (Console Implementation)

- [ ] Console library code in shimmy workspace
- [ ] Tool registry implementation (file ops, git, command, analysis)
- [ ] WebSocket handler for real-time streaming
- [ ] CLI commands (`chat`, `edit`, `analyze`)
- [ ] License validation integration

---

## 5. Integration Points

### Architecture

```
User → Chat UI (browser/CLI)
       ↓
    shimmy serve (with console feature)
       ↓
    WebSocket /ws/console
       ↓
    shimmy-context (token budgeting)
       ↓
    airframe (inference)
       ↓
    Response back through WebSocket
```

### shimmy CLI Integration (existing spec)

```rust
// Feature-gated commands
#[cfg(feature = "console")]
Chat { model: Option<String>, message: Option<String> }

#[cfg(feature = "console")]
Edit { file: String, instruction: String, model: Option<String> }

#[cfg(feature = "console")]
Analyze { path: String, model: Option<String> }
```

---

## 6. Implementation Roadmap

### Phase 1: CLI First (Minimal Viable)

1. Enable console feature flag in shimmy
2. Implement basic chat command (one-shot mode)
3. Wire to shimmy HTTP API
4. Add shimmy-context for token budgeting

### Phase 2: Interactive Chat

1. Add WebSocket endpoint `/ws/console`
2. Implement WebSocket handler
3. Add streaming response support

### Phase 3: Tool Execution

1. Implement ToolRegistry trait
2. Add file operation tools (read, write, search)
3. Add git integration tools
4. Add command execution tool

### Phase 4: Polish

1. Session history storage
2. Context optimization
3. License validation (optional)

---

## 7. Key Files to Modify

### shimmy (this repo)

| File | Changes |
|------|---------|
| `Cargo.toml` | Enable `console` feature |
| `src/cli.rs` | Add chat/edit/analyze commands |
| `src/server.rs` | Add WebSocket endpoint |

### shimmy-console (reference)

| File | Purpose |
|------|---------|
| `specs/001-shimmy-console-integration/spec.md` | Full technical spec |
| `shimmy-context/src/lib.rs` | Reuse for token budgeting |
| `console/src/commands/` | Reuse CLI patterns |

---

## 8. Dependencies

- **tokio-tungstenite** - WebSocket
- **crossterm** - Terminal UI
- **shimmy-context** - Token budgeting
- **reqwest** - HTTP client for local API calls

---

## 9. Notes

- shimmy-console repo has the full spec and some implementation
- Vision work is in shimmy-workspace (NOT going public)
- Airframe already supports WebGPU inference
- Use shimmy's existing OpenAI-compatible API as backend

---

## 10. Action Items

- [ ] Verify shimmy serves models correctly
- [ ] Test shimmy-context library
- [ ] Review console spec in shimmy-console
- [ ] Implement Phase 1 (basic chat CLI)
- [ ] Test local AI chat works

---

*End of Spike Document*
---

## Session Notes (2026-06-07)

### What Was Done

1. **Airframe Hygiene**
   - Removed 4MB of artifacts, chat sessions, internal docs
   - Used git-filter-repo to rewrite history
   - Applied clippy fixes (0 warnings)
   - Created backup branch: `release/v0.2.2-filtered-backup`

2. **Repository Mapping**
   - Cloned schoolmarm into ~/repos/schoolmarm
   - Created skill: `~/.kiro/skills/airframe-dev-workflow.md`
   - Documented all repos in stack

3. **CI Status**
   - Branch `release/v0.2.2-clean` pushed to private
   - Waiting for CI to pass before public release

### Secrets Scanning
- ✅ airframe src/: 0 secrets
- ✅ shimmy src/: 0 secrets

### For Next Session

- Run CI on airframe release/v0.2.2-clean
- If CI passes: publish airframe to crates.io
- Update shimmy dependency version
- Review shimmy documentation (EN, zh-CN, zh-TW, wiki)
- Push shimmy to public

### Key Branches

| Branch | Purpose |
|--------|---------|
| `release/v0.2.2-clean` | Airframe release (clippy + hygiene) |
| `feat/starcoder-triage` | Original clippy work (backup) |
| `release/airframe-update` | Shimmy release branch (waiting for airframe) |