# Airframe Brain Handbook

## Purpose

This repository uses a local MCP memory server named `airframe-brain` for document and source code retrieval.

Airframe is a FP32-first inference core for Llama-family models — a pure Rust physics engine with WebGPU acceleration. This brain indexes all source code, shaders, and documentation for AI-assisted development.

## Current Index Snapshot

Last verified build (March 5, 2026):
- Total documents: 97
- Total chat sessions: 0 (new repo)
- Database size: 1.25 MB

Document breakdown:
- 1 markdown file
- 94 source files (`.rs`, `.wgsl`, `.toml`, `.lock`)
- 2 operational artifacts

Storage locations:
- SQLite source database: `.mcp/airframe-brain.db`
- Chroma persistent index: `.mcp/chroma_db`
- MCP server script: `.mcp/airframe-brain-mcp.py`
- Rebuild script: `.mcp/build_airframe_brain.py`
- Index generator: `.mcp/generate_brain_index.sh`

## What Is Indexed

The ingest process includes five categories:

**1. Markdown documents** — All `.md` files.

**2. Source files** — All `.rs` (Rust), `.wgsl` (WebGPU shaders), `.toml`, and `.lock` files. This is the primary content for a code-heavy repo.

**3. Operational artifacts** — `.vscode/tasks.json`, `.vscode/settings.json`, `.vscode/mcp.json`, and files containing `hetzner`, `vm`, `ssh`, `deploy`, `gpu`, `benchmark`, `parity`, or `conformance`.

**4. Chat-like files** — Files matching `chat`, `conversation`, `session`, `claude`, `grok`, or `copilot`.

**5. VS Code chat sessions** — From local and hetzner-airframe remote workspace storage.

## Key Source Architecture

- `src/core/` — GGUF model loading, tensor types, weight IDs, dequantization
- `src/ops/reference/` — Pure Rust reference ops (matmul, RMSNorm, RoPE, attention, FFN, softmax)
- `src/backend/bindless/` — WebGPU bindless pipeline, WGSL shaders, KV cache
- `src/runtime/` — Engine, KV cache, sampling, multi-token engine
- `src/family/llama.rs` — Llama model family adapter
- `src/validation/` — Conformance testing, evidence collection, slice validation
- `src/conformance/` — Diff tooling, fixture management

## AI Retrieval Workflow

1. Run `search_documents` with technical terms (e.g., "rmsnorm", "kv_cache", "dequant q4_0").
2. Use returned IDs with `get_document` for full source code.
3. Cite implementation from retrieved source before proposing changes.
4. If content appears stale after code changes, run `reindex`.

## MCP Tools

- `search_documents` — semantic search across all 97 documents
- `search_chats` — semantic search across chat sessions
- `get_document` — retrieve full document by ID
- `get_chat` — retrieve full chat session by ID
- `list_recent_documents` — latest documents by modification date
- `get_stats` — index statistics
- `reindex` — force Chroma re-indexing

## Local Operations

Rebuild index:

```powershell
C:\Users\micha\AppData\Local\Programs\Python\Python313\python.exe .mcp\build_airframe_brain.py
```

Manual MCP smoke test:

```powershell
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | C:\Users\micha\AppData\Local\Programs\Python\Python313\python.exe .mcp\airframe-brain-mcp.py
```

## VS Code Integration

Workspace MCP wiring:
- `.vscode/settings.json`
- `.vscode/mcp.json`

If tools are not visible:
1. Confirm both files still point to absolute Python and script paths.
2. Reload the VS Code window.
3. Confirm MCP server initializes with the smoke test above.
4. Re-run rebuild script if content changed heavily.

## Startup Behavior

The MCP server responds to `initialize` in ~4 seconds. Chroma indexing is deferred until the first tool call. For a small DB like this, the first search should be near-instant.

The builder writes `chroma_index_state.json` after each rebuild. If the DB has not changed, the server skips re-indexing.
