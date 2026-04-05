---
description: "Use when working on GPU pipeline, WGSL shader, conformance, inference, or GGUF questions that need indexed repo retrieval."
---

# Airframe Brain

This repo has a local MCP memory server (`airframe-brain`) indexing Rust source, WGSL shaders, and docs.

## When to Use

Before answering GPU pipeline, WGSL shader, conformance, inference, or GGUF questions:
1. Call `search_documents` with specific terms (e.g., "rmsnorm", "kv_cache", "dequant q4_0").
2. Use `get_document` by ID for full source code.
3. Source files (.rs, .wgsl) are indexed — not just markdown.

## Tools

`search_documents` `search_chats` `get_document` `get_chat` `get_stats` `list_recent_documents` `reindex`

## Offline Index

For a file-by-file card catalog, read `.mcp/brain-index/brain-index-00-overview.instructions.md`.
That directory contains 14 reference files with document listings, topics, and timelines.
Do NOT read them all — scan the overview first, then read specific sections as needed.

## Reference

See `AIRFRAME_BRAIN_HANDBOOK.md` for full setup, rebuild commands, and troubleshooting.
