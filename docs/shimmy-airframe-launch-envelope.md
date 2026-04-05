# Shimmy x Airframe Launch Envelope

## Purpose

This document freezes the truthful launch envelope for the Airframe-powered Shimmy preview.

Use it as the source of truth for:

- provider metadata
- release notes
- smoke testing
- launch messaging

## Public Promise

- Product: Shimmy
- Engine: Airframe
- Launch model: TinyLlama 1.1B Chat Q4_0 GGUF
- Truthful advertised context length: 2048 tokens

Do not advertise 4096 or 8192 context for this launch.

## Runtime Reality

- Model GGUF metadata reports 2048 context
- Effective ordinary content horizon is 2048 tokens
- Session replay budget is 2048 tokens
- Rolling KV cache allocation is 4096 slots
- Helical shifting allows continued generation beyond the active horizon
- Continued generation is not the same thing as a larger fully visible context window

## Operational Interpretation

- `2048` is the user-facing context number
- `4096` is an internal cache-management number
- `8192` should not appear in public-facing comments or release claims for this preview

## Tonight's Release Checklist

- Provider default registration reports 2048 context
- Registry fallback paths report 2048 when no explicit model context is known
- `/v1/models` output is checked against the launch model
- One end-to-end chat completion succeeds on the Airframe path
- Known limitation is documented: rolling generation exists, but full ordinary-token visibility is still 2048

## Out Of Scope For Tonight

- changing the underlying attention mask behavior
- widening the truthful context claim beyond 2048
- broad backend refactors across non-Airframe engines
- OpenClaw-specific expansion work