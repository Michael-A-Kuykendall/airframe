# Airframe Optimization Checklist

This document tracks the current optimization pass as a sequence of small, testable, revertable changes.

## Baseline Hygiene

- [ ] Resolve the `.beads` repository ID mismatch before using `bd` for work updates or sync.
- [ ] Keep commits scoped to this optimization track only; do not bundle unrelated workspace changes.
- [ ] Capture a repeatable baseline before each change:
  - `cargo check --bin shimmy_server_gpu`
  - `cargo check -p shimmy-console`
  - One manual chat run against the local GPU server
  - Record first-token latency and full-response latency

## Track 1: Session Sliding Window

- [ ] Add explicit server-side chat sessions instead of resetting KV for every request.
- [ ] Store a rolling context window per session with a target ceiling of 2048 active tokens.
- [ ] Keep the existing helical shift / compaction behavior once the live session reaches the ceiling.
- [ ] Send only delta user input from the console after the session is established.
- [ ] Preserve assistant responses back into session state so follow-up turns remain coherent.
- [ ] Add validation for multi-turn consistency across at least 5 consecutive prompts.

## Track 2: Real Token Streaming

- [ ] Replace snapshot polling with true server push streaming.
- [ ] Evaluate transport in this order:
  - HTTP chunked response
  - Server-Sent Events
  - WebSocket fallback only if the simpler options are awkward in this codebase
- [ ] Emit token or chunk deltas directly from the decode loop.
- [ ] Keep snapshot status endpoints only for queue inspection, reconnect, and debugging.
- [ ] Measure first-token latency before and after the transport change.

## Track 3: Low-Risk Decode Loop Optimizations

- [ ] Replace O(n) `recent_tokens.remove(0)` with a ring buffer or `VecDeque`.
- [ ] Stop cloning the full generated string into job state on every token.
- [ ] Gate per-token debug logging behind a verbose flag.
- [ ] Collapse CPU-side logits metric scans into a single pass accumulator if it stays readable.
- [ ] Reuse buffers where possible in the streaming adapter to reduce short-lived string allocation.

## Track 4: Prefill / Ingest Optimizations

- [ ] Batch or deduplicate prompt embedding dequant requests during prefill.
- [ ] Replace repeated KV `increment()` loops with the existing bulk completion API when valid.
- [ ] Review whether static prompt scaffolding can be prefused or cached per session.

## Experimental Protocol

- [ ] Make one optimization change per commit.
- [ ] Run the baseline checks after every change.
- [ ] If latency or correctness regresses, revert that single commit immediately.
- [ ] Keep notes on qualitative effects: streaming smoothness, tool-call stability, and multi-turn memory quality.

## Commit Sequence

- [ ] Commit 0: repo hygiene and checklist only.
- [ ] Commit 1: server-side session sliding window.
- [ ] Commit 2: real token streaming transport.
- [ ] Commit 3: low-risk decode loop cleanup.
- [ ] Commit 4: prefill / ingest optimizations.