# Airframe Current Stack Audit

## Current Status

- Treat the current TinyLlama exact-story repro path as functionally correct unless new contrary evidence appears.
- The latest clean full-run comparison against the historical extracted artifact matched story content completely.
- The remaining file-level mismatch collapsed to a single trailing newline in the historical artifact.
- No substantive current blocker is confirmed on that repro path.

## Working Assumption

- Treat the original golden-trace / TinyLlama baseline as correct.
- Treat the main remaining open question as helical-shift and long-run edge-case characterization, not basic repro failure.
- Do not treat older truncation runs as evidence of current mathematical divergence unless reproduced under clean conditions.

## Verified Today

- Current tip matches the historical `7777` short-prefix baseline for the first `128` tokens.
- Verified short-prefix hash: `f82a1ad07e5f74415a3121821e580998eecda4edd30b43efc9b294aa591c7974`
- `b4baaea` also matches that same short-prefix hash.
- Helical-off, `im_end`-off, and session-window-off short checks also matched that same short-prefix hash.
- A clean full current-tip exact rerun completed to `eos` and matched the historical extracted story content.
- Conclusion: current tip and `b4baaea` are both bit-perfect at the front of the run, and the current full repro does not show a substantive content divergence.

## Remaining Work

- Characterize helical-shift behavior under deliberate long-run edge cases.
- Record one canonical end-to-end provider proof through the Shimmy surface.
- Keep the public launch envelope honest at `2048` tokens.

## Airframe Repo Server History

Only four commits touched the current Airframe server file.

1. `82f28e1`
- Introduced `src/bin/shimmy_server_gpu.rs` into this repo.
- This commit imported almost the entire current server surface in one shot.

2. `8b66198`
- Added session-based rolling chat context.

3. `a7d9621`
- Added chunked job-stream transport for queued jobs.

4. `639895c`
- No meaningful functional server change. One-line checkpoint commit.

## What Exists On Tip Right Now

### 1. Transport Layer

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

- Server binds async HTTP listener on `SHIMMY_PORT` or `8080`.
- `POST /` no longer returns direct story output for normal requests.
- `POST /` queues a job and returns JSON with `job_id`.
- `GET /api/repro/queue` returns queue state.
- `GET /api/repro/job-status?job_id=...` returns final result JSON.
- `GET /api/repro/job-stream?job_id=...` returns chunked plain-text stream or completed text.
- Queue depth is `15` jobs.

### 2. Request Surface

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

Current request struct includes:

- `task`
- `prompt`
- `session_id`
- `prompt_mode`
- `max_tokens`
- `min_tokens`
- `ignore_eos`
- `temperature`
- `top_p`
- `repetition_penalty`
- `seed`
- `stream`
- `expose_candidate`

### 3. Prompt Modes

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

- `creative`
  - current default
  - uses the legacy TinyLlama creative wrapper
  - this is the correct mode for the historical story repro path

- `creative-chatml`
  - explicit ChatML creative mode
  - not the default anymore
  - still present as an alternate path

- `raw`
  - prompt passed through with no wrapping

- `developer`
  - ChatML developer wrapper
  - grammar-constrained output path
  - sanitizer / compile-check / fail-closed policy path

### 4. Queue / Worker Model

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

- A single async worker drains queued jobs.
- Job state tracks:
  - `status`
  - `position`
  - `eta_seconds`
  - `partial_text`
  - `result`
  - `error`
- Broadcast channels are used to stream token deltas / chunks to clients.

### 5. Session Windowing

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

- Session support was added in `8b66198`.
- `session_id` causes prior tokens to be prepended into a rolling window.
- Session window size is `2048` tokens.
- Session state is stored server-side in memory.
- If a session is reused, prompt tokens are no longer just the fresh request.

### 6. Decode-Time Stop Paths

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

Current stop reasons include:

- `max_tokens`
- `eos`
- `im_end`
- `grammar_reject`
- `grammar_accept`
- `end_marker`

Important point:

- `im_end` is a real stop path on tip.
- That stop path did not exist in the old plain creative story path.
- It can terminate generation if `<|im_end|>` is sampled and `ignore_eos` is not set.

### 7. Long-Context / Helical Logic

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

- Helical shift check runs inside decode.
- Trigger condition:
  - `current_len >= cache.max_len() - 4`
- Current compaction behavior:
  - `keep_sink = 4`
  - `shift_amt = cache.max_len() / 4`
- Shift is executed through `RopeShiftPipeline`.
- This is one of the main long-run structural suspects.

### 8. Developer-Only Output Controls

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

Developer mode adds all of this:

- grammar mask
- grammar accept / reject stopping
- end marker stopping on `// END_RUST_FILE`
- sanitizer
- optional raw-text exposure
- compile check
- fail-closed policy response

This is probably irrelevant to normal creative story generation unless the wrong `prompt_mode` is being used.

### 9. Eval Side-Path Inside Server

File: [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

- `task = wikitext2` or `lambada` shells out to `shimmy_eval` from inside the worker.
- This is separate from normal creative story inference but adds more branching surface to the server.

## What Changed After Airframe Stood Up

If the question is specifically "what piled on after this repo stood up", the answer is smaller than it seemed.

### Added after initial import

1. Session rolling window
- commit `8b66198`

2. Chunked job stream over queued transport
- commit `a7d9621`

### Already present at initial import

1. Queue architecture
2. Job status / job stream API shape
3. Prompt modes
4. Developer grammar / sanitizer / compile-check path
5. Helical shift logic
6. Eval-task branching inside server
7. Large `InferenceResponse` / `JobState` surface

So the honest summary is:

- later pile-on inside this repo is real, but limited
- most of the current complexity arrived at repo birth in `82f28e1`

## Most Relevant Truncation Suspects

These are the current non-math suspects worth auditing first.

1. Queued transport versus direct-response expectations
- Current tip does not behave like the old direct-streaming server.
- A client that expects old behavior can misread a queued response as a failed or empty run.

2. Helical shift / compaction during long decode
- This is the highest-value long-run structural suspect.

3. `im_end` stop path
- If the generation path samples ChatML terminators, tip can stop on `im_end`.

4. Session contamination
- Reused `session_id` changes the actual prompt token stream.

5. Chunked job-stream path
- Long runs now pass through queue state and chunked transport rather than the old direct SSE path.

## Lower-Priority Suspects

1. Early-token math divergence
- Current evidence argues against this.

2. Developer grammar / sanitizer path for creative runs
- Only relevant if requests accidentally route through `developer` mode.

## Practical Working Conclusion

- The current tip still reproduces the short known-good prefix exactly.
- The live problem is therefore best treated as a long-run structural / transport / window-extension problem.
- The right next audit targets are:
  1. helical shift behavior during long runs
  2. current job transport / stream path
  3. any accidental `im_end` / wrong prompt-mode termination on story jobs

## Suspect Table

Comparison baseline: `b4baaea` on 2026-03-02.

| Feature / behavior | In `b4baaea` | On current tip | Likely to affect long-run precision | Likely to affect truncation / output shape | Notes |
|---|---|---|---|---|---|
| F32 core decode path | Yes | Yes | Low | Low | Short-prefix parity still matches, so early math path looks stable. |
| Legacy creative prompt wrapper | Yes | Yes | Medium | Medium | Current default is back on legacy creative path. |
| ChatML creative mode | No | Yes | Medium | High | Present as `creative-chatml`; wrong routing can change token path and stop behavior. |
| `im_end` stop path | No | Yes | Low | High | New explicit termination path on tip. |
| Helical shift / compaction | Yes | Yes | High | High | Exists in both, but still a prime long-run suspect because it only matters later in decode. |
| Direct POST response model | Yes | No | Low | High | `b4` responds directly; tip queues jobs. Client assumptions can break. |
| Queue / worker architecture | No | Yes | Low | High | Added structural surface between request and result. |
| Chunked job-stream transport | No | Yes | Low | Medium | Added after repo stand-up in `a7d9621`. |
| Session rolling window | No | Yes | Medium | High | Added in `8b66198`; reused session state changes prompt token stream. |
| Developer grammar mask | No | Yes | Medium | High | Only relevant if wrong `prompt_mode` is used. |
| Developer sanitizer / compile gate | No | Yes | Low | Medium | Only relevant on developer path, but can zero or alter output. |
| Expanded response / job state fields | No | Yes | Low | Low | Mostly observability / transport metadata. |
| Eval task branching in server | No | Yes | Low | Low | Separate branch, unlikely for story path unless wrong task is sent. |

## Highest-Value Suspects

If the question is "what changed between known-good repro checkpoint and now that is most credible as the remaining precision/truncation problem," the short list is:

1. Helical shift / compaction in long decode
2. Queue plus job-stream transport replacing direct request/response
3. `im_end` stop path on the creative stack
4. Session rolling window contamination

## Lower-Value Suspects

1. Early F32 decode math
2. Basic sampling path
3. Legacy creative wrapper itself

Those lower-value items are not impossible, but current short-prefix parity makes them poor primary suspects.