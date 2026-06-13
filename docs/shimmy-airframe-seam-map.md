# Shimmy x Airframe Seam Map

This document maps the major seams in Shimmy and Airframe so the integration can proceed deliberately instead of by ad hoc refactor.

## High-Level Position

- Shimmy is the product surface.
- Airframe is the engine/runtime.
- The correct first move is not a rewrite.
- The correct first move is to replace Shimmy's inference seam one layer at a time.

## System Roles

### Shimmy Role

Shimmy already owns:

- CLI product surface
- OpenAI-compatible HTTP surface
- WebSocket and SSE streaming surfaces
- model registry and discovery
- binary packaging and release process
- user-facing brand and installation story

Key files:

- [vendor/shimmy/src/cli.rs](vendor/shimmy/src/cli.rs)
- [vendor/shimmy/src/server.rs](vendor/shimmy/src/server.rs)
- [vendor/shimmy/src/api.rs](vendor/shimmy/src/api.rs)
- [vendor/shimmy/src/openai_compat.rs](vendor/shimmy/src/openai_compat.rs)
- [vendor/shimmy/src/model_registry.rs](vendor/shimmy/src/model_registry.rs)

### Airframe Role

Airframe currently owns:

- GPU server execution path
- bindless pipeline runtime
- request queue and job lifecycle
- token streaming transport
- session rolling context window
- GPU-specific inference implementation

Key files:

- [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)
- [src/backend/bindless/pipeline.rs](src/backend/bindless/pipeline.rs)
- [src/backend/bindless/pipeline_shift.rs](src/backend/bindless/pipeline_shift.rs)
- [src/backend/bindless/kv_cache.rs](src/backend/bindless/kv_cache.rs)
- [crates/console/src/adapters/shimmy_adapter.rs](crates/console/src/adapters/shimmy_adapter.rs)

## Shimmy Seam Inventory

### Product Surfaces

Shimmy entrypoints and protocols:

- CLI commands in [vendor/shimmy/src/cli.rs](vendor/shimmy/src/cli.rs)
- native API routes in [vendor/shimmy/src/server.rs](vendor/shimmy/src/server.rs)
- OpenAI chat completions in [vendor/shimmy/src/openai_compat.rs](vendor/shimmy/src/openai_compat.rs)
- SSE streaming and websocket generation in [vendor/shimmy/src/api.rs](vendor/shimmy/src/api.rs)

### Engine Boundary

Shimmy's integration seam is already formalized:

- `InferenceEngine` in [vendor/shimmy/src/engine/mod.rs](vendor/shimmy/src/engine/mod.rs)
- `LoadedModel` in [vendor/shimmy/src/engine/mod.rs](vendor/shimmy/src/engine/mod.rs)
- backend router in [vendor/shimmy/src/engine/adapter.rs](vendor/shimmy/src/engine/adapter.rs)

This is the main seam that should absorb Airframe.

### Current Llama Coupling

The current llama.cpp path is concentrated rather than spread everywhere:

- feature flags in [vendor/shimmy/Cargo.toml](vendor/shimmy/Cargo.toml)
- backend implementation in [vendor/shimmy/src/engine/llama.rs](vendor/shimmy/src/engine/llama.rs)
- backend selection in [vendor/shimmy/src/engine/adapter.rs](vendor/shimmy/src/engine/adapter.rs)

That is good news. It means a clean historical separation is realistic.

## Airframe Seam Inventory

### External Request Surface

Airframe's currently exposed server seam is the GPU repro server:

- request/response schema in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)
- queue, status, and stream endpoints in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)
- streaming path via `/api/repro/job-stream` in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

### Runtime Core

Airframe's actual compute seam is below the server layer:

- `BindlessPipeline` in [src/backend/bindless/pipeline.rs](src/backend/bindless/pipeline.rs)
- `RopeShiftPipeline` in [src/backend/bindless/pipeline_shift.rs](src/backend/bindless/pipeline_shift.rs)
- `KVCache` in [src/backend/bindless/kv_cache.rs](src/backend/bindless/kv_cache.rs)

### Local Console Layer

The current console stack is useful, but it is not the long-term Shimmy seam:

- backend trait in [crates/console/src/websocket/mod.rs](crates/console/src/websocket/mod.rs)
- GPU adapter in [crates/console/src/adapters/shimmy_adapter.rs](crates/console/src/adapters/shimmy_adapter.rs)
- CLI chat loop in [crates/console/src/commands/chat.rs](crates/console/src/commands/chat.rs)

This is the proving harness seam, not the product seam.

## Direct Seam Matching

### Match 1: Public API Layer

Shimmy side:

- [vendor/shimmy/src/server.rs](vendor/shimmy/src/server.rs)
- [vendor/shimmy/src/api.rs](vendor/shimmy/src/api.rs)
- [vendor/shimmy/src/openai_compat.rs](vendor/shimmy/src/openai_compat.rs)

Airframe side:

- [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

Assessment:

- Do not replace Shimmy's public API layer with Airframe's repro server.
- Use Shimmy's API layer as the permanent public surface.
- Treat Airframe's current server as a temporary proof harness and reference implementation.

### Match 2: Engine Abstraction

Shimmy side:

- `InferenceEngine`
- `LoadedModel`
- backend dispatch in [vendor/shimmy/src/engine/adapter.rs](vendor/shimmy/src/engine/adapter.rs)

Airframe side:

- the runtime currently buried in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)
- bindless runtime in [src/backend/bindless/pipeline.rs](src/backend/bindless/pipeline.rs)

Assessment:

- This is the primary integration seam.
- Airframe needs to become a Shimmy backend implementation, not a replacement server.
- The current Airframe runtime is too binary-centered and should be progressively lifted into a reusable engine module.

### Match 3: Streaming

Shimmy side:

- SSE and websocket token callbacks in [vendor/shimmy/src/api.rs](vendor/shimmy/src/api.rs)
- OpenAI chunk streaming in [vendor/shimmy/src/openai_compat.rs](vendor/shimmy/src/openai_compat.rs)

Airframe side:

- token broadcast + chunked stream in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)

Assessment:

- Shimmy already has the right external streaming abstractions.
- Airframe should emit tokens through Shimmy's existing `LoadedModel::generate(..., on_token)` callback model.
- Airframe's current HTTP streaming transport should not be the final transport inside Shimmy.

### Match 4: Session / Context Handling

Shimmy side:

- currently request-oriented, prompt/template driven through [vendor/shimmy/src/api.rs](vendor/shimmy/src/api.rs)
- no equivalent Airframe-style rolling session window at the engine seam today

Airframe side:

- rolling token window in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs)
- KV and helical memory machinery in [src/backend/bindless/kv_cache.rs](src/backend/bindless/kv_cache.rs) and [src/backend/bindless/pipeline_shift.rs](src/backend/bindless/pipeline_shift.rs)

Assessment:

- Session continuity is a likely Airframe value-add to Shimmy.
- But it should be integrated behind Shimmy's engine/backend seam rather than introduced first at the API layer.

### Match 5: Model and Backend Selection

Shimmy side:

- backend selection logic in [vendor/shimmy/src/engine/adapter.rs](vendor/shimmy/src/engine/adapter.rs)
- packaging flags in [vendor/shimmy/Cargo.toml](vendor/shimmy/Cargo.toml)

Airframe side:

- no formal feature-gated Shimmy-style backend entry yet

Assessment:

- Add an Airframe backend to Shimmy's adapter instead of replacing the adapter.
- Preserve the llama backend during transition.
- Use a feature flag or explicit backend selector for preview integration.

## Recommended Integration Order

### Phase 1: Map and Freeze

Goal:

- no more architectural guessing

Deliverables:

- this seam map
- historical llama baseline in `vendor/shimmy` on `historical-llama-cpp`

### Phase 2: Introduce Airframe Backend Skeleton In Shimmy

Create:

- `vendor/shimmy/src/engine/airframe.rs`

Add:

- a new backend choice in [vendor/shimmy/src/engine/adapter.rs](vendor/shimmy/src/engine/adapter.rs)
- a feature flag in [vendor/shimmy/Cargo.toml](vendor/shimmy/Cargo.toml)

Initial implementation can be pragmatic:

- call into Airframe through a thin adapter
- potentially even via process/HTTP bridge first if that is the shortest proof path
- but keep the integration behind Shimmy's engine trait

### Phase 3: Match Generation Semantics

Adapt Airframe to Shimmy's generation contract:

- input: prompt + `GenOptions`
- output: final string
- streaming: `on_token` callback

This is the exact contract in [vendor/shimmy/src/engine/mod.rs](vendor/shimmy/src/engine/mod.rs).

### Phase 4: Lift Runtime Out Of The Binary

Longer term, move the runtime that is currently embedded in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs) into a library-facing module so it can be used directly by the Shimmy backend without the repro server wrapper.

This is the main cleanup move, but it should follow the first integration proof, not precede it.

## Seam Risk Notes

### Lowest-Risk Seams

- Shimmy API layer unchanged
- Shimmy CLI unchanged
- Shimmy registry/model discovery unchanged
- Shimmy streaming protocols unchanged externally

### Highest-Risk Seams

- Airframe runtime currently lives in a binary-oriented server path
- session and KV behavior are not yet shaped as a clean reusable library API
- console work can distract from the proper Shimmy engine seam

## Working Rule

When in doubt:

- integrate at Shimmy's engine seam
- not at Shimmy's server surface
- not at Airframe's console layer

## Immediate Next Step

The next concrete move should be:

1. create a Shimmy integration branch off `historical-llama-cpp`
2. add an `airframe` backend skeleton in `vendor/shimmy/src/engine/airframe.rs`
3. wire it into Shimmy's `InferenceEngineAdapter`
4. keep the current llama backend intact as the historical fallback