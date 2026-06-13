# Card System Architecture

See also: `docs/card-system-fail-fast-experiments.md` for the pre-implementation experiment plan and controller-side forcing options.

## Purpose

This document defines a minimum-complexity architecture for an optional card-processing layer that helps a small local model preserve operative task state without consuming excessive prompt budget.

The design goal is not to create a full memory framework. The goal is to add the smallest useful layer that:

- extracts durable task state early
- preserves constraints and negative space
- reduces transcript replay pressure
- can be turned on and off for direct comparison
- is easy to inspect before and after processing

This document is intended to be implementation-facing and audit-friendly.

## Design Goals

### Primary Goals

1. Improve task continuity for small models across multi-turn work.
2. Reduce prompt waste by replacing bulk transcript replay with a compact state object.
3. Preserve hard constraints, exact inventories, and explicit negative space.
4. Make the feature optional and observable so it can be tested against baseline behavior.
5. Keep the first implementation small enough that it does not create new architecture debt.

### Non-Goals

1. This is not a general long-term memory system.
2. This is not a project-management or issue-graph system.
3. This is not a retrieval framework.
4. This is not a replacement for the base prompt or the local immediate context.
5. This is not intended to preserve full conversational history.

## Core Thesis

The card system should behave like a compact operative state layer between raw conversation input and model action.

Instead of repeatedly feeding the model a growing transcript, the system should:

1. compress the first substantive task prompt into a structured card
2. patch the card as state changes
3. use the card as the primary durable task state
4. keep the feature optional so outcomes can be compared with and without card processing

## Architectural Principles

### 1. Replacement Over Addition

The card must replace transcript bulk, not simply add another layer of prompt text.

### 2. Fixed Shape Over Freeform Summary

The card must use a small closed schema so the model compresses into stable semantic slots rather than producing arbitrary summaries.

### 3. Negative Space Is First-Class

The system must preserve what must not happen, not just what should happen.

### 4. Incremental Updates Over Full Rewrites

The card should be patched when possible and only compacted when drift or size requires it.

### 5. Optional and Observable

The system must be easy to disable, inspect, and benchmark.

## High-Level Architecture

The minimum architecture has five logical components:

1. `Card Controller`
2. `Bootstrap Extractor`
3. `Card Store`
4. `Card Patcher`
5. `Card Renderer`

### 1. Card Controller

The controller decides whether card processing is enabled, whether a task requires a card, and whether the current card should be created, patched, compacted, or bypassed.

Responsibilities:

- read card-processing configuration
- detect first substantive prompt
- decide whether a card should be created
- select between baseline and card-enabled execution
- trigger refresh or compaction when needed

### 2. Bootstrap Extractor

The extractor runs once at task start when card processing is enabled and a task qualifies for card creation.

Responsibilities:

- read the current user task prompt
- extract durable operative state into named spaces
- capture hard constraints, exact sets, and negative space
- record unresolved items explicitly instead of guessing
- produce the initial card

### 3. Card Store

The store holds the current active card and a minimal change history.

Responsibilities:

- persist the current card for the active task
- retain lightweight updates or events if needed
- expose current and prior card snapshots for inspection
- support before/after comparisons during testing

### 4. Card Patcher

The patcher updates the card after meaningful new information arrives.

Responsibilities:

- patch only changed fields when possible
- append tiny events if needed
- compact the card when it becomes stale, contradictory, or oversized
- avoid repeating unchanged content

### 5. Card Renderer

The renderer prepares the active card for model consumption and optionally for human inspection.

Responsibilities:

- render the card into a compact prompt-safe form
- expose human-readable card state for debugging and auditing
- produce before/after views when card processing is enabled

## Named Spaces

The first implementation should use a small set of named spaces. These are the minimum useful state compartments.

### Objective Space

- `goal`
- `success`

### Constraint Space

- `must`
- `must_not`
- `exact`

### Assumption Space

- `assume`
- `unknown`

### Execution Space

- `now`
- `next`
- `blocked`

### Evidence Space

- `facts`
- `done`
- `risks`

These spaces are intentionally small. If a future design needs more, that should be justified by repeated testing rather than added preemptively.

## Canonical Card Shape

The first implementation should be structurally equivalent to:

```json
{
  "goal": "one-sentence objective",
  "success": ["observable completion criteria"],
  "must": ["hard positive constraints"],
  "must_not": ["hard negative constraints or scope limits"],
  "exact": ["inventories or sets that must not drift"],
  "assume": ["assumptions currently relied on"],
  "unknown": ["unresolved questions or missing facts"],
  "facts": ["durable facts earned in-session"],
  "now": "single active step",
  "next": ["next 1 to 3 immediate actions"],
  "blocked": ["current blockers if any"],
  "done": ["completed items worth preserving"],
  "risks": ["ways the task can go wrong"]
}
```

## Execution Lifecycle

### Phase 1: Baseline Intake

The system receives a user prompt.

The controller decides:

1. is card processing enabled?
2. is this a substantive task?
3. does this task merit a card?

If the answer is no, the system uses baseline execution.

### Phase 2: Bootstrap

If card processing is enabled and the prompt qualifies:

1. extract the initial card before task execution
2. lint the card for missing key dimensions
3. store the card
4. render the card for the working prompt

The model should begin task execution only after the initial card exists.

### Phase 3: Active Work

During execution, the model works from:

1. stable policy or system prompt
2. current card
3. immediate local evidence

The full transcript is not the primary working state.

### Phase 4: Update

When new durable information appears:

1. patch changed fields
2. record blockers, done state, or new facts
3. compact if the card exceeds budget or becomes internally messy

### Phase 5: Inspect or Compare

For debugging and evaluation, the system should allow:

1. view current card
2. view prior card snapshot
3. compare baseline vs card-enabled runs
4. compare before vs after card compaction

## First-Turn Bootstrap Contract

This is the most important control rule in the architecture.

The model must be explicitly instructed that, on the first substantive task prompt, it creates the card before normal action begins.

Required behavior:

1. identify the prompt as task-bearing
2. extract durable task state into the card schema
3. preserve `must_not`, `unknown`, and `exact` information explicitly
4. avoid copying the prompt verbatim
5. begin work from the card

Without this explicit bootstrap rule, the model will tend to act first and summarize later, which defeats the purpose.

## Qualification Rules

The controller should create a card only when worthwhile.

### Create a Card When

1. the task is expected to span multiple turns
2. the prompt contains several constraints or prohibitions
3. the task likely requires tools, code changes, or research
4. exact sets or inventories matter
5. continuity across turns matters

### Bypass or Minimize Card Creation When

1. the request is trivial or one-shot
2. there is no durable state worth preserving
3. the user request is simple enough that card overhead would cost more than it saves

## Prompt Budget Policy

The card only makes sense if it consumes less context than it saves.

### Budget Targets

Initial targets for a small-model implementation:

- initial card: `150-400` tokens
- patch update: `20-80` tokens
- next-step list: capped and short
- no repeated rationale unless it changed meaningfully

### Budget Rules

1. the card must remain smaller than the context bulk it replaces
2. unchanged fields should not be re-emitted unnecessarily
3. event history should not be injected wholesale
4. raw tool output should only enter the card when it becomes a durable fact
5. the card must be compacted if it grows beyond budget

## Card Processing Modes

The feature must be easy to turn on and off.

### Required Modes

1. `off`
2. `shadow`
3. `on`

### `off`

No card is generated or consumed.

Use cases:

- baseline behavior
- direct A/B testing
- failure isolation

### `shadow`

The system generates and updates the card, but the model does not use it as working input.

Use cases:

- measuring card quality without affecting task behavior
- debugging extraction and update logic
- comparing what the card would have said versus what the model actually did

### `on`

The card is generated and used as the primary durable task-state layer.

Use cases:

- production evaluation
- outcome testing against baseline
- prompt-budget reduction testing

### Configuration Surface

The first implementation should expose a simple switch such as:

- `card_processing = off|shadow|on`

If a runtime flag or UI control exists, it should map directly to one of these values.

## Inspectability Requirements

The system should be auditable by inspection.

Minimum inspectability features:

1. show active card
2. show previous card snapshot
3. show card creation time and last update time
4. show whether current run is `off`, `shadow`, or `on`
5. show before/after compaction when compaction occurs

This is necessary for both debugging and evaluation.

## Lint Rules

Before a card becomes active, validate:

1. `goal` exists and is specific
2. `must` exists when hard constraints are present
3. `must_not` exists when scope limits or prohibitions are present
4. `unknown` captures unresolved uncertainty instead of silent guessing
5. `exact` captures exact inventories when applicable
6. `now` is actionable
7. card size is within budget

If lint fails, the bootstrap extractor or patcher should revise the card before activation.

## Minimal Persistence Model

To avoid unnecessary debt, the first version should persist only what is needed for active-task continuity and comparison.

Recommended minimal persistence:

1. current card snapshot
2. previous card snapshot
3. optional tiny event list
4. mode flag (`off|shadow|on`)

Do not add a large database or generalized memory service until testing proves it is necessary.

## Failure Modes

### 1. Card Costs More Than It Saves

Symptom:

- prompt usage increases without measurable continuity benefit

Mitigation:

- cap card size
- patch instead of rewrite
- remove low-value fields

### 2. Card Becomes Summary Slop

Symptom:

- fields become verbose prose instead of operative state

Mitigation:

- closed schema
- lint for verbosity
- hard caps on list sizes and field lengths

### 3. Negative Space Gets Lost

Symptom:

- the model remembers what to do but forgets what must not happen

Mitigation:

- mandatory `must_not`
- explicit lint for missing negative space when present in the prompt

### 4. Exact Inventories Drift

Symptom:

- file sets, allowed tools, routes, commands, or output requirements mutate over turns

Mitigation:

- dedicated `exact` field
- explicit bootstrap extraction rule

### 5. Card Bootstrap Is Inconsistent

Symptom:

- sometimes the first-turn card exists, sometimes it does not

Mitigation:

- explicit controller-side first-turn rule
- qualification logic that is simple and deterministic

## Testing Strategy

The architecture must support direct testing against baseline.

### Evaluation Axes

1. task completion quality
2. constraint retention
3. negative-space retention
4. exact-set retention
5. token usage
6. multi-turn continuity
7. inspectability of state

### Required Comparisons

1. `off` vs `on`
2. `off` vs `shadow`
3. first-turn behavior with card bootstrap enabled vs disabled
4. before vs after card compaction

### Suggested Test Cases

1. coding task with multiple constraints and prohibitions
2. research task with exact required outputs
3. multi-turn debugging task with changing facts and blockers
4. small one-shot task to verify bypass behavior
5. transcript growth test to measure whether card mode saves tokens over time

## Implementation Order

The minimum-risk order is:

1. define schema
2. implement `off|shadow|on` control path
3. implement first-turn bootstrap extraction
4. implement lint
5. implement patching of changed fields only
6. add inspection views
7. run baseline and card-enabled test sets

This order keeps the architecture honest and avoids premature complexity.

## Acceptance Criteria

The first implementation is successful if:

1. card processing can be turned `off`, `shadow`, and `on`
2. first-turn bootstrap reliably creates a card for qualifying tasks
3. card-enabled runs retain constraints better than baseline in targeted tests
4. prompt growth pressure is reduced or at least not materially worsened
5. card state is inspectable before and after updates
6. the implementation does not require a heavy persistence or retrieval subsystem

## Open Questions

1. Should `facts` and `done` remain separate in version one, or merge under one evidence field?
2. Should `shadow` mode save full card snapshots on every patch or only on compaction?
3. What is the best threshold for qualifying a prompt as substantive enough to warrant card creation?
4. Should the card be stored as JSON internally and rendered as a terse textual form for the model?
5. How aggressive should compaction be before it risks dropping useful nuance?

## Current Recommendation

Build the first version as a small optional controller layer with:

1. one compact card schema
2. one deterministic first-turn bootstrap rule
3. one simple mode flag: `off|shadow|on`
4. one patch-based update path
5. one inspection surface

Do not add shards, retrieval, project-memory graphs, or generalized long-term memory until this simpler system proves insufficient.