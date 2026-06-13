# Card System Findings

See also: `docs/card-system-architecture.md` for the implementation-facing architecture derived from these findings.

## Current Goal

Design a minimal assistive card system for small local models that reduces token waste, preserves operative task state across turns, and improves rehydration after prompt growth or compaction.

This is not a full external work tracker. The target is a compact working state for the current task.

## What Beads Is Good At

- Persistent work memory across sessions
- Structured dependency tracking
- Explicit ready versus blocked work queries
- Append-only audit trail of actions and decisions
- Re-priming agents after context compaction
- Multi-agent coordination and lifecycle reporting

## What To Borrow From Beads

### 1. Prime / Re-prime

Beads has an explicit `prime` concept for restoring essential workflow context after compaction.

Transferable idea:

- Generate a compact card early
- Re-inject the card instead of the whole transcript
- Refresh the card when the conversation drifts or token pressure rises

### 2. Named State Dimensions

Beads treats state as explicit dimensions rather than one giant status blob.

Transferable idea:

- Keep the card split into small stable fields such as `goal`, `constraints`, `known_facts`, `open_questions`, `current_step`, `risks`, and `done`
- Update only the dimensions that changed when possible

### 3. Append-only Event Trail

Beads records operational changes as events and exposes current state as a fast lookup layer.

Transferable idea:

- Keep a small event log behind the current card
- Use the card as the active short form
- Rebuild or refresh the card from the event log when needed

### 4. Ready / Blocked Distinction

Beads cleanly separates actionable work from blocked work.

Transferable idea:

- The card should always say what can be done now
- The card should also say what is blocked, and why

## What Not To Borrow Yet

- Full issue graph and dependency engine
- Multi-agent orchestration model
- Database-first persistence
- Workflow ceremony that requires maintaining an issue tracker for every prompt

These are useful at project-memory scale, but they are the wrong starting point for a small-model prompt-state aid.

## Current Working Model

### Card

The card is the compact operative state for the current task.

Suggested fields:

```json
{
  "goal": "one-sentence objective",
  "success": ["observable completion criteria"],
  "constraints": ["hard limits, truths, prohibitions"],
  "known_facts": ["durable facts needed for this task"],
  "open_questions": ["missing information or assumptions to verify"],
  "plan": ["next 3 to 5 actions only"],
  "current_step": "single active step",
  "risks": ["ways the current approach can fail"],
  "done": ["important completed items"]
}
```

### Event Log

Keep a tiny append-only stream of changes behind the card.

Examples:

- `fact_added`
- `constraint_changed`
- `step_completed`
- `blocked_on_user`
- `assumption_made`
- `risk_discovered`

### Rehydration Loop

Minimal loop:

1. Receive prompt
2. Generate card
3. Work from `card + current local context`
4. Emit small events as facts or state change
5. Refresh the card when enough drift accumulates

## Current Thesis

The smallest useful memory aid is probably:

1. A compact card
2. A tiny event log
3. A re-prime step

This should be enough to improve a small development helper without importing a full external project-memory system.

## Open Questions

- How much of the card should be symbolic versus plain natural language?
- Can a small fixed symbol set improve compression and rehydration quality?
- What is the minimum field set that still preserves task fidelity?
- When should the system refresh the card automatically versus preserving it as-is?
- How should negative space be represented so the card preserves what must not happen, not just what should happen?

## Next Input To Fold In

The Sorcery repository may contain useful ideas for:

- compact symbolic expression of session logic
- hydration and dehydration of complex task state
- explicit representation of constraints and negative space
- block composition that remains easy to re-expand

Those findings should be merged into this note and then converted into a concrete card-system action list.

## Sorcery Findings

### What Sorcery Proved

Sorcery validated a strong version of the same core idea:

- high-context reasoning can be compressed into a smaller transmissible artifact
- low-context execution can work well if the compressed artifact is sealed tightly enough
- the biggest failures come from underspecified casting, not from invocation defects

This is directly relevant to a card system for small models.

### The Most Transferable Sorcery Ideas

#### 1. Closed Grammar Beats Loose Summary

Sorcery's strongest practical result is not the branding or the spellbook workflow. It is the proof that a very small closed grammar can preserve a surprising amount of architectural state.

The important lesson for the card system is:

- do not let the card become a freeform blob
- keep a fixed, small set of semantic slots
- make the model compress into those slots on purpose

#### 2. Negative Space Must Be Explicit

Sorcery repeatedly found that what was forbidden mattered as much as what was required.

Transferable rule for the card:

- the card must explicitly preserve `must_not` or equivalent negative-space fields
- ambiguity should not live only in absence; it should be named

This is one of the highest-value ideas to steal.

#### 3. Hydration and Dehydration Need Stable Shape

Sorcery's dehydration/rehydration cycle works because the compressed artifact stays structurally regular.

Transferable rule for the card:

- the card should dehydrate a large prompt into a regular structure
- rehydration should expand from that structure, not from improvised recollection

In other words, the card should act like a compact control object, not like a paragraph summary.

#### 4. Open Questions Must Block or Degrade Confidence

Sorcery treats unresolved `?` items as slice gates.

Transferable rule for the card:

- open questions must not be silently dropped
- unresolved questions should either block action or reduce certainty for downstream steps

#### 5. Exact Sets Need Explicit Exactness

Sorcery found that inventory drift happened when sets were described loosely.

Transferable rule for the card:

- if a list is exact, mark it exact
- if a set is only illustrative, mark it approximate

This matters for:

- required outputs
- allowed tools
- known constraints
- relevant files or modules
- completion criteria

#### 6. Compression Quality Depends on Casting Discipline

Sorcery's meta-analysis kept landing on the same point: the notation was sufficient, but the casting discipline was not.

Transferable rule for the card:

- the quality bottleneck is the first-pass compression step
- card generation needs a deliberate review pass or lint, not just a one-shot summary

### Sorcery Findings That Should Not Be Copied Directly

- Do not import the whole spellbook workflow into normal agent usage
- Do not require per-step formal verification for ordinary prompt work
- Do not force users to learn a bespoke notation before the system is useful
- Do not confuse architectural compression with full application capture

The card system should remain lighter than Sorcery.

## Implications For A Small-Model Card

The card should not be a prose summary. It should be a compressed state object with a small closed grammar.

### Recommended Card Dimensions

Current best candidate field set:

- `goal`
- `success`
- `must`
- `must_not`
- `assumptions`
- `dependencies`
- `open_questions`
- `current_step`
- `next_steps`
- `done`
- `risks`

This is the smallest set that captures:

- positive intent
- negative space
- uncertainty
- exact versus approximate lists
- action continuity

### Recommended Compression Rule

When turning a prompt or transcript into a card, the system should ask:

1. What is the actual objective?
2. What must be true?
3. What must not happen?
4. What assumptions are being relied on?
5. What depends on what?
6. What is still unresolved?
7. What exact sets must not drift?
8. What is the next action?

That is the Sorcery-style casting pass, but adapted to a lightweight agent card.

### Recommended Rehydration Rule

Action should be driven from:

1. stable policy or system context
2. the current card
3. immediate local evidence

Not from the whole conversation transcript by default.

## Minimal Card Grammar

The simplest useful card grammar likely needs only a handful of operators or slots. It does not need Sorcery's full symbolic syntax, but it should preserve the same distinctions.

Possible minimal card grammar:

- `goal:` what this work is trying to achieve
- `must:` hard positive constraints
- `must_not:` hard negative constraints
- `assume:` runtime or environment assumptions
- `depends_on:` explicit prerequisites or required objects
- `unknown:` unresolved questions or missing facts
- `exact:` sets that must not drift
- `now:` current action
- `next:` immediate queued actions
- `done:` completed state worth preserving

This is plain enough to be readable and rigid enough to compress consistently.

## Prompt Budget Discipline

The card only helps if it is cheaper than the context it replaces.

That means the system needs hard budget rules.

### Budget Rules

1. The card must be shorter than the prompt material it is standing in for.
2. The card must not be regenerated on every turn unless drift requires it.
3. Only the changed fields should be updated when possible.
4. The event log should not be injected wholesale; it should be compacted back into the card.
5. The card should carry only durable operative state, not full conversational detail.

### Practical Budget Target

For a small local model, the first version should target something like:

- card: 150 to 400 tokens
- incremental update: 20 to 80 tokens
- refresh trigger: after meaningful state drift, not after every reply

If the card grows past its budget, the system should compact it instead of appending more prose.

### What Must Never Live In The Card

- long examples unless the example itself is the task
- repeated rationale already captured by a rule
- verbose history of every turn
- raw tool output unless it becomes a durable fact
- duplicate constraints stated in multiple places

The card must preserve working shape, not transcript bulk.

## Named Spaces

Named spaces are important because they let the model separate kinds of state instead of mixing everything into one summary blob.

The first implementation does not need many spaces. It just needs the right ones.

### Recommended Named Spaces

#### 1. Objective Space

What the task is trying to accomplish.

Fields:

- `goal`
- `success`

#### 2. Constraint Space

What must hold and what must not happen.

Fields:

- `must`
- `must_not`
- `exact`

#### 3. Assumption Space

What the current plan is relying on but has not fully proven.

Fields:

- `assume`
- `unknown`

#### 4. Execution Space

What is being done right now and what comes next.

Fields:

- `now`
- `next`
- `blocked`

#### 5. Evidence Space

Facts earned from the actual session that should survive prompt loss.

Fields:

- `done`
- `facts`
- `risks`

These spaces are enough for a first version. They are also legible to a human reviewer.

### Why Named Spaces Matter

Without named spaces, small models tend to do three bad things:

1. blend facts, guesses, and goals together
2. lose negative space because it is not explicitly typed
3. keep re-explaining instead of updating state

Named spaces reduce all three failure modes.

## First-Turn Card Bootstrap

This is the part that has to be explicit. The model should not be expected to invent the card workflow on its own.

The system prompt or controller has to tell it what to do on the first substantive user prompt.

### Required First-Turn Rule

On the first task-bearing prompt, before normal task execution, the model must:

1. extract the operative state into the card
2. check the card for missing negative space or unresolved unknowns
3. begin work using the card as the operative state object

If the user prompt is too small to justify a card, the system may skip card creation. But that threshold should be explicit.

### Suggested Bootstrap Policy

Create a card when any of the following are true:

- the prompt contains multiple constraints
- the task is expected to span multiple turns
- the task involves code changes, research, or tool use
- the task includes explicit do and do-not rules
- the task introduces exact inventories or important assumptions

Skip or minimize the card when:

- the request is one-shot and trivial
- there is no durable state worth carrying forward

### Suggested First-Turn Flow

1. Receive user request
2. Run a short extraction pass into named spaces
3. Emit or store compact card
4. Lint for missing `must_not`, `unknown`, or `exact` information
5. Start task work from the card

This avoids the common failure mode where the model starts acting first and only later tries to summarize what it is doing.

### Suggested Bootstrap Prompt Shape

The system-side instruction should be something close to:

"When the first substantive task prompt arrives, create a compact task card before acting. Fill only the defined named spaces. Preserve hard constraints, explicit negative space, assumptions, exact sets, current action, and next actions. Do not copy the full prompt. Compress it into durable operative state. If key uncertainty remains, record it under unknown instead of guessing."

That instruction is probably more important than the exact card schema.

## Update Strategy

The model should not rewrite the full card on every turn.

Preferred order:

1. patch changed fields
2. append a tiny event if needed
3. compact only when the card becomes stale, contradictory, or oversized

This is how the system avoids spending prompt budget on its own bookkeeping.

## Lint Rules For Reliability

Before the card is accepted, check:

1. Is there a clear `goal`?
2. Is there at least one `must` when constraints exist?
3. Is there at least one `must_not` when prohibitions or scope limits exist?
4. Are unresolved questions recorded under `unknown` instead of being silently inferred?
5. Are exact inventories marked under `exact` when the user specified a precise set?
6. Is `now` actionable rather than vague?
7. Is the card within token budget?

If one of these fails, the card should be revised before it becomes the primary working state.

## Actionable Next Steps

1. Freeze a first-pass card schema with a small closed field set.
2. Define a card-generation prompt that explicitly extracts negative space, assumptions, and exact sets.
3. Define a card-refresh rule based on drift or event count.
4. Add a tiny event log so card changes are incremental instead of full rewrites.
5. Add a lint pass for missing fields such as unresolved unknowns, absent negative space, or vague exact sets.
6. Prototype the card on real local-dev tasks with a small model and compare token usage and task coherence against baseline.
7. Keep all of this independent of a larger shard or project-memory system until the card proves it is insufficient.

## Current Conclusion

The best hybrid so far is:

- Beads for the idea of re-prime, state dimensions, ready/blocked, and append-only events
- Sorcery for the idea of a small closed grammar, explicit negative space, exact-set discipline, and hydration/dehydration

That combination points to a simple card system, not a giant framework.

The likely winning move is a card that is:

- compact
- structured
- negative-space aware
- exact-set aware
- refreshable from small events
- readable by humans and small models