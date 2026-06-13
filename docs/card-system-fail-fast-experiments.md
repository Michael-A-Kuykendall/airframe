# Card System Fail-Fast Experiments

## Purpose

This document exists to answer the practical question before deeper architecture work continues:

How do we force first-turn prompt-to-card behavior in a way that materially changes model behavior, without disturbing the current 1B model weights unless later testing proves that retraining is necessary?

The emphasis here is fail-fast experimentation using existing exposed control surfaces.

## Current Reality In The Code

The current runtime already has a real request lifecycle choke point in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs#L895).

Inside `process_inference_job`, the server already:

1. reads the raw user prompt
2. chooses prompt templating mode
3. builds the effective prompt text
4. tokenizes it
5. optionally folds in prior session state from `SessionState.token_window`
6. runs prefill and decode

Relevant current surfaces:

- [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs#L220): current session state is only a rolling token window
- [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs#L932): prompt text is constructed before tokenization
- [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs#L959): prior session tokens are concatenated before the current prompt
- [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs#L1399): session state is updated after response generation
- [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs#L367): FSE / CrewChief is active, but currently as a lower-level enforcement hook

This means the system already has a place where an external controller can force prompt shaping and session substitution before inference begins.

## Clarifying "Controller Responsibility"

When I say controller responsibility, I mean:

The model should not be trusted to voluntarily remember to create and obey a card.

Instead, some non-optional runtime layer must decide:

1. whether a card is required
2. how the first-turn card is created
3. what prompt material the model actually receives afterward
4. what state gets persisted between turns

That responsibility should live outside the base model weights whenever possible.

This is important because there are three very different ways to influence behavior:

1. **Instruction-only**: ask the model nicely to make a card
2. **Controller-enforced**: external runtime makes card creation and reuse part of the request pipeline
3. **Weight-level behavior change**: train or fine-tune the model so carding becomes a learned default

The fail-fast goal is to determine whether option 2 is sufficient before touching option 3.

## Decision Framework

We should test candidate mechanisms in order of least weight disturbance and least tech debt.

### Desired Priority Order

1. runtime/controller enforcement
2. prompt-template reinforcement
3. lightweight sidecar or planner model
4. fine-tune or adapter training on top of the current 1B model

Do not start with training unless the runtime/controller path fails.

## Candidate Mechanisms

### Experiment A: Instruction-Only Prompt Contract

#### Mechanism

Use the existing prompt templating layer to inject a strong instruction such as:

"Before solving the task, compress the task into a card and use that card as your operative state."

#### How It Would Work

- implemented inside the current prompt templating logic
- no second pass
- no structural change to session storage

#### Why It Is Attractive

- almost no engineering work
- fastest to test

#### Why It Is Weak

- cannot force compliance
- card may be verbose or skipped
- model may still carry prompt bulk internally instead of obeying the card discipline
- no guarantee that later turns use the card rather than transcript residue

#### Expected Result

Useful as a baseline, but unlikely to be sufficient.

### Experiment B: Single-Pass Visible Card Prefix

#### Mechanism

Externally render a card-shaped prefix into the prompt before the actual task body, so the model sees a fixed card structure first.

Example pattern:

- controller creates a card shell or compressed card text
- prompt sent to model becomes `CARD + TASK`

#### How It Would Work

- card created outside the model or by deterministic extraction logic
- no second inference pass required
- current prompt-mode templating can be extended to prepend a card block

#### Why It Is Attractive

- still low complexity
- easy to inspect
- controller can force the shape of the state object

#### Why It Is Limited

- if the card is not genuinely derived well, it becomes decorative prompt clutter
- model may still attend heavily to the trailing raw task body
- session window may still retain transcript-style content rather than card-first state

#### Expected Result

Worth testing, especially as a fast comparison against instruction-only behavior.

### Experiment C: Two-Pass Controller-Enforced Card Bootstrap

#### Mechanism

On the first substantive prompt:

1. run a card-generation pass
2. capture structured card output
3. run the actual task using the card as the primary state object
4. persist card-centered session state rather than raw prior prompt bulk

This can use the same model twice or a separate smaller sidecar planner.

#### How It Would Work

- the controller intercepts the first task prompt before normal execution
- pass 1 asks for structured card extraction only
- pass 2 performs the actual task with `card + local evidence`
- later turns patch or compact the card instead of replaying bulk transcript

#### Why It Is Strong

- actually forces first-turn card creation
- creates a crisp before/after artifact for inspection
- lets the controller replace session memory with card memory
- changes behavior at the root without requiring weight changes

#### Why It Costs More

- requires an extra inference pass on bootstrap turns
- requires card validation and storage
- introduces a controller layer that must be implemented carefully

#### Expected Result

This is the most promising mechanism if the goal is reliable behavior change without retraining.

### Experiment D: Session-State Substitution Only

#### Mechanism

Do not force card creation at first prompt. Instead, after a turn completes, replace stored session state with a card-shaped state object for the next turn.

#### How It Would Work

- current `SessionState.token_window` logic is replaced or augmented
- later requests consume card-derived state instead of the whole rolling token window

#### Why It Is Useful

- attacks context bloat directly
- touches a real existing surface in the server

#### Why It Is Not Enough Alone

- does not guarantee the first turn was card-shaped
- still allows the initial response to happen without card discipline

#### Expected Result

Good as a follow-on experiment, but not the primary answer to first-turn forcing.

### Experiment E: Adapter / Fine-Tune for Card Reflex

#### Mechanism

Train a lightweight adapter or fine-tune so the model naturally emits and obeys card structure on task-bearing prompts.

#### Why It Might Help

- reduces reliance on explicit controller instructions
- can make card behavior feel more native

#### Why It Should Come Later

- disturbs the current model behavior surface
- makes it harder to isolate whether the runtime mechanism was already enough
- adds real maintenance cost and training debt

#### Expected Result

Only worth considering if controller-enforced experiments prove insufficient.

## Recommended Fail-Fast Order

The best order is:

1. `A` instruction-only baseline
2. `B` visible card prefix
3. `C` two-pass controller-enforced bootstrap
4. `D` session-state substitution with card persistence
5. `E` training only if the controller path still fails

But in practical terms, `C` is the real candidate to beat.

## What We Are Really Testing

The critical question is not whether the model can output a card.

The critical questions are:

1. can the runtime force the model to start from a card on first turn?
2. can the runtime keep subsequent turns card-centered rather than transcript-centered?
3. does this improve constraint retention and reduce prompt waste?

Any experiment that does not answer those three questions is probably noise.

## Minimal Experimental Surface In The Existing Server

The existing server is already sufficient for a first fail-fast prototype.

### Minimal Runtime Changes Needed

1. add a `card_processing_mode` request or env flag
2. add a first-turn bootstrap branch before prompt tokenization
3. add a card store to `SessionState`
4. optionally replace `token_window` persistence with `card + local tail`
5. expose debug response fields to inspect the card used for a run

This means we do not need a new service just to learn whether the mechanism works.

## Proposed Experimental Modes

For fail-fast testing, use these modes:

### `off`

Current behavior.

### `instruction`

Prompt contract only. No external bootstrap. Used to measure how far pure prompting gets us.

### `prefix`

Externally supplied visible card prefix before the task body.

### `bootstrap`

Two-pass controller-enforced card creation on the first substantive prompt.

### `bootstrap-session`

Same as `bootstrap`, plus session persistence is card-centered rather than transcript-centered.

These modes are better than a single `on` switch for the experiment phase because they isolate causal mechanisms.

## Success Criteria For The Fail-Fast Phase

The experiment phase succeeds if we can determine:

1. whether instruction-only is too weak
2. whether visible card prefix materially changes behavior
3. whether two-pass bootstrap reliably forces first-turn card creation and reuse
4. whether card-centered session persistence reduces transcript bloat without hurting quality
5. whether any of the above are strong enough that training can be deferred

## Prediction

My current prediction is:

1. instruction-only will be unreliable
2. visible prefix may help somewhat but will not fully force behavior
3. two-pass bootstrap will be the first mechanism that actually changes behavior at the root
4. session-state substitution will be necessary if we want the gains to survive across turns
5. training will probably not be necessary for a first working system if bootstrap plus session substitution is done well

## Practical Recommendation

Do not start by training the 1B model.

Start by proving whether the runtime can force card behavior using a controller-side bootstrap at the request choke point in [src/bin/shimmy_server_gpu.rs](src/bin/shimmy_server_gpu.rs#L895).

If that works, the model weights stay intact and the behavior change lives in a reversible, inspectable control layer.

If that fails, then consider a lightweight adapter or fine-tune specifically for card reflex behavior.

## Next Step

Implement the smallest possible experiment harness around the server request path that supports:

1. `off`
2. `instruction`
3. `prefix`
4. `bootstrap`

Then run the same prompt set across all four and inspect:

1. whether a card exists
2. whether the card was actually used
3. token cost
4. constraint retention
5. multi-turn state quality