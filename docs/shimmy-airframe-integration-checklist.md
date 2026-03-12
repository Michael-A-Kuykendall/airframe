# Shimmy x Airframe Integration Checklist

This document is the execution checklist for integrating Airframe into Shimmy without turning the repo into a refactor swamp.

## How To Use This Document

This document is intended to be executable by a lower-cost model or by a tired human without relying on memory.

Rules for use:

- Complete phases in order unless this document explicitly says otherwise.
- Do not silently skip a checkbox because it feels implied.
- After each phase, run the optimization review loop before moving on.
- If a checkbox cannot be completed, add a short note directly under that phase before proceeding.
- If a checkbox reveals a new dependency, update this document first and code second.

Required output for each phase:

- a short summary of what changed
- the exact files touched
- the exact verification run
- the result of the optimization review loop

## Optimization Review Loop

Run this loop at the end of every phase.

### Phase-End Questions

- [ ] Is any step in this phase ambiguous enough that a cheaper model could do it out of order?
- [ ] Is any step missing a concrete file anchor?
- [ ] Is any acceptance criterion too subjective to verify quickly?
- [ ] Is any hidden assumption still living only in the operator's head?
- [ ] Is there any smaller sequence that would make the same phase safer?
- [ ] Is there any validation step missing that would catch a likely failure earlier?

### If The Answer Is Yes

- [ ] Rewrite the phase before continuing.
- [ ] Add missing file anchors.
- [ ] Add missing verification commands or concrete verification descriptions.
- [ ] Tighten acceptance criteria.
- [ ] Remove fluff and replace it with operational detail.

### Exit Rule

Do not move to the next phase until you can say:

- from my perspective this phase is as dense, ordered, and context-laden as it needs to be
- I do not see a simpler or safer phrasing that would materially improve execution

## Working Position

- Shimmy remains the public product surface.
- Airframe becomes an internal engine/backend.
- Shimmy's CLI, HTTP API, OpenAI-compatible API, and model discovery remain the stable public contract.
- Airframe is intentionally obscured behind Shimmy's engine seam.
- The first proof does not need architectural purity.
- The first proof does need clean boundaries, explicit acceptance criteria, and limited scope.

## Complexity Assessment

This is not a freeform coding task.

It is also not so complex that it requires a deeply nested issue graph before work can begin.

The correct control level for this project is:

- one careful markdown execution document
- one narrow working branch
- one integration seam at a time
- one verification gate per phase

If this checklist starts to feel incomplete or ambiguous during implementation, stop and update this document before continuing.

## Non-Goals

- Do not rewrite Shimmy's server layer.
- Do not replace Shimmy's public API surface with Airframe's repro server.
- Do not merge every Airframe experiment into the integration branch.
- Do not redesign the entire Airframe runtime before the first integration proof.
- Do not delete or destabilize the current llama backend during the first pass.
- Do not expose Airframe as a separately branded public engine in the initial Shimmy-next release.

## Success Definition

The project is successful when all of the following are true:

- Shimmy can select an internal Airframe backend.
- Shimmy's public API surface stays intact.
- A request can flow through Shimmy into Airframe and return valid text.
- Streaming still works through Shimmy's existing external streaming surfaces.
- The llama backend still exists as fallback.
- The integration is narrow enough that future runtime cleanup can happen without redoing the public interface.

## Architecture Rule

When there is a choice, prefer:

- Shimmy API unchanged
- Shimmy engine seam changed
- Airframe runtime adapted behind the seam

When there is a conflict, do not let:

- Airframe's current binary structure
- Airframe's console harness
- Airframe's repro HTTP transport

become the permanent public interface.

## Phase 0: Freeze Scope

- [ ] Confirm the integration target branch.
- [ ] Confirm the exact repo and branch where Shimmy integration code will live.
- [ ] Confirm that Shimmy is still the product identity.
- [ ] Confirm that Airframe remains intentionally internal for this release pass.
- [ ] Confirm that the first milestone is a preview-quality proof, not a universal backend rewrite.

Evidence to record:

- active branch name
- target remote branch name
- exact repo path where Shimmy integration code will be edited
- one-sentence restatement of product identity and internal-engine policy

Acceptance criteria:

- There is exactly one branch where the real integration work happens.
- The team is not simultaneously trying to solve product branding, monetization packaging, and engine refactoring in the same coding pass.

Do not proceed until:

- there is no uncertainty about which repository owns the integration code
- there is no uncertainty about whether Airframe is public-facing in this pass

## Phase 1: Capture The Existing Seam

- [ ] Read Shimmy engine traits and adapter flow.
- [ ] Read the Airframe runtime entrypoints currently embedded in the GPU server binary.
- [ ] Identify the smallest callable generation surface Airframe can expose without dragging the console layer with it.
- [ ] Write down the input contract Airframe must satisfy.
- [ ] Write down the output and streaming contract Shimmy expects.

Files to anchor:

- `shimmy_integration/src/engine/mod.rs`
- `shimmy_integration/src/engine/adapter.rs`
- `src/bin/shimmy_server_gpu.rs`
- `src/backend/bindless/pipeline.rs`
- `src/backend/bindless/pipeline_shift.rs`
- `src/backend/bindless/kv_cache.rs`

Artifacts to produce before leaving this phase:

- one plain-language paragraph describing the Shimmy-side engine contract
- one plain-language paragraph describing the minimum Airframe callable surface
- one explicit statement of what is not part of the seam

Acceptance criteria:

- The exact integration seam is described in plain language.
- There is no ambiguity about whether the first implementation is library-direct or bridge-based.

Do not proceed until:

- a cheaper model could identify the seam by reading only this phase and the anchored files

## Phase 2: Choose The First Integration Mode

There are two realistic first-pass modes.

### Option A: Thin Bridge First

Shimmy calls Airframe through a controlled bridge.

Possible bridge forms:

- in-process adapter if Airframe runtime can be lifted fast enough
- process adapter if direct lift is not yet clean
- local HTTP adapter only as a temporary proof, never as the permanent architecture

### Option B: Direct Library Integration First

Shimmy links directly to an extracted Airframe runtime module.

Decision checklist:

- [ ] Estimate how much Airframe code is currently trapped inside the binary.
- [ ] Estimate how much work is required to expose a reusable runtime API.
- [ ] Decide whether direct integration is faster than a narrow bridge.
- [ ] Record the chosen path here before implementation begins.

Decision rule:

- choose the path with the smallest amount of irreversible refactoring required to get one-shot generation working through Shimmy
- prefer temporary ugliness behind the seam over public-surface churn
- prefer explicit technical debt over accidental architecture

Chosen path:

- [ ] Thin bridge first
- [ ] Direct library integration first

Acceptance criteria:

- One path is chosen.
- The unchosen path is explicitly deferred.

Do not proceed until:

- the first implementation mode can be explained in two sentences without hand-waving

## Phase 3: Define The Minimal Airframe Backend Contract

The first Airframe backend should satisfy only the contract Shimmy actually needs.

- [ ] Input prompt support
- [ ] Generation options support
- [ ] Final text output support
- [ ] Token callback support for streaming
- [ ] Clear error propagation
- [ ] Explicit unsupported-feature behavior where necessary

Contract details to write down explicitly:

- exact input type coming from Shimmy
- exact output type expected by Shimmy
- how stop tokens are handled
- how streaming callback ownership works
- which generation options are mapped directly versus approximated
- what happens when Airframe cannot honor a requested option

Do not include on day one unless required:

- [ ] exotic model switching behavior
- [ ] generalized session persistence beyond what is necessary
- [ ] every existing Airframe experiment
- [ ] every future runtime knob

Acceptance criteria:

- The backend contract is small enough to implement in one pass.
- The contract is large enough to power Shimmy's normal generation flow.

Do not proceed until:

- the contract can be implemented without pulling in Shimmy's API layer or Airframe's console layer

## Phase 4: Add Shimmy Backend Skeleton

- [ ] Create `shimmy_integration/src/engine/airframe.rs`.
- [ ] Add an Airframe backend type or selector path.
- [ ] Wire the new backend into `shimmy_integration/src/engine/adapter.rs`.
- [ ] Preserve the current llama backend intact.
- [ ] Gate the backend behind an explicit selection mechanism.
- [ ] Make failure mode obvious when Airframe is selected but not configured correctly.

Implementation constraints:

- do not edit Shimmy API files in this phase unless backend selection requires it indirectly
- do not remove existing backends
- do not start runtime extraction in this phase unless the skeleton cannot compile otherwise

Expected touched files:

- `shimmy_integration/src/engine/airframe.rs`
- `shimmy_integration/src/engine/adapter.rs`
- possibly `shimmy_integration/src/engine/mod.rs`
- possibly `shimmy_integration/Cargo.toml`

Acceptance criteria:

- Shimmy builds with the Airframe backend code present.
- Shimmy can still build and run with the legacy llama path untouched.
- Backend selection is explicit and testable.

Do not proceed until:

- the code compiles even if the Airframe backend only returns a clear placeholder error

## Phase 5: Implement Non-Streaming Text Generation

Start with the smallest useful milestone.

- [ ] Make a non-streaming call from Shimmy into Airframe.
- [ ] Return a complete text result through Shimmy's existing generation path.
- [ ] Normalize generation options where the two systems use different names or defaults.
- [ ] Handle prompt formatting without letting Airframe's console harness leak into Shimmy.
- [ ] Handle model selection or model-path assumptions explicitly.

Questions that must be answered during implementation:

- is the prompt raw text or already template-expanded when it reaches the backend
- who owns stop-token trimming
- does the first proof use one hardcoded known-good model or configurable model routing
- what exact error is returned when the Airframe path is selected without a valid model

Acceptance criteria:

- A single text generation request succeeds end to end.
- The response returns through Shimmy's normal API path.
- No public API changes are required.

Minimum verification:

- one CLI or test invocation that proves the response path works
- one failure-case invocation that proves the error path is intelligible

## Phase 6: Implement Streaming Through Shimmy's Existing Surface

Streaming must remain Shimmy-native externally.

- [ ] Identify Shimmy's current token callback path.
- [ ] Adapt Airframe token emission into Shimmy's callback model.
- [ ] Keep SSE/OpenAI chunk behavior owned by Shimmy.
- [ ] Verify stream termination behavior.
- [ ] Verify that partial output is emitted steadily, not buffered until the end.

Streaming invariants:

- Shimmy owns the outward chunk shape
- Airframe emits token pieces, not final transport frames
- end-of-stream is explicit and testable
- callback backpressure does not deadlock generation

Acceptance criteria:

- Streaming works from the same Shimmy endpoint shape as before.
- The outer client cannot tell whether llama or Airframe produced the tokens except through behavior and performance.

Minimum verification:

- one streaming CLI or HTTP test
- one explicit end-of-stream check
- one check for duplicated or buffered output

## Phase 7: Session And Context Strategy

This is where accidental complexity can explode.

Questions to resolve before coding deeply:

- [ ] Does the first integration expose Airframe's rolling session behavior at all?
- [ ] If yes, is it internal-only or request-visible?
- [ ] If no, what is the fallback behavior for multi-turn continuity?
- [ ] How are prompt templates and token history managed between Shimmy and Airframe?
- [ ] Which layer owns conversation continuity in the first release?

Recommended first-pass posture:

- [ ] Keep session behavior minimal.
- [ ] Avoid inventing new public request semantics unless absolutely necessary.
- [ ] Let Airframe's more advanced continuity features stay mostly internal until basic generation is stable.

Default recommendation unless disproven:

- first pass should keep session semantics behind the backend boundary
- first pass should avoid exposing Airframe-specific session ids in Shimmy's public contract
- first pass should prefer correctness over maximal continuity features

Acceptance criteria:

- Multi-turn behavior is understandable.
- No hidden duplicate-history bug is introduced.
- No accidental API shape explosion occurs.

Do not proceed until:

- ownership of history, prompt templating, and continuity is stated in one paragraph here

## Phase 8: Model Selection And Backend Routing

- [ ] Decide how a model is routed to Airframe versus llama.
- [ ] Decide whether routing is feature-flag-based, config-based, explicit backend-based, or model-family-based.
- [ ] Document the first supported model matrix.
- [ ] Reject unsupported model requests clearly.

Supported-model matrix template:

- backend selector
- model family
- known-good quantization
- supported hardware target
- current support status
- fallback behavior

Recommended first release rule:

- [ ] Support a small known-good set only.
- [ ] Do not imply universal model compatibility until validated.

Acceptance criteria:

- A user can tell when they are using the Airframe path.
- Unsupported combinations fail clearly instead of producing silent nonsense.

Do not proceed until:

- the first supported matrix is small enough to test manually in one sitting

## Phase 9: Runtime Extraction Work

This phase happens only to the degree required by the chosen first integration mode.

- [ ] Identify the minimum Airframe code that must move out of `src/bin/shimmy_server_gpu.rs`.
- [ ] Extract only the runtime pieces that are needed by the Shimmy backend.
- [ ] Keep extraction surgical.
- [ ] Avoid dragging the repro server HTTP layer into the new backend boundary.

Extraction rules:

- move runtime code, not product surface
- prefer wrappers over renames when possible
- do not chase elegance if a stable bridge proves the seam faster

Acceptance criteria:

- The reusable runtime surface exists.
- It is smaller than the entire current GPU server binary logic.
- It does not force a massive rename-and-move project.

Do not proceed until:

- the extracted surface has a name, a call shape, and a deliberately small responsibility set

## Phase 10: Verification Gates

### Build Gates

- [ ] Airframe builds.
- [ ] Shimmy integration build passes.
- [ ] Legacy llama backend still builds.

Record exact commands used:

- [ ] command for Airframe build
- [ ] command for Shimmy integration build
- [ ] command for legacy backend verification

### Functional Gates

- [ ] One-shot text generation works.
- [ ] Streaming works.
- [ ] Error path is intelligible.
- [ ] Fallback backend path still works.

Record exact proof points:

- [ ] non-streaming proof command or test
- [ ] streaming proof command or test
- [ ] fallback proof command or test

### Behavioral Gates

- [ ] No public Shimmy endpoint changes required.
- [ ] No obvious performance collapse versus current proof path.
- [ ] No hang at end-of-stream.
- [ ] No duplicate prompt re-ingestion bug.

### Safety Gates

- [ ] Unsupported models fail fast.
- [ ] Unsupported quantization paths fail fast.
- [ ] Misconfiguration is obvious from logs.

Do not leave this phase with implied proof only.

Every checked item in this phase must be backed by:

- a command
- a test
- or a clear manual verification step recorded next to the work summary

## Phase 11: Preview Packaging Constraints

Before calling it preview-ready:

- [ ] Narrow the supported model set.
- [ ] Narrow the supported hardware story.
- [ ] Write the user-facing caveats honestly.
- [ ] Preserve the current fallback backend.
- [ ] Keep Airframe branding intentionally subordinate or invisible in the public release surface.

Acceptance criteria:

- The release promise is small and believable.
- The implementation does not force a full rewrite before feedback can be gathered.

Required public framing for preview:

- what is supported
- what is intentionally not supported yet
- what fallback exists
- how users should think about the feature without needing to know Airframe branding

## Phase 12: Decision Gate Before Deeper Refactor

Only after preview proof works:

- [ ] Decide whether to deepen library extraction.
- [ ] Decide whether to expose more Airframe-specific capabilities.
- [ ] Decide whether to broaden model support.
- [ ] Decide whether to optimize for more quant families.
- [ ] Decide whether to de-emphasize or retire the older backend path.

## Known Risk Areas

- [ ] Airframe runtime is currently binary-centered.
- [ ] Streaming semantics could diverge between the two systems.
- [ ] Session continuity could introduce hidden duplication or ownership confusion.
- [ ] Quantization support in Airframe is narrower than universal GGUF support.
- [ ] TinyLlama-based assumptions may leak into generic integration if not controlled.
- [ ] Console-harness code may tempt shortcut reuse in the wrong layer.

## Red Flags That Mean Stop And Re-Plan

Stop and update this document if any of these happen:

- [ ] The integration suddenly requires replacing Shimmy's server layer.
- [ ] The only viable path appears to route public traffic through Airframe's repro HTTP server permanently.
- [ ] Airframe runtime extraction balloons into a repo-wide architectural migration.
- [ ] Model/session semantics become unclear enough that test expectations cannot be written simply.
- [ ] The fallback backend starts breaking while the Airframe path is still incomplete.

## Execution Order

The recommended implementation order is:

1. Freeze branch and scope.
2. Re-read seam files and finalize the first integration mode.
3. Add backend skeleton in Shimmy.
4. Make one-shot generation work.
5. Make streaming work.
6. Handle backend routing and supported model gating.
7. Run verification gates.
8. Package the preview story.

## End-Of-Day Update Template

Use this template after each work session:

### Session Summary

- Phase worked on:
- Files touched:
- Checks completed:
- Verification run:
- Result:

### Optimization Review

- What was unclear before this pass:
- What was tightened:
- What still feels risky:
- Does the document need more density before the next session:

### Next Exact Move

- one sentence only

## Current Status Snapshot

- [x] Strategy documented
- [x] Seam map documented
- [x] Airframe proof harness exists
- [x] OpenClaw/provider packaging branch exists
- [ ] Shimmy-side Airframe backend skeleton implemented
- [ ] End-to-end Shimmy to Airframe generation path implemented
- [ ] Shimmy streaming path backed by Airframe implemented
- [ ] Preview-quality integration validated

## Final Rule

The goal is not to finish every possible cleanup move.

The goal is to reach a working Shimmy-next proof where:

- Shimmy still feels like Shimmy
- Airframe is doing the important hidden work
- the public interface remains stable
- the integration branch remains understandable
- and the next step after the preview is obvious instead of chaotic