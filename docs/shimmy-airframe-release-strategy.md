# Shimmy x Airframe Release Strategy

## Executive Decision

Do not do a big refactor first.

The current state argues for a staged release, not a purity rewrite.

Recommended position:

- `Shimmy` remains the product, brand, distribution, and OpenAI-compatible server.
- `Airframe` becomes the engine/runtime layer that powers the next major Shimmy release.
- The current Airframe console work is treated as an internal proving harness, not the primary public surface.
- The old `llama.cpp` route becomes a compatibility path and later a historical branch, but not on day one.

That gives the fastest path to market without turning the current branch situation into a full rewrite.

## Core Thesis

You are not deciding between two equivalent products.

- Shimmy already has audience, stars, install expectations, release mechanics, and a clear public promise.
- Airframe already has the performance and technical novelty.

So the move is not `replace Shimmy with Airframe`.

The move is:

- `Ship Shimmy Next powered by Airframe`

This captures the upside of the Shimmy brand without forcing an immediate invasive refactor of every branch where console work drifted.

## What To Avoid

Avoid these traps:

1. Do not stop and re-aim the whole repo around a clean architecture before shipping.
2. Do not make the console branch the mandatory path to release.
3. Do not force a full merge of all divergent experimental branches before proving the market-facing version.
4. Do not delete the old backend immediately.

Those are high-drama moves with low near-term shipping value.

## Recommended Release Model

### Track A: Near-Term Public Release

Ship a release that is effectively:

- Shimmy-branded
- Airframe-powered
- intentionally scoped
- marketed as the next major generation

Suggested public framing:

- `Shimmy v2.0`
- subtitle: `Powered by Airframe`

This is the best use of the existing market pull.

### Track B: Internal Engine Productization

In parallel, keep shaping Airframe into a cleaner reusable library/runtime:

- stable engine boundary
- reusable API surface
- isolated runtime crates
- future paid/private/closed options if desired

This work should happen behind the release, not in front of it.

## Reality Check On The Console Work

The current console work is valuable, but it is not the right center of gravity for the launch.

Treat it as:

- a proving harness
- an internal operator UI
- a debugging and demo surface
- a staging area for interaction patterns

Do not make it the required public integration layer for the first release.

That is the key simplification.

In release terms:

- Keep the console if it helps testing and demos.
- Do not block shipping on perfectly reconciling every console branch.
- Lift only the parts that are clearly valuable to the public Shimmy release.

## Product Recommendation

### What The Public Gets

Publicly, the user should experience:

- the familiar Shimmy binary and API identity
- faster generation
- smoother streaming
- stronger GPU behavior
- better multi-turn continuity
- benchmark proof on the launch site

### What Changes Under The Hood

Under the hood, Airframe provides:

- the runtime and GPU execution path
- the streaming behavior
- the session behavior
- the performance story

### What Stays Legacy

Keep these as legacy paths for a while:

- llama.cpp backend
- older fallback paths
- historical branch for users who need the previous architecture

## Release Sequence

### Phase 0: Freeze The Story

Make one decision and stop re-litigating it:

- `Shimmy is the product`
- `Airframe is the engine`

Once that is fixed, every technical choice becomes easier.

### Phase 1: Stabilize The Current Airframe Proof

Goal:

- prove the current Airframe server/chat stack is dependable enough to demo and benchmark

Scope:

- server startup and shutdown cleanly
- chat works end-to-end
- token streaming works reliably
- session continuity is acceptable
- benchmark harness is reproducible

Do not refactor broadly here. Stabilize the existing path.

### Phase 2: Define The Shimmy Integration Seam

Use Shimmy's existing engine abstraction as the seam.

Target:

- add an `Airframe` backend beside the current `Llama` backend
- keep the Shimmy API server and CLI surface intact
- route generation through Airframe where available

This avoids redoing Shimmy's public server/API identity.

### Phase 3: Ship A Public Preview

Suggested launch form:

- `Shimmy v2 preview`
- `Shimmy experimental Airframe release`
- or `Shimmy Next`

The preview should intentionally ship with a narrow promise:

- one or a few known-good models
- one or a few GPU targets
- a benchmark page that proves the delta

This is not the final universal packaging moment. It is the credibility moment.

### Phase 4: Deprecation Path

Only after the new path is clearly stable:

- mark the old llama.cpp route as legacy
- create a historical branch for it
- keep fallback support for one or two release cycles
- then reduce its prominence

## Branch Strategy

Because the branch situation is already messy, use a release branch strategy that minimizes merge pain.

### Recommended Branches

1. `airframe-stabilization`
2. `shimmy-airframe-spike`
3. `shimmy-v2-preview`
4. `legacy-llama-cpp` later, when you are ready to freeze it

### Practical Rule

Do not merge every experimental branch into one mega-branch.

Instead:

- pick the smallest working Airframe path
- stabilize it
- port only the necessary parts into a Shimmy integration spike
- leave the rest of the experiments behind unless they are clearly needed

That is the correct antidote to branch chaos.

## Release Scope Recommendation

### First Release Should Include

- GPU server path
- chat/generation path
- streaming
- session continuity
- benchmark evidence
- a simple run/install story

### First Release Should Not Depend On

- perfectly unified console architecture
- every experimental branch being reconciled
- maximum backend compatibility
- total internal crate purity

## Marketing Positioning

Suggested message:

- `Shimmy has a new engine.`
- `Shimmy v2 is powered by Airframe.`
- `Faster streaming, better GPU execution, stronger runtime.`

This is much stronger than introducing an unrelated new name and asking the market to care.

Airframe still matters publicly, but as a credibility amplifier:

- `Powered by Airframe`
- technical writeups
- benchmarks
- developer docs later

## Release Risks

### Risk 1: Branch Drift

Mitigation:

- do not merge everything
- ship from the smallest coherent slice

### Risk 2: Console Refactor Spiral

Mitigation:

- treat console as internal harness for now
- do not make console cleanup a release gate

### Risk 3: Over-Promising Generality

Mitigation:

- launch narrowly
- promise what is proven

### Risk 4: Legacy Backend Breakage

Mitigation:

- keep the old backend as fallback during transition
- use a preview release before hard deprecation

## Concrete Recommendation

If the question is:

- `Should I refactor everything first?`

The answer is no.

If the question is:

- `Should I release Airframe completely on its own right now?`

The answer is probably not as the primary go-to-market vehicle.

If the question is:

- `Should I release a new Shimmy generation powered by Airframe while continuing to harden Airframe underneath?`

The answer is yes.

## Immediate Next Steps

1. Freeze the public product decision: Shimmy product, Airframe engine.
2. Stabilize the current Airframe path enough for reproducible demos and benchmarks.
3. Build a minimal Shimmy integration spike using Shimmy's existing engine abstraction.
4. Launch a preview release rather than waiting for architectural perfection.
5. Deprecate the old path only after the preview proves itself.

## Go / No-Go Standard

Go when all of these are true:

- the GPU path starts reliably
- chat works reliably
- streaming works reliably
- benchmark results are reproducible
- the launch story is simple to explain in one sentence

That sentence should be:

- `Shimmy v2 is powered by Airframe and it is faster.`