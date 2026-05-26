# Airframe Top-to-Bottom Scrub

This document is the release-audit pass for a perfectionist review of Airframe and the Shimmy integration surface.

The goal is not to be encouraging. The goal is to find imperfection, document it precisely, and force every claim to stay honest.

## Audit Rule

Treat every file, comment, docstring, public claim, and helper as suspect until it is verified against the code that actually runs.

A line is only acceptable when at least one of the following is true:

- it is necessary and correct
- it is documented clearly enough that a second reader will not misread it
- it is covered by a test or release check
- it is deliberately exempted with a justified reason

No silent assumptions.
No hand-wavy language.
No "should be fine".

## Primary Questions

For every reviewed area, ask:

- Is the documentation accurate right now, not aspirational?
- Is the inline comment explaining real behavior or papering over confusion?
- Is the API shaped idiomatically, or only made to work?
- Is there a hidden warning suppression, dead path, or stale helper hiding in plain sight?
- Does the release story match the actual behavior and gates?
- Would an outside auditor think this was built cleanly, or patched to stop the compiler from complaining?

## Audit Scope

Review in this order:

1. README and release-facing docs
2. Architecture docs and validation plans
3. Public API and CLI surfaces
4. HTTP/server surfaces
5. Inline comments and doc comments in Rust code
6. Error handling and fallible paths
7. Test code and fixtures
8. Warning suppressions and dead-code exemptions
9. Release envelope and feature claims

## What Counts As An Imperfection

A finding should be logged if any of these are true:

- the doc says one thing and the code does another
- a comment is stale, redundant, or misleading
- a function signature is awkward without justification
- a helper exists only to work around a type or lint issue
- a `#[allow(...)]` hides a problem without a written reason
- a release claim overstates what has been validated
- a path is test-only but lives in production code with no clear boundary
- an error path swallows context or returns vague diagnostics
- a public name or module boundary makes the architecture harder to understand than it should be

## Known Imperfection Ledger

These are not all necessarily bugs, but they are the first things an auditor should inspect closely.

### 1. Warning suppressions exist across the tree

The repository still contains many `#[allow(...)]` attributes. Some are justified, but they are still audit targets because they can hide drift if they are not reviewed.

Examples to re-check:

- `#[allow(dead_code)]`
- `#[allow(clippy::too_many_arguments)]`
- `#[allow(unused_mut)]`
- `#[allow(non_camel_case_types)]`

Audit question:

- Is each suppression still necessary, and is the justification still accurate?

### 2. Some docs are framing documents, not evidence

Several release docs are useful, but they are still planning or framing documents until they are backed by current validation.

Cross-check these against the actual code and current test results:

- [docs/airframe-current-stack-audit.md](docs/airframe-current-stack-audit.md)
- [docs/shimmy-airframe-integration-checklist.md](docs/shimmy-airframe-integration-checklist.md)
- [docs/shimmy-airframe-release-strategy.md](docs/shimmy-airframe-release-strategy.md)
- [docs/shimmy-airframe-launch-envelope.md](docs/shimmy-airframe-launch-envelope.md)
- [docs/helical-shift-validation-plan.md](docs/helical-shift-validation-plan.md)

Audit question:

- Does each document describe a verified state, or a hoped-for one?

### 3. Long-run behavior is still the most audit-sensitive path

The repo has strong short-run parity and a green test suite, but long-run behavior is still the place where structural mistakes usually hide.

Audit question:

- Are the long-context, helical-shift, session-window, and release-envelope claims still exactly aligned with the latest validation?

### 4. Inline documentation must be checked for drift

Doc comments and inline comments can silently rot even when tests pass.

Inspect especially:

- server behavior comments
- release-envelope comments
- comments describing temporary branches or test-only helpers
- comments near `#[allow(...)]` attributes

Audit question:

- Does the comment explain the current code, or the reason an old version used to be different?

### 5. Test-only helpers living in production modules should be justified

A helper inside production code is fine if the boundary is explicit. It is not fine if it exists because the codebase never got cleaned up after a temporary debug pass.

Audit question:

- Could this helper be moved, removed, or isolated without making the code worse?

## Review Checklist

### Documentation

- [ ] README does not overstate capability or completeness.
- [ ] Release docs match the current command behavior.
- [ ] Architecture docs do not promise a boundary that the code does not actually enforce.
- [ ] Every operational claim has a current, runnable verification path.

### Inline Documentation

- [ ] Each non-obvious comment explains why the code exists, not just what the code literally does.
- [ ] No comment describes a path that no longer exists.
- [ ] No comment is used as a band-aid for weak naming.
- [ ] Doc comments on public APIs reflect the actual contract.

### Idiomatic Rust

- [ ] Error paths return useful context.
- [ ] `Result` values are not ignored unless the ignore is explicit and justified.
- [ ] Helper structs exist to improve clarity, not to hide awkward signatures.
- [ ] Public functions are split when the signature becomes a maintenance hazard.
- [ ] Temporary workarounds are not left behind as permanent structure.

### Lint Hygiene

- [ ] Every `#[allow(...)]` has a reason that still holds.
- [ ] Every `#[allow(dead_code)]` is either test-only, documented, or removed.
- [ ] Every `#[allow(clippy::too_many_arguments)]` has been re-evaluated.
- [ ] No warning suppression exists solely to make CI quiet.

### Surface Integrity

- [ ] CLI flags map to real behavior.
- [ ] Server endpoints match their documentation.
- [ ] Public-facing launch language matches the validated behavior.
- [ ] Internal-only paths are not accidentally advertised as product surface.

### Test Integrity

- [ ] The full suite still passes.
- [ ] Ignored tests are documented as such and are not being mistaken for coverage.
- [ ] Determinism and long-context checks are still visible in the release story.
- [ ] Fixtures used for parity are clearly named and obviously tied to their intended baseline.

## Current Verification Baseline

As of the latest validation run:

- `cargo clippy -- -D warnings` passes.
- `cargo test --no-run` passes without warnings.
- `cargo test` passes with zero failures.
- The determinism test completes successfully.

This is a good baseline, but it does not end the audit.

A perfectionist pass still has to check whether the repository is clean by inspection, not just by compiler output.

## What To Inspect Next

If you are doing the scrub manually, start here:

1. [docs/airframe-current-stack-audit.md](docs/airframe-current-stack-audit.md)
2. [docs/shimmy-airframe-release-strategy.md](docs/shimmy-airframe-release-strategy.md)
3. [docs/shimmy-airframe-integration-checklist.md](docs/shimmy-airframe-integration-checklist.md)
4. `src/bin/shimmy_server_gpu.rs`
5. `src/bin/shimmy_eval.rs`
6. `src/backend/bindless/pipeline/`
7. `src/runtime/gpu.rs`
8. `tests/`

## Output Standard For Findings

When you find an imperfection, write it like this:

- file
- exact line or nearest anchor
- what is wrong
- why it matters
- the smallest clean fix
- whether it is a doc drift, style issue, architectural mismatch, or real bug

Avoid vague statements like "needs cleanup".
Say exactly what is wrong.

## Exit Criteria

This scrub is only finished when all of the following are true:

- documentation matches behavior
- inline comments are current and useful
- no unjustified suppression remains
- public claims match validated gates
- the reviewer can explain the release story without relying on memory or chat history

If any one of those is false, the scrub is not done.
