# Helical Shift Validation Plan

## Purpose

Validate helical shift as an edge-case and long-run behavior question without reopening settled claims about the current TinyLlama exact-story repro.

This plan exists because the main story repro is currently in a no-known-blocker state, but helical shift still deserves bounded, explicit validation before wider release.

## What Is Already Established

- current tip matches the historical short-prefix baseline at `128` tokens
- `b4baaea` matches that same baseline
- helical-off also matches that same short-prefix baseline
- the latest clean full exact-story rerun on current tip matched historical story content, with only a trailing newline difference at the artifact level

## Validation Goal

Answer a narrower question:

- does helical shift preserve expected behavior in long-run generation and edge-case conditions, or does it introduce repeatable degradation that should be documented or fixed before broader release?

## Test Rules

1. Keep tests one variable at a time.
2. Keep the shortest discriminating SHA check as the front-door sanity test.
3. Do not generalize from a long-run failure until the short check passes in the same environment.
4. Save final text outputs for any long-run comparison, not only hashes.
5. Record stop reason, token count, and whether repetition or collapse appears.

## Baseline

- request: `artifacts/story_seed7777_128tok_request.json`
- expected short SHA: `f82a1ad07e5f74415a3121821e580998eecda4edd30b43efc9b294aa591c7974`
- helper: `scripts/short_story_sha_check.ps1`

## Test Sequence

### 1. Short Sanity Check

- Run current tip with default settings.
- Confirm the short SHA matches expected.

### 2. Long Default Run

- Run the full exact-story request on current tip.
- Save the final plain text output.
- Compare against the historical extracted story file.
- Record first difference location if any.

### 3. Long Helical-Off Run

- Run the same full exact-story request with helical shift disabled.
- Save the final plain text output.
- Compare against both the default long run and the historical extracted story file.

### 4. Stress Cases

Run a small set of deliberately hostile cases:

- generation that crosses the compaction boundary multiple times
- repeated-seed runs to verify determinism under long decode
- prompts designed to expose repetition collapse or narrative loopback
- session reuse versus fresh session for the same prompt

### 5. Classification

Classify each result as one of:

- no issue
- expected limitation requiring documentation
- reproducible defect requiring code change

## Minimum Output Record Per Run

- server/backend variant
- seed
- prompt file
- stop reason
- tokens generated
- output character count
- SHA if applicable
- saved final text path
- first diff location versus baseline if applicable
- short note on repetition, collapse, or quality drift

## Exit Criteria

Helical shift is sufficiently validated for the current release pass when all of the following are true:

- short sanity checks remain stable
- the long default run does not show a newly reproducible content failure on the canonical repro path
- any helical-specific long-run weakness is either fixed or documented as a known limitation
- the release decision is based on recorded outputs, not memory or chat archaeology