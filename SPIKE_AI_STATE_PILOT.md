# AI State Pilot™ — Product Spike

**Status:** Spike (pre-roadmap)  
**Priority:** Project #2 (after Shimmy Vision)  
**Patent Portfolio:** 15 provisionals filed, USPTO Application #63/823,535 through #63/824,076+  
**Core Infrastructure:** airframe (engine) + shimmy (interface layer)  

---

## The One-Sentence Version

AI State Pilot is a cryptographically-auditable, fail-closed enforcement layer that wraps an LLM inference engine — making AI safe enough to run in banks, hospitals, and regulated government environments by guaranteeing that every token can be traced, every decision can be rolled back, and no AI action can escape human review.

---

## Why This Exists Now

You have been thinking about this for years. The patents were filed in June–July 2025. The underlying infrastructure is finally real:

- **airframe** owns the KV cache and the token commit loop — the only place enforcement can be implemented cleanly
- **libfse** (Patent #5, Fail-Closed Enforcement) is built and published
- **schoolmarm** (Patent #9, Model-Checker/GBNF) is built and published  
- **`InferenceControl` trait** is live in airframe — every token passes through it before commit
- **`KvSnapshot`** is live — cryptographic rollback has a foundation

The pieces are on the board. The Crew Chief stack just needs to be assembled.

---

## The Crew Chief Concept

In Air Force maintenance, the Crew Chief has more authority over an aircraft than any general. The pilot suggests; the crew chief clears. If the crew chief says no, the aircraft does not fly.

The `InferenceControl::intervene()` hook **is** that position in the inference stack. It sits between the sampler (the pilot, who thinks they're in charge) and the KV cache commit (the aircraft leaving the ground). Nothing gets past it. Not the model weights, not the temperature, not the prompt. It is the last word on every token.

That is what Patent #5 (Fail-Closed Enforcement Mechanism) describes. The Rust implementation is `FseControl`, already shipped.

---

## Product Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           shimmy                                    │
│         (model discovery, GGUF loading, user-facing API)            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                        airframe                                     │
│             (GPU inference engine, KV cache, token loop)            │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                  ChainControl (combinator)                   │  │
│   │                                                              │  │
│   │  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌───────────┐  │  │
│   │  │Schoolmarm  │ │ FseControl │ │ Budget   │ │  Audit    │  │  │
│   │  │(GBNF cage) │ │(libfse DFA)│ │(token/   │ │ Control  │  │  │
│   │  │Patent #9   │ │Patent #5   │ │sentence  │ │(hash-    │  │  │
│   │  │            │ │            │ │bounds)   │ │chained)  │  │  │
│   │  │            │ │            │ │Patent #4 │ │Patent #2 │  │  │
│   │  └────────────┘ └────────────┘ └──────────┘ └───────────┘  │  │
│   │                                                              │  │
│   │  ControlDecision: Allow | EarlyExit | BlockAndTerminate      │  │
│   │                       | HoldForHumanReview (Enterprise)      │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│   KvSnapshot ←→ CheckpointedWriteQueue (Patent #14)                │
│   Cryptographic rollback: HMAC(prev_hash || token || step)          │
└─────────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│              Human Gate (Enterprise tier — Patent #7)               │
│   BlockAndTerminate → hold stream → notify reviewer → await ACK     │
│   Model cannot proceed without human clearance on flagged events     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Product Tiers

### Tier 1 — Open (Free, current state)
- NoopControl (pass-through, everything permitted)
- Standard shimmy inference, no policy enforcement
- Existing behavior — nothing changes

### Tier 2 — Guarded (Premium)
- `ChainControl` with configurable rule set
- GBNF grammar constraints per endpoint (SchoolmarmControl)
- libfse pattern DFA — block known-bad patterns at token level (FseControl)
- Token/budget limits (BudgetControl — Patent #4)
- Tamper-evident audit log for every enforcement event (AuditControl — Patent #2)
- Policy loaded from a signed config file — cannot be changed at runtime without re-signing (Patent #1)
- **Target:** Power users, agentic workflows, anyone who needs an output cage

### Tier 3 — Enterprise (The Full Crew Chief Stack)
Everything in Tier 2 plus:
- `HoldForHumanReview` ControlDecision — stream suspends, event queued for human ACK
- Cryptographically verifiable KV write queue — every token commit is HMAC-chained (Patent #14)
- Full state rollback with cryptographic proof — "prove to your auditor exactly what the model said and that it was not modified after the fact"
- Policy Privilege Boundary Tables — role-based control over which rules apply to which users/endpoints (Patent #10)
- Air-gap capable — runs entirely offline, no telemetry, no cloud calls (Patent #6)
- **Target:** Banks, hospitals, defense contractors, regulated government — anywhere AI currently cannot be used because there is no audit trail and no rollback

---

## The Development Pod

This is the killer enterprise use case. Here is how it works:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Air-Gapped Dev Pod                               │
│                                                                     │
│   ┌───────────┐    ┌────────────┐    ┌─────────────────────────┐   │
│   │  shimmy   │───▶│  airframe  │───▶│   AI State Pilot        │   │
│   │(model     │    │(inference  │    │(ChainControl + Audit +  │   │
│   │ console)  │    │ engine)    │    │ HumanGate + CryptoQueue)│   │
│   └───────────┘    └────────────┘    └─────────────────────────┘   │
│                                                                     │
│   ONLY ALLOWED OUTPUT:  ──────────────────────────────────▶  PR    │
│                                                                     │
│   Everything else (filesystem writes, network calls, exec) = BLOCKED│
└─────────────────────────────────────────────────────────────────────┘
              │ PR
              ▼
    Human reviews diff
              │ Approve
              ▼
    Changes land in production
```

The pod is a sealed container. shimmy is the model interface. airframe is the engine. AI State Pilot is the enforcement wrapper. The pod is permitted exactly one output channel: a pull request. It cannot touch production. It cannot execute anything. It cannot exfiltrate data. Every single token it generated is cryptographically logged.

A compliance officer at a bank can look at that audit trail and answer:
- What did the model generate?
- Was any output blocked?
- Was the output modified after generation?
- Who approved the PR?
- What was the chain of custody from model output to production?

None of those questions have answers with any current AI tooling. This system answers all of them.

The human gating mechanism makes AI development compatible with change management processes that have existed in regulated industries for decades. You are not asking the bank to trust AI. You are putting the AI inside a cage the bank already knows how to audit, and handing the keys to the compliance team.

---

## Why This Works for Regulated Industries

| Pain point today | AI State Pilot answer |
|---|---|
| "We can't prove the AI didn't modify the output after the fact" | Cryptographically verifiable KV write queue — Patent #14 |
| "We can't audit what the AI decided not to say" | Tamper-evident audit log chains every enforcement event — Patent #2 |
| "AI could execute arbitrary code in our environment" | Fail-closed enforcement at the token level, not the output level — Patent #5 |
| "We can't roll back an AI decision that caused a problem" | KvSnapshot + CheckpointedWriteQueue — rollback with proof |
| "Regulatory change management requires human sign-off on every production change" | HumanGate: model stream holds until human ACKs — Patent #7 |
| "We can't have AI touching production systems" | Dev pod with PR-only output — Patent #6 (air-gapped) |
| "Our grammar/schema must be exactly followed" | SchoolmarmControl — GBNF forces valid output shape — Patent #9 |

The pitch is not "trust AI." The pitch is "we built a cage your compliance team can audit."

---

## Integration with Shimmy

AI State Pilot is not a separate product — it is the enforcement layer that activates on top of shimmy + airframe.

From a user perspective:
- Standard shimmy session = Tier 1 (NoopControl, current behavior)
- Premium shimmy session = Tier 2 (policy file loaded, enforcement active)
- Enterprise pod = Tier 3 (full stack, human gate, crypto audit, PR-only output)

The user doesn't interact with InferenceControl directly. They configure a `policy.toml` (signed), start their shimmy session, and enforcement happens transparently. The audit log is the artifact they hand to compliance.

For the agentic case (shimmy-console running in a loop), AI State Pilot is the reason the agent cannot go off the rails. The grammar constraints (SchoolmarmControl) ensure it only outputs valid tool calls. The FseControl blocks known injection patterns. The budget limits prevent runaway loops. The human gate holds on anything ambiguous. The crypto log proves everything that happened.

---

## Implementation Roadmap (post-Shimmy Vision)

The work is organized in five sprints, each buildable independently:

**Sprint 1 — Wire what already exists** (highest leverage, lowest risk)
- `ChainControl` combinator — run controls in sequence, first non-Allow wins
- Server wiring: `grammar` field on generate request → SchoolmarmControl
- Server wiring: per-tenant `FseMap` config → FseControl
- `BudgetControl` — token cap enforcement

**Sprint 2 — Audit layer** (Patent #2)
- `AuditControl` wrapper — SHA-256 hash chain over every enforcement event
- Append-only log writer (memory-mapped file or SQLite)
- `KvSnapshot.version` as sequence number

**Sprint 3 — Cryptographic write queue** (Patent #14)
- Extend `KvSnapshot`: add `prev_hash: [u8; 32]`
- HMAC-SHA256 on every token commit: `HMAC(prev_hash || token_id || step)`
- Verification CLI: given a log, recompute chain and assert it matches

**Sprint 4 — Human gate** (Patent #7)
- `ControlDecision::HoldForHumanReview(String)` variant
- Tokio channel to a review queue in the server
- HTTP endpoint: `GET /v1/review/queue`, `POST /v1/review/{id}/approve`, `POST /v1/review/{id}/deny`
- Stream resumes on approve, terminates on deny

**Sprint 5 — Policy layer** (Patent #1)
- `policy.toml` schema: grammar rules, fse patterns, budget limits, human-gate thresholds
- Ed25519 signature verification on load — policy cannot be hot-modified without re-signing
- `PolicyStore` struct loaded at server startup, immutable for lifetime of process

---

## Patent Portfolio Alignment

| Implementation | Patent # | USPTO App # | Filed |
|---|---|---|---|
| FseControl (libfse) | #5 — Fail-Closed Enforcement | 63/823,554 | Jun 13, 2025 |
| InferenceControl trait | #13 — Command Interception Engine | 63/824,076 | Jun 15, 2025 |
| ChainControl + PolicyStore | #1 — Immutable Policy/Config Layer | 63/823,535 | Jun 13, 2025 |
| AuditControl | #2 — Tamper-Evident Audit Logging | 63/823,544 | Jun 13, 2025 |
| BudgetControl | #4 — Mathematical Boundedness Engine | 63/823,551 | Jun 13, 2025 |
| Air-gapped dev pod | #6 — Air-Gapped Local-Only AI Deployment | 63/823,560 | Jun 13, 2025 |
| HumanGate | #7 — Human-Gated Command Pipeline | 63/823,562 | Jun 13, 2025 |
| SchoolmarmControl | #9 — Model-Checker Integration | 63/823,570 | Jun 13, 2025 |
| Privilege tiers | #10 — Policy Privilege Boundary Tables | 63/823,575 | Jun 13, 2025 |
| SDK packaging | #11 — SDK Middleware for Compliance AI | 63/823,577 | Jun 13, 2025 |
| KvSnapshot extension | #12 — Memory State Enforcement Engine | 63/824,063 | Jun 15, 2025 |
| CheckpointedWriteQueue | #14 — Cryptographic Write Queue | (Jun 17, 2025) | Jun 17, 2025 |

---

## Open Questions (to resolve during spike)

1. **Naming**: "AI State Pilot" is the patent brand. Does it become a shimmy feature flag, a separate crate (`airframe-policy`), or a standalone product with its own repo?
2. **Pricing model**: Per-seat enterprise license? Usage-based (tokens through the enforcement layer)? SDK licensing fee?
3. **Dev pod delivery**: Docker container (fastest to ship), or a shimmy CLI flag that activates the pod runtime?
4. **Crypto audit log format**: Append-only SQLite (queryable) vs. rolling HMAC-chained binary file (tamper-evident but less convenient)?
5. **SchoolmarmControl on the server**: Accept raw GBNF string per request, or pre-compiled grammar IDs registered at server startup? (Per-request is more flexible; registered is faster and allows policy signing.)
6. **`patent pilot` conversion deadline**: All 15 provisionals are 12-month provisionals. June–July 2025 filings = June–July 2026 deadline for nonprovisional conversion. **The clock is running.** Sprints 1–3 should be complete and documented before conversion to give the strongest "reduction to practice" evidence.

---

## Summary

You spent years working out why AI keeps going sideways. The answer you landed on — a fail-closed, cryptographically auditable, human-gateable enforcement layer that operates at the token level, not the output level — is genuinely novel and now patented across 15 claims.

You also happened to build the only pure-Rust local inference engine that owns its KV cache and token loop, which is the only implementation surface where this enforcement can actually work correctly.

Shimmy Vision ships first. Then this. The regulated-industry market — banks, hospitals, defense contractors — cannot use current AI tooling precisely because it has no audit trail and no rollback. This is the product that unblocks them.
