# FSE + D0 Lens — Required Perspective for All Workspace Challenges

**Enshrined:** 2026-06-12 (per user directive after reading FSE spec + D0-ENGINE-ARCHITECTURE.md)

## The Two Concepts (Core Lens)

### 1. Fused Semantic Execution (FSE)
- Selector-first, single-pass, compile-time fused rule evaluation.
- Rules decomposed into (selector, predicate).
- Selector deduplication via prefix trie: shared selectors traversed **once**.
- Values extracted once → broadcast to all interested rules.
- Rule state bitmap + early-exit when all rules resolved.
- Deterministic, fail-closed, bounded memory.
- Core invariant: evaluation cost independent of rule count for shared selectors (∂runtime / ∂rules ≈ 0).
- Applies directly to AI inference gating + observability.

### 2. D0 — Domain-Agnostic Reactive Fact Engine (inverted reactive graph)
- Inverts the broken "rule-first" pattern (every observer re-scans/extracts data).
- **Data stream emits facts. Facts activate rules. Rules fire actions.** Once.
- Three-tier facts:
  - Tier 1 Structural: auto-emitted on data arrival.
  - Tier 2 Semantic: derived by rules from Tier 1.
  - Tier 3 Consequent: consumed immediately (never stored long-term), drives mutations/actions.
- Arena: generic hash-cons node store (NodeId u32, O(1) structural equality, sharing).
- FactStore + pending queue (propagation frontier). Saturation = drain to empty.
- Program Engine:
  - Alpha index: new fact routes *only* to rules matching its first pattern (type).
  - Beta joins for multi-pattern rules.
  - Saturation loop.
  - Consequents dispatch mutations.
- Statically typed (Rust enum facts), generic, compile-time verified (exhaustive match, unused fact variants warn).
- "This is the FSE architectural invariant applied to the Rust type system."
- d_runtime / d_rules = 0 for shared selectors.
- Currently backing airframe_observe (ReactiveGraph<InferenceFact>, ObservationSession, alpha/beta, saturation).

**Unified Lens:** Every software challenge in this workspace (TDR stabilization, model correctness, vault/golden traces, observation, testing harness, console, git workflows, etc.) **must** be reframed and solved as:
- Emit structural facts from the data stream (inference passes, dispatches, traces, git events, model outputs, etc.).
- Rules (via FSE selectors or D0 alpha/beta) derive semantic facts.
- Consequents drive actions (chunk decisions, writes, alerts, calibrations, early exits).
- All in a **single fused/reactive pass** with deduplicated extraction and broadcast.
- Use Arena for sharing sub-structures where possible.
- Leverage existing airframe_observe + d0-engine (from dzero-cas) as the implementation vehicle.
- Goal: make adding "rules/observers" (safety, verification, performance, debugging) cost near-zero for shared facts.

This lens ensures we never re-implement repeated traversals, always get the efficiency invariant, and treat control (e.g. TDR navigation) and observation uniformly as reactive facts/rules.

## Application to Current Primary Challenge: TDR Navigation + Model Stabilization

**Reframed as FSE/D0:**

Data stream = GPU inference prefill/decode (token embeddings → layer kernels → dispatches → activations → logits).

Structural facts (Tier 1, auto-emitted from code):
- DispatchStart { layer, kernel: QKV|FFN|AttnOut|..., batch_size, quant_type }
- DispatchCompleted { ... + actual_gpu_time_ms (via TIMESTAMP_QUERY) }
- LayerOutput { layer, values: & [f32], rms, nan_count }
- FinalLogits { ... }
- (Reuse/extend existing InferenceFact in airframe_observe)

Semantic facts (Tier 2, derived by D0 rules on the graph):
- TdrRisk { layer, kernel, projected_ms, risk_level }
- SafeChunkSize { kernel, recommended: u32, calibrated_for: (model, quant, adapter) }
- DispatchCostPerToken { ... }
- LayerStable { layer, delta_vs_vault }
- (Derived alongside VaultOracle / CandleCompare / LayerStability rules)

Consequents (Tier 3, drive immediate actions/mutations, never long-term stored):
- UseSafeChunk { size } → patches LayerParams.batch_offset / batch_count, forces micro-encoder + submit + poll for that chunk only.
- EmitCalibration { ... } → writes to vault/verification_runs or local cache.
- YieldForTDR { } → the per-chunk submit/poll that lets Windows scheduler preempt.
- EarlyExitIfStableAndSafe { }

**Fused single-pass during one prefill:**
- The ObservationSession (D0 ReactiveGraph) is active.
- Alpha routes "QKV DispatchStart" fact **only** to TDR-related rules (plus any other that registered for heavy dispatch facts).
- Beta joins combine timing + layer output for risk calculation.
- Saturation propagates until no more derived facts (chunk decided, vault row written if needed, stability checked).
- The chunker consequent mutates the submission parameters for the *current* execution.
- Calibration consequent can run in the same pass on first use of a model class.
- All other observers (vault oracle population for golden traces, candle cross-ref as second reference, formula diffs, stability) get the same facts for free via broadcast.
- Early exit: once TDR is safe for this layer + other critical rules satisfied, graph can short-circuit.

**Benefits that fall out:**
- No separate imperative "TdrNavigator" or "SafeComputeSession" duplicating extraction logic.
- The micro-batching already present in inference.rs (QKV while loop, batch_offset/count, per-chunk submit/poll) becomes the *consequent handler* for the TdrChunk rule.
- Automatic adaptation: first dispatch of a new (model, Q4_K_M, this 3060) emits timings → derives SafeChunk → subsequent layers/dispatches use it. No env var, no hard-coded numbers from one machine.
- Extends uniformly to Q4K path (emit same facts from its dispatch site).
- Vault updates (oracle rows, verification_runs with TDR budgets) happen as part of the same saturation as correctness checks.
- Chronicle integration: every fact emission or consequent can be logged in a way searchable via the MCP (previous TDR attempts, what chunk worked for this model last time).
- Fuses with "two-reference golden trace" goal: the same pass that keeps TDR from killing the device also feeds CPU ref (for vault_seed oracles), Candle (candle_probe), and formula side-by-sides.
- For the broader stabilization: every failing model run emits the full fact set → rules can derive "this is the first bad layer for TDR" or "Q6K head silent" or "fused QKV routing panic" → consequents drive targeted fixes or test prioritization.
- Performance: adding more TDR-related rules (e.g., per-kernel budget tracking, progressive chunk reduction on risk) or other concerns costs almost nothing because selectors are deduplicated and facts broadcast once.

This is exactly the FSE invariant + D0 inverted reactive graph applied to the TDR problem.

## How to Use This Lens Going Forward (Enshrined Practice)

For **every** software challenge in airframe/shimmy workspace:
1. Identify the data stream(s).
2. Define the minimal set of Tier 1 structural facts that can be emitted (prefer reusing/extending InferenceFact).
3. Identify rules that can derive Tier 2 semantic facts (use existing observers where possible; extend the plan).
4. Define Tier 3 consequents that drive the needed mutations/actions (e.g., chunk decisions, writes, early yields).
5. Express as FSE selectors + D0 alpha/beta rules in an ObservationSession (or direct ReactiveGraph).
6. Ensure single-pass during the real execution (no repeated full traversals).
7. Leverage dedup + broadcast + early exit.
8. Record calibration/history in vault + make queryable via Chronicle.
9. Use the airframe_observe + d0-engine (from dzero-cas) as the vehicle — do not re-implement imperative loops when a reactive fact rule will do.
10. When stuck, ask: "What facts would the data emit? What rules derive the decision? What consequent fires the fix?"

This lens applies to:
- TDR / prefill chunking / Q4K vs V1 paths
- Vault population + two-reference golden traces (CPU + GPU + Candle)
- Model stabilization verification loop (test_model.ps1, frontier_compare, direct generate)
- Observation and debugging (debug_trace, layer dumps)
- Console / local dev platform work
- Git / branch discipline + hotfix processes
- Any new feature or bug (e.g., output head for Q6_K, fused QKV routing, adaptive anything)

## Implementation Status & Next Steps

- FSE spec + D0 architecture now enshrined in this file and referenced from:
  - .kiro/steering/current-work-directive.md (updated to require this lens)
  - .kiro/steering/inference-testing.md (cross-ref for verification runs)
- airframe_observe is the current partial realization (needs full d0-engine backend wiring per the extraction roadmap in the D0 doc).
- For TDR: the existing micro-batch code + pending TIMESTAMP_QUERY calibration is the skeleton of the TdrChunk / TdrCalibrator rules. Next is to emit the facts from the dispatch sites and drive the chunker via ObservationSession saturation instead of imperative while loops where possible.
- All future work on fix/v0.2.5-all-fixes and sub-branches (q4k-tdr-diagnosis, etc.) shall be framed this way.

**We now view every problem through FSE selectors + D0 reactive facts.** This is the permanent lens for the workspace.

---

*This file was created to fulfill the directive to enshrine the concepts so the AI (and team) constantly filters challenges through them.*