# feat/inference-fabric-core — Bead Spec (frozen backup)

> **AUTHORITATIVE SOURCE = the `bd` issue tracker.** Anchor bead `airframe-v9w` + the 10
> task beads (`airframe-69c`, `airframe-ppo`, `airframe-0zh`, `airframe-c90`, `airframe-6gc`,
> `airframe-0q2`, `airframe-y26`, `airframe-ed6`, `airframe-937`, `airframe-3z9`). Run
> `bd show airframe-v9w` to recover everything. This committed markdown is a **frozen backup
> only** — the `bd` issues are the live source of truth (everything needed to do the work is
> in the beads; done-work and rejected approaches are not presented as tasks).
> Branch: `feat/inference-fabric-core`.

---

## 0. TL;DR

Promote `airframe_observe::isf` (the patent-pending **Inference Saturation Fabric**, a D0
reactive fact engine) from an *observation* layer into the *control core* of inference
dispatch. Dispatch becomes **data-driven and reactive**: GGUF-derived facts flow through the
fabric; rules derive which shader/formula each tensor uses; the forward pass executes from
the saturated facts. The hardcoded `if (qt==)` ladders in WGSL/Rust are retired. Adoption in
`gpu.rs` is a single 1-liner swap.

**Validation is algebraic against the GGUF/GGML spec — NOT against any external engine.**
Golden traces (candle / llama.cpp outputs) are OUT. The math is the referenced core.

---

## 1. GOLDEN RULES (do not violate — these are the guardrails)

These exist because a prior session violated them and wasted a full cycle. Bake them in.

1. **Math, not golden traces.** The canonical dequant/attention/RoPE/RMSNorm formulas come
   from the **GGUF/GGML spec**, written once per quant type, audited algebraically. They do
   NOT come from candle, llama.cpp, or any other engine's output. We stand on our own
   mathematical feet. (Prior mistake: chasing a "GPU vs airframe-CPU" divergence on Qwen3 —
   airframe-CPU was NEVER a validated Qwen3 reference; the comparison was meaningless.)
2. **airframe-CPU is NOT a Qwen3 reference.** Only **TinyLlama (q4_0 and q6_k)** was ever
   validated against airframe's own CPU path. Every other model (Qwen2/3, Llama 3.x,
   DeepSeek, …) is self-derived shader/math work with NO validated CPU golden. Any
   `vault.duckdb.layer_oracles` rows for those models were seeded from airframe's own CPU
   path and are UNTRUSTED for those models. The vault is a control-plane/fact store, not a
   Qwen3 oracle.
3. **Dispatch is fabric-driven, not if/then.** Selection of a tensor's dequant/shader is a
   *rule* fired by the alpha index on a fact, not a hardcoded `if qt==` ladder in WGSL or
   Rust. The dispatch *logic* (quant_type → formula) lives in an auditable, spec-cited
   registry — never inline in a shader.
4. **The saturation fabric is the control core.** D0 model: data emits facts → facts
   activate rules → rules fire actions, once, regardless of observer count. `airframe_observe::isf`
   already implements this engine. We extend its fact vocabulary and write dispatch rules;
   we do NOT rebuild the fabric.
5. **candle/llama.cpp are at most optional cross-checks, never the standard.** They may be
   used LATER to spot-check our numbers. They never define correctness.

---

## 2. ARCHITECTURE (discovered, not assumed)

From the B0 spike (`docs/bead-b0-dispatch-map.md`):

- **One shared forward pass.** Both `gpu.rs::generate` (imperative) and `gpu.rs::generate_isf`
  (reactive) call the SAME `run_full_model_prefill_chunked_with_cache_state`. The quant
  dispatch lives entirely inside that forward pass. ⇒ The "dual orchestrators" are just two
  token-loops; B4's 1-liner swap is safe; B7 = delete the imperative loop.
- **Control plane already exists.** `ModelRoutePlan` (`src/core/routing.rs`) is a typed,
  spec+tensor-derived plan (NormKind, QkvLayout, FfnKind, digest, reasons/warnings). The
  B1 quant→formula registry hooks onto it as a new field — we extend, not build fresh.
- **Shader is already partly data-driven.** `QUANT_ELEMS`/`QUANT_BYTES` const arrays in
  `sh_layer_v1.wgsl:343-346` index block sizes by quant_type. The ONLY genuine if/then
  ladder is `dequant_dispatch()` at `sh_layer_v1.wgsl:353-361`.
- **D0 → inference mapping (the control core):**
  | D0 tier | inference |
  |---|---|
  | Tier 1 Structural (auto-emitted on data) | `TensorFact{quant_type, shape, offset, arch}` asserted on GGUF load (B2) |
  | Tier 2 Semantic (derived by rules) | `DispatchFact{shader, params, offset}` derived from `TensorFact` via B1 registry (B3a) |
  | Tier 3 Consequent (drives mutation) | forward pass reads saturated `DispatchFact`s and executes — no if/then |
  | Alpha index | each `TensorFact` fans to exactly its dispatch rule |

  ⇒ B3b: `dequant_dispatch`'s `if qt==` ladder becomes a **fabric rule** that emits
  `DispatchFact{formula_index}`; the shader consumes that as a uniform.

---

## 3. CURRENT INVESTIGATION STATE (as of branch start)

- **Q6_K nibble flip REVERTED.** A prior session flipped the Q6_K nibble polarity in
  `sh_dequant_any.wgsl` / `sh_layer_v1.wgsl` to "fix" a Layer-6 FFN spike. A freshly rebuilt
  flipped binary made `ffn_out` MAE *worse* at every layer vs baseline (Layer 6: 6.02→6.24;
  Layer 35: 30.8→35.2; logits MAE 1.83→2.19). So the original polarity was correct and the
  spike is a still-open, undiagnosed bug — NOT a nibble issue. Do NOT re-flip.
- **Layer-6 FFN (Q6_K) spike REMAINS OPEN.** Root cause not found. Re-audit algebraically
  (B6) once the formula registry exists; do NOT assume it is a nibble/dequant bug.
- **Hard-copy evidence** (committed on branch): `fc_qwen3_4b_v{2,3,4}.json`,
  `fc_qwen3_0_6b.json`, `build_frontier.log`, `fc_run_v4.log`, plus `loader.rs` dummy-blob
  change retained as-is.
- **Confidence on architecture: ~100%.** Remaining work is execution, not design.

---

## 4. THE BEADS (all Fibonacci ≤ 8, self-contained)

Each bead: WHY / WHAT / ACCEPTANCE / POINTS / DEPENDS / GUARDRAIL. Dependencies also encoded
in `bd` (see tracker). IDs are `airframe-*` in `.beads`.

### airframe-69c — B0: Map current quant/arch dispatch call sites (SPIKE, CLOSED)
- **Why:** unknown exact dispatch locations risk breaking multi-quant support when replacing.
- **What:** read loader/preflight/pipeline/gpu.rs + WGSL; produce call-site map.
- **Acceptance:** `docs/bead-b0-dispatch-map.md` lists each site + what it selects.
- **Points:** 3. **Guardrail:** findings already captured; this spike is the basis for all
  other beads. Do not re-derive dispatch by intuition.

### airframe-ppo — B1: Canonical quant-type formula registry (spec-derived)
- **Why:** the referenced core must be GGUF/GGML spec math, written once per quant type,
  auditable algebraically, zero external-engine dependency. Original quantizers were built
  from the format spec, not golden traces.
- **What:** new `quant_formula` module in `airframe_observe`; table keyed by GGML quant enum
  (0=F32,1=F16,2=Q4_0,6=Q5_0,8=Q8_0,12=Q4_K,13=Q5_K,14=Q6_K, + any others present). Each
  entry: canonical dequant/attn/RoPE/RMSNorm formula as Rust fn + doc-comment citing the
  GGUF/GGML spec. Hang the `quant_type → formula_index` mapping onto `ModelRoutePlan`.
- **Acceptance:** compiles; each entry cites spec; unit test hand-computes one Q6_K block and
  asserts the registry fn matches.
- **Points:** 5. **Depends:** B0. **Guardrail:** formulas cite the SPEC, never candle. This
  is the math the whole stack stands on.

### airframe-0zh — B2: GGUF facts → fabric assertions (control-plane facts)
- **Why:** control plane must be fact-driven, not if/then. GGUF header already has per-tensor
  quant_type/shape/offset; asserting as facts is mechanical + model-independent.
- **What:** define `TensorFact{quant_type, shape, offset, arch_params}` (alpha-keyed) in
  `airframe_observe`; on load assert one per tensor from GGUF header — no hardcoding.
- **Acceptance:** loader asserts N `TensorFact`s; unit test asserts count/fields == GGUF meta.
- **Points:** 5. **Depends:** none. **Guardrail:** facts are derived from GGUF, not from any
  hardcoded quant list in Rust.

### airframe-c90 — B3a: Fabric dispatch rule TensorFact → DispatchFact
- **Why:** replace the WGSL/Rust if/then ladder with a reactive rule; given `TensorFact`, emit
  `DispatchFact` selecting the shader via B1 registry.
- **What:** ISF rule in `airframe_observe`: on `TensorFact`, look up B1 by quant_type, emit
  `DispatchFact{shader, params, offset}`.
- **Acceptance:** rule fires per tensor; test asserts `DispatchFact` emitted for each quant
  type, selecting correct registry entry.
- **Points:** 5. **Depends:** B1, B2. **Guardrail:** this rule IS the replacement for
  `dequant_dispatch`; the dispatch *logic* lives here (spec-cited), not in WGSL.

### airframe-6gc — B3b: Retire WGSL `if (qt==)` dispatch ladder
- **Why:** `dequant_dispatch()` (`sh_layer_v1.wgsl:353-361`) duplicates dispatch logic that
  now lives in the registry; keeping it risks drift.
- **What:** replace the per-quant branches with consumption of `DispatchFact{formula_index}`
  (a uniform from B3a). Dequant functions (`dequant_q6k_elem`, etc.) stay. Mirrored ladders in
  `sh_dequant_any.wgsl` / `sh_head_blob.wgsl` retired the same way.
- **Acceptance:** shaders compile; every quant type our models use still dequantizes (verified
  by B6); no `qt ==` ladder remains in dequant entry points.
- **Points:** 5. **Depends:** B0, B1. **Guardrail:** do NOT re-implement the ladder inline.
  The shader receives a registry-derived index; it does not re-derive quant→formula.

### airframe-0q2 — B4: Wire fabric dispatch into gpu.rs (1-liner swap)
- **Why:** inference must execute through the fabric path; acceptance = 1-line change in gpu.rs.
- **What:** make `generate` call the fabric-driven path (`generate_isf`) fed by B2/B3 dispatch
  facts; keep old imperative `generate` behind a flag for A/B. The swap is one call site.
- **Acceptance:** model runs end-to-end via fabric path; output matches old path within
  tolerance on TinyLlama (the ONE validated reference).
- **Points:** 5. **Depends:** B2, B3a. **Guardrail:** both paths share the same forward pass,
  so output MUST match on TinyLlama — if it doesn't, the fabric path is wrong, not the old one.

### airframe-y26 — B5: DuckDB persistent fact store for dispatch control plane
- **Why:** one empirical DB driving dispatch; makes load reproducible + auditable; realizes
  the vault as a control plane (not a golden-trace store).
- **What:** persist `TensorFact` per model into `vault.duckdb`; on load hydrate from DB (or
  parse GGUF → assert → store). Document schema column.
- **Acceptance:** facts load from DB; doc describes schema; test shows GGUF→DB→facts round-trip.
- **Points:** 5. **Depends:** B2. **Guardrail:** the DB stores FACTS (quant_type/shape/offset),
  never golden outputs.

### airframe-ed6 — B6: Algebraic audit harness (formula vs shader per quant type)
- **Why:** self-validation without golden traces — the "stand on our own mathematical feet"
  gate. Prove each shader implements the spec formula, independent of any external engine.
- **What:** extend invariant cage/tests to assert each shader's `PerTensorOutput` matches the
  B1 canonical formula numerically (unit test per quant type: hand-computed block vs shader).
- **Acceptance:** passes per quant type; this — not candle — is the certification gate.
- **Points:** 5. **Depends:** B1. **Guardrail:** the comparison is formula-vs-shader, BOTH
  ours. External engines are not the arbiter.

### airframe-937 — B7: Retire dual orchestrators / dead forward-pass paths
- **Why:** `docs/architecture-map.md` debt #1 — `generate` (imperative) stacked on `generate_isf`
  (reactive); two control styles for one pipeline.
- **What:** after B6 green, remove old imperative `generate` body (or keep only behind the
  flag from B4); single inference facade.
- **Acceptance:** single orchestrator; `cargo test` green; no dead duplicate forward-pass code.
- **Points:** 3. **Depends:** B4 (and B6). **Guardrail:** keep `generate_isf`; delete the
  imperative twin. Do not keep both as "fallbacks" — that is the haunted house.

### airframe-3z9 — B8: TinyLlama certification gate (trusted reference)
- **Why:** TinyLlama q4_0/q6_k is the ONLY model where airframe's own CPU path was validated;
  the one model we can certify against our own math.
- **What:** run full fabric-driven pipeline on TinyLlama q4_0 and q6_k; assert output matches
  the vault golden within tolerance.
- **Acceptance:** green on both quant variants.
- **Points:** 3. **Depends:** B4, B6. **Guardrail:** this is the ONLY accepted reference. Do
  not certify against Qwen3/candle.

---

## 5. REFERENCES
- D0 engine spec: copied to `.beads/reference/D0-ENGINE-ARCHITECTURE.md` (gitignored). Source:
  `dzero-cas/D0-ENGINE-ARCHITECTURE.md`.
- Dispatch map: `docs/bead-b0-dispatch-map.md` (committed).
- GGUF/GGML quant spec: quant enum + block layouts; block sizes already in
  `sh_layer_v1.wgsl:343-346`. Canonical dequant fns currently in `dequant_q*_elem` (sh_layer_v1
  + sh_dequant_any) — the CURRENT (to-be-audited) impls; B1 writes spec-cited canonical forms,
  B6 proves they match.
- Control plane: `src/core/routing.rs` (`ModelRoutePlan`) — home for the quant→formula mapping.
