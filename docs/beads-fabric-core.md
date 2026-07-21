# Inference Fabric-Core Refactor — Bead Plan

Promote `airframe_observe::isf` (the patent-pending Inference Saturation Fabric) from an
*observation* layer into the *control core* of inference dispatch. Dispatch becomes
data-driven: a `quant_type → canonical formula → shader/kernel` registry, selected by
reactive fabric rules from GGUF-derived facts, with DuckDB as the persistent fact store.
The hardcoded `if (qt==)` ladders in the WGSL/Rust are retired. Adoption in `gpu.rs` is a
single 1-liner swap. Validation is algebraic against the GGUF/GGML spec — NOT against any
external engine (candle/llama.cpp). Golden traces are out; the math is the referenced core.

---

## Map current quant/arch dispatch call sites (exploration spike)

**Why:** The quant_type and architecture currently drive shader/kernel selection through a
scattered set of hardcoded branches. Their exact locations and what each selects are unknown.
We must map them before replacing the paths, or we will break multi-quant support.

**What:** Read `src/backend/bindless/loader.rs`, `preflight.rs`, `pipeline/mod.rs`,
`pipeline/dequant.rs`, and `gpu.rs` (`generate` + `generate_isf`). Produce a written
call-site map (file:line) of: (a) every `quant_type` branch, (b) every architecture branch
(qwen/llama/rope-base/norm-eps/head_dim), (c) where `generate` and `generate_isf` diverge.
No code change.

**Acceptance:** A doc (`docs/bead-b0-dispatch-map.md`) lists each call site, the
quant_type/arch it handles, and what shader/kernel it selects. No code modified.

**Fibonacci points:** 3

**Depends on:** none (do first)

---

## Canonical quant-type formula registry (spec-derived)

**Why:** The referenced core must be the GGUF/GGML spec math, written once per quant type,
auditable algebraically, with zero dependency on any external engine. This is how quantizers
were originally built — from the format spec, not golden traces. Every dispatch rule and the
audit harness reference this table.

**What:** In `airframe_observe` (new `quant_formula` module), create a table keyed by the GGML
quant enum (0=F32, 1=F16, 2=Q4_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K, plus any others present
in our models). Each entry: the canonical dequant/attention/RoPE/RMSNorm formula as a Rust fn
+ a doc-comment citing the exact GGUF/GGML spec source. Enumerate the types from the GGUF
headers found in the B0 map. Independent of candle/llama.cpp.

**Acceptance:** Table compiles; each entry carries a spec citation; a unit test hand-computes
one Q6_K block from the spec and asserts the registry fn matches.

**Fibonacci points:** 5

**Depends on:** B0 (uses the enumerated quant types)

---

## GGUF facts → fabric assertions (control-plane facts)

**Why:** The control plane must be fact-driven, not if/then. The GGUF header already contains,
per tensor, its quant_type, shape, and byte offset; asserting these as facts is mechanical and
model-independent. This is the single empirical "database" that drives dispatch.

**What:** Define `TensorFact { quant_type, shape, offset, arch_params }` (alpha-keyed) in
`airframe_observe`. On model load, assert one `TensorFact` per tensor, derived from the GGUF
header — no hardcoding of quant types in code.

**Acceptance:** Loader asserts N `TensorFact`s for a known model; a unit test asserts the count
and fields equal the GGUF metadata.

**Fibonacci points:** 5

**Depends on:** none (can run parallel with B1)

---

## Fabric dispatch rule: TensorFact → DispatchFact

**Why:** Replace the WGSL/Rust if/then ladder with a reactive rule. Given a `TensorFact`, the
rule emits a `DispatchFact` selecting the shader/kernel registered for that quant_type. This is
the core move that makes the saturation fabric drive inference.

**What:** Write an ISF rule in `airframe_observe` that, on `TensorFact`, looks up B1's registry
by `quant_type` and emits `DispatchFact { shader, params, offset }`.

**Acceptance:** Rule fires per tensor; a test asserts a `DispatchFact` is emitted for each
quant_type present, selecting the correct registry entry.

**Fibonacci points:** 5

**Depends on:** B1, B2

---

## Retire WGSL `if (qt==)` dispatch ladder

**Why:** The hardcoded `if (qt==14u){Q6_K} else if (qt==13u){Q5_K} …` chains in
`sh_dequant_any.wgsl`, `sh_layer_v1.wgsl`, `sh_head_blob.wgsl` duplicate dispatch logic that
now lives in the registry. Keeping them risks drift between shader and registry.

**What:** Replace the per-quant branches in those shaders with a single path that receives the
selected shader via `DispatchFact` (or, if per-quant shaders are kept, select them via the
registry and delete the inline `qt` ladder). Ensure all quant types still dequantize.

**Acceptance:** Shaders compile; every quant type our models use still dequantizes correctly
(verified by the B6 unit test); no `qt ==` ladder remains in the dequant entry point.

**Fibonacci points:** 5

**Depends on:** B0, B1

---

## Wire fabric dispatch into gpu.rs (1-liner swap)

**Why:** Inference must actually execute through the fabric path. The acceptance criterion is
that adopting it is a single call-site change in `gpu.rs`.

**What:** In `gpu.rs`, make `generate` call the fabric-driven path (`generate_isf`) fed by the
B2/B3 dispatch facts; keep the old imperative `generate` reachable behind a flag for A/B
comparison. The swap is one line.

**Acceptance:** A model runs end-to-end via the fabric path; output matches the old path within
tolerance on TinyLlama (the one validated reference).

**Fibonacci points:** 5

**Depends on:** B2, B3a

---

## DuckDB persistent fact store for dispatch control plane

**Why:** The user wants one empirical DB driving dispatch. Persisting the `TensorFact` table per
model makes load reproducible and the dispatch auditable, and realizes the vault as a control
plane (not a golden-trace store).

**What:** Persist `TensorFact` (quant_type, shape, offset, arch) per model into `vault.duckdb`;
on load, hydrate facts from the DB (or parse GGUF → assert, then store). Document the added
schema column.

**Acceptance:** A model's facts load from DB; doc describes the schema; a test shows facts
round-trip GGUF → DB → facts.

**Fibonacci points:** 5

**Depends on:** B2

---

## Algebraic audit harness: formula vs shader per quant type

**Why:** Self-validation without golden traces — the "stand on our own mathematical feet" gate.
Prove each shader implements the spec formula, independent of any external engine.

**What:** Extend the invariant cage / tests to assert each shader's emitted `PerTensorOutput`
matches the B1 canonical formula numerically (a unit test per quant type: hand-computed block
vs shader output).

**Acceptance:** Test passes per quant type; this is the certification gate, not candle.

**Fibonacci points:** 5

**Depends on:** B1

---

## Retire dual orchestrators / dead forward-pass paths

**Why:** `docs/architecture-map.md` debt #1: `generate` (imperative) is stacked on `generate_isf`
(reactive) — two control styles for one pipeline. After B6 proves the fabric path, the
duplicate must go.

**What:** After B6 is green, remove the old imperative `generate` body (or keep it only behind
the flag from B6); single inference facade.

**Acceptance:** Single orchestrator; `cargo test` green; no dead duplicate forward-pass code.

**Fibonacci points:** 3

**Depends on:** B6 (and B4)

---

## TinyLlama certification gate (trusted reference)

**Why:** TinyLlama q4_0/q6_k is the ONLY model where airframe's own CPU path was validated;
every other model is self-derived. It is the one model we can certify against our own math.

**What:** Run the full fabric-driven pipeline on TinyLlama q4_0 and q6_k; assert output matches
the vault golden within tolerance.

**Acceptance:** Green on both quant variants.

**Fibonacci points:** 3

**Depends on:** B6, B4
