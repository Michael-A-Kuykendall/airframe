# airframe-conformance

Independent conformance verification for the Airframe GPU inference engine.

**This crate is TEST-ONLY and MUST NOT be a production dependency.**

## Purpose

This crate provides an independent conformance verification system that can:
- Capture tensor values from production Airframe engines (via telemetry only)
- Capture tensor values from reference engines (candle, llama.cpp)
- Compare captures against numerical tolerances
- Produce evidence packages with full provenance

The key architectural principle: **conformance code must never import production Airframe implementation code**. It may only depend on specification APIs.

## Dependency Boundary

### Allowed Dependencies

Conformance code MAY import from these specification-only modules:

- `airframe::capture::spec` — Capture point definitions and coordinate plans (specification only)
- `airframe::capture::telemetry` — Telemetry emission traits (no implementation)

These modules contain only type definitions, traits, and constants — no implementation logic.

### Forbidden Dependencies

Conformance code MUST NEVER import from these production modules:

- `airframe::semantic` — Model semantic analysis
- `airframe::loader` — GGUF loading
- `airframe::dispatch` — Kernel dispatch
- `airframe::offset` — Offset computation
- `airframe::cache` — KV cache management
- `airframe::capture::production` — Production capture implementation
- `airframe::inference` — Inference pipeline
- `airframe::quant` — Quantization/dequantization
- `airframe::rope` — RoPE implementation
- `airframe::rms_norm` — RMSNorm implementation
- `airframe::attention` — Attention implementation
- `airframe::ffn` — FFN implementation
- `airframe::lm_head` — LM head implementation

**Rationale:** If conformance code imports production implementation, a bug in production becomes a bug in the oracle — the conformance check becomes meaningless.

## Telemetry-Only Capture

Production Airframe engines emit capture data through a **telemetry trait** defined in `airframe::capture::telemetry`. The conformance crate consumes this telemetry but has no access to the production capture implementation.

This ensures:
1. Capture points are defined by specification (in `airframe::capture::spec`)
2. Production engines implement the telemetry trait to emit values
3. Conformance only sees the emitted values, never the capture logic
4. A bug in production capture logic cannot silently corrupt the oracle

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        airframe-conformance                      │
│  (test-only crate, not published as production dependency)       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │   Manifest  │  │   Capture    │  │   Comparison           │  │
│  │   Schema    │  │   Protocol   │  │   Engine               │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘  │
│         │                │                      │                │
│         ▼                ▼                      ▼                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Evidence Package Generator                     │  │
│  │  (manifest + captures + declared_inputs + comparisons)     │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ telemetry trait (spec only)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Production Airframe                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Inference  │  │   Capture    │  │  Telemetry Emitter     │  │
│  │  Pipeline   │──│  Implementation│─▶│  (implements spec)     │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Schemas

All data structures are versioned with JSON Schema (Draft 2020-12):

| Schema | File | Purpose |
|--------|------|---------|
| Manifest | `schemas/manifest.schema.json` | Conformance run configuration |
| Capture | `schemas/capture.schema.json` | Engine capture output |
| Declared Input | `schemas/declared_input.schema.json` | Tokenizer provenance |
| Build Provenance | `schemas/build_provenance.schema.json` | Shimmy binary provenance |
| Comparison | `schemas/comparison.schema.json` | Engine comparison results |
| Evidence | `schemas/evidence.schema.json` | Complete evidence package |

## Validation

Run all validation gates:

```bash
# From airframe workspace root
cargo check -p airframe-conformance
cargo test -p airframe-conformance dependency_policy capture_protocol -- --test-threads=1
python docs/conformance/validate_schemas.py
python scripts/check_conformance_docs.py
```

## Gates

| Gate | Command | Purpose |
|------|---------|---------|
| 1 | `cargo check -p airframe-conformance` | Crate compiles |
| 2 | `cargo test -p airframe-conformance dependency_policy` | Dependency boundary enforced |
| 3 | `python docs/conformance/validate_schemas.py` | All schemas valid |
| 4 | `cargo test -p airframe-conformance capture_protocol` | Capture protocol works |
| 5 | `python scripts/check_conformance_docs.py` | Architecture docs correct |

## Versioning

Schema versions are embedded in the `$schema` field (e.g., `airframe.conformance.manifest.v1`).
Breaking changes require a new schema version (v2, v3, ...).
The conformance crate version tracks the implementation version.

## Provenance Binding

Every conformance run binds:
- **Model provenance**: GGUF metadata + file hash
- **Tokenizer provenance**: Tokenizer name, version, config hash, chat template
- **Build provenance**: Shimmy version, git commit, Airframe version, features
- **Capture provenance**: Engine, version, git commit, config hash

This ensures evidence packages are fully reproducible and auditable.