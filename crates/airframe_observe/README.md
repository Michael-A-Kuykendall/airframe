# airframe-observe 0.1.4

Inference facts, observers and host-side coordination for Airframe.

## Release responsibilities

ObservationSession registers observers, accepts layer/logit/text and dispatch-timing facts, and runs saturation. ISF uses Dzero FactStore, ClosureProgram, RunBudget and SaturationFabric with caller-provided embedding, prefill, decode and control callbacks. This includes coordination, not merely passive logging. Observers consume facts that the engine actually emits; enabling observation does not independently validate the producer.

The Cargo requirement is dzero = "0.1"; this coordinated candidate resolves **Dzero 0.1.0**. airframe-observe does not depend on Airframe: the engine consumes these contracts. See [CHANGELOG](CHANGELOG.md) and the [workspace roadmap](https://github.com/Michael-A-Kuykendall/airframe/blob/main/ROADMAP.md).

## Correctness and performance boundary

These APIs do not implement GPU matrix arithmetic, validate model metadata, certify logits, fix driver failures or guarantee latency. Registered callbacks and fact processing have costs; observer count is not a universal constant-cost promise. Numerical certification requires a separately executed engine/reference comparison on exact fixtures and configurations.

libfse performs byte-pattern policy scanning and returns violations; it has separate noncommercial licensing. This crate coordinates inference facts and observers using Dzero. Their responsibilities and licenses are distinct.

```toml
[dependencies]
airframe-observe = "0.1.4"
```

## Patent Notice

This crate implements the Fused Semantic Execution (FSE) Inference Observation
architecture - a selector-first, compile-once, single-pass model for capturing and
broadcasting inference state to multiple independent observers simultaneously. This
architecture and its application to AI inference observability is covered by a pending
US patent by Michael A. Kuykendall. All rights reserved.

Contact: michaelallenkuykendall@gmail.com

## License

MIT
