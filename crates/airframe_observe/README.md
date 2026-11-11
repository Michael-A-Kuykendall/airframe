# airframe-observe 0.1.4

**Languages / 语言 / 語言:** English · [简体中文](docs/zh-CN/README.md) · [繁體中文](docs/zh-TW/README.md)

Inference facts, observers and host-side coordination for Airframe.

## Release responsibilities

ObservationSession registers observers, accepts layer/logit/text and dispatch-timing facts, and runs saturation. ISF uses Dzero FactStore, ClosureProgram, RunBudget and SaturationFabric with caller-provided embedding, prefill, decode and control callbacks. This includes coordination, not merely passive logging. Observers consume facts that the engine actually emits; enabling observation does not independently validate the producer.

The Cargo requirement is dzero = "0.1"; this coordinated candidate resolves **Dzero 0.1.0**. airframe-observe does not depend on Airframe: the engine consumes these contracts. See [CHANGELOG](CHANGELOG.md) and the [workspace roadmap](https://github.com/Michael-A-Kuykendall/airframe/blob/main/ROADMAP.md).

## Correctness and performance boundary

These APIs do not implement GPU matrix arithmetic, validate model metadata, certify logits, fix driver failures or guarantee latency. Registered callbacks and fact processing have costs; observer count is not a universal constant-cost promise. Numerical certification requires a separately executed engine/reference comparison on exact fixtures and configurations.

libfse performs byte-pattern policy scanning and returns violations; its [additional Apache-2.0 grant](https://github.com/Michael-A-Kuykendall/airframe/blob/8efb1e7a830b0822cfc5ac1b121e149948d5ffe3/crates/libfse/LICENSE-GRANT-0.1.4.md) permits commercial use without separate permission. This crate coordinates inference facts and observers using Dzero. Their responsibilities and licenses are distinct.

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

[MIT](LICENSE). See [stack licensing](https://github.com/Michael-A-Kuykendall/airframe/blob/366c8aa429116cf305e04c9dde182a842040b4b1/LICENSING.md) for the distinct component licenses, third-party obligations and immutable archive/additional-grant boundary.
