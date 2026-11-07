# Changelog

## 0.1.4 additional license grant — 2026-11-07

- Michael A. Kuykendall additionally licenses owned libfse code under Apache-2.0, including commercial use, modification, redistribution and incorporation into products. No separate commercial permission is required. See [the exact version/checksum grant](LICENSE-GRANT-0.1.4.md), [complete license](LICENSE) and [NOTICE](NOTICE).
- The published 0.1.4 archive, original license metadata and original tag remain unchanged. Earlier restrictions described in the original archive history below are superseded by the additional grant to the extent of the licensor's rights. Third-party rights remain applicable. No runtime, dependency or version change.

## [0.1.4] — 2026-11-06

- Documentation errata: clarify caller enforcement responsibility, action-dependent scanning cost and distinct airframe-observe responsibilities. Remove unsupported performance and whole-system guarantees without changing executable code or license rights.

- Correct license metadata to the FSE Noncommercial Evaluation License in LICENSE, preserving existing commercial, product-embedding and derivative-work permission requirements, patent and third-party notices. This package is source available, not ordinary MIT.


- Version the exact library payload used by the certified Airframe workspace instead of substituting the different published 0.1.3 payload.
- Preserve existing scanner/metrics implementation and formatting; no new behavior is introduced by this release preparation.


All notable changes to `libfse` will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.1] — 2026-05-27

### Changed
- **Patent notice**: Updated all headers to clearly state pending US patent status and commercial licensing requirements.
- **README**: Complete rewrite — added design rationale, code examples, API reference, performance methodology, security properties, and use case documentation.
- **Copyright**: Updated from "AI State Pilot" to "Michael Kuykendall / DZERO".

### Added
- `examples/policy_scan.rs` — canonical quickstart demonstrating all three opcodes, ScanCursor pattern, and fail-closed semantics.

---

## [0.1.0] — 2026-05-26

### Added
- Initial public release.
- `FseMap::compile(rules)` — compile a rule set into a fused DFA + action table.
- `FseMap::new_cursor()` — borrowless persistent DFA walker state.
- `FseMap::scan_with_cursor(cursor, input)` — single-pass fail-closed scan.
- Three opcodes: `Reject`, `Record`, `Ignore`.
- `FseOpcode::Control(ControlOp)` for mode switching and rule state reset.
- DoS protection: `RuleId` hard-capped at 65535 (max ~8KB per cursor).
- Zero-allocation hot path — verified by `test_zero_alloc_in_hot_loop`.
- Benchmarks: ~27% lower latency than `aho-corasick` iterator on 7KB payload.
- `ScanSummary` returned on success: `match_states_seen`, `pattern_hits`, `rules_recorded`.
- `Violation::PolicyReject` and `Violation::IntegrityError` — fail-closed error taxonomy.

---

[0.1.1]: https://github.com/Michael-A-Kuykendall/airframe/compare/libfse-v0.1.0...libfse-v0.1.1
[0.1.0]: https://github.com/Michael-A-Kuykendall/airframe/releases/tag/libfse-v0.1.0
