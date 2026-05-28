<div align="center">

# libfse

### Fused Semantic Execution — Fail-Closed Policy Engine

[![Crates.io](https://img.shields.io/crates/v/libfse.svg)](https://crates.io/crates/libfse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://rustup.rs/)
[![Part of Airframe](https://img.shields.io/badge/part%20of-Airframe-blueviolet)](https://github.com/Michael-A-Kuykendall/airframe)

**Policy is data. Decisions are fused. No evaluation logic can be skipped.**

</div>

---

> **Patent Notice**: This crate implements inventions covered by a pending US patent application for Fail-Closed Semantic Enforcement methods. Open-source use is permitted under the MIT license for non-commercial, evaluation, or internal research purposes. Commercial use or embedding in products requires a separate license — contact michaelallenkuykendall@gmail.com.

Copyright © 2026 Michael Kuykendall / DZERO. All rights reserved.

---

## The Core Idea

Most policy engines follow this pattern:

```
scan(input) → Vec<Match> → caller decides what to do
```

The problem: the caller's decision logic can have bugs. A missed branch, an off-by-one, an early return — and a policy rule is silently skipped.

FSE inverts this:

```
compile(rules) → FseMap (fused DFA + opcode table)
scan(input, map) → Ok(summary) | Err(Violation)   ← no decision left to the caller
```

Rules are compiled into a fused executable. The scanning kernel **is** the decision. There is no post-processing step where enforcement can be bypassed.

This is not an optimization. It is an architectural guarantee.

---

## Quick Start

```toml
[dependencies]
libfse = "0.1"
```

```rust
use libfse::{FseMap, FseScanner, Rule, FseOpcode};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define rules
    let rules = vec![
        Rule { pattern: b"DROP TABLE".to_vec(),  opcode: FseOpcode::Reject(0) },
        Rule { pattern: b"SELECT *".to_vec(),    opcode: FseOpcode::Record(1) },
        Rule { pattern: b"-- comment".to_vec(),  opcode: FseOpcode::Ignore },
    ];

    // 2. Compile once (reuse FseMap across threads via Arc)
    let map = FseMap::compile(rules)?;

    // 3. Create a cursor (per-stream state, no borrow of map)
    let mut cursor = map.new_cursor()?;

    // 4. Scan — fails closed on Reject, records on Record
    let summary = map.scan_with_cursor(&mut cursor, b"SELECT * FROM users")?;
    println!("rules fired: {}", summary.rules_recorded);

    // This returns Err(Violation::PolicyReject { .. })
    let result = map.scan_with_cursor(&mut cursor, b"DROP TABLE users");
    assert!(result.is_err());

    Ok(())
}
```

---

## Design

### Inverted Control Flow

Traditional scanner → caller decides:
```
for m in aho_corasick.find_iter(input) {
    match m.pattern_id() {        // ← can forget a case
        0 => reject(),
        1 => record(),
        _ => {}                   // ← silent miss
    }
}
```

FSE: the opcode is baked into the DFA state table at compile time:
```
for byte in input {
    state = dfa.next(state, byte);
    for action in state_action_table[state] {   // ← precomputed, exhaustive
        match action {
            Reject { .. } => return Err(Violation),  // immediate, no escape
            Record { .. } => set_bit(cursor, rule),
            Ignore        => {}
        }
    }
}
```

No match object is created. No iterator is returned. No caller logic is involved in enforcement.

### ScanCursor: Borrowless Persistent State

```rust
// Store alongside Arc<FseMap> without lifetime parameters
struct StreamGuard {
    map:    Arc<FseMap>,
    cursor: Mutex<ScanCursor>,
}

impl StreamGuard {
    fn check(&self, delta: &[u8]) -> Result<ScanSummary, Violation> {
        self.map.scan_with_cursor(&mut self.cursor.lock(), delta)
    }
}
```

The cursor holds only the DFA walker state and rule bitset — no reference to the `FseMap`. This makes it trivial to store in structs that also hold an `Arc<FseMap>` without fighting the borrow checker.

### Fail-Closed Integrity

Two error modes, both treated as hard failures:

| Error | Trigger | Meaning |
|---|---|---|
| `Violation::PolicyReject` | A `Reject` rule matched | Policy explicitly blocked the input |
| `Violation::IntegrityError` | Bounds error in action table | Compiler or runtime state is corrupt — fail closed |

After any `Err`, the cursor is considered poisoned. Create a new one.

---

## Performance

Benchmarked against raw `aho-corasick` iterator on a 7KB payload with 12 rules (rigorous methodology: full automaton + rule state reset per iteration):

```
Scanner Comparison/libfse_scan          time: [89.6 µs  90.1 µs  90.7 µs]
Scanner Comparison/aho_corasick_find_iter  time: [118.4 µs 120.1 µs 121.7 µs]
```

**~27% lower latency** than the standard iterator path. The advantage comes from eliminating the `Match` object allocation and the caller dispatch loop — both are fused into the DFA state table at compile time.

Rule count has near-zero impact on per-byte scan cost once compiled, because all rules share the same DFA traversal.

Run benchmarks:
```bash
cargo bench
```

---

## Security Properties

- **DoS protection**: `RuleId` is hard-capped at 65535. `FseMap::compile` returns `Err(BuildError::RuleIdTooLarge)` on violation. Max memory per scanner: ~8KB regardless of rule count.
- **No unsafe code** in the hot path.
- **Zero heap allocation** during scanning. Verified by `test_zero_alloc_in_hot_loop` (custom global allocator panics on any alloc in scan scope).
- **No regex backtracking**: DFA guarantees linear scan time — no catastrophic backtracking possible.

---

## API Reference

### `FseMap` — the compiled policy

```rust
FseMap::compile(rules: Vec<Rule>) -> Result<FseMap, BuildError>
FseMap::new_cursor(&self) -> Result<ScanCursor, ScanError>
FseMap::scan_with_cursor(&self, cursor: &mut ScanCursor, input: &[u8])
    -> Result<ScanSummary, Violation>
```

### `Rule` — input definition

```rust
Rule {
    pattern: Vec<u8>,   // byte pattern to match (case-sensitive)
    opcode:  FseOpcode, // what to do when matched
}
```

### `FseOpcode`

| Variant | Effect |
|---|---|
| `Ignore` | Pattern matched; no action. Useful for suppressing sub-patterns. |
| `Record(RuleId)` | Set bit `RuleId` in cursor's rule bitset. Counted in `ScanSummary`. |
| `Reject(RuleId)` | Immediately return `Err(Violation::PolicyReject)`. |
| `Control(ControlOp)` | Reset rule state or switch mode. |

### `ScanSummary`

```rust
pub struct ScanSummary {
    pub match_states_seen: u64,  // DFA states with at least one action
    pub pattern_hits:      u64,  // total Record actions fired
    pub rules_recorded:    u32,  // distinct RuleIds set in bitset
}
```

---

## Technical Background

The FSE architecture compiles independent semantic rules into a single fused DFA. The key invariant:

> **∂runtime / ∂rules ≈ 0** (for shared selectors)

This is formalized in the patent-pending specification. The full technical reconstruction (including FIG. 1–6 diagrams from the patent drawings) is at [`fused_semantic_execution_full_markdown_reconstruction.md`](../../fused_semantic_execution_full_markdown_reconstruction.md).

---

## Use Cases

- **LLM output filtering**: Scan token stream incrementally; reject on policy violation before the token is emitted
- **SQL injection detection**: Compile known injection patterns as Reject rules; scan every query in microseconds
- **Content moderation**: Compile keyword/phrase policy; scan at ingest with guaranteed enforcement
- **Audit logging**: Use Record rules to build a tamper-evident event log of which policies fired

---

## Related

- [**Airframe**](https://github.com/Michael-A-Kuykendall/airframe) — GPU inference engine that uses libfse for policy enforcement during token generation
- [**Shimmy**](https://github.com/Michael-A-Kuykendall/shimmy) — OpenAI-compatible inference server built on Airframe
