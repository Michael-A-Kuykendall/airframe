<div align="center">
  <img src="https://raw.githubusercontent.com/Michael-A-Kuykendall/airframe/main/assets/libfse-logo.png" alt="libfse" width="480" height="auto" />

  ### libfse 0.1.4 — byte-pattern policy scanning

  [![Crates.io](https://img.shields.io/crates/v/libfse.svg)](https://crates.io/crates/libfse)
</div>

**Languages / 语言 / 語言:** English · [简体中文](docs/zh-CN/README.md) · [繁體中文](docs/zh-TW/README.md)

> **License:** [Apache-2.0](LICENSE), including commercial use, modification, redistribution and embedding. The [additional license grant for libfse 0.1.4](LICENSE-GRANT-0.1.4.md) identifies the exact published archive and supersedes its earlier noncommercial restrictions for the licensor's rights. No separate commercial permission is required. The immutable crates.io archive and historical registry metadata remain unchanged.
>
> **Patent scope:** Apache-2.0 section 3 applies to covered claims licensable by contributors. See [NOTICE](NOTICE) for original attribution; third-party rights remain applicable.

Copyright © 2026 Michael Kuykendall / DZERO. All rights reserved.

## What 0.1.4 provides

`FseMap::compile` compiles byte patterns and actions into a shared automaton.
`FseMap::scan_with_cursor` traverses incoming bytes and executes the actions
associated with each reached state. It returns a scan summary or a violation.
The caller must actually invoke the scanner and honor its result before
emitting or acting on rejected data. This API cannot force a caller to do so.

This source documents the existing scanner and additional permissive grant; it does not
add an inference engine, a general semantic understanding system, or a new
security certification. See [CHANGELOG](CHANGELOG.md) and the [workspace roadmap](https://github.com/Michael-A-Kuykendall/airframe/blob/main/ROADMAP.md).

## Library example

Review the license before adding the dependency:

```toml
[dependencies]
libfse = "0.1.4"
```

```rust
use libfse::{FseMap, FseOpcode, Rule};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rules = vec![
        Rule { pattern: b"DROP TABLE".to_vec(), opcode: FseOpcode::Reject(0) },
        Rule { pattern: b"SELECT *".to_vec(), opcode: FseOpcode::Record(1) },
    ];
    let map = FseMap::compile(rules)?;
    let mut cursor = map.new_cursor()?;
    let summary = map.scan_with_cursor(&mut cursor, b"SELECT * FROM users")?;
    println!("distinct recorded rules: {}", summary.rules_recorded);
    assert!(map.scan_with_cursor(&mut cursor, b"DROP TABLE users").is_err());
    Ok(())
}
```

This is a library, with no installed command-line executable.

## State, errors and limits

Use a separate `ScanCursor` per logical stream. The cursor retains automaton
position and rule-recording state across chunks without borrowing the map.
Discard or deliberately reset request state at request boundaries; do not
reuse one stream's state as if it belonged to another.

- `Reject` returns `Violation::PolicyReject`; the caller must stop or reject the
  affected operation.
- Inconsistent action-table indices return `Violation::IntegrityError`; treat
  this as a hard failure, not permission to continue.
- `Record` records a rule in the cursor; `Ignore` performs no policy action.
  Neither creates a tamper-evident audit log.
- This is byte-pattern matching, not a complete SQL injection, moderation,
  authorization or adversarial-input defense.

One automaton traversal is shared by compiled rules, but matched-state actions
are still visited. Total work depends on input bytes **and actions executed**;
there is no constant-in-rule-count or universal throughput guarantee. Cursor
bitsets, automaton construction and map storage have separate memory costs.
Benchmarks in this repository are workload-specific experiments, not a 0.1.4
performance promise.

## Relationship to Airframe Observe

libfse implements **policy scanning**: bytes, patterns, actions and violations.
**airframe-observe 0.1.4** coordinates inference facts, observation and host-side
callbacks using Dzero; those contracts do not implement or prove GPU numerical
correctness. Airframe's policy controller consumes libfse results during text
generation. These libraries have different consumed APIs and licensing; they
are not interchangeable or a claim of unbypassable whole-system enforcement.

- [Airframe](https://github.com/Michael-A-Kuykendall/airframe)
- [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy)
