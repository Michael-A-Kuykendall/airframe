# libfse — Fused Semantic Execution Kernel

> **Patent Notice**: This crate implements inventions covered by a pending US patent application for Fail-Closed Semantic Enforcement methods. Commercial use, embedding in products, or creation of derivative works requires a separate commercial license from the author. Open-source use is permitted under the MIT license for non-commercial, evaluation, or internal research purposes only. Contact michaelallenkuykendall@gmail.com for licensing inquiries.

Copyright © 2026 Michael Kuykendall / DZERO. All rights reserved.

## Overview
LibFSE is a "Fail-Closed" policy enforcement engine designed for high-security stream scanning. Unlike traditional regex engines (`aho-corasick`) that return matches for the caller to handle, FSE *inverts* this relationship: the scanning kernel itself executes policy opcodes (Record, Reject, Ignore) inside the hot loop.

This ensures that no policy decision can be skipped due to logic errors in the glue code.

## Key Features
- **Inverted Control Flow**: Policy is data, not code.
- **Zero-Allocation**: Hot loop uses `BitVec` and static cursors. No heap defaults.
- **Fail-Closed**: Unknown opcodes or "Reject" instructions terminate scan immediately (`Violation::IntegrityError`).
- **Fused Action Tables (V4)**: State-to-Action mapping is precomputed. No runtime `Match` object overhead.
- **Precomputed Bitmasks**: Rule recording uses single-instruction bitwise ORs (no bit-shift arithmetic).

## Performance
- **Optimized**: ~3-4% faster than standard `aho-corasick` on match-heavy workloads.
- **Constant Time Policy**: Policy complexity (number of rules) has minimal impact on per-byte overhead due to fused execution.

## Usage
```rust
use libfse::{FseMap, Rule, FseOpcode, FseScanner};

let rules = vec![
    Rule::new("admin", FseOpcode::Reject),
    Rule::new("log", FseOpcode::Record(1)),
];
let map = FseMap::compile(rules)?;
let mut scanner = FseScanner::new(&map)?;

// Will return Violation::IntegrityError if "admin" is found
scanner.scan(b"user: admin login")?; 
```

## Testing
Run the suite with strict allocation checks:
```bash
cargo test
```
The suite includes a custom `#[global_allocator]` test to prove zero heap activity during scanning.
