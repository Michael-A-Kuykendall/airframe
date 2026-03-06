# The Redo Switch: Implementation Status

## Critical Analysis
The original request to "just fix it" revealed a structural blocker in `shimmy_server.rs`.
The **Two-Phase Commit** (Propose -> Evaluate -> Commit) is strictly required for the Redo Switch to work.

### Current State (Before Edit)
```rust
let next = sampler.sample(&logits)?;
// ... bits of logic ...
logits = engine.decode(next, ...); // COMMIT happens here
```

### The Blocker
To implement `Redo`, we need to inspect the `logits` and the `candidate` *before* deciding to call `engine.decode`.
If we detect a violation, we must:
1.  **NOT** call `engine.decode` with the bad token.
2.  **Pick** a different token (e.g., via a different sampling strategy) from the *same* logits.
3.  **Then** call `engine.decode` with the new token.

### The Path Forward
I have edited `shimmy_server.rs` to clearly visualize this flow, but I did **not** enable the Redo logic yet because the `Sampler` struct in `libshimmy` is opaque (I cannot easily ask it for "Top-2" without modifying the library or manually parsing logits).

To fully realize Experiment 4 ("Self-Healing"), we must:
1.  Expose `argmax` or a `sample_with_strategy` method on the `logits` vector directly.
2.  Use that in the `else` block of the safety check.

For now, the **Invariant Contract** is documented in `contracts/INVARIANT_REDO_SWITCH.md` and the server code is structurally ready for the logic injection.
