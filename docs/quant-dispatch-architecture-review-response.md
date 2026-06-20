# Response to Grok Review — Quant Dispatch Architecture

## Review received 2026-06-19. Responses inline.

---

## Overall Assessment

Agreed with both problem statements and the proposed direction.

---

## Answers Feedback

### Question 1 (WGSL switch support)

Fair that it needs testing on my hardware. I will add a compile verification step to the implementation plan.

**My call:** I slightly prefer the `build.rs` codegen approach now, for the same reason the reviewer states — full control, no WGSL-vendor uncertainty, and the generated code is trivially verifiable (it's just if/else chains, same as today, but emitted from one source of truth). The FSE invariant (one place to edit) is preserved regardless of whether the output is a `switch` or generated `if/else`.

### Question 2 (Build.rs vs switch)

Agreed. `build.rs` codegen from `quant_types.toml` is the safer bet given the WGSL uncertainty. The generated code is boring if/else — easy to debug, easy to read in the output artifact.

### Question 3 (Selector array design)

Strong agreement with explicit struct. `qt.attn_out` > `qt[3]`. I'll use:

```wgsl
struct QuantSelectors {
    qk:    u32,
    v:     u32,
    attn_out: u32,
    gate:  u32,
    up:    u32,
    down:  u32,
}
```

### Question 4 (Together vs separate)

The reviewer says "do them together" then immediately says "stage it internally." Slight contradiction.

**My call:** Do them as **one coordinated change** (single MR, single branch), but with two clear internal phases:

1. Phase A: Add `dequant_dispatch()` switch table (or generated equivalent), keeping the old packed u32 selector. All existing models pass unchanged.
2. Phase B: Replace packed u32 with `QuantSelectors`, update all kernel selector reads. Fixes the proxy problem.

This gives a clean intermediate checkpoint between phases while maintaining atomicity at the MR level.

---

## Folding In: Changes to the Plan

### 1. Assertion for Unknown Quant Types (Accepted)

Add a compile-time or runtime guard. When a quant type is encountered that no branch handles, **panic/error instead of silently falling through to Q4_0**. This would have caught the Q5_0 miss in under 5 minutes instead of 2 weeks.

**Implementation sketch** (in Rust, pre-shader):
```rust
// In inference.rs / layer dispatch
let supported = [0u8, 1, 2, 6, 8, 12, 13, 14];
assert!(
    supported.contains(&quant_type),
    "Unsupported quant type {} for tensor {}",
    quant_type, tensor_name
);
```

### 2. Regression Test (Accepted)

Add Qwen2-0.5B to the smoke test suite. Include a check:
- All 24 layers produce finite output (`gpu_non_finite == 0`)
- Logits are finite

### 3. Build.rs Codegen (New Preferred Path)

Design the `quant_types.toml` schema:

```toml
[Q5_0]
type_num = 6
elems_per_block = 32
bytes_per_block = 22
dequant_fn = "dequant_q5_0_elem"

[Q8_0]
type_num = 8
elems_per_block = 32
bytes_per_block = 34
dequant_fn = "dequant_q8_0_elem"
# ...
```

`build.rs` reads this and emits `dispatch_generated.wgsl` containing:
```wgsl
fn dequant_dispatch(qt: u32, block_base: u32, elem: u32) -> f32 {
    if (qt == 6u) { return dequant_q5_0_elem(block_base, elem); }
    else if (qt == 8u) { return dequant_q8_0_elem(block_base, elem); }
    // ... all from config
    else { /* assertion path or panic in WGSL (debug) */ }
}
```

---

## Clarifying Question

The reviewer mentions the staged plan but doesn't address **how the generated dispatch function handles non-uniform activation expressions** across kernels. Current code shows:

- QKV: `dot += temp_state[temp_base + col] * dequant(...)` — no RMS/norm
- AttnProj: `dot += temp_state[temp_base + col] * dequant(...)` — same
- FFNProj: `dot += activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col] * dequant(...)` — has RMS + norm
- FFNDown: `dot += (val_gate * val_up) * dequant(...)` — product of two values
- head_blob: `dot += act_in[b * 32u + e] * dequant(...)` — uses `act_in`, not `temp_state`

The dispatch function can only encapsulate the **dequant arithmetic** (reading bytes, extracting nibbles, computing scale * value). The **activation multiplication** stays in each kernel because it differs. This means the dispatch function signature is:

```wgsl
fn dequant_dispatch(qt: u32, block_base: u32, elem: u32) -> f32
```

And each kernel calls:
```wgsl
// QKV:
dot += temp_state[temp_base + col] * dequant_dispatch(qt, bb, e);

// FFNProj:
dot += (activation_in[act_base + col] * rms * norm_bank[norm_offset_base + col])
        * dequant_dispatch(qt, bb, e);

// FFNDown:
dot += (val_gate * val_up) * dequant_dispatch(qt, bb, e);
```

This is fine — the duplication elimination is in the dequant branching (the 5 copies of the if/else chain), not in the outer multiply. **The dispatch function is correct as specified.**

---

## Summary of Folds

| Change | Source | Status |
|--------|--------|--------|
| Prefer `build.rs` codegen over raw WGSL `switch` | Review Q2 | Accepted |
| Use explicit `QuantSelectors` struct over array | Review Q3 | Accepted |
| Stage as one MR, two phases (not simultaneous) | Review Q4 | Clarified |
| Add assertion for unknown quant types | Review Rec 3 | Accepted & drafted |
| Add regression test for Q5_0 models | Review Rec 4 | Accepted |
| Dequant-only dispatch (activation stays in kernel) | My analysis | Confirmed correct |
