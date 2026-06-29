# Quant Type Dispatch Architecture — Problem Space & Proposed Fix

## Request for Cloud Review

---

## 1. Executive Summary

The airframe inference engine has a two-layer architecture problem in how it selects quantized weight dequant routines at GPU shader runtime:

**Layer 1 — Dispatch Duplication:** Five GPU compute kernels each maintain an independent if/else chain over quant type codes. Adding a quant type requires 5 edits. We recently missed a type (Q5_0) because of this — a textbook FSE violation.

**Layer 2 — Selector Fragmentation:** The selectors these chains branch on are not uniform. Different kernels read different bit-fields of a packed u32, and one kernel silently reuses another weight's quant type as a proxy (FFNProj reads attn_output's type for ffn_gate/up weights). This is architecturally unsound: it works only because current models happen to make these types identical.

This document presents both problems with evidence, proposes FSE-aligned fixes for both tiers, and asks for review before implementation.

---

## 2. Current Architecture

### 2.1 Metadata Encoding

Each transformer layer has a single packed u32 `quant_type_packed` field:

```
bits 0-7   (>>  0):  main_qkv quant type  (Q, K projection weights, V projection weight)
bits 8-15  (>>  8):  V projection quant type (attn_v.weight — alternates per-layer in Q4_K_M)
bits 16-23 (>> 16):  ffn_down quant type  (ffn_down.weight)
bits 24-31 (>> 24):  attn_output quant type (attn_output.weight — ALSO used for ffn_gate/up)
```

This is set at compile time in `metadata.rs`:
```rust
// src/backend/bindless/metadata.rs
let lqt_v = t(&tensor_types, layer_idx, "attn_v.weight");
let lqt_down = t(&tensor_types, layer_idx, "ffn_down.weight");
let lqt_attn_out = t(&tensor_types, layer_idx, "attn_output.weight");
compiled_layers.push(CompiledLayerEntry {
    offsets,
    quant_type_packed: lqt_main
        | (lqt_v << 8)
        | (lqt_down << 16)
        | (lqt_attn_out << 24),
});
```

### 2.2 Shader Dispatch (Current, Rule-First)

Five kernels in two WGSL shader files read different bit-fields:

**`sh_layer_v1.wgsl` — 4 kernels:**

```
Kernel                Selector Bits    Weight(s) Dispatched
─────────────────────────────────────────────────────────────
main_qkv              >> 0 (qt var)    Q, K, V weights
main_attn_proj        >> 24            attn_output.weight
main_ffn_proj         >> 24            ffn_gate.weight, ffn_up.weight
main_ffn_down         >> 16            ffn_down.weight
```

**`sh_head_blob.wgsl` — 1 kernel:**

```
Kernel                Selector         Weight Dispatched
─────────────────────────────────────────────────────────
main_lm_head          params.quant_type  output.weight / token_embd.weight
```

Each kernel has an identical if/else chain (5 copies total):

```wgsl
// Example: main_qkv in sh_layer_v1.wgsl (line 418)
if (qt == 14u) { /* Q6_K: 210 bytes/block, 256 elems */ }
else if (qt == 13u) { /* Q5_K: 176 bytes/block, 256 elems */ }
else if (qt == 12u) { /* Q4_K: 144 bytes/block, 256 elems */ }
else if (qt == 6u) {  /* Q5_0: 22 bytes/block, 32 elems */ }
else if (qt == 8u) {  /* Q8_0: 34 bytes/block, 32 elems */ }
else if (qt == 1u) {  /* F16:  2 bytes/elem */ }
else if (qt == 0u) {  /* F32:  4 bytes/elem */ }
else { /* Q4_0 fallback: 18 bytes/block, 32 elems */ }
```

The chain differs slightly per kernel (dim var, activation expression) but the structure is identical.

---

## 3. Problem 1: Dispatch Duplication (FSE Violation)

### 3.1 Evidence: The Q5_0 Bug

Qwen2-0.5B uses quant type 6 (Q5_0) for 6 of 9 weight types per layer:
- attn_q.weight, attn_k.weight, attn_output.weight
- ffn_gate.weight, ffn_up.weight, ffn_k.weight
- attn_v.weight = Q8_0 (type 8) — different
- ffn_down.weight alternates Q4_K / Q6_K per layer
- token_embd.weight = Q8_0 (type 8)

The `dequant_q5_0_elem` function existed in both shaders (added 2026-06-18) but was **never wired into any of the 5 if/else chains**. All Q5_0 tensors fell through to the `else { Q4_0 fallback }`, which reads 18-byte blocks as if they're 22-byte Q5_0 blocks → garbage → NaN from layer 0.

**Time to catch:** 2 weeks. **Cost:** O(5) edits, missed 5 times.

### 3.2 FSE Analysis

This is a textbook **rule-first** architecture:

```
Rule-first (current):
  Kernel 1: evaluate quant_type → dispatch to dequant
  Kernel 2: evaluate quant_type → dispatch to dequant  (DUPLICATED)
  ...
  Kernel 5: evaluate quant_type → dispatch to dequant  (DUPLICATED)

Selector-first (FSE invariant):
  quant_type evaluated once → broadcast to all dequant paths
  ∂runtime/∂quant_types ≈ 0 per kernel
```

Rule-first violates the core FSE invariant: the selector (quant_type) is extracted N times, not once. Adding a type requires N edits. We proved this empirically by missing one.

### 3.3 Proposed Fix: WGSL Switch Table

A single `dequant_dispatch()` function in each shader that all kernels call:

```wgsl
// One dispatch point per shader — FSE selector-first
fn dequant_dispatch(qt: u32, block_base: u32, elem: u32) -> f32 {
    switch qt {
        case 14u: { return dequant_q6k(block_base, elem); }
        case 13u: { return dequant_q5k(block_base, elem); }
        case 12u: { return dequant_q4k(block_base, elem); }
        case 6u:  { return dequant_q5_0(block_base, elem); }
        case 8u:  { return dequant_q8_0(block_base, elem); }
        case 1u:  { return dequant_f16_at(block_base + elem * 2u); }
        case 0u:  { return bitcast<f32>(read_word(...)); }
        default:  { return dequant_q4_0(block_base, elem); }
    }
}
```

Each kernel calls:
```wgsl
dot += activation * dequant_dispatch(qt, block_base + b * block_stride, e);
```

**Effect:** Adding a new quant type = 1 branch in 1 place (per shader, so 2 locations). Not 5. The switch statement is compiled to a jump table by most GPU compilers — no runtime cost difference from the if/else chain.

**Open question:** WGSL `switch` support is optional per the spec. Real-world support on Vulkan (RTX 3060, Nvidia driver 32.0.15.9649) needs verification.

---

## 4. Problem 2: Selector Fragmentation

### 4.1 The Packed u32 Design

The current design encodes 4 quant types into a single u32. Each kernel reads different bits:

| Bits | Field | Used By | For Weight(s) |
|------|-------|---------|---------------|
| 0-7 | main | main_qkv | Q, K, V |
| 8-15 | v | *(not used in shader — debug only)* | V |
| 16-23 | ffn_down | main_ffn_down | ffn_down |
| 24-31 | attn_output | main_attn_proj, main_ffn_proj | attn_output, ffn_gate, ffn_up |

### 4.2 The Proxy Problem

The `>> 24u` selector is used by both `main_attn_proj` AND `main_ffn_proj`. This means:

- `main_attn_proj(attn_output.weight)` uses `>> 24u` → correct
- `main_ffn_proj(ffn_gate.weight, ffn_up.weight)` uses `>> 24u` → **proxied from attn_output.weight**

This works ONLY because current models happen to make attn_output, ffn_gate, and ffn_up all the same quant type. For Qwen2-0.5B, they are all Q5_0 (type 6). There is no architectural guarantee this holds across all models.

### 4.3 The V Alternation Problem

Q4_K_M models alternate V weight quant between Q6_K (14) and Q4_K (12) per layer. The `>> 8` field exists for this, but `main_qkv` uses `>> 0` (the main field), not `>> 8`. The per-layer V fix (airframe-mbc 2026-06-19) worked around this by encoding the correct per-layer V type into `>> 0`, but the packing scheme itself remains fragile.

### 4.4 Proposed Fix: Per-Tensor Selector Array

Instead of a packed u32, pass a small array of quant types indexed by weight role:

```wgsl
struct LayerQuantTypes {
    qk:    u32,  // Q, K projection weights
    v:     u32,  // V projection weight
    attn_out: u32,  // attn_output.weight
    gate:  u32,  // ffn_gate.weight
    up:    u32,  // ffn_up.weight
    down:  u32,  // ffn_down.weight
}
```

Each kernel dispatches on its tensor's actual type:

| Kernel | Selector | Weight Used |
|--------|----------|-------------|
| main_qkv | `qt.qk` for QK, `qt.v` for V | Per-component dispatch within QKV |
| main_attn_proj | `qt.attn_out` | attn_output.weight |
| main_ffn_proj | `qt.gate` for gate, `qt.up` for up | Separate dispatch per sub-weight |
| main_ffn_down | `qt.down` | ffn_down.weight |
| main_lm_head | `head_quant_type` (already separate) | output.weight |

This eliminates both the proxy problem and the bit-field packing. The switch table from Problem 1 operates on these scalar fields.

**Cost:** 6 u32s instead of 1 per layer — negligible (24 bytes vs 4 bytes per layer, ~144 bytes total for 24 layers).

---

## 5. Quant Type Reference

All GGML quant types relevant to airframe:

| Code | Name | Block Bytes | Elems/Block | Dequant Function | Seen In |
|------|------|-------------|-------------|-----------------|---------|
| 0 | F32 | 4/elem | 1 | `bitcast<f32>` | All models (biases, norms) |
| 1 | F16 | 2/elem | 1 | `unpack2x16float` | All models (token_embd, rare) |
| 2 | Q4_0 | 18 | 32 | `dequant_q4_0` | TinyLlama, Phi (fallback) |
| 3 | Q4_1 | 20 | 32 | *(not implemented)* | Unverified |
| 6 | Q5_0 | 22 | 32 | `dequant_q5_0` | **Qwen2 family** — newly added |
| 7 | Q5_1 | 24 | 32 | *(not implemented)* | Unverified |
| 8 | Q8_0 | 34 | 32 | `dequant_q8_0` | Qwen2 V proj, token_embd |
| 10 | Q2_K | 80 | 256 | *(not implemented)* | Edge, recent llama.cpp |
| 11 | Q3_K | 112 | 256 | *(not implemented)* | Edge, recent llama.cpp |
| 12 | Q4_K | 144 | 256 | `dequant_q4_k` | Most models (ffn_down) |
| 13 | Q5_K | 176 | 256 | `dequant_q5_k` | Some models |
| 14 | Q6_K | 210 | 256 | `dequant_q6_k` | Common (projections) |
| 16+ | IQ2/3/4_* | varies | 256 | *(not implemented)* | llama.cpp >= b3xxx |

Types 2, 6, 8, 12, 13, 14, 0, 1 are currently handled. Types 3, 7, 10, 11, 16+ are unimplemented and would silently fall through to Q4_0 (producing garbage).

---

## 6. Model Coverage Matrix

Current smoke test status by model:

| Model | Quant | Weights Used | Dispatch Status |
|-------|-------|-------------|-----------------|
| TinyLlama 1.1B | Q4_0 | Q4_0 only | ✅ PASS |
| TinyLlama 1.1B | Q6_K | Q6_K all | ✅ PASS |
| Llama 3.2 1B | Q4_K_M | Q6_K+V, Q4_K+ffn_down, Q8_0+token | ✅ PASS |
| Llama 3.2 3B | Q4_K_M | Same pattern | ✅ PASS (smoke pending) |
| Qwen2 0.5B | Q4_K_M | Q5_0(Q,K,gate,up,out), Q8_0(V,embd), Q4_K/Q6_K(ffn_down) | ✅ PASS (after Q5_0 fix) |
| Qwen2 1.5B | Q4_K_M | Q5_0 as above | ✅ PASS |
| Qwen3 0.6B | Q4_K_M | Q4_K + Q6_K + Q8_0 + Q5_K | ❌ FAIL (QK-norm) |
| Qwen3 1.7B | Q4_K_M | Same pattern | ❌ FAIL (QK-norm) |
| Phi-3.5 | Q4_K_M | Fused QKV (unsupported layout) | ❌ FAIL (loader) |
| StarCoder2 3B | Q4_K_M | Fused FFN gate (unsupported layout) | ❌ FAIL (loader) |
| DeepSeek-Coder-V2 | Q4_K_M | MLA (unsupported architecture) | ❌ FAIL (loader) |
| Gemma 2 2B | Q4_K_M | All types | ❌ FAIL (OOM) |

---

## 7. Risk Analysis

### Dispatch Table Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| WGSL `switch` unsupported on RTX 3060/Vulkan | Must use if/else chain, losing FSE benefit | Verify with `naga` or compile test first; fallback is `if/else if/else` generated by build.rs |
| Compiler fails to optimize `switch` to jump table | Slight perf regression vs hand-written chain | Profile-check; the chain is not in the inner loop (block-level dispatch, not element-level) |
| Table function call overhead per block | ~1 instruction per block from non-inline call | Mark `@must_use` / rely on WGSL inlining; if a concern, inline via `build.rs` codegen |

### Selector Array Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Adding 5 u32s (20 bytes) to LayerParams | ~24 bytes extra per layer, ~600B total for 24 layers | Negligible vs ~400MB model weights |
| Rust-side metadata must unpack 6 types | Small refactor in `metadata.rs` | Straightforward: store separate fields instead of packing |
| Head dispatch uses unified API | Must merge head_quant_type into same scheme | head_blob is separate pipeline; can adopt array separately |

---

## 8. Open Questions for Review

1. **WGSL `switch` support:** Does `naga` (the WGSL compiler used by wgpu) fully support `switch` statements on all backends? Specifically Vulkan + Nvidia drivers. Need a compile test.

2. **Build.rs codegen vs WGSL switch:** If `switch` is unsupported, the alternative is a `build.rs` that reads a `quant_types.toml` and emits the if/else chain into each kernel. This eliminates code duplication at compile time without requiring `switch` support. Is this cleaner or worse?

3. **Selector array design:** Is 6 explicit u32 fields the right granularity, or should it be a `HashMap<WeightRole, u32>` passed as a uniform? WGSL doesn't support hashmaps, so an enum-indexed array is the practical choice:

   ```wgsl
   const ROLE_QK: u32 = 0u;
   const ROLE_V: u32 = 1u;
   // ...
   var qt: array<u32, 6>;  // uniform buffer
   // dispatches: dequant_dispatch(qt[ROLE_QK], ...)
   ```

4. **Should the dispatch table and selector array be done as one change or two?** They are logically independent. The switch table fixes the FSE violation in isolation; the selector array fixes the underlying metadata architecture. Doing both together is safer (one touch to all 5 kernels) but riskier (more moving parts).

---

## 9. Recommended Implementation Order

1. **Phase A** — Verify WGSL `switch` support. If yes, implement dispatch table in `sh_layer_v1.wgsl` and `sh_head_blob.wgsl`. Verify all passing models still pass.

2. **Phase B** — Replace packed u32 with per-tensor selector array in `metadata.rs`, `LayerParams`, and all 5 kernels. Requires updating the Rust-side metadata, the Rust->GPU uniform layout, and all shader selectors.

3. **Phase C** — Wire the unified dispatch + selector into the QK-norm path (airframe-dna) once divergence is understood.

---

*Prepared 2026-06-19 for cloud review. Repo: `C:\Users\micha\repos\airframe`. Branch: `feat/phase4-pingpong-activation`.*
