# B0 — Current Quant/Arch Dispatch Call-Site Map

Spike to de-risk `feat/inference-fabric-core`. Goal: locate every place quant_type and
architecture currently drive shader/kernel selection, so B1 (formula registry) and B3b
(retire WGSL ladder) can replace them without breaking multi-quant support.

**Headline finding (raises confidence):** the forward pass and its quant dispatch are
**already centralized and mostly data-driven**. Two structures already exist that the
refactor extends rather than builds from scratch:
- `ModelRoutePlan` (`src/core/routing.rs`) — a typed, spec+tensor-derived control plane.
- `QUANT_ELEMS` / `QUANT_BYTES` const arrays in `sh_layer_v1.wgsl` — a quant_type-indexed
  table of block sizes (the seed of the data-driven pattern).

The only genuine "if/then ladder" is `dequant_dispatch()` in `sh_layer_v1.wgsl:353-361`.

---

## 1. Orchestration layer (`src/runtime/gpu.rs`)

Both token-loops call the **same shared forward pass**:
`self.pipeline.run_full_model_prefill_chunked_with_cache_state(...)`.

| Location | Function | Calls forward pass at | Notes |
|---|---|---|---|
| `gpu.rs:435`, `gpu.rs:497` | `generate_isf` (reactive) | `run_full_model_prefill_chunked_with_cache_state` | Token loop driven by `airframe_observe::isf` (saturation fabric). Also calls `invariant_capture_sink_mut()` to emit facts. |
| `gpu.rs:680`, `gpu.rs:857` | `generate` (imperative) | `run_full_model_prefill_chunked_with_cache_state` | Manual token loop. |

**Implication for B4/B7:** the "dual orchestrators" are two *token loops* over ONE forward
pass. The quant dispatch lives entirely inside that forward pass, NOT in `gpu.rs`. So the
1-liner swap (B4) is safe — both paths produce identical forward-pass output; the cert gate
(B8, TinyLlama) will confirm. B7 = delete the imperative token loop, keep `generate_isf`.

---

## 2. Forward pass — per-layer loop (`src/backend/bindless/pipeline/inference.rs`)

| Location | What it does | Quant/Arch selected |
|---|---|---|
| `inference.rs:366-379` | Reads `get_tensor_type("blk.0.attn_q/k/v/ffn_down.weight")` from GGUF metadata to seed `weight_quant_type`, `qt_v`, `qt_ffn_down`. | Per-tensor quant_type from GGUF header. |
| `inference.rs:427-444` | Builds `ModelRoutePlan::from_spec_and_tensors(spec, has_tensor)` → `ffn_kind_policy`, `qkv_layout_policy`. Env-gated by `SHIMMY_ROUTE_V2_LAYER_PARAMS`. | Arch/FFN/QKV policy. |
| `inference.rs:446-459` | Builds `LayerParams { ..., quant_qk, quant_v, quant_attn_out, quant_ffn_down, quant_ffn_gate, ... }` from `ModelSpec`. | Per-layer quant types (uniforms). |
| `inference.rs:1047-1081` | The layer loop. **All models use V1 pipelines** (`layer_pipeline_attn_norm`, `layer_pipeline_qkv`, `layer_pipeline_ffn_down`, etc.). No per-quant shader selection in Rust. | Fixed V1 pipeline set; quant dispatched inside shader via `LayerParams` uniform. |
| `inference.rs:1056-1058` | Comment: "V1 handles all quant types (Q4_0, Q4_K, Q5_K, Q6_K, F16, F32) via per-kernel quant_type branch checks". | Confirms dispatch is in-shader. |
| `inference.rs:919-925` | `head_quant_type` check against `supported` set; panics if unsupported. | Head quant gating. |

**Implication for B3b:** Rust does NOT branch on quant_type to pick a shader. It always
binds the V1 pipeline and passes quant types as `LayerParams` uniforms. The branch is in WGSL.

---

## 3. WGSL dispatch ladder (`src/backend/bindless/sh_layer_v1.wgsl`)

| Location | What | Quant handled |
|---|---|---|
| `sh_layer_v1.wgsl:343-346` | `QUANT_ELEMS[16]`, `QUANT_BYTES[16]` const arrays indexed by quant_type (0=F32,1=F16,2=Q4_0,6=Q5_0,8=Q8_0,12=Q4_K,13=Q5_K,14=Q6_K). **Already data-driven.** | Block size table. |
| `sh_layer_v1.wgsl:353-361` | `dequant_dispatch(qt, ...)` — THE ladder: `if qt==14 Q6_K elif 13 Q5_K elif 12 Q4_K elif 6 Q5_0 elif 8 Q8_0 elif 2 Q4_0 else 0`. **This is the only hardcoded if/then to retire (B3b).** | qt → dequant fn. |
| `sh_layer_v1.wgsl:456` | `let qt = select(params.quant_qk, params.quant_v, target_stage==2u);` — quant type from `LayerParams` uniform. | Per-tensor qt. |
| `sh_layer_v1.wgsl:458-468` | `if qt==1 F16 elif qt==0 F32 else dequant_dispatch(qt)` in `main_qkv` matmul. | F16/F32/block-quant. |

Mirrored ladder exists in `sh_dequant_any.wgsl:223-238` (`qt == 14u Q6_K / 13u Q5_K / 12u Q4_K / 8u Q8_0 / 1u F16 ...`) and `sh_head_blob.wgsl` (lm_head dequant).

**Implication for B3b:** retire `dequant_dispatch`'s `if qt==` ladder. The dequant
functions (`dequant_q6k_elem`, etc.) stay; only the selection moves to the registry.

---

## 4. Existing control plane (`src/core/routing.rs`)

`ModelRoutePlan` — already the typed control plane the refactor needs:
- Derived once from `ModelSpec` + a `has_tensor(name)` closure (GGUF tensor manifest).
- Computes `NormKind` (LayerNorm/RmsNorm), `QkvLayout` (Separate/Fused), `FfnKind`
  (Gated/NonGated), `qk_norm_enabled`, `post_norm_enabled`, `arch`.
- Carries `reasons` / `warnings` / `hard_errors` and a `digest` (deterministic hash).
- Emits policy codes (`qkv_layout_policy_code`, `ffn_kind_policy_code`) passed to shaders.

**Implication for B2/B3:** `ModelRoutePlan` is where the per-tensor `quant_type →
canonical-formula-index` mapping should live (new field alongside `ffn_kind`/`qkv_layout`).
This is the natural home for the B1 registry hook — not a new parallel structure.

---

## 5. Open design decision (B3b mechanism) — document, do not block

Replacing `dequant_dispatch`'s ladder has two faithful options:
- **(a) Registry-derived index + WGSL `switch(formula_index)`:** Rust registry (B1) owns
  `quant_type → formula_index`; WGSL switches on the index. The dispatch *logic* moves out
  of WGSL into the auditable, spec-cited Rust table. Minimal change; WGSL still has a switch.
- **(b) WGSL override-constant specialization:** bake `formula_index` at pipeline-creation
  time per quant type; no runtime branch. Branchless, but requires one specialized pipeline
  per distinct quant type present (heavier; one layer shader handles 6 tensors of possibly
  different quant types, so this multiplies pipelines).

Recommendation: **(a)** as the primary target (moves dispatch logic into the spec-derived
registry, which is the user's "stand on our own mathematical feet" requirement); **(b)** as a
later optimization if branch cost matters. Either satisfies "retire the if/then ladder" in the
sense that the *dispatch logic* is no longer hardcoded in WGSL.

---

## 6. References (for B1)

- GGUF/GGML quant spec: quant enum values and block layouts (Q4_0=2, Q5_0=6, Q8_0=8,
  Q4_K=12, Q5_K=13, Q6_K=14, F16=1, F32=0). Block sizes already encoded in
  `sh_layer_v1.wgsl:343-346` (`QUANT_ELEMS`/`QUANT_BYTES`).
- Canonical dequant formulas: currently implemented in `dequant_q6k_elem` / `dequant_q5k_elem`
  / `dequant_q4k_elem` / `dequant_q4_0_elem` / `dequant_q8_0_elem` / `dequant_q5_0_elem`
  (sh_layer_v1.wgsl) and mirrored in `sh_dequant_any.wgsl`. These are the CURRENT (to-be-
  audited) implementations; B1 writes the spec-cited canonical forms and B6 proves they match.
- Control plane: `ModelRoutePlan` (`src/core/routing.rs`) — the home for the quant→formula
  mapping.
