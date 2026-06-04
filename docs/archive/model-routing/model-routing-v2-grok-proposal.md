# Model Routing v2 Architecture Spec (Grok Draft)

Source: user-provided Grok response captured for local evaluation.

---

**Yes — now with concrete current-state data, I can give you a tight, implementation-ready design.**

Below is the full deliverable matching what you asked for (and what section 11/12 of the preload pack requested). It is grounded in the exact shapes, inference logic, shader branches, and split you described.

---

# Model Routing v2 Architecture Spec

**Goal**: One canonical, load-time `ModelRoutePlan` that drives **both** prompt rendering and math/layer execution. llama.cpp-comparable decision quality, FSE-aligned (selector-first, extract-once, broadcast-many), fully explicit, auditable, and rollback-friendly.

**Rollout recommendation**: **Option A (Incremental)** with dual-population + feature-flag cutover. This matches your current reality and risk tolerance.

---

## 1. Canonical `ModelRoutePlan` Schema (v2)

```rust
// src/core/routing.rs

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormKind {
    LayerNorm,
    RmsNorm,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum QkvLayout {
    Separate,   // attn_q, attn_k, attn_v present
    Fused,      // attn_qkv present
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FfnKind {
    Gated,      // ffn_gate + ffn_up + ffn_down (SwiGLU-style)
    NonGated,   // ffn_up + ffn_down only
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RopeKind {
    None,
    Norm,       // standard RoPE
    Neox,       // interleaved (most Llama/Mistral/Qwen)
    MRope,      // multimodal (future)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PromptRendererMode {
    Jinja,      // tokenizer.chat_template present and used
    Family,     // arch/name-based fallback template
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRoutePlan {
    // Identity
    pub arch: String,
    pub model_name: String,
    pub gguf_file_type: String,
    pub route_version: u32,           // bump on schema changes

    // Norm policy
    pub norm_kind: NormKind,
    pub norm_eps: f32,
    pub qk_norm_enabled: bool,
    pub post_norm_enabled: bool,

    // Attention policy
    pub qkv_layout: QkvLayout,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub attn_logit_softcap: Option<f32>,
    pub final_logit_softcap: Option<f32>,
    pub attention_key_length: Option<u32>,

    // Positional policy
    pub rope_kind: RopeKind,
    pub rope_dim: u32,
    pub rope_base: f32,
    pub rope_scale: f32,
    pub context_train: u32,
    pub context_runtime: u32,

    // FFN policy
    pub ffn_kind: FfnKind,
    pub ffn_activation: String,       // "silu", "gelu", etc. (from hidden_activation or default)

    // Prompt policy (now unified)
    pub prompt_renderer_mode: PromptRendererMode,
    pub prompt_family: String,
    pub prompt_template_source: String, // "embedded" | "fallback:<family>"

    // Quant / KV policy (execution concern but declared here)
    pub kv_quant_mode: String,        // "f32" | "int4" etc.

    // Validation & Audit
    pub validation: RouteValidation,
    pub digest: String,               // short hash of key decisions for traces
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteValidation {
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
    pub hard_errors: Vec<String>,
    pub strict_mode_pass: bool,
}
```

**Key improvements over current shape**:
- `ffn_kind` and `qkv_layout` are now **explicit enum policy**, not inferred from tensor presence at runtime.
- Prompt policy is **inside** the same artifact.
- Softcapping + `attention.key_length` are first-class (you already parse them in `spec.rs`).
- `validation` + `digest` give you the audit trail you want.
- `route_version` for future schema evolution.

---

## 2. Deterministic Decision Tree (Pseudocode)

```rust
pub fn from_spec_and_tensors(
    spec: &ModelSpec,
    has_tensor: impl Fn(&str) -> bool,
    runtime_ctx_override: Option<u32>,
) -> Result<ModelRoutePlan, RouteError> {
    let mut reasons = vec![];
    let mut warnings = vec![];
    let mut hard_errors = vec![];

    // === 1. Identity & Arch Family (explicit match first) ===
    let arch = spec.architecture.clone();
    let model_name = spec.name.clone();

    // === 2. Norm Policy ===
    let norm_kind = if spec.uses_layer_norm() {
        NormKind::LayerNorm
    } else {
        NormKind::RmsNorm
    };
    let norm_eps = spec.layer_norm_rms_epsilon.unwrap_or(1e-5);
    let qk_norm_enabled = spec.has_qk_norm;
    let post_norm_enabled = match arch.as_str() {
        "qwen2" | "qwen3" | "gemma2" => true,   // known post-norm families
        _ => false,
    };

    // === 3. QKV Layout (still from tensor presence, but now explicit) ===
    let qkv_layout = if has_tensor("blk.0.attn_q.weight") &&
                       has_tensor("blk.0.attn_k.weight") &&
                       has_tensor("blk.0.attn_v.weight") {
        QkvLayout::Separate
    } else if has_tensor("blk.0.attn_qkv.weight") {
        QkvLayout::Fused
    } else {
        warnings.push("Could not determine qkv_layout from layer-0 tensors".into());
        QkvLayout::Unknown
    };

    // === 4. FFN Kind — explicit, no more shader heuristic ===
    let ffn_kind = if has_tensor("blk.0.ffn_gate.weight") {
        FfnKind::Gated
    } else {
        FfnKind::NonGated
    };

    // === 5. Rope Policy ===
    let rope_kind = match arch.as_str() {
        "llama" | "mistral" | "qwen2" | "qwen3" | "starcoder2" => RopeKind::Neox,
        "gemma" | "gemma2" => RopeKind::Norm,
        _ => RopeKind::None,
    };
    let rope_dim = spec.rope_dimension_count.unwrap_or(spec.embedding_length / spec.head_count);
    let rope_base = spec.rope_freq_base.unwrap_or(10000.0);
    let rope_scale = 1.0; // TODO: parse rope_scaling dict when present

    // === 6. Softcapping & Key Length (already parsed in spec) ===
    let attn_logit_softcap = spec.attn_logit_softcapping;
    let final_logit_softcap = spec.final_logit_softcapping;
    let attention_key_length = spec.attention_key_length;

    // === 7. Prompt Renderer (unified) ===
    let (prompt_renderer_mode, prompt_family, prompt_template_source) =
        if spec.chat_template.is_some() {
            (PromptRendererMode::Jinja, arch.clone(), "embedded".into())
        } else {
            (PromptRendererMode::Family, arch.clone(), format!("fallback:{}", arch))
        };

    // === 8. Build + Validate ===
    let plan = ModelRoutePlan { /* all fields above */ };

    // Strict compatibility matrix (example)
    if plan.qkv_layout == QkvLayout::Unknown {
        hard_errors.push("Unknown QKV layout — cannot safely route".into());
    }
    if plan.ffn_kind == FfnKind::NonGated && arch == "llama" {
        warnings.push("Llama-family model with non-gated FFN is unusual".into());
    }

    let strict_pass = hard_errors.is_empty() &&
                      std::env::var("SHIMMY_ROUTE_CHECK_STRICT").is_err();

    if !strict_pass && std::env::var("SHIMMY_ROUTE_CHECK_STRICT").is_ok() {
        return Err(RouteError::StrictModeViolation(hard_errors));
    }

    Ok(plan)
}
```

This replaces the current heuristic-heavy `from_spec_and_tensors`.

---

## 3. Per-Variable Source + Fallback Table (First Cut)

| Field                    | Primary Source                  | Fallback / Derivation                  | Fail-Fast Boundary                  | Notes |
|--------------------------|---------------------------------|----------------------------------------|-------------------------------------|-------|
| `norm_kind`              | `spec.uses_layer_norm()`        | —                                      | —                                   | Direct from metadata |
| `qk_norm_enabled`        | `spec.has_qk_norm`              | `false`                                | —                                   | Already reliable |
| `post_norm_enabled`      | Arch match (qwen/gemma)         | `false`                                | —                                   | Move from heuristic to explicit arch arm |
| `qkv_layout`             | Layer-0 tensor presence         | `Unknown` + warning                    | Strict mode fails on `Unknown`      | Still tensor-driven but now typed |
| `ffn_kind`               | `blk.0.ffn_gate.weight` presence| —                                      | —                                   | **Critical**: removes shader `offsets.ffn_gate == 0` branch |
| `rope_kind`              | Arch match                      | `None`                                 | —                                   | Explicit first |
| `attn_logit_softcap`     | `spec.attn_logit_softcapping`   | `None`                                 | —                                   | Already parsed |
| `prompt_renderer_mode`   | `spec.chat_template.is_some()`  | `Family`                               | —                                   | Unifies the split you have today |
| `gqa_ratio`              | `head_count / head_count_kv`    | `1`                                    | Must divide evenly or hard error    | Add validation |

---

## 4. Migration / Cutover Plan (Phased, Low Risk)

**Phase 0 (Current – this week)**
- Keep existing `ModelRoutePlan` + heuristic paths.
- Add new `ModelRoutePlanV2` (or just evolve in place behind a flag) in parallel.
- Emit both old and new in `ROUTE_CHECK` JSON for comparison.

**Phase 1 (Next 1–2 weeks) – Dual Population**
- Build full `ModelRoutePlan` v2 in `from_spec_and_tensors`.
- Populate **both** old fields and new typed enums.
- Update `ROUTE_CHECK` to include `route_plan_v2` + `digest`.
- Add `SHIMMY_ROUTE_CHECK_STRICT=1` already fails on new hard errors.
- **No shader changes yet**.

**Phase 2 (Week 3) – Wire Consumers (Data Path Only)**
- Change `LayerParams` construction to derive from `ModelRoutePlan` (feature flag `route_v2_layer_params`).
- Pass explicit `ffn_kind`, `qkv_layout`, `norm_kind`, softcaps, etc. into bind group / push constants.
- **Remove** the `if offsets.ffn_gate == 0` branch in `sh_layer_v1.wgsl` and `sh_layer_v1_int4.wgsl` **only behind the flag**.

**Phase 3 (Week 4+) – Prompt Unification**
- Move `PromptRenderer` decision fully into `ModelRoutePlan`.
- Delete the old split path once tests pass.

**Phase 4 – Cutover**
- Default `route_v2_*` flags to `true`.
- Remove legacy heuristic code.
- Bump `route_version` to 2.

This gives you clean rollback at every gate.

---

## 5. CI Trait Matrix (Minimal but Powerful)

Test **combinations**, not just model names.

**Core Matrix (must pass in strict mode)**

| Combination                          | Expected `ffn_kind` | `qkv_layout` | `norm_kind` | `post_norm` | `qk_norm` | Rope     | Status |
|--------------------------------------|---------------------|--------------|-------------|-------------|-----------|----------|--------|
| non_gated + layer_norm + separate    | NonGated            | Separate     | LayerNorm   | false       | false     | None/Neox| Pass   |
| gated + rms_norm + separate          | Gated               | Separate     | RmsNorm     | false       | false     | Neox     | Pass   |
| gated + rms_norm + qk_norm           | Gated               | Separate     | RmsNorm     | false       | true      | Neox     | Pass   |
| gated + rms_norm + post_norm         | Gated               | Separate     | RmsNorm     | true        | false     | Neox     | Pass   |
| fused_qkv + gated                    | Gated               | Fused        | RmsNorm     | false       | false     | Neox     | Pass   |
| softcap present (Qwen-style)         | Gated               | Separate     | RmsNorm     | true        | false     | Neox     | Pass   |

**Pass criteria**:
- `ROUTE_CHECK` JSON contains correct enums + no unexpected warnings in strict mode.
- No math divergence on smoke matrix (Phi + TinyLlama + Llama-3.2-1B/3B + Qwen3-0.6B).
- `digest` is stable for identical manifests.

---

## 6. Risk Register + Mitigation Gates

| Risk | Likelihood | Impact | Mitigation | Gate |
|------|------------|--------|------------|------|
| Shader still branches on `ffn_gate` offset after v2 | Medium | High (silent math divergence) | Phase 2 explicitly removes the branch behind flag + adds WGSL compile-time assert or runtime check | Must pass before Phase 3 |
| New arch appears with unknown qkv/ffn combo | Medium | Medium | Strict mode + hard error on `Unknown` layout/kind | CI fails on unknown in strict |
| Prompt template source drifts from math route | Low | Medium | Single `ModelRoutePlan` is source of truth | Phase 3 cutover test |
| Provenance too verbose (perf) | Low | Low | Keep `digest` + top-level lists; field-level provenance is v2.1 | Measure ROUTE_CHECK size |
| Rollback difficulty | Low | High | Dual population + feature flags at every consumer | Every phase has explicit rollback commit |

---

## 7. Ready-to-Paste `routing.rs` Skeleton (Start Here)

```rust
// src/core/routing.rs  (new file or major refactor of existing)

use crate::core::spec::ModelSpec;

#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    #[error("Strict mode violation: {0:?}")]
    StrictModeViolation(Vec<String>),
    // add more variants as needed
}

pub fn build_route_plan(
    spec: &ModelSpec,
    has_tensor: impl Fn(&str) -> bool,
) -> Result<ModelRoutePlan, RouteError> {
    // paste the deterministic decision tree from section 2 here
    // return the fully populated ModelRoutePlan
}
```

You can drop the enums + struct from section 1 directly into this file.

---

**Immediate recommended next internal steps** (after you review this):

1. Create `src/core/routing.rs` with the enums + `ModelRoutePlan` + `build_route_plan` skeleton.
2. Wire the new plan into `shimmy_server_gpu.rs` route check (dual emit for now).
3. Add the `ffn_kind` / `qkv_layout` fields to `LayerParams` (behind feature flag).
4. Run your existing smoke matrix with `SHIMMY_ROUTE_CHECK_STRICT=1` and capture diffs.

This design directly eliminates the remaining heuristic branches you identified in the shaders and in `post_norm_enabled` / `ffn_kind` inference while keeping migration safe.

Would you like me to:
- Expand any section (especially the WGSL integration or full `LayerParams` evolution)?
- Generate the actual diff-style migration PR checklist as a separate artifact?
- Start drafting the updated `inference.rs` / `LayerParams` population code?

Just say the word and we keep moving. This is now concrete enough to code against.
