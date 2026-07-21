use crate::core::spec::ModelSpec;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    LayerNorm,
    RmsNorm,
}

impl NormKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LayerNorm => "layer_norm",
            Self::RmsNorm => "rms_norm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QkvLayout {
    Separate,
    Fused,
    Unknown,
}

impl QkvLayout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::Fused => "fused",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnKind {
    Gated,
    NonGated,
}

impl FfnKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gated => "gated",
            Self::NonGated => "non_gated",
        }
    }
}

/// Typed model routing plan derived once from GGUF model metadata and tensor manifest.
///
/// This is intentionally lightweight so call sites can adopt it incrementally.
#[derive(Debug, Clone)]
pub struct ModelRoutePlan {
    pub route_version: u32,
    pub model_name: String,
    pub arch: String,
    pub prompt_renderer_mode: String,
    pub prompt_renderer_family: Option<String>,
    pub prompt_template_source: String,
    pub norm_kind: NormKind,
    pub qk_norm_enabled: bool,
    pub post_norm_enabled: bool,
    pub qkv_layout: QkvLayout,
    pub ffn_kind: FfnKind,
    pub reasons: Vec<String>,
    pub warnings: Vec<String>,
    pub hard_errors: Vec<String>,
    pub strict_mode_pass: bool,
    pub digest: String,
}

impl ModelRoutePlan {
    pub const ROUTE_VERSION: u32 = 2;

    pub const QKV_LAYOUT_INFER: u32 = 0;
    pub const QKV_LAYOUT_SEPARATE: u32 = 1;
    pub const QKV_LAYOUT_FUSED: u32 = 2;

    pub const FFN_KIND_INFER: u32 = 0;
    pub const FFN_KIND_GATED: u32 = 1;
    pub const FFN_KIND_NON_GATED: u32 = 2;

    /// Build route plan from a model spec plus tensor presence function.
    pub fn from_spec_and_tensors<F>(spec: &ModelSpec, has_tensor: F) -> Self
    where
        F: Fn(&str) -> bool,
    {
        let mut reasons = Vec::new();
        let mut warnings = Vec::new();
        let mut hard_errors = Vec::new();

        let qkv_layout = if has_tensor("blk.0.attn_q.weight")
            && has_tensor("blk.0.attn_k.weight")
            && has_tensor("blk.0.attn_v.weight")
        {
            reasons.push("separate Q/K/V tensors are present".to_string());
            QkvLayout::Separate
        } else if has_tensor("blk.0.attn_qkv.weight") {
            reasons.push("fused attn_qkv tensor is present".to_string());
            QkvLayout::Fused
        } else {
            warnings.push(
                "neither separate Q/K/V nor fused attn_qkv tensor set was found for layer 0"
                    .to_string(),
            );
            QkvLayout::Unknown
        };

        let ffn_kind = if has_tensor("blk.0.ffn_gate.weight") {
            reasons.push("ffn_gate.weight is present -> gated FFN route".to_string());
            FfnKind::Gated
        } else {
            reasons.push("ffn_gate.weight is absent -> non-gated FFN route".to_string());
            FfnKind::NonGated
        };

        let norm_kind = if spec.uses_layer_norm() {
            reasons.push("ModelSpec::uses_layer_norm() selected LayerNorm".to_string());
            NormKind::LayerNorm
        } else {
            reasons.push("ModelSpec::uses_layer_norm() selected RMSNorm".to_string());
            NormKind::RmsNorm
        };

        if spec.has_qk_norm {
            reasons.push("has_qk_norm=true from ModelSpec derived traits".to_string());
        }

        let arch = spec.arch_string().to_string();

        // Lightweight consistency checks for currently supported families.
        if arch == "qwen3"
            && (!has_tensor("blk.0.attn_q_norm.weight") || !has_tensor("blk.0.attn_k_norm.weight"))
        {
            warnings
                .push("qwen3 route expects attn_q_norm.weight and attn_k_norm.weight".to_string());
        }

        if arch == "phi" && qkv_layout != QkvLayout::Fused {
            warnings.push("phi route usually expects fused attn_qkv layout".to_string());
        }

        if qkv_layout == QkvLayout::Unknown {
            hard_errors.push("unable to determine qkv_layout from model tensors".to_string());
        }

        if arch.contains("gpt2") && qkv_layout == QkvLayout::Fused {
            reasons.push("gpt2-family fused QKV layout detected".to_string());
        }

        let strict_mode_pass = hard_errors.is_empty();

        let mut route = Self {
            route_version: Self::ROUTE_VERSION,
            model_name: spec.model_name.clone(),
            arch,
            prompt_renderer_mode: "unknown".to_string(),
            prompt_renderer_family: None,
            prompt_template_source: "unknown".to_string(),
            norm_kind,
            qk_norm_enabled: spec.has_qk_norm,
            post_norm_enabled: spec.post_norm_enabled,
            qkv_layout,
            ffn_kind,
            reasons,
            warnings,
            hard_errors,
            strict_mode_pass,
            digest: String::new(),
        };
        route.update_digest();
        route
    }

    pub fn apply_prompt_routing(
        &mut self,
        mode: String,
        family: Option<String>,
        template_source: String,
    ) {
        self.prompt_renderer_mode = mode;
        self.prompt_renderer_family = family;
        self.prompt_template_source = template_source;
        self.reasons.push(format!(
            "prompt renderer selected mode={} source={}",
            self.prompt_renderer_mode, self.prompt_template_source
        ));
        self.update_digest();
    }

    pub fn qkv_layout_policy_code(&self) -> u32 {
        match self.qkv_layout {
            QkvLayout::Separate => Self::QKV_LAYOUT_SEPARATE,
            QkvLayout::Fused => Self::QKV_LAYOUT_FUSED,
            QkvLayout::Unknown => Self::QKV_LAYOUT_INFER,
        }
    }

    pub fn ffn_kind_policy_code(&self) -> u32 {
        match self.ffn_kind {
            FfnKind::Gated => Self::FFN_KIND_GATED,
            FfnKind::NonGated => Self::FFN_KIND_NON_GATED,
        }
    }

    pub fn update_digest(&mut self) {
        self.digest = self.compute_digest();
    }

    /// Resolve a GGML quant type id to its canonical dispatch formula slot.
    ///
    /// The slot is the stable index the shader consumes (`formula_index`),
    /// owned by `airframe_observe::quant_formula` — NOT the raw GGML type id.
    /// This is the `quant_type → formula_index` mapping the dispatch control
    /// plane hangs on (replaces the WGSL `if qt==` ladder in B3b).
    ///
    /// Gated behind `isf` because the authoritative formula registry lives in
    /// the `airframe-observe` crate (optional dependency).
    #[cfg(feature = "isf")]
    pub fn quant_formula_slot(&self, quant_type: u32) -> Option<u32> {
        airframe_observe::quant_formula::slot_for_type(quant_type).map(|s| s.as_u32())
    }

    fn compute_digest(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.route_version.hash(&mut hasher);
        self.model_name.hash(&mut hasher);
        self.arch.hash(&mut hasher);
        self.norm_kind.as_str().hash(&mut hasher);
        self.qk_norm_enabled.hash(&mut hasher);
        self.post_norm_enabled.hash(&mut hasher);
        self.qkv_layout.as_str().hash(&mut hasher);
        self.ffn_kind.as_str().hash(&mut hasher);
        self.prompt_renderer_mode.hash(&mut hasher);
        self.prompt_renderer_family.hash(&mut hasher);
        self.prompt_template_source.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::spec::{GgufFileType, ModelArch};

    fn base_spec() -> ModelSpec {
        ModelSpec {
            n_vocab: 32000,
            n_embd: 2048,
            n_layer: 22,
            n_head: 32,
            n_head_kv: 4,
            ff_dim: 5632,
            rms_eps: 1e-5,
            rope_base: 10000.0,
            rope_scale: 1.0,
            rope_dim: 64,
            yarn_alpha: 1.0,
            yarn_beta: 32.0,
            n_ctx: 2048,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            has_qk_norm: false,
            post_norm_enabled: false,
            head_dim: 64,
            gqa_ratio: 8,
            kv_dim: 256,
            arch: ModelArch::Llama,
            file_type: GgufFileType::Q4_0,
            model_name: "unit-test-model".to_string(),
            chat_template: None,
            temp_buffer_size: 16384,
            kv_cache_size_per_layer: 2048 * 4 * 64 * 4,
        }
    }

    #[test]
    fn digest_is_deterministic_for_same_inputs() {
        let spec = base_spec();
        let has = |name: &str| {
            matches!(
                name,
                "blk.0.attn_q.weight"
                    | "blk.0.attn_k.weight"
                    | "blk.0.attn_v.weight"
                    | "blk.0.ffn_gate.weight"
            )
        };

        let a = ModelRoutePlan::from_spec_and_tensors(&spec, has);
        let b = ModelRoutePlan::from_spec_and_tensors(&spec, has);

        assert_eq!(a.digest, b.digest);
    }

    #[test]
    fn digest_changes_when_prompt_policy_changes() {
        let spec = base_spec();
        let has = |name: &str| {
            matches!(
                name,
                "blk.0.attn_q.weight"
                    | "blk.0.attn_k.weight"
                    | "blk.0.attn_v.weight"
                    | "blk.0.ffn_gate.weight"
            )
        };

        let mut route = ModelRoutePlan::from_spec_and_tensors(&spec, has);
        let before = route.digest.clone();
        route.apply_prompt_routing(
            "family".to_string(),
            Some("ChatML".to_string()),
            "fallback".to_string(),
        );
        let after = route.digest;

        assert_ne!(before, after);
    }

    #[test]
    fn strict_mode_pass_false_when_hard_errors_present() {
        let spec = base_spec();
        let has_none = |_name: &str| false;
        let route = ModelRoutePlan::from_spec_and_tensors(&spec, has_none);

        assert!(!route.hard_errors.is_empty());
        assert!(!route.strict_mode_pass);
    }

    #[cfg(feature = "isf")]
    #[test]
    fn quant_formula_slot_maps_ggml_type_to_registry_index() {
        let spec = base_spec();
        let has = |name: &str| {
            matches!(
                name,
                "blk.0.attn_q.weight" | "blk.0.attn_k.weight" | "blk.0.attn_v.weight"
            )
        };
        let route = ModelRoutePlan::from_spec_and_tensors(&spec, has);

        // Q6_K (14) -> slot 7; Q4_0 (2) -> slot 2; unsupported -> None
        assert_eq!(route.quant_formula_slot(14), Some(7));
        assert_eq!(route.quant_formula_slot(2), Some(2));
        assert_eq!(route.quant_formula_slot(99), None);
    }
}
