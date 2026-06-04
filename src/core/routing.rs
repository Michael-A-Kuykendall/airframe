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
            warnings.push("qwen3 route expects attn_q_norm.weight and attn_k_norm.weight".to_string());
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
            post_norm_enabled: spec.arch_string().contains("gemma"),
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
