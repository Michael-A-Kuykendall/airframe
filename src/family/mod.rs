pub mod llama;

use crate::core::spec::ModelSpec;
use crate::core::{error::Result, tensor::Tensor, weight_id::WeightId};
use crate::ops::dispatch::OpDispatcher;
use crate::runtime::kvcache::KvCache;
use std::collections::HashMap;

/// Interface every model family must implement.
/// dyn-safe — use `Box<dyn ModelFamily>` in `Engine`.
pub trait ModelFamily: Send + Sync {
    /// Execute full model forward pass.
    /// Input: token IDs. Returns logits [seq_len, n_vocab].
    fn forward(
        &self,
        input_ids: &[usize],
        weights: &HashMap<WeightId, Tensor>,
        kv_cache: &mut KvCache,
        ops: &OpDispatcher,
    ) -> Result<Tensor>;

    /// All weight IDs this model requires.
    fn required_weights(&self) -> Vec<WeightId>;

    /// Validate all required weights are present.
    fn validate_weights(&self, weights: &HashMap<WeightId, Tensor>) -> Result<()>;

    /// Reference to the model spec.
    fn spec(&self) -> &ModelSpec;

    // ── Arch-specific properties ─────────────────────────────────────
    // Default implementations delegate to ModelSpec.
    // Override when a family needs different behavior.

    fn arch_string(&self) -> &str {
        self.spec().arch_string()
    }

    fn has_qk_norm(&self) -> bool {
        self.spec().has_qk_norm
    }

    fn uses_layer_norm(&self) -> bool {
        self.spec().uses_layer_norm()
    }

    /// Gemma-2 has post-attention and post-FFW norms.
    fn has_post_norms(&self) -> bool {
        false
    }

    fn layer_prefix(&self) -> &str {
        "blk"
    }

    /// Gemma-2 scales embeddings by sqrt(n_embd).
    fn embedding_scale(&self) -> Option<f32> {
        None
    }

    fn attn_logit_softcap(&self) -> f32 {
        self.spec().attn_logit_softcap
    }

    fn final_logit_softcap(&self) -> f32 {
        self.spec().final_logit_softcap
    }

    fn chat_template(&self) -> Option<&str> {
        self.spec().chat_template.as_deref()
    }

    /// Phi may share ffn_norm with attn_norm (no distinct ffn_norm tensor).
    fn phi_fallback_ffn_norm(&self) -> bool {
        false
    }

    /// Qwen3 uses separate packed weight_k for Q and K in preflight.
    fn requires_q_weight_k(&self) -> bool {
        false
    }
}
