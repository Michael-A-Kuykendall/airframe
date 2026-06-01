//! SigLIP-So400M Vision Transformer encoder for MiniCPM-V-2.6.
//!
//! Architecture (from GGUF tensor map — `mmproj-model-f16.gguf`):
//!   hidden_dim  = 1152
//!   n_layers    = 27
//!   n_heads     = 16   (head_dim = 72)
//!   mlp_dim     = 4304
//!   patch_size  = 14
//!   image_size  = 448
//!   n_patches   = (448/14)^2 = 1024
//!   pos_embed   = [4900, 1152]  (NO CLS; sliced to n_patches at runtime)
//!
//! Key differences from the LLM (LlamaModel):
//!   - LayerNorm (not RMSNorm)
//!   - GELU activation (not SwiGLU)
//!   - Q/K/V projections have bias terms
//!   - Bidirectional attention (no causal mask)
//!   - NO CLS token — SigLIP uses all patch features
//!   - NO pre-encoder LayerNorm (only v.post_ln applied after blocks)
//!   - Positional embedding via learned `v.position_embd.weight` [4900, 1152]

use crate::core::{error::Result, tensor::Tensor};
use crate::ops::dispatch::OpDispatcher;

// ─── Per-layer config ─────────────────────────────────────────────────────────

/// SigLIP-So400M default architecture dimensions.
pub struct SigLipConfig {
    pub hidden_dim: usize, // 1152
    pub n_layers:   usize, // 27
    pub n_heads:    usize, // 16
    pub head_dim:   usize, // 72  (= hidden_dim / n_heads)
    pub mlp_dim:    usize, // 4304
    pub patch_size: usize, // 14
    pub image_size: usize, // 448
    pub layer_norm_eps: f32, // 1e-6
}

impl Default for SigLipConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 1152,
            n_layers:   27,
            n_heads:    16,
            head_dim:   72,
            mlp_dim:    4304,
            patch_size: 14,
            image_size: 448,
            layer_norm_eps: 1e-6,
        }
    }
}

// ─── Transformer block ────────────────────────────────────────────────────────

/// One SigLIP ViT transformer block.
///
/// Each block applies:
///   1. LayerNorm (pre-norm)
///   2. Bidirectional self-attention with Q/K/V bias
///   3. Residual connection
///   4. LayerNorm (post-norm before FFN)
///   5. GELU FFN: fc1 → GELU → fc2
///   6. Residual connection
pub struct SigLipBlock {
    // Attention weights and biases
    pub attn_q_weight: Tensor, // [hidden, hidden]
    pub attn_q_bias:   Tensor, // [hidden]
    pub attn_k_weight: Tensor, // [hidden, hidden]
    pub attn_k_bias:   Tensor, // [hidden]
    pub attn_v_weight: Tensor, // [hidden, hidden]
    pub attn_v_bias:   Tensor, // [hidden]
    pub attn_o_weight: Tensor, // [hidden, hidden]
    pub attn_o_bias:   Tensor, // [hidden]

    // Layer norms (each has weight + bias)
    pub ln1_weight: Tensor,    // [hidden]
    pub ln1_bias:   Tensor,    // [hidden]
    pub ln2_weight: Tensor,    // [hidden]
    pub ln2_bias:   Tensor,    // [hidden]

    // MLP (fc1: hidden → mlp_dim, fc2: mlp_dim → hidden)
    pub mlp_fc1_weight: Tensor, // [hidden, mlp_dim]
    pub mlp_fc1_bias:   Tensor, // [mlp_dim]
    pub mlp_fc2_weight: Tensor, // [mlp_dim, hidden]
    pub mlp_fc2_bias:   Tensor, // [hidden]
}

impl SigLipBlock {
    /// Forward pass for one transformer block.
    ///
    /// `x`: `[seq, hidden]` → returns `[seq, hidden]`
    pub fn forward(
        &self,
        x: &Tensor,
        ops: &OpDispatcher,
        cfg: &SigLipConfig,
    ) -> Result<Tensor> {
        // 1. Pre-attention LayerNorm
        let normed = ops.layernorm(x, &self.ln1_weight, Some(&self.ln1_bias), cfg.layer_norm_eps)?;

        // 2. Q, K, V projections with bias
        //    normed: [seq, hidden], weight: [hidden, hidden] → [seq, hidden]
        let q = ops.add_bias(&ops.matmul(&normed, &self.attn_q_weight)?, &self.attn_q_bias)?;
        let k = ops.add_bias(&ops.matmul(&normed, &self.attn_k_weight)?, &self.attn_k_bias)?;
        let v = ops.add_bias(&ops.matmul(&normed, &self.attn_v_weight)?, &self.attn_v_bias)?;

        // 3. Bidirectional multi-head attention (no RoPE, no causal mask)
        let attn_out = ops.vit_attention(
            &q, &k, &v,
            &self.attn_o_weight, &self.attn_o_bias,
            cfg.n_heads, cfg.head_dim,
        )?;

        // 4. Residual
        let x = ops.add(x, &attn_out)?;

        // 5. Pre-FFN LayerNorm
        let normed2 = ops.layernorm(&x, &self.ln2_weight, Some(&self.ln2_bias), cfg.layer_norm_eps)?;

        // 6. GELU FFN: fc1 → gelu → fc2
        let h = ops.add_bias(&ops.matmul(&normed2, &self.mlp_fc1_weight)?, &self.mlp_fc1_bias)?;
        let h = ops.gelu(&h)?;
        let h = ops.add_bias(&ops.matmul(&h, &self.mlp_fc2_weight)?, &self.mlp_fc2_bias)?;

        // 7. Residual
        ops.add(&x, &h)
    }
}

// ─── Full encoder ─────────────────────────────────────────────────────────────

/// SigLIP-So400M image encoder.
///
/// Processes one 448×448 image tile into a sequence of 1024 feature vectors
/// (patch_0, …, patch_1023), shape [1024, 1152].
///
/// Key differences from the HuggingFace config assumption:
///   - No CLS token — SigLIP uses all patch features directly
///   - Positional embedding is [4900, 1152] (supports up to 70×70 tiles);
///     sliced to n_patches at runtime
///   - No pre-encoder LayerNorm; only post-encoder LayerNorm (v.post_ln)
///
/// Multiple tiles are processed independently; the Resampler then aggregates
/// them into 64 visual tokens per tile.
pub struct SigLipEncoder {
    // Patch embedding
    pub patch_weight: Tensor,  // [1152, 3, 14, 14] — v.patch_embd.weight

    pub patch_bias:   Tensor,  // [1152]            — v.patch_embd.bias

    // Learned positional embedding — sliced to [n_patches, 1152] at runtime
    pub pos_embed: Tensor,     // [4900, 1152]       — v.position_embd.weight

    // Post-encoder LayerNorm (no pre-LN in this model)
    pub post_ln_weight: Tensor, // [1152]            — v.post_ln.weight
    pub post_ln_bias:   Tensor, // [1152]            — v.post_ln.bias

    // Transformer blocks
    pub layers: Vec<SigLipBlock>,

    pub cfg: SigLipConfig,
}

impl SigLipEncoder {
    /// Encode one 448×448 image tile.
    ///
    /// `image_pixels`: `[3, 448, 448]` float32, normalised to SigLIP mean/std
    /// (mean=0.5, std=0.5 → range ≈ [−1, 1]).
    ///
    /// Returns `[n_patches, 1152]` — 1024 patch features (no CLS token).
    /// The Resampler (Phase 2.2) then compresses these into 64 visual tokens.
    pub fn forward(&self, image_pixels: &Tensor, ops: &OpDispatcher) -> Result<Tensor> {
        let n_patches = {
            let h = self.cfg.image_size / self.cfg.patch_size;
            h * h  // (448/14)^2 = 1024
        };

        // 1. Patch embedding: [3, 448, 448] → [1024, 1152]
        let patches = ops.patch_embed(
            image_pixels,
            &self.patch_weight,
            &self.patch_bias,
            self.cfg.patch_size,
        )?;

        // 2. Slice positional embedding to [n_patches, hidden_dim] and add
        //    pos_embed is stored as [4900, 1152]; we take the first n_patches rows.
        let pos_slice_data = self.pos_embed.data[..n_patches * self.cfg.hidden_dim].to_vec();
        let pos_slice = Tensor::new(pos_slice_data, vec![n_patches, self.cfg.hidden_dim])?;
        let mut x = ops.add(&patches, &pos_slice)?;

        // 3. No pre-encoder LayerNorm in this model (v.post_ln only)

        // 4. 27 transformer blocks
        for layer in &self.layers {
            x = layer.forward(&x, ops, &self.cfg)?;
        }

        // 5. Post-encoder LayerNorm
        x = ops.layernorm(&x, &self.post_ln_weight, Some(&self.post_ln_bias), self.cfg.layer_norm_eps)?;

        Ok(x) // [1024, 1152]
    }
}

// ─── Test helpers ─────────────────────────────────────────────────────────────

/// Build a single `SigLipBlock` with all-zero weights (useful for unit tests).
pub fn zero_block(hidden: usize, mlp_dim: usize) -> SigLipBlock {
    let ww = Tensor::zeros(vec![hidden, hidden]);
    let wb = Tensor::zeros(vec![hidden]);
    let fw = Tensor::zeros(vec![hidden, mlp_dim]);
    let fb = Tensor::zeros(vec![mlp_dim]);
    let bw = Tensor::zeros(vec![mlp_dim, hidden]);
    let bb = Tensor::zeros(vec![hidden]);
    // For LN weights, use ones (zero weight → all-zero output which is less useful)
    let ln_w = Tensor::new(vec![1.0f32; hidden], vec![hidden]).unwrap();
    let ln_b = Tensor::zeros(vec![hidden]);
    SigLipBlock {
        attn_q_weight: ww.clone(), attn_q_bias: wb.clone(),
        attn_k_weight: ww.clone(), attn_k_bias: wb.clone(),
        attn_v_weight: ww.clone(), attn_v_bias: wb.clone(),
        attn_o_weight: ww.clone(), attn_o_bias: wb.clone(),
        ln1_weight: ln_w.clone(), ln1_bias: ln_b.clone(),
        ln2_weight: ln_w.clone(), ln2_bias: ln_b.clone(),
        mlp_fc1_weight: fw,       mlp_fc1_bias: fb,
        mlp_fc2_weight: bw,       mlp_fc2_bias: bb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::dispatch::OpDispatcher;

    fn ops() -> OpDispatcher { OpDispatcher::new() }

    #[test]
    fn test_siglip_block_output_shape() {
        // A block with zero weights should not change shape
        let cfg = SigLipConfig { hidden_dim: 8, n_heads: 2, head_dim: 4, mlp_dim: 16, ..SigLipConfig::default() };
        let block = zero_block(cfg.hidden_dim, cfg.mlp_dim);
        let x = Tensor::zeros(vec![5, cfg.hidden_dim]);
        let out = block.forward(&x, &ops(), &cfg).unwrap();
        assert_eq!(out.shape, vec![5, cfg.hidden_dim]);
    }

    #[test]
    fn test_siglip_block_residual_with_identity_ln() {
        // With all-zero attention + FFN weights, output = layernorm(input) + layernorm(layernorm(input))
        // At minimum, output shape must match and values must be finite
        let cfg = SigLipConfig { hidden_dim: 4, n_heads: 1, head_dim: 4, mlp_dim: 8, ..SigLipConfig::default() };
        let block = zero_block(cfg.hidden_dim, cfg.mlp_dim);
        let x = Tensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 4]).unwrap();
        let out = block.forward(&x, &ops(), &cfg).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        assert!(out.data.iter().all(|v| v.is_finite()), "output contains non-finite values");
    }
}
