//! Perceiver Resampler projector for MiniCPM-V-2.6.
//!
//! Compresses 1025 ViT patch features ([1025, 1152]) into 64 visual tokens
//! in Qwen2-7B embedding space ([64, 3584]).
//!
//! Architecture:
//!   - 64 learned query embeddings [64, 3584]
//!   - 1 cross-attention layer: queries attend to ViT key/values
//!   - Single attention head count derived from d_model (typically 16 heads × 224 head_dim = 3584)
//!   - LayerNorm on queries (pre) and ViT features (pre-KV)
//!   - Optional final linear projection [3584 → 3584]
//!
//! Key distinction from self-attention: Q comes from the learned query bank,
//! while K and V come from the ViT output.  The sequence length of the output
//! is always 64 (number of learned queries), regardless of the number of tiles.

use crate::core::{error::Result, tensor::Tensor};
use crate::ops::dispatch::OpDispatcher;

/// Perceiver Resampler configuration.
pub struct ResamplerConfig {
    pub n_queries:  usize,  // 64
    pub d_model:    usize,  // 3584  (Qwen2-7B hidden dim)
    pub kv_dim:     usize,  // 1152  (SigLIP hidden dim — ViT output dim, input to kv_weight)
    pub n_heads:    usize,  // 16
    pub head_dim:   usize,  // 224   (= d_model / n_heads)
    pub layer_norm_eps: f32, // 1e-6
}

impl Default for ResamplerConfig {
    fn default() -> Self {
        Self {
            n_queries: 64,
            d_model:   3584,
            kv_dim:    1152,
            n_heads:   16,
            head_dim:  224,  // 3584 / 16
            layer_norm_eps: 1e-6,
        }
    }
}

/// Perceiver Resampler.
///
/// Inputs:
///   `vit_features`: `[n_vit_tokens, kv_dim]` — all tokens from the ViT encoder
///   (typically [1024, 1152] for a single 448×448 tile — no CLS in this model).
///
/// Output: `[n_queries, d_model]` — 64 visual tokens ready for injection into
/// the Qwen2-7B LLM embedding sequence.
///
/// Architecture (from GGUF tensor map):
///   1. kv_proj     = vit_features  @ kv_weight        [kv_dim → d_model]
///   2. kv_normed   = layernorm(kv_proj, ln_kv)
///   3. q_normed    = layernorm(query_embeds, ln_q)
///   4. K = (kv_normed + pos_embed_k[:n_vit]) @ attn_k_weight + attn_k_bias
///   5. V = kv_normed @ attn_v_weight + attn_v_bias
///   6. Q = q_normed @ attn_q_weight + attn_q_bias
///   7. out = cross_attn(Q, K, V) → output projection
///   8. out = layernorm(query_embeds + out, ln_post)
///   9. out = out @ proj_weight
pub struct Resampler {
    /// Learned query embeddings: [n_queries, d_model]
    pub query_embeds: Tensor,

    /// Initial linear projection of ViT features from kv_dim → d_model space.
    /// resampler.kv.weight [kv_dim, d_model] (GGUF: 3584×1152 = out×in reversed)
    pub kv_weight: Tensor,  // [kv_dim, d_model]

    /// LayerNorm on query embeddings before Q projection
    pub ln_q_weight: Tensor,  // [d_model]
    pub ln_q_bias:   Tensor,  // [d_model]

    /// LayerNorm on kv-projected ViT features (operates in d_model space)
    pub ln_kv_weight: Tensor, // [d_model]
    pub ln_kv_bias:   Tensor, // [d_model]

    /// Cross-attention projections — all operate in d_model × d_model space
    pub attn_q_weight: Tensor, // [d_model, d_model]
    pub attn_q_bias:   Tensor, // [d_model]
    pub attn_k_weight: Tensor, // [d_model, d_model]
    pub attn_k_bias:   Tensor, // [d_model]
    pub attn_v_weight: Tensor, // [d_model, d_model]
    pub attn_v_bias:   Tensor, // [d_model]
    pub attn_o_weight: Tensor, // [d_model, d_model]
    pub attn_o_bias:   Tensor, // [d_model]

    /// Positional embedding for keys — added to kv_projected before attn_k_weight.
    /// Supports up to 4900 positions (70×70); sliced to n_vit at runtime.
    pub pos_embed_k: Tensor,  // [4900, d_model]  (resampler.pos_embed_k)

    /// Post-attention LayerNorm
    pub ln_post_weight: Tensor, // [d_model]
    pub ln_post_bias:   Tensor, // [d_model]

    /// Final linear projection (identity if weight = I)
    pub proj_weight: Tensor,   // [d_model, d_model]

    pub cfg: ResamplerConfig,
}

impl Resampler {
    /// Compress ViT features into 64 visual tokens.
    ///
    /// `vit_features`: `[n_vit_tokens, kv_dim]`  
    /// Returns: `[n_queries, d_model]`
    pub fn forward(&self, vit_features: &Tensor, ops: &OpDispatcher) -> Result<Tensor> {
        let cfg = &self.cfg;
        let n_vit = vit_features.shape[0];

        // 1. Project ViT features from kv_dim → d_model space
        //    vit_features [n_vit, kv_dim] @ kv_weight [kv_dim, d_model] → [n_vit, d_model]
        let kv_proj = ops.matmul(vit_features, &self.kv_weight)?;

        // 2. Normalise kv-projected features (ln_kv operates in d_model space)
        let kv_normed = ops.layernorm(
            &kv_proj,
            &self.ln_kv_weight,
            Some(&self.ln_kv_bias),
            cfg.layer_norm_eps,
        )?;

        // 3. Normalise queries
        let q_normed = ops.layernorm(
            &self.query_embeds,
            &self.ln_q_weight,
            Some(&self.ln_q_bias),
            cfg.layer_norm_eps,
        )?;

        // 4. Add positional embedding to K input (sliced to n_vit rows)
        let pos_k_data = self.pos_embed_k.data[..n_vit * cfg.d_model].to_vec();
        let pos_k = Tensor::new(pos_k_data, vec![n_vit, cfg.d_model])?;
        let kv_with_pos = ops.add(&kv_normed, &pos_k)?;

        // 5. Project Q, K, V  (all in d_model × d_model space now)
        let q = ops.add_bias(&ops.matmul(&q_normed,    &self.attn_q_weight)?, &self.attn_q_bias)?;
        let k = ops.add_bias(&ops.matmul(&kv_with_pos, &self.attn_k_weight)?, &self.attn_k_bias)?;
        let v = ops.add_bias(&ops.matmul(&kv_normed,   &self.attn_v_weight)?, &self.attn_v_bias)?;

        // 6. Cross-attention: Q attends to K/V
        let attn_out = cross_attention_f32(
            &q, &k, &v,
            &self.attn_o_weight,
            &self.attn_o_bias,
            cfg.n_heads,
            cfg.head_dim,
        )?;

        // 7. Residual: original query_embeds + attention output
        let x = ops.add(&self.query_embeds, &attn_out)?;

        // 8. Post-attention LayerNorm
        let x = ops.layernorm(&x, &self.ln_post_weight, Some(&self.ln_post_bias), cfg.layer_norm_eps)?;

        // 9. Final linear projection
        ops.matmul(&x, &self.proj_weight)
        // Returns [n_queries, d_model] = [64, 3584]
    }
}

// ─── Cross-attention helper ───────────────────────────────────────────────────

/// Scaled dot-product cross-attention.
///
/// Q queries over K/V from a different sequence.
///   q: [q_len, n_head * head_dim]
///   k: [kv_len, n_head * head_dim]
///   v: [kv_len, n_head * head_dim]
///
/// Returns [q_len, out_dim] after output projection.
fn cross_attention_f32(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o_weight: &Tensor,
    o_bias: &Tensor,
    n_head: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let q_len  = q.shape[0];
    let kv_len = k.shape[0];
    let d_model = n_head * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut out = vec![0.0f32; q_len * d_model];

    for h in 0..n_head {
        let h_off = h * head_dim;

        let q_h: Vec<f32> = (0..q_len)
            .flat_map(|s| q.data[s * d_model + h_off..s * d_model + h_off + head_dim].iter().copied())
            .collect();
        let k_h: Vec<f32> = (0..kv_len)
            .flat_map(|s| k.data[s * d_model + h_off..s * d_model + h_off + head_dim].iter().copied())
            .collect();
        let v_h: Vec<f32> = (0..kv_len)
            .flat_map(|s| v.data[s * d_model + h_off..s * d_model + h_off + head_dim].iter().copied())
            .collect();

        // Scores: [q_len, kv_len]
        let mut scores = vec![0.0f32; q_len * kv_len];
        for i in 0..q_len {
            for j in 0..kv_len {
                let dot: f32 = (0..head_dim)
                    .map(|d| q_h[i * head_dim + d] * k_h[j * head_dim + d])
                    .sum();
                scores[i * kv_len + j] = dot * scale;
            }
        }

        // Softmax over kv_len for each query
        for i in 0..q_len {
            let row = &mut scores[i * kv_len..(i + 1) * kv_len];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
            for x in row.iter_mut() {
                *x = (*x - max).exp() / sum;
            }
        }

        // Context: [q_len, head_dim]
        for i in 0..q_len {
            for d in 0..head_dim {
                let val: f32 = (0..kv_len)
                    .map(|j| scores[i * kv_len + j] * v_h[j * head_dim + d])
                    .sum();
                out[i * d_model + h_off + d] = val;
            }
        }
    }

    // Output projection: [q_len, d_model] @ [d_model, out_dim] + bias
    let out_dim = o_weight.shape[1];
    let mut projected = vec![0.0f32; q_len * out_dim];
    for i in 0..q_len {
        for j in 0..out_dim {
            let dot: f32 = (0..d_model)
                .map(|kk| out[i * d_model + kk] * o_weight.data[kk * out_dim + j])
                .sum();
            projected[i * out_dim + j] = dot + o_bias.data[j];
        }
    }

    Tensor::new(projected, vec![q_len, out_dim])
}

// ─── Test helpers ─────────────────────────────────────────────────────────────

/// Build a Resampler where everything is identity / ones (useful for unit tests).
/// Returns (resampler, cfg).
pub fn identity_resampler(cfg: ResamplerConfig) -> Resampler {
    fn eye(n: usize) -> Tensor {
        let mut d = vec![0.0f32; n * n];
        for i in 0..n { d[i * n + i] = 1.0; }
        Tensor::new(d, vec![n, n]).unwrap()
    }
    let ln_w_d  = Tensor::new(vec![1.0f32; cfg.d_model], vec![cfg.d_model]).unwrap();
    let ln_b_d  = Tensor::zeros(vec![cfg.d_model]);
    let zeros_d = Tensor::zeros(vec![cfg.d_model]);

    // kv_weight: [kv_dim, d_model] — zero initial projection (no ViT signal in identity test)
    let kv_w = Tensor::zeros(vec![cfg.kv_dim, cfg.d_model]);
    // pos_embed_k: [4900, d_model] — zeros (same max size as real model, safe for any n_vit)
    let pos_embed_k = Tensor::zeros(vec![4900, cfg.d_model]);

    let q_w  = eye(cfg.d_model);
    let o_w  = eye(cfg.d_model);
    let o_b  = Tensor::zeros(vec![cfg.d_model]);

    Resampler {
        query_embeds:   Tensor::zeros(vec![cfg.n_queries, cfg.d_model]),
        kv_weight:      kv_w,
        ln_q_weight:    ln_w_d.clone(),
        ln_q_bias:      ln_b_d.clone(),
        ln_kv_weight:   ln_w_d.clone(),
        ln_kv_bias:     ln_b_d.clone(),
        attn_q_weight:  q_w.clone(),
        attn_q_bias:    zeros_d.clone(),
        attn_k_weight:  eye(cfg.d_model),
        attn_k_bias:    zeros_d.clone(),
        attn_v_weight:  eye(cfg.d_model),
        attn_v_bias:    zeros_d.clone(),
        attn_o_weight:  o_w.clone(),
        attn_o_bias:    o_b,
        pos_embed_k,
        ln_post_weight: ln_w_d,
        ln_post_bias:   ln_b_d,
        proj_weight:    o_w,
        cfg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::dispatch::OpDispatcher;

    fn ops() -> OpDispatcher { OpDispatcher::new() }

    #[test]
    fn test_resampler_output_shape() {
        // Minimal dims: 4 queries, d_model=8, kv_dim=6, 2 heads, head_dim=4
        let cfg = ResamplerConfig {
            n_queries: 4, d_model: 8, kv_dim: 6, n_heads: 2, head_dim: 4, layer_norm_eps: 1e-5,
        };
        let r = identity_resampler(cfg);
        let vit_feats = Tensor::zeros(vec![10, 6]); // 10 ViT tokens
        let out = r.forward(&vit_feats, &ops()).unwrap();
        assert_eq!(out.shape, vec![4, 8]); // n_queries × d_model
    }

    #[test]
    fn test_resampler_output_finite() {
        let cfg = ResamplerConfig {
            n_queries: 4, d_model: 8, kv_dim: 6, n_heads: 2, head_dim: 4, layer_norm_eps: 1e-5,
        };
        let r = identity_resampler(cfg);
        let vit_feats = Tensor::new(
            (0..60).map(|i| i as f32 * 0.1).collect(),
            vec![10, 6],
        ).unwrap();
        let out = r.forward(&vit_feats, &ops()).unwrap();
        assert!(out.data.iter().all(|v| v.is_finite()),
            "Resampler output contains non-finite values");
    }

    #[test]
    fn test_resampler_query_count_independent_of_vit_tokens() {
        // 64 queries should always produce 64 output rows regardless of input length
        let cfg = ResamplerConfig {
            n_queries: 8, d_model: 4, kv_dim: 4, n_heads: 1, head_dim: 4, layer_norm_eps: 1e-5,
        };
        for n_vit in [1, 5, 25, 100] {
            let r = identity_resampler(ResamplerConfig {
                n_queries: 8, d_model: 4, kv_dim: 4, n_heads: 1, head_dim: 4, layer_norm_eps: 1e-5,
            });
            let vit = Tensor::zeros(vec![n_vit, 4]);
            let out = r.forward(&vit, &ops()).unwrap();
            assert_eq!(out.shape[0], 8,
                "n_vit={n_vit}: expected 8 output queries, got {}", out.shape[0]);
        }
        let _ = cfg;
    }
}
