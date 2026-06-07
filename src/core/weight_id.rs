//! Canonical weight identifiers for Llama model tensors.
//!
//! Maps semantic weight names (e.g., `AttnQ { layer: 0 }`) to
//! GGUF tensor names (e.g., `blk.0.attn_q.weight`).

/// Weight tensor identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WeightId {
    // Token embedding
    TokenEmbed,

    // Per-layer weights (layer index included)
    AttnQ {
        layer: usize,
    },
    AttnK {
        layer: usize,
    },
    AttnV {
        layer: usize,
    },
    AttnO {
        layer: usize,
    },
    AttnNorm {
        layer: usize,
    },
    /// Per-head Q RMSNorm weight (Qwen3). Shape: [head_dim]. GGUF: blk.N.attn_q_norm.weight
    AttnQNorm {
        layer: usize,
    },
    /// Per-head K RMSNorm weight (Qwen3). Shape: [head_dim]. GGUF: blk.N.attn_k_norm.weight
    AttnKNorm {
        layer: usize,
    },

    FfnGate {
        layer: usize,
    },
    FfnUp {
        layer: usize,
    },
    FfnDown {
        layer: usize,
    },
    FfnNorm {
        layer: usize,
    },

    // Final layer
    OutputNorm, // GGUF: output_norm.weight
    OutputProj,

    // ── Vision encoder (SigLIP ViT) ──────────────────────────────────────────
    // GGUF prefix: "v."
    // Patch embedding
    VisionPatchEmbedWeight, // v.patch_embd.weight   [1152, 3, 14, 14] F16
    VisionPatchEmbedBias,   // v.patch_embd.bias     [1152]            F32
    // Positional embedding (shared across all input sizes via slicing)
    VisionPosEmbedWeight, // v.position_embd.weight [4900, 1152]     F16
    // Post-encoder LayerNorm (no pre-LN in this GGUF)
    VisionPostNormWeight, // v.post_ln.weight      [1152]            F32
    VisionPostNormBias,   // v.post_ln.bias        [1152]            F32

    // Per ViT block  (layer index = 0..26)
    VisionBlkAttnQWeight(usize), // v.blk.N.attn_q.weight   [1152, 1152]  F16
    VisionBlkAttnQBias(usize),   // v.blk.N.attn_q.bias     [1152]        F32
    VisionBlkAttnKWeight(usize), // v.blk.N.attn_k.weight   [1152, 1152]  F16
    VisionBlkAttnKBias(usize),   // v.blk.N.attn_k.bias     [1152]        F32
    VisionBlkAttnVWeight(usize), // v.blk.N.attn_v.weight   [1152, 1152]  F16
    VisionBlkAttnVBias(usize),   // v.blk.N.attn_v.bias     [1152]        F32
    VisionBlkAttnOutWeight(usize), // v.blk.N.attn_out.weight [1152, 1152] F16
    VisionBlkAttnOutBias(usize), // v.blk.N.attn_out.bias   [1152]        F32
    VisionBlkLn1Weight(usize),   // v.blk.N.ln1.weight      [1152]        F32
    VisionBlkLn1Bias(usize),     // v.blk.N.ln1.bias        [1152]        F32
    VisionBlkLn2Weight(usize),   // v.blk.N.ln2.weight      [1152]        F32
    VisionBlkLn2Bias(usize),     // v.blk.N.ln2.bias        [1152]        F32
    VisionBlkFfnUpWeight(usize), // v.blk.N.ffn_up.weight   [1152, 4304]  F16
    VisionBlkFfnUpBias(usize),   // v.blk.N.ffn_up.bias     [4304]        F32
    VisionBlkFfnDownWeight(usize), // v.blk.N.ffn_down.weight [4304, 1152] F16
    VisionBlkFfnDownBias(usize), // v.blk.N.ffn_down.bias   [1152]        F32

    // ── Perceiver Resampler ───────────────────────────────────────────────────
    // GGUF prefix: "resampler."
    ResamplerQuery,         // resampler.query          [64, 3584]      F32
    ResamplerKvWeight,      // resampler.kv.weight      [1152, 3584]    F16  (ViT→LLM space)
    ResamplerLnQWeight,     // resampler.ln_q.weight    [3584]          F32
    ResamplerLnQBias,       // resampler.ln_q.bias      [3584]          F32
    ResamplerLnKvWeight,    // resampler.ln_kv.weight   [3584]          F32  (post kv-proj)
    ResamplerLnKvBias,      // resampler.ln_kv.bias     [3584]          F32
    ResamplerAttnQWeight,   // resampler.attn.q.weight  [3584, 3584]    F16
    ResamplerAttnQBias,     // resampler.attn.q.bias    [3584]          F32
    ResamplerAttnKWeight,   // resampler.attn.k.weight  [3584, 3584]    F16
    ResamplerAttnKBias,     // resampler.attn.k.bias    [3584]          F32
    ResamplerAttnVWeight,   // resampler.attn.v.weight  [3584, 3584]    F16
    ResamplerAttnVBias,     // resampler.attn.v.bias    [3584]          F32
    ResamplerAttnOutWeight, // resampler.attn.out.weight [3584, 3584]   F16
    ResamplerAttnOutBias,   // resampler.attn.out.bias  [3584]          F32
    ResamplerPosEmbedK, // resampler.pos_embed_k    [4900, 3584]    F32  (positional embed for K)
    ResamplerLnPostWeight, // resampler.ln_post.weight [3584]          F32
    ResamplerLnPostBias, // resampler.ln_post.bias   [3584]          F32
    ResamplerProjWeight, // resampler.proj.weight    [3584, 3584]    F16
}

impl WeightId {
    /// Generate all WeightIds for a model with n_layer layers
    pub fn all_for_layers(n_layer: usize) -> Vec<WeightId> {
        let mut weights = vec![WeightId::TokenEmbed];

        for layer in 0..n_layer {
            weights.extend([
                WeightId::AttnQ { layer },
                WeightId::AttnK { layer },
                WeightId::AttnV { layer },
                WeightId::AttnO { layer },
                WeightId::AttnNorm { layer },
                WeightId::FfnGate { layer },
                WeightId::FfnUp { layer },
                WeightId::FfnDown { layer },
                WeightId::FfnNorm { layer },
            ]);
        }

        weights.extend([WeightId::OutputNorm, WeightId::OutputProj]);

        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_id_enumeration_for_tinylama() {
        let weights = WeightId::all_for_layers(22); // TinyLlama has 22 layers

        // Should have: 1 token_embed + 22*9 layer weights + 2 output weights = 201 total
        assert_eq!(weights.len(), 1 + 22 * 9 + 2);

        // Check first few weights
        assert_eq!(weights[0], WeightId::TokenEmbed);
        assert_eq!(weights[1], WeightId::AttnQ { layer: 0 });
        assert_eq!(weights[2], WeightId::AttnK { layer: 0 });

        // Check last few weights
        assert_eq!(weights[weights.len() - 2], WeightId::OutputNorm);
        assert_eq!(weights[weights.len() - 1], WeightId::OutputProj);

        // Verify layer 21 (last layer) weights are present
        assert!(weights.contains(&WeightId::AttnQ { layer: 21 }));
        assert!(weights.contains(&WeightId::FfnNorm { layer: 21 }));
    }

    #[test]
    fn test_weight_id_enumeration_small_model() {
        let weights = WeightId::all_for_layers(2); // 2-layer model

        // Should have: 1 + 2*9 + 2 = 21 total
        assert_eq!(weights.len(), 21);

        // Check structure
        assert_eq!(weights[0], WeightId::TokenEmbed);
        assert_eq!(weights[1], WeightId::AttnQ { layer: 0 });
        assert_eq!(weights[10], WeightId::AttnQ { layer: 1 });
        assert_eq!(weights[19], WeightId::OutputNorm);
        assert_eq!(weights[20], WeightId::OutputProj);
    }
}
