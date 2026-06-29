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
    AttnQ { layer: usize },
    AttnK { layer: usize },
    AttnV { layer: usize },
    AttnO { layer: usize },
    AttnNorm { layer: usize },
    /// Per-head Q RMSNorm weight (Qwen3). Shape: [head_dim]. GGUF: blk.N.attn_q_norm.weight
    AttnQNorm { layer: usize },
    /// Per-head K RMSNorm weight (Qwen3). Shape: [head_dim]. GGUF: blk.N.attn_k_norm.weight
    AttnKNorm {
        layer: usize,
    },
    /// Per-head attention logit scale (Qwen3). Shape: [n_head]. GGUF: blk.N.attention.scale
    /// Multiplies attention scores after 1/sqrt(head_dim). Defaults to 1.0/sqrt(head_dim) when absent.
    AttentionScale {
        layer: usize,
    },

    FfnGate { layer: usize },
    FfnUp { layer: usize },
    FfnDown { layer: usize },
    FfnNorm { layer: usize },

    // Final layer
    OutputNorm, // GGUF: output_norm.weight
    OutputProj,
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
