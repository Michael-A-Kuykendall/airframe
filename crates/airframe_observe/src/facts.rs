//! InferenceFact — the domain fact vocabulary for inference observation.
//!
//! This is the domain-specific Fact enum that plugs into d0-engine's
//! generic ReactiveGraph<InferenceFact>.
//!
//! Each variant corresponds to a data point in the inference graph.
//! When the forward pass produces one of these, it is asserted into
//! the ReactiveGraph, which broadcasts it to all registered observers.

use d0_engine::AlphaKey;

/// A fact about an inference run.
///
/// Three tiers following d0-engine conventions:
/// - Structural (Tier 1): emitted as data is produced
/// - Semantic (Tier 2): derived by rules (e.g. "layer output is stable")
/// - Consequent (Tier 3): drives external actions (not stored)
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum InferenceFact {
    // ── Tier 1: Structural ────────────────────────────────────────────────
    /// Hidden state output after transformer layer N (position P).
    /// Emitted by vault_seed after each LlamaBlock::forward() call.
    LayerOutput {
        layer_idx: u32,
        position: u32,
        rms_bits: u32, // f32::to_bits() — Hash-safe encoding
        checksum: i64,
    },

    /// Final logits after output_norm + output_proj (vocab-size vector).
    /// Emitted by vault_seed after the full forward pass.
    /// This is the comparison point for candle cross-validation.
    FinalLogits {
        position: u32,
        rms_bits: u32, // f32::to_bits()
        checksum: i64,
    },

    /// Decoded output text (post-sampling, incremental).
    /// Emitted by the inference server per token.
    /// This is what FseControl uses for policy scanning.
    OutputToken { step: u32, token_id: u32 },

    // ── Tier 2: Semantic ─────────────────────────────────────────────────
    /// Layer output RMS is within expected bounds (no explosion/collapse).
    LayerOutputStable { layer_idx: u32 },

    /// Final logits contain no NaN or Inf values.
    LogitsClean,

    // ── Tier 3: Consequent ───────────────────────────────────────────────
    // (Not stored — consumed immediately by the session loop)
    /// Vault oracle row should be written for this layer.
    WriteOracleRow { layer_idx: u32 },

    /// Candle cross-validation should be triggered for these logits.
    TriggerCandleCompare { rms_bits: u32, checksum: i64 },
}

/// Discriminant constants for alpha indexing.
/// Each variant family gets a unique u64 key.
pub const KEY_LAYER_OUTPUT: u64 = 1;
pub const KEY_FINAL_LOGITS: u64 = 2;
pub const KEY_OUTPUT_TOKEN: u64 = 3;
pub const KEY_LAYER_STABLE: u64 = 4;
pub const KEY_LOGITS_CLEAN: u64 = 5;
pub const KEY_WRITE_ORACLE: u64 = 6;
pub const KEY_CANDLE_COMPARE: u64 = 7;

/// Map an InferenceFact to its AlphaKey for d0-engine dispatch.
///
/// This is the discriminant function — maps fact variants to index keys.
/// Rules registered on a key only fire when a fact with that key arrives.
/// None = fact should not trigger any rules (terminal/consequent facts).
pub fn alpha_key_of(fact: &InferenceFact) -> Option<AlphaKey> {
    match fact {
        InferenceFact::LayerOutput { .. } => Some(AlphaKey(KEY_LAYER_OUTPUT)),
        InferenceFact::FinalLogits { .. } => Some(AlphaKey(KEY_FINAL_LOGITS)),
        InferenceFact::OutputToken { .. } => Some(AlphaKey(KEY_OUTPUT_TOKEN)),
        InferenceFact::LayerOutputStable { .. } => Some(AlphaKey(KEY_LAYER_STABLE)),
        InferenceFact::LogitsClean => Some(AlphaKey(KEY_LOGITS_CLEAN)),
        // Tier 3 consequents — don't trigger further rules
        InferenceFact::WriteOracleRow { .. } => None,
        InferenceFact::TriggerCandleCompare { .. } => None,
    }
}

/// Helper: encode f32 as u32 bits for hash-safe storage.
pub fn f32_to_bits(v: f32) -> u32 {
    v.to_bits()
}

/// Helper: decode u32 bits back to f32.
pub fn bits_to_f32(bits: u32) -> f32 {
    f32::from_bits(bits)
}

/// Compute RMS of a float slice.
pub fn rms(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let sq: f32 = v.iter().map(|x| x * x).sum();
    (sq / v.len() as f32).sqrt()
}

/// Compute row-wise checksum — deterministic, catches silent corruption.
pub fn checksum(v: &[f32]) -> i64 {
    v.iter()
        .map(|x| x.to_bits() as i64)
        .fold(0i64, |a, b| a.wrapping_add(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tier1_facts_have_alpha_keys() {
        let facts = vec![
            InferenceFact::LayerOutput {
                layer_idx: 0,
                position: 1,
                rms_bits: 0,
                checksum: 0,
            },
            InferenceFact::FinalLogits {
                position: 1,
                rms_bits: 0,
                checksum: 0,
            },
            InferenceFact::OutputToken {
                step: 0,
                token_id: 1,
            },
        ];
        for f in &facts {
            assert!(
                alpha_key_of(f).is_some(),
                "Tier 1 fact must have alpha key: {:?}",
                f
            );
        }
    }

    #[test]
    fn tier3_consequents_have_no_alpha_key() {
        let facts = vec![
            InferenceFact::WriteOracleRow { layer_idx: 0 },
            InferenceFact::TriggerCandleCompare {
                rms_bits: 0,
                checksum: 0,
            },
        ];
        for f in &facts {
            assert!(
                alpha_key_of(f).is_none(),
                "Tier 3 fact must not have alpha key: {:?}",
                f
            );
        }
    }

    #[test]
    fn distinct_variants_have_distinct_keys() {
        let k1 = alpha_key_of(&InferenceFact::LayerOutput {
            layer_idx: 0,
            position: 0,
            rms_bits: 0,
            checksum: 0,
        });
        let k2 = alpha_key_of(&InferenceFact::FinalLogits {
            position: 0,
            rms_bits: 0,
            checksum: 0,
        });
        assert_ne!(
            k1, k2,
            "LayerOutput and FinalLogits must have distinct alpha keys"
        );
    }
}
