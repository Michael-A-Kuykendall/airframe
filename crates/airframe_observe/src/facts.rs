//! InferenceFact — the domain fact vocabulary for inference observation.
//!
//! This is the domain-specific Fact enum that plugs into d0-engine's
//! generic ReactiveGraph<InferenceFact>.
//!
//! Each variant corresponds to a data point in the inference graph.
//! When the forward pass produces one of these, it is asserted into
//! the ReactiveGraph, which broadcasts it to all registered observers.

use d0_engine::AlphaKey;

/// Why generation halted.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum HaltReason {
    EosToken,
    MaxTokensReached,
    ExtraStopToken,
}


#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum KernelKind {
    Qkv,
    AttnOut,
    FfnDown,
    FfnProj,
    FullLayer, // combined when not measuring per-kernel
}

/// Why a yield point was requested.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum YieldReason {
    TdrBudgetExceeded,
    LayerBoundary,
}

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

    // ── Tier 1: Dispatch timing (for TDR scheduling via Saturation Fabric) ──
    /// CPU-side elapsed time for a GPU kernel dispatch at a given layer.
    /// Emitted by inference.rs after each queue.submit()+poll() completes.
    /// The fabric uses this to decide whether to yield on the next dispatch.
    DispatchTiming {
        layer: u32,
        kernel: KernelKind,
        elapsed_ms: u32,
    },

    /// A GPU dispatch completed — carries layer index, what ran, and wall time.
    /// This is the primary TDR input fact. Rules accumulate elapsed_ms and
    /// derive TdrRiskHigh when budget is approached.
    DispatchCompleted {
        layer: u32,
        kernel: KernelKind,
        elapsed_ms: u32,
    },

    // ── Tier 2: TDR risk derived from accumulated dispatch timings ────────
    /// Accumulated dispatch time exceeds TDR threshold — yield required.
    TdrRiskHigh { layer: u32 },

    /// Accumulated dispatch time is safe — continue batching.
    TdrRiskLow { layer: u32 },

    // ── Tier 3: Yield decisions (consequents — not stored long-term) ──────
    /// Inference loop must submit+poll now.
    YieldNow { layer: u32, reason: YieldReason },

    // ── ISF: Inference Saturation Fabric facts ───────────────────────────
    // These facts drive the generate() loop via D0 reactive graph.
    // The loop itself disappears — replaced by fact assertion + saturation.

    // Tier 1: Input stream — one fact per prompt token
    /// A single token from the input prompt. Emitted by generate() before prefill.
    PromptToken {
        position: u32,
        token_id: u32,
    },

    // Tier 2: Embedding extracted for a token position
    /// The embedding vector for a token position is ready (dequanted from VRAM).
    /// Rules assert this after GPU dequant completes.
    /// Deduplicated: if token_id appears N times, dequant fires once.
    EmbeddingReady {
        position: u32,
        token_id: u32,
    },

    // Tier 2: All prompt embeddings collected — prefill can fire
    PrefillBatchReady { token_count: u32 },

    // Tier 2: Prefill complete — first logits available
    PrefillComplete { position: u32 },

    // Tier 1: One decode step — self-asserted by the fabric after each token
    /// Unique per step (step field ensures dedup doesn't block re-assertion).
    DecodeStep {
        step: u32,
        token_id: u32,
    },

    // Tier 3: Decode step produced logits — sample and emit
    DecodeLogitsReady {
        step: u32,
    },

    // Tier 3: Halt the generation loop (EOS hit or max_tokens reached)
    GenerationHalt { reason: HaltReason },

    // New for model family workshop using Saturation Fabric and vault
    FamilyContext {
        family: String,
        quant: String,
        has_qk_norm: bool,
    },
    PerTensorOutput {
        layer_idx: u32,
        position: u32,
        q_rms_bits: u32,
        k_rms_bits: u32,
        v_rms_bits: u32,
        post_rms_bits: u32,
        ffn_rms_bits: u32,
        output_rms_bits: u32,
        q_checksum: i64,
        k_checksum: i64,
        v_checksum: i64,
        post_checksum: i64,
        ffn_checksum: i64,
        output_checksum: i64,
    },
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
pub const KEY_FAMILY_CONTEXT: u64 = 8;
pub const KEY_PER_TENSOR_OUTPUT: u64 = 9;
pub const KEY_DISPATCH_TIMING: u64 = 10;
pub const KEY_TDR_RISK_HIGH: u64 = 11;
pub const KEY_DISPATCH_COMPLETED: u64 = 18;
// ISF: Inference Saturation Fabric keys
pub const KEY_PROMPT_TOKEN: u64 = 12;
pub const KEY_EMBEDDING_READY: u64 = 13;
pub const KEY_PREFILL_BATCH_READY: u64 = 14;
pub const KEY_PREFILL_COMPLETE: u64 = 15;
pub const KEY_DECODE_STEP: u64 = 16;
pub const KEY_DECODE_LOGITS_READY: u64 = 17;

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
        InferenceFact::FamilyContext { .. } => Some(AlphaKey(KEY_FAMILY_CONTEXT)),
        InferenceFact::PerTensorOutput { .. } => Some(AlphaKey(KEY_PER_TENSOR_OUTPUT)),
        InferenceFact::DispatchTiming { .. } => Some(AlphaKey(KEY_DISPATCH_TIMING)),
        InferenceFact::DispatchCompleted { .. } => Some(AlphaKey(KEY_DISPATCH_COMPLETED)),
        InferenceFact::TdrRiskHigh { .. } => Some(AlphaKey(KEY_TDR_RISK_HIGH)),
        // Tier 2 derived — no further rules fire on these
        InferenceFact::TdrRiskLow { .. } => None,
        // ISF facts
        InferenceFact::PromptToken { .. } => Some(AlphaKey(KEY_PROMPT_TOKEN)),
        InferenceFact::EmbeddingReady { .. } => Some(AlphaKey(KEY_EMBEDDING_READY)),
        InferenceFact::PrefillBatchReady { .. } => Some(AlphaKey(KEY_PREFILL_BATCH_READY)),
        InferenceFact::PrefillComplete { .. } => Some(AlphaKey(KEY_PREFILL_COMPLETE)),
        InferenceFact::DecodeStep { .. } => Some(AlphaKey(KEY_DECODE_STEP)),
        InferenceFact::DecodeLogitsReady { .. } => Some(AlphaKey(KEY_DECODE_LOGITS_READY)),
        // Tier 3 consequents — don't trigger further rules
        InferenceFact::WriteOracleRow { .. } => None,
        InferenceFact::TriggerCandleCompare { .. } => None,
        InferenceFact::YieldNow { .. } => None,
        InferenceFact::GenerationHalt { .. } => None,
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
            InferenceFact::FamilyContext {
                family: "test".to_string(),
                quant: "Q4_K_M".to_string(),
                has_qk_norm: true,
            },
            InferenceFact::PerTensorOutput {
                layer_idx: 0,
                position: 1,
                q_rms_bits: 0,
                k_rms_bits: 0,
                v_rms_bits: 0,
                post_rms_bits: 0,
                ffn_rms_bits: 0,
                output_rms_bits: 0,
                q_checksum: 0,
                k_checksum: 0,
                v_checksum: 0,
                post_checksum: 0,
                ffn_checksum: 0,
                output_checksum: 0,
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
