//! ObservationSession — the live runtime for a single inference pass.
//!
//! Wraps d0-engine's ReactiveGraph<InferenceFact> with the inference domain's
//! fact vocabulary and observer registration API.
//!
//! Usage:
//! 1. Create a session
//! 2. Register observers (vault oracle, candle compare, stability, etc.)
//! 3. During forward pass: call emit() for each layer output and final logits
//! 4. Call saturate() to run d0-engine to fixpoint
//! 5. Read results from observers

use crate::facts::{alpha_key_of, InferenceFact};
use crate::observers::{CandleCompareObserver, LayerStabilityObserver, VaultOracleObserver};
use d0_engine::{ClosureProgram, ReactiveGraph, RunBudget, RunResult};

/// The live observation session for one inference pass.
pub struct ObservationSession {
    graph: ReactiveGraph<InferenceFact>,
    vault_oracle: Option<VaultOracleObserver>,
    candle_compare: Option<CandleCompareObserver>,
}

impl ObservationSession {
    /// Create a new session. Observers must be registered before the first emit().
    pub fn new() -> Self {
        let program = ClosureProgram::new();
        let graph = ReactiveGraph::new(program, alpha_key_of);
        Self {
            graph,
            vault_oracle: None,
            candle_compare: None,
        }
    }

    /// Register the VaultOracleObserver.
    /// Also registers LayerStabilityObserver on the same key (free broadcast).
    pub fn register_vault_oracle(&mut self) -> &VaultOracleObserver {
        let observer = VaultOracleObserver::new();

        // Register oracle capture rule
        self.graph
            .program
            .register(VaultOracleObserver::alpha_key(), observer.rule());

        // Register stability check on the SAME alpha key — zero extra cost (FSE broadcast)
        self.graph.program.register(
            LayerStabilityObserver::alpha_key(),
            LayerStabilityObserver::rule(),
        );

        self.vault_oracle = Some(observer);
        self.vault_oracle.as_ref().unwrap()
    }

    /// Register the CandleCompareObserver.
    pub fn register_candle_compare(&mut self) -> &CandleCompareObserver {
        let observer = CandleCompareObserver::new();
        self.graph
            .program
            .register(CandleCompareObserver::alpha_key(), observer.rule());
        self.candle_compare = Some(observer);
        self.candle_compare.as_ref().unwrap()
    }

    /// Emit a fact into the session.
    /// Queues it for rule activation during the next saturate() call.
    /// Can also be called during forward pass — facts accumulate until saturate().
    pub fn emit(&mut self, fact: InferenceFact) {
        self.graph.assert(fact);
    }

    /// Convenience: emit a LayerOutput fact from raw values.
    pub fn emit_layer_output(&mut self, layer_idx: u32, position: u32, values: &[f32]) {
        let rms = crate::facts::rms(values);
        let cs = crate::facts::checksum(values);
        self.emit(InferenceFact::LayerOutput {
            layer_idx,
            position,
            rms_bits: rms.to_bits(),
            checksum: cs,
        });
    }

    /// Convenience: emit a FinalLogits fact from raw logit values.
    pub fn emit_final_logits(&mut self, position: u32, logits: &[f32]) {
        let rms = crate::facts::rms(logits);
        let cs = crate::facts::checksum(logits);
        self.emit(InferenceFact::FinalLogits {
            position,
            rms_bits: rms.to_bits(),
            checksum: cs,
        });
    }

    /// Run d0-engine to fixpoint. All observers receive their data.
    ///
    /// Returns RunResult with statistics.
    /// Call this after all facts for the current pass have been emitted.
    pub fn saturate(&mut self) -> RunResult {
        self.graph.run(RunBudget::default())
    }

    /// Access vault oracle captures after saturation.
    pub fn vault_oracle(&self) -> Option<&VaultOracleObserver> {
        self.vault_oracle.as_ref()
    }

    /// Access candle compare capture after saturation.
    pub fn candle_compare(&self) -> Option<&CandleCompareObserver> {
        self.candle_compare.as_ref()
    }

    /// Number of facts currently in the store.
    pub fn fact_count(&self) -> usize {
        self.graph.fact_count()
    }

    /// Check if a specific fact is currently asserted.
    pub fn contains(&self, fact: &InferenceFact) -> bool {
        self.graph.store.contains(fact)
    }

    /// Register for family workshop mode (uses Saturation Fabric for auto family analysis and vault recording).
    pub fn register_family_workshop(&mut self) {
        // Rules for divergence and formula will be added via Saturation Fabric in future extensions.
        // For now, enables emit of new facts for per-family debug passes.
        // Example rule for divergence (in full impl, compare to vault).
        self.graph.program.register(
            crate::facts::alpha_key_of(&InferenceFact::LayerOutput { layer_idx: 0, position: 0, rms_bits: 0, checksum: 0 }).unwrap(),
            |fact, _store| {
                if let InferenceFact::LayerOutput { layer_idx, .. } = fact {
                    // In full, derive divergence if not stable.
                    if *layer_idx == 0 {
                        // Placeholder for family divergence fact.
                    }
                }
                vec![]
            },
        );
    }

    /// Emit family context fact.
    pub fn emit_family_context(&mut self, family: String, quant: String, has_qk_norm: bool) {
        self.emit(InferenceFact::FamilyContext {
            family,
            quant,
            has_qk_norm,
        });
    }

    /// Emit per-tensor output fact (for Q/K/V/post/ffn/output stats per layer/position).
    pub fn emit_per_tensor_output(
        &mut self,
        layer_idx: u32,
        position: u32,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        post: &[f32],
        ffn: &[f32],
        output: &[f32],
    ) {
        let q_rms = crate::facts::rms(q);
        let k_rms = crate::facts::rms(k);
        let v_rms = crate::facts::rms(v);
        let post_rms = crate::facts::rms(post);
        let ffn_rms = crate::facts::rms(ffn);
        let output_rms = crate::facts::rms(output);
        let q_cs = crate::facts::checksum(q);
        let k_cs = crate::facts::checksum(k);
        let v_cs = crate::facts::checksum(v);
        let post_cs = crate::facts::checksum(post);
        let ffn_cs = crate::facts::checksum(ffn);
        let output_cs = crate::facts::checksum(output);
        self.emit(InferenceFact::PerTensorOutput {
            layer_idx,
            position,
            q_rms_bits: q_rms.to_bits(),
            k_rms_bits: k_rms.to_bits(),
            v_rms_bits: v_rms.to_bits(),
            post_rms_bits: post_rms.to_bits(),
            ffn_rms_bits: ffn_rms.to_bits(),
            output_rms_bits: output_rms.to_bits(),
            q_checksum: q_cs,
            k_checksum: k_cs,
            v_checksum: v_cs,
            post_checksum: post_cs,
            ffn_checksum: ffn_cs,
            output_checksum: output_cs,
        });
    }
}

impl Default for ObservationSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::bits_to_f32;

    #[test]
    fn vault_oracle_captures_layer_outputs() {
        let mut session = ObservationSession::new();
        session.register_vault_oracle();

        // Emit 3 layer outputs
        for i in 0..3 {
            session.emit_layer_output(i, 1, &vec![0.1f32; 64]);
        }

        let result = session.saturate();
        assert!(result.saturated);

        let captures = session.vault_oracle().unwrap().drain();
        assert_eq!(captures.len(), 3, "should capture 3 layer outputs");
        assert_eq!(captures[0].layer_idx, 0);
        assert_eq!(captures[2].layer_idx, 2);
    }

    #[test]
    fn candle_compare_captures_final_logits() {
        let mut session = ObservationSession::new();
        session.register_candle_compare();

        let logits: Vec<f32> = (0..32000).map(|i| i as f32 * 0.001).collect();
        session.emit_final_logits(1, &logits);

        let result = session.saturate();
        assert!(result.saturated);

        let capture = session.candle_compare().unwrap().take();
        assert!(capture.is_some(), "should capture final logits");
        let c = capture.unwrap();
        assert!(c.rms > 0.0, "RMS should be positive");
    }

    #[test]
    fn fse_broadcast_both_observers_fire_from_layer_output() {
        // VaultOracleObserver AND LayerStabilityObserver both registered on KEY_LAYER_OUTPUT.
        // One LayerOutput fact should trigger BOTH — FSE broadcast property.
        let mut session = ObservationSession::new();
        session.register_vault_oracle();

        session.emit_layer_output(0, 1, &vec![0.5f32; 32]);
        let result = session.saturate();
        assert!(result.saturated);

        // Oracle capture happened
        let captures = session.vault_oracle().unwrap().drain();
        assert_eq!(captures.len(), 1);

        // Stability fact was derived (from the stability observer on same key)
        assert!(
            session.contains(&InferenceFact::LayerOutputStable { layer_idx: 0 }),
            "LayerOutputStable should be derived by stability observer"
        );

        // d_runtime / d_rules = 0: both rules fired from ONE alpha lookup
        // Verified: result.facts_derived includes WriteOracleRow AND LayerOutputStable
        assert!(
            result.facts_derived >= 2,
            "at least 2 derived facts from one LayerOutput emission (oracle + stability)"
        );
    }

    #[test]
    fn session_reports_correct_fact_count() {
        let mut session = ObservationSession::new();
        session.register_vault_oracle();

        session.emit_layer_output(0, 1, &vec![1.0f32; 4]);
        session.emit_layer_output(1, 1, &vec![1.0f32; 4]);
        session.saturate();

        // 2 LayerOutput + 2 WriteOracleRow + 2 LayerOutputStable = 6 facts
        assert_eq!(session.fact_count(), 6);
    }
}
