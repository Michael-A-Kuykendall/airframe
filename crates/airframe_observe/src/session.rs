//! ObservationSession — the live runtime for a single inference pass.
//!
//! Wraps dzero's ReactiveGraph<InferenceFact> with the inference domain's
//! fact vocabulary and observer registration API.
//!
//! Usage:
//! 1. Create a session
//! 2. Register observers (vault oracle, candle compare, stability, etc.)
//! 3. During forward pass: call emit() for each layer output and final logits
//! 4. Call saturate() to run dzero to fixpoint
//! 5. Read results from observers

use crate::facts::{alpha_key_of, InferenceFact, KernelKind, YieldReason, KEY_DISPATCH_TIMING};
use crate::observers::{
    CandleCompareObserver, CertificationObserver, LayerStabilityObserver, VaultOracleObserver,
};
use dzero::{AlphaKey, ClosureProgram, ReactiveGraph, RunBudget, RunResult};
use std::sync::{Arc, Mutex};

/// TDR scheduler — holds mutable timing state between layer dispatches.
/// Lives inside ObservationSession. After each `saturate()`, check
/// `scheduler.should_yield` and act accordingly in the inference loop.
pub struct TdrScheduler {
    /// Accumulated CPU-side ms since last yield. Reset when yield fires.
    pub accumulated_ms: u32,
    /// Budget before a yield is required (default: 1500ms, conservative for Windows TDR ~2s limit)
    pub budget_ms: u32,
    /// Set by the YieldNow consequent rule. Inference loop reads this.
    pub should_yield: bool,
    /// Last layer that triggered a yield (for logging).
    pub last_yield_layer: Option<u32>,
}

impl TdrScheduler {
    pub fn new(budget_ms: u32) -> Self {
        Self {
            accumulated_ms: 0,
            budget_ms,
            should_yield: false,
            last_yield_layer: None,
        }
    }

    /// Add timing and determine if yield is needed.
    /// Returns the YieldNow fact if threshold exceeded.
    pub fn accumulate(&mut self, layer: u32, elapsed_ms: u32) -> Option<InferenceFact> {
        self.should_yield = false;
        self.accumulated_ms += elapsed_ms;
        if self.accumulated_ms >= self.budget_ms {
            self.should_yield = true;
            self.last_yield_layer = Some(layer);
            self.accumulated_ms = 0; // reset after yield
            Some(InferenceFact::YieldNow {
                layer,
                reason: YieldReason::TdrBudgetExceeded,
            })
        } else {
            None
        }
    }

    /// Force reset (call after actual GPU poll completes).
    pub fn reset(&mut self) {
        self.accumulated_ms = 0;
        self.should_yield = false;
    }
}

/// The live observation session for one inference pass.
pub struct ObservationSession {
    graph: ReactiveGraph<InferenceFact>,
    vault_oracle: Option<VaultOracleObserver>,
    candle_compare: Option<CandleCompareObserver>,
    certification: Option<CertificationObserver>,
    /// TDR scheduler — drives adaptive yield decisions via saturation fabric.
    pub scheduler: Option<TdrScheduler>,
    /// Arc-backed live TDR scheduler state (shared with the registered rule closure).
    _tdr_sched_arc: Option<Arc<Mutex<TdrScheduler>>>,
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
            certification: None,
            scheduler: None,
            _tdr_sched_arc: None,
        }
    }

    /// Register the TDR scheduler with the saturation fabric.
    ///
    /// `budget_ms`: accumulated CPU-side ms before a YieldNow consequent fires.
    /// Default 1500ms is conservative for Windows TDR (~2s watchdog).
    ///
    /// Registers a rule on KEY_DISPATCH_TIMING. Every DispatchTiming fact
    /// broadcasts to this rule — zero additional extraction cost per dispatch.
    /// The rule derives TdrRiskHigh or TdrRiskLow and emits YieldNow when needed.
    pub fn register_tdr_scheduler(&mut self, budget_ms: u32) {
        let scheduler = Arc::new(Mutex::new(TdrScheduler::new(budget_ms)));
        let sched_ref = scheduler.clone();

        self.graph
            .program
            .register(AlphaKey(KEY_DISPATCH_TIMING), move |fact, _store| {
                if let InferenceFact::DispatchTiming {
                    layer, elapsed_ms, ..
                } = fact
                {
                    let mut s = sched_ref.lock().unwrap();
                    if let Some(yield_fact) = s.accumulate(*layer, *elapsed_ms) {
                        // Derive TdrRiskHigh + YieldNow consequent
                        vec![InferenceFact::TdrRiskHigh { layer: *layer }, yield_fact]
                    } else {
                        vec![InferenceFact::TdrRiskLow { layer: *layer }]
                    }
                } else {
                    vec![]
                }
            });

        // Store the scheduler so the inference loop can read should_yield
        self.scheduler = Some(TdrScheduler::new(budget_ms));
        // Keep the Arc-based one as the live state, replace the simple one
        // with a wrapper that reads from the Arc after saturation
        self._tdr_sched_arc = Some(scheduler);
    }

    /// After calling saturate(), check if the fabric decided a yield is needed.
    /// Returns true if the inference loop should submit+poll the GPU now.
    pub fn should_yield(&self) -> bool {
        if let Some(arc) = &self._tdr_sched_arc {
            arc.lock().unwrap().should_yield
        } else {
            false
        }
    }

    /// Reset the TDR scheduler after the inference loop has completed a poll.
    pub fn reset_tdr(&self) {
        if let Some(arc) = &self._tdr_sched_arc {
            arc.lock().unwrap().reset();
        }
    }

    /// Emit a DispatchTiming fact — the primary input to the TDR scheduler.
    /// Call this after each GPU kernel dispatch+poll in the inference loop.
    pub fn emit_dispatch_timing(&mut self, layer: u32, kernel: KernelKind, elapsed_ms: u32) {
        self.emit(InferenceFact::DispatchTiming {
            layer,
            kernel,
            elapsed_ms,
        });
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

    /// Register the CertificationObserver (V2).
    ///
    /// Registers two rules — one on KEY_LAYER_OUTPUT and one on KEY_FINAL_LOGITS
    /// — that compare each live observation against the VaultOracle reference
    /// facts in the store, emitting CertificationPass / CertificationFail.
    /// VaultOracle facts (V1) must be emitted into the session before the
    /// observation facts they certify.
    ///
    /// `layer_tolerance` / `logits_tolerance` are the max relative RMS deltas
    /// for a pass; use `CertificationObserver::with_defaults()` tolerances
    /// (2.0 / 4.0) for the standard V2 gate via `register_certification_default`.
    pub fn register_certification(
        &mut self,
        layer_tolerance: f32,
        logits_tolerance: f32,
    ) -> &CertificationObserver {
        let observer = CertificationObserver::new(layer_tolerance, logits_tolerance);
        self.graph.program.register(
            CertificationObserver::layer_alpha_key(),
            observer.layer_rule(),
        );
        self.graph.program.register(
            CertificationObserver::logits_alpha_key(),
            observer.logits_rule(),
        );
        self.certification = Some(observer);
        self.certification.as_ref().unwrap()
    }

    /// Register the CertificationObserver with the standard V2 tolerances.
    pub fn register_certification_default(&mut self) -> &CertificationObserver {
        self.register_certification(
            CertificationObserver::DEFAULT_LAYER_TOLERANCE,
            CertificationObserver::DEFAULT_LOGITS_TOLERANCE,
        )
    }

    /// Emit VaultOracle reference facts into the session.
    /// Call before emitting the LayerOutput facts to be certified.
    pub fn emit_vault_oracles(&mut self, oracles: impl IntoIterator<Item = InferenceFact>) {
        for oracle in oracles {
            self.emit(oracle);
        }
    }

    /// Access certification results after saturation.
    pub fn certification(&self) -> Option<&CertificationObserver> {
        self.certification.as_ref()
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

    /// Run dzero to fixpoint. All observers receive their data.
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
            crate::facts::alpha_key_of(&InferenceFact::LayerOutput {
                layer_idx: 0,
                position: 0,
                rms_bits: 0,
                checksum: 0,
            })
            .unwrap(),
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
    #[allow(clippy::too_many_arguments)]
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
    fn certification_passes_within_tolerance() {
        let mut session = ObservationSession::new();
        session.register_certification_default();

        // Oracle: layer 0, position 1, expected RMS 0.5, matching checksum.
        let values = vec![0.5f32; 32];
        session.emit(InferenceFact::VaultOracle {
            model_id: 0,
            layer_idx: 0,
            position: 1,
            expected_rms_bits: 0.5f32.to_bits(),
            checksum: crate::facts::checksum(&values),
        });
        // Live output: all 0.5 → RMS 0.5 → exact match (rel_delta 0.0).
        session.emit_layer_output(0, 1, &values);

        let result = session.saturate();
        assert!(result.saturated);

        assert!(
            session.contains(&InferenceFact::CertificationPass {
                layer_idx: 0,
                position: 1,
                rms_delta_bits: 0.0f32.to_bits(),
            }),
            "exact RMS match should certify PASS"
        );

        let results = session.certification().unwrap().drain();
        assert_eq!(results.len(), 1);
        assert!(results[0].passed);
        assert!(results[0].checksum_match, "identical values → checksum match");
        assert_eq!(results[0].rel_delta, 0.0);
    }

    #[test]
    fn certification_passes_at_ratio_below_two() {
        let mut session = ObservationSession::new();
        session.register_certification_default();

        // expected 0.5, observed 1.0 → rel_delta = 0.5/0.5 = 1.0 < 2.0 → PASS.
        session.emit(InferenceFact::VaultOracle {
            model_id: 0,
            layer_idx: 2,
            position: 1,
            expected_rms_bits: 0.5f32.to_bits(),
            checksum: 7,
        });
        session.emit_layer_output(2, 1, &vec![1.0f32; 8]);
        assert!(session.saturate().saturated);

        let results = session.certification().unwrap().drain();
        assert_eq!(results.len(), 1);
        assert!(results[0].passed, "rel_delta 1.0 < 2.0 must PASS");
        assert!((results[0].rel_delta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn certification_fails_outside_tolerance() {
        let mut session = ObservationSession::new();
        session.register_certification_default();

        // expected 0.5, observed 2.0 → rel_delta = 1.5/0.5 = 3.0 >= 2.0 → FAIL.
        session.emit(InferenceFact::VaultOracle {
            model_id: 0,
            layer_idx: 5,
            position: 1,
            expected_rms_bits: 0.5f32.to_bits(),
            checksum: 999,
        });
        session.emit_layer_output(5, 1, &vec![2.0f32; 16]);

        let result = session.saturate();
        assert!(result.saturated);

        let results = session.certification().unwrap().drain();
        assert_eq!(results.len(), 1);
        assert!(!results[0].passed, "rel_delta 3.0 must FAIL");
        assert!(!results[0].checksum_match, "checksums differ");
        assert!((results[0].rel_delta - 3.0).abs() < 1e-6);

        // A CertificationFail fact must be present with the divergent layer.
        assert!(
            session.contains(&InferenceFact::CertificationFail {
                layer_idx: 5,
                position: 1,
                rms_delta_bits: 3.0f32.to_bits(),
                observed_rms_bits: 2.0f32.to_bits(),
                expected_rms_bits: 0.5f32.to_bits(),
                checksum_match: false,
            }),
            "CertificationFail fact should be asserted"
        );
    }

    #[test]
    fn certification_final_logits_uses_logits_tolerance() {
        let mut session = ObservationSession::new();
        session.register_certification_default();

        // Final-logits oracle is stored with layer_idx == -1.
        // expected 1.0, observed 4.0 → rel_delta 3.0 < 4.0 → PASS (would FAIL
        // under the 2.0 layer tolerance, proving the logits path is used).
        session.emit(InferenceFact::VaultOracle {
            model_id: 0,
            layer_idx: -1,
            position: 1,
            expected_rms_bits: 1.0f32.to_bits(),
            checksum: 42,
        });
        session.emit_final_logits(1, &vec![4.0f32; 8]);
        assert!(session.saturate().saturated);

        let results = session.certification().unwrap().drain();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_final_logits);
        assert!(results[0].passed, "rel_delta 3.0 < 4.0 logits tolerance → PASS");
        assert_eq!(
            results[0].layer_idx,
            crate::observers::FINAL_LOGITS_LAYER
        );
    }

    #[test]
    fn certification_ignores_uncovered_layers() {
        let mut session = ObservationSession::new();
        session.register_certification_default();

        // No oracle for layer 3 → no certification fact, no result.
        session.emit_layer_output(3, 1, &vec![0.5f32; 32]);
        let result = session.saturate();
        assert!(result.saturated);

        assert!(session.certification().unwrap().drain().is_empty());
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
