//! Built-in observers for the vault pipeline.
//!
//! Each observer is a dzero rule registered on specific InferenceFact keys.
//! When the forward pass emits a fact, all rules indexed on that fact's alpha key
//! fire simultaneously — zero additional extraction cost per shared selector.

use crate::facts::{bits_to_f32, InferenceFact, KEY_FINAL_LOGITS, KEY_LAYER_OUTPUT};
use dzero::{AlphaKey, FactStore};
use std::sync::{Arc, Mutex};

/// Captured oracle data from a single layer.
#[derive(Clone, Debug)]
pub struct OracleCapture {
    pub layer_idx: u32,
    pub position: u32,
    pub rms: f32,
    pub checksum: i64,
}

/// Captured logit data for candle cross-validation.
#[derive(Clone, Debug)]
pub struct LogitCapture {
    pub position: u32,
    pub rms: f32,
    pub checksum: i64,
}

/// VaultOracleObserver captures layer outputs and stores them for vault import.
///
/// Registered on KEY_LAYER_OUTPUT. Every LayerOutput fact broadcasts here.
pub struct VaultOracleObserver {
    pub captures: Arc<Mutex<Vec<OracleCapture>>>,
}

impl VaultOracleObserver {
    pub fn new() -> Self {
        Self {
            captures: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns the rule closure to register with dzero.
    /// The closure fires on every LayerOutput fact.
    pub fn rule(
        &self,
    ) -> impl Fn(&InferenceFact, &FactStore<InferenceFact>) -> Vec<InferenceFact> + Send + Sync
    {
        let captures = self.captures.clone();
        move |fact, _store| {
            if let InferenceFact::LayerOutput {
                layer_idx,
                position,
                rms_bits,
                checksum,
            } = fact
            {
                captures.lock().unwrap().push(OracleCapture {
                    layer_idx: *layer_idx,
                    position: *position,
                    rms: bits_to_f32(*rms_bits),
                    checksum: *checksum,
                });
                // Emit a consequent to trigger oracle row write
                vec![InferenceFact::WriteOracleRow {
                    layer_idx: *layer_idx,
                }]
            } else {
                vec![]
            }
        }
    }

    /// Alpha key this observer registers on.
    pub fn alpha_key() -> AlphaKey {
        AlphaKey(KEY_LAYER_OUTPUT)
    }

    /// Drain captured oracle data after saturation.
    pub fn drain(&self) -> Vec<OracleCapture> {
        self.captures.lock().unwrap().drain(..).collect()
    }
}

impl Default for VaultOracleObserver {
    fn default() -> Self {
        Self::new()
    }
}

/// CandleCompareObserver captures final logits for cross-validation.
///
/// Registered on KEY_FINAL_LOGITS. Fires once per forward pass.
/// The candle_probe binary generates the reference to compare against.
pub struct CandleCompareObserver {
    pub capture: Arc<Mutex<Option<LogitCapture>>>,
}

impl CandleCompareObserver {
    pub fn new() -> Self {
        Self {
            capture: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns the rule closure to register with dzero.
    pub fn rule(
        &self,
    ) -> impl Fn(&InferenceFact, &FactStore<InferenceFact>) -> Vec<InferenceFact> + Send + Sync
    {
        let capture = self.capture.clone();
        move |fact, _store| {
            if let InferenceFact::FinalLogits {
                position,
                rms_bits,
                checksum,
            } = fact
            {
                *capture.lock().unwrap() = Some(LogitCapture {
                    position: *position,
                    rms: bits_to_f32(*rms_bits),
                    checksum: *checksum,
                });
                // Emit consequent to trigger candle comparison
                vec![InferenceFact::TriggerCandleCompare {
                    rms_bits: *rms_bits,
                    checksum: *checksum,
                }]
            } else {
                vec![]
            }
        }
    }

    /// Alpha key this observer registers on.
    pub fn alpha_key() -> AlphaKey {
        AlphaKey(KEY_FINAL_LOGITS)
    }

    /// Take the captured logit data.
    pub fn take(&self) -> Option<LogitCapture> {
        self.capture.lock().unwrap().take()
    }
}

impl Default for CandleCompareObserver {
    fn default() -> Self {
        Self::new()
    }
}

/// Sentinel `layer_idx` used in certification facts for the final-logits check
/// (the vault stores final logits as an oracle row with `layer_idx = -1`).
pub const FINAL_LOGITS_LAYER: u32 = u32::MAX;

/// One certification comparison result: a live observation vs its VaultOracle.
#[derive(Clone, Debug)]
pub struct CertResult {
    /// Layer index, or `FINAL_LOGITS_LAYER` for the final-logits comparison.
    pub layer_idx: u32,
    pub position: u32,
    pub observed_rms: f32,
    pub expected_rms: f32,
    /// Relative delta: `|observed - expected| / expected`.
    pub rel_delta: f32,
    pub checksum_match: bool,
    pub passed: bool,
    pub is_final_logits: bool,
}

/// Relative RMS delta `|observed - expected| / |expected|`.
/// Zero when both are zero; +inf when only expected is zero.
fn rel_delta(observed: f32, expected: f32) -> f32 {
    if expected == 0.0 {
        if observed == 0.0 {
            0.0
        } else {
            f32::INFINITY
        }
    } else {
        (observed - expected).abs() / expected.abs()
    }
}

/// CertificationObserver compares live observation facts against the
/// `VaultOracle` reference facts loaded from the golden vault (V1).
///
/// Two rules, sharing one results buffer:
/// - on KEY_LAYER_OUTPUT: each `LayerOutput` is matched to the `VaultOracle`
///   with the same `(layer_idx, position)` (layer_idx >= 0). PASS if the
///   relative RMS delta is below `layer_tolerance` (default 2.0).
/// - on KEY_FINAL_LOGITS: each `FinalLogits` is matched to the `VaultOracle`
///   final-logits row (`layer_idx == -1`, same position). PASS if the relative
///   RMS delta is below `logits_tolerance` (default 4.0). Emitted certification
///   facts use `FINAL_LOGITS_LAYER` as the layer index.
///
/// No matching oracle → no certification fact (that point is uncovered).
pub struct CertificationObserver {
    /// Max relative RMS delta for a layer-output PASS.
    pub layer_tolerance: f32,
    /// Max relative RMS delta for a final-logits PASS.
    pub logits_tolerance: f32,
    /// All comparison results, in fire order. Drained after saturation.
    pub results: Arc<Mutex<Vec<CertResult>>>,
}

impl CertificationObserver {
    /// Default relative-delta tolerance for a layer-output pass (per bead V2 /
    /// AGENTS.md: per-layer RMS ratio < 2.0 = OK).
    pub const DEFAULT_LAYER_TOLERANCE: f32 = 2.0;
    /// Default relative-delta tolerance for a final-logits pass.
    pub const DEFAULT_LOGITS_TOLERANCE: f32 = 4.0;

    pub fn new(layer_tolerance: f32, logits_tolerance: f32) -> Self {
        Self {
            layer_tolerance,
            logits_tolerance,
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Construct with the standard V2 gate tolerances (2.0 / 4.0).
    pub fn with_defaults() -> Self {
        Self::new(
            Self::DEFAULT_LAYER_TOLERANCE,
            Self::DEFAULT_LOGITS_TOLERANCE,
        )
    }

    /// Rule for KEY_LAYER_OUTPUT: certify each LayerOutput against its oracle.
    pub fn layer_rule(
        &self,
    ) -> impl Fn(&InferenceFact, &FactStore<InferenceFact>) -> Vec<InferenceFact> + Send + Sync
    {
        let results = self.results.clone();
        let tolerance = self.layer_tolerance;
        move |fact, store| {
            let InferenceFact::LayerOutput {
                layer_idx,
                position,
                rms_bits,
                checksum,
            } = fact
            else {
                return vec![];
            };
            for f in store.facts.iter() {
                let InferenceFact::VaultOracle {
                    layer_idx: oracle_layer,
                    position: oracle_pos,
                    expected_rms_bits,
                    checksum: oracle_checksum,
                    ..
                } = f
                else {
                    continue;
                };
                if *oracle_layer != *layer_idx as i32 || *oracle_pos != *position {
                    continue;
                }
                return certify(
                    &results,
                    *layer_idx,
                    *position,
                    *rms_bits,
                    *checksum,
                    *expected_rms_bits,
                    *oracle_checksum,
                    tolerance,
                    false,
                );
            }
            vec![]
        }
    }

    /// Rule for KEY_FINAL_LOGITS: certify FinalLogits against the -1 oracle row.
    pub fn logits_rule(
        &self,
    ) -> impl Fn(&InferenceFact, &FactStore<InferenceFact>) -> Vec<InferenceFact> + Send + Sync
    {
        let results = self.results.clone();
        let tolerance = self.logits_tolerance;
        move |fact, store| {
            let InferenceFact::FinalLogits {
                position,
                rms_bits,
                checksum,
            } = fact
            else {
                return vec![];
            };
            for f in store.facts.iter() {
                let InferenceFact::VaultOracle {
                    layer_idx: oracle_layer,
                    position: oracle_pos,
                    expected_rms_bits,
                    checksum: oracle_checksum,
                    ..
                } = f
                else {
                    continue;
                };
                if *oracle_layer != -1 || *oracle_pos != *position {
                    continue;
                }
                return certify(
                    &results,
                    FINAL_LOGITS_LAYER,
                    *position,
                    *rms_bits,
                    *checksum,
                    *expected_rms_bits,
                    *oracle_checksum,
                    tolerance,
                    true,
                );
            }
            vec![]
        }
    }

    /// Alpha key for the layer-output rule.
    pub fn layer_alpha_key() -> AlphaKey {
        AlphaKey(KEY_LAYER_OUTPUT)
    }

    /// Alpha key for the final-logits rule.
    pub fn logits_alpha_key() -> AlphaKey {
        AlphaKey(KEY_FINAL_LOGITS)
    }

    /// Drain the accumulated certification results after saturation.
    pub fn drain(&self) -> Vec<CertResult> {
        self.results.lock().unwrap().drain(..).collect()
    }
}

/// Shared comparison: record a result and emit the Pass/Fail consequent.
#[allow(clippy::too_many_arguments)]
fn certify(
    results: &Arc<Mutex<Vec<CertResult>>>,
    layer_idx: u32,
    position: u32,
    observed_rms_bits: u32,
    observed_checksum: i64,
    expected_rms_bits: u32,
    oracle_checksum: i64,
    tolerance: f32,
    is_final_logits: bool,
) -> Vec<InferenceFact> {
    let observed_rms = bits_to_f32(observed_rms_bits);
    let expected_rms = bits_to_f32(expected_rms_bits);
    let delta = rel_delta(observed_rms, expected_rms);
    let checksum_match = observed_checksum == oracle_checksum;
    let passed = delta < tolerance;

    results.lock().unwrap().push(CertResult {
        layer_idx,
        position,
        observed_rms,
        expected_rms,
        rel_delta: delta,
        checksum_match,
        passed,
        is_final_logits,
    });

    if passed {
        vec![InferenceFact::CertificationPass {
            layer_idx,
            position,
            rms_delta_bits: delta.to_bits(),
        }]
    } else {
        vec![InferenceFact::CertificationFail {
            layer_idx,
            position,
            rms_delta_bits: delta.to_bits(),
            observed_rms_bits,
            expected_rms_bits,
            checksum_match,
        }]
    }
}

/// LayerStabilityObserver derives semantic facts about layer health.
///
/// Registered on KEY_LAYER_OUTPUT. Fires alongside VaultOracleObserver
/// (same alpha key, zero additional extraction cost — FSE broadcast).
pub struct LayerStabilityObserver;

impl LayerStabilityObserver {
    pub fn alpha_key() -> AlphaKey {
        AlphaKey(KEY_LAYER_OUTPUT) // shares the key — free broadcast
    }

    pub fn rule(
    ) -> impl Fn(&InferenceFact, &FactStore<InferenceFact>) -> Vec<InferenceFact> + Send + Sync
    {
        |fact, _store| {
            if let InferenceFact::LayerOutput {
                layer_idx,
                rms_bits,
                ..
            } = fact
            {
                let rms = bits_to_f32(*rms_bits);
                // Derive stability fact if RMS is in sane range
                if rms > 0.0 && rms < 1000.0 && !rms.is_nan() && !rms.is_infinite() {
                    vec![InferenceFact::LayerOutputStable {
                        layer_idx: *layer_idx,
                    }]
                } else {
                    vec![] // No stability fact — something is wrong
                }
            } else {
                vec![]
            }
        }
    }
}
