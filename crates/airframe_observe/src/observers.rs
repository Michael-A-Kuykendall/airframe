//! Built-in observers for the vault pipeline.
//!
//! Each observer is a d0-engine rule registered on specific InferenceFact keys.
//! When the forward pass emits a fact, all rules indexed on that fact's alpha key
//! fire simultaneously — zero additional extraction cost per shared selector.

use crate::facts::{bits_to_f32, InferenceFact, KEY_FINAL_LOGITS, KEY_LAYER_OUTPUT};
use d0_engine::{AlphaKey, FactStore};
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

    /// Returns the rule closure to register with d0-engine.
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

    /// Returns the rule closure to register with d0-engine.
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
