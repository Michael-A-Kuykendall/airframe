//! Observer — anything that wants to receive inference data.
//!
//! In FSE terms: an observer is a "rule action" — it fires when its
//! registered selector is broadcast during the single-pass execution.

use crate::output::ObservationData;

/// Unique name for an observer. Used in the broadcast list.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObserverId(pub String);

impl ObserverId {
    pub fn new(name: impl Into<String>) -> Self {
        ObserverId(name.into())
    }
}

impl std::fmt::Display for ObserverId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Trait implemented by anything that wants to receive inference observations.
///
/// Called by the execution module when a registered selector fires.
/// Multiple observers may receive the same data in one broadcast — zero-copy
/// via shared reference.
pub trait Observer: Send + Sync {
    fn id(&self) -> &ObserverId;

    /// Called when a selector this observer registered for fires.
    /// `data` is a shared reference — no copy, no allocation.
    fn observe(&self, data: &ObservationData);
}

/// Built-in observers for the vault pipeline
///
/// VaultOracleObserver: captures layer outputs and logits for vault_seed.
pub struct VaultOracleObserver {
    pub id: ObserverId,
    pub captured: std::sync::Mutex<Vec<ObservationData>>,
}

impl VaultOracleObserver {
    pub fn new() -> Self {
        Self {
            id: ObserverId::new("vault_oracle"),
            captured: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn drain(&self) -> Vec<ObservationData> {
        self.captured.lock().unwrap().drain(..).collect()
    }
}

impl Default for VaultOracleObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for VaultOracleObserver {
    fn id(&self) -> &ObserverId {
        &self.id
    }

    fn observe(&self, data: &ObservationData) {
        self.captured.lock().unwrap().push(data.clone());
    }
}

/// CandleCompareObserver: captures final logits for cross-validation.
pub struct CandleCompareObserver {
    pub id: ObserverId,
    pub logits: std::sync::Mutex<Option<Vec<f32>>>,
}

impl CandleCompareObserver {
    pub fn new() -> Self {
        Self {
            id: ObserverId::new("candle_compare"),
            logits: std::sync::Mutex::new(None),
        }
    }

    pub fn take_logits(&self) -> Option<Vec<f32>> {
        self.logits.lock().unwrap().take()
    }
}

impl Default for CandleCompareObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for CandleCompareObserver {
    fn id(&self) -> &ObserverId {
        &self.id
    }

    fn observe(&self, data: &ObservationData) {
        if let ObservationData::FinalLogits { values, .. } = data {
            *self.logits.lock().unwrap() = Some(values.clone());
        }
    }
}
