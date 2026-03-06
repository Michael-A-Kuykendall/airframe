//! Validation-specific error types for V2 slice gates.

use thiserror::Error;

/// Slice gate validation error.
#[derive(Debug, Error)]
pub enum SliceValidationError {
    #[error("Slice gate failed: {gate_command} - {reason}")]
    GateFailed {
        gate_command: String,
        reason: String,
    },

    #[error("Artifact generation failed: {artifact_path} - {error}")]
    ArtifactFailed {
        artifact_path: String,
        error: String,
    },

    #[error("Evidence checklist incomplete: missing {missing_items:?}")]
    EvidenceIncomplete { missing_items: Vec<String> },

    #[error("Determinism validation failed: run1 != run2 at step {step}")]
    DeterminismFailed { step: usize },

    #[error("Unknown tensor type {ggml_type} encountered - fail-closed behavior activated")]
    UnknownTensorType { ggml_type: u32 },

    #[error("Invariant violation detected: {description}")]
    InvariantViolation { description: String },

    #[error("Unsupported model feature: {feature} - explicit capability boundary")]
    UnsupportedFeature { feature: String },

    #[error("Validation rule violated: {rule} - {diagnostic}")]
    ValidationRuleViolated { rule: String, diagnostic: String },

    #[error("System boundary exceeded: {boundary} - {details}")]
    SystemBoundaryExceeded { boundary: String, details: String },
}

/// KV cache invariant violation.
#[derive(Debug, Error)]
pub enum KVCacheError {
    #[error("Monotonic growth violated: expected {expected}, got {actual} at step {step}")]
    MonotonicGrowthViolated {
        expected: usize,
        actual: usize,
        step: usize,
    },

    #[error(
        "Attention history not used: score vector length {score_len} != cache length {cache_len}"
    )]
    HistoryNotUsed { score_len: usize, cache_len: usize },

    #[error("Single-step illusion detected: attention only uses latest token")]
    SingleStepIllusion,

    #[error("Prefill/decode equivalence failed at step {step}: prefill={prefill_token}, stepwise={stepwise_token}")]
    EquivalenceFailed {
        step: usize,
        prefill_token: u32,
        stepwise_token: u32,
    },
}

/// Oracle conformance error.
#[derive(Debug, Error)]
pub enum OracleError {
    #[error("Oracle fixture not found: {fixture_id}")]
    FixtureNotFound { fixture_id: String },

    #[error("Model SHA256 mismatch: expected {expected}, oracle used {oracle}")]
    ModelMismatch { expected: String, oracle: String },

    #[error("Token mismatch at step {step}: lib={lib_token}, oracle={oracle_token}")]
    TokenMismatch {
        step: usize,
        lib_token: u32,
        oracle_token: u32,
    },
}

pub type ValidationResult<T> = std::result::Result<T, SliceValidationError>;
