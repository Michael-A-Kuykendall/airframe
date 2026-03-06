//! Error types for libshimmy operations.
//!
//! All fallible operations return `Result<T, LibshimmyError>`.

use crate::validation::errors::{KVCacheError, OracleError, SliceValidationError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LibshimmyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unsupported: {0}")]
    Unsupported(String),

    #[error("Invariant violation: {msg}")]
    InvariantViolation { msg: String },

    #[error("Invalid model spec field '{field}': expected {expected}, got {got}")]
    InvalidModelSpec {
        field: String,
        expected: String,
        got: String,
    },

    #[error("Missing tensor: {name}. Ensure GGUF file contains all required model weights")]
    MissingTensor { name: String },

    #[error("Shape mismatch for tensor '{tensor}': expected {expected:?}, got {got:?}. Verify model architecture matches GGUF metadata")]
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Unsupported quantization type {ggml_type} ({type_name}) for tensor '{tensor_name}'. Supported types: F32 (0), Q4_0 (2), Q4_K (12), Q6_K (14)")]
    QuantUnsupported {
        tensor_name: String,
        ggml_type: u32,
        type_name: String,
    },

    #[error("Tensor bounds error: {tensor_name} type {ggml_type} ({type_name}) extends beyond file (computed end: {computed_end}, file size: {file_size}). Check GGUF file integrity or tensor metadata")]
    TensorBounds {
        tensor_name: String,
        ggml_type: u32,
        type_name: String,
        computed_end: u64,
        file_size: u64,
    },

    #[error("Fixture error: {msg}")]
    FixtureError { msg: String },

    #[error("Missing required weight: {weight_id}")]
    WeightMissing { weight_id: String },

    #[error("GGML type resolution failed for type {ggml_type}: {reason}. Valid types: F32 (0), Q4_0 (2), Q4_K (12), Q6_K (14)")]
    GgmlTypeError { ggml_type: u32, reason: String },

    #[error(
        "Dequantization failed for tensor '{tensor_name}' type {ggml_type} ({type_name}): {reason}"
    )]
    DequantizationError {
        tensor_name: String,
        ggml_type: u32,
        type_name: String,
        reason: String,
    },

    // V2 Validation errors
    #[error("Slice validation error: {0}")]
    SliceValidation(#[from] SliceValidationError),

    #[error("KV cache error: {0}")]
    KVCache(#[from] KVCacheError),

    #[error("Oracle error: {0}")]
    Oracle(#[from] OracleError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, LibshimmyError>;

#[macro_export]
macro_rules! ensure {
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            return Err($crate::core::error::LibshimmyError::InvariantViolation {
                msg: format!($($arg)+),
            });
        }
    };
}
