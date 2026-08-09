//! Error types for the raw GGUF reader.

use thiserror::Error;

/// Result type for raw GGUF operations.
pub type RawGgufResult<T> = Result<T, RawGgufError>;

/// Errors that can occur when reading a GGUF file.
#[derive(Debug, Error)]
pub enum RawGgufError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid magic bytes: expected 'GGUF', got {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("Unsupported GGUF version: {0} (expected 2 or 3)")]
    UnsupportedVersion(u32),

    #[error("Alignment must be a multiple of 8, got {0}")]
    InvalidAlignment(u64),

    #[error("Metadata key-value count exceeds limit: {0} > {1}")]
    MetadataCountExceeded(u64, u64),

    #[error("Tensor count exceeds limit: {0} > {1}")]
    TensorCountExceeded(u64, u64),

    #[error("Tensor name too long: {0} bytes > {1} bytes")]
    TensorNameTooLong(usize, usize),

    #[error("Tensor dimension count too large: {0} > {1}")]
    TooManyDimensions(u32, u32),

    #[error("Invalid tensor dimension: {0}")]
    InvalidDimension(u64),

    #[error("Unknown GGML type: {0} (raw directory value preserved)")]
    UnknownGgmlType(u32),

    #[error("Tensor offset out of bounds: offset {0} + upper_bound {1} > file size {2}")]
    TensorOffsetOutOfBounds(u64, u64, u64),

    #[error("Tensor storage ranges overlap: {0} overlaps {1}")]
    OverlappingTensors(String, String),

    #[error("Tensor data extends past end of file: {0} + {1} > {2}")]
    TensorPastEndOfFile(u64, u64, u64),

    #[error("Data start offset not aligned: {0} not aligned to {1}")]
    DataStartNotAligned(u64, u64),

    #[error("Header parsing failed: {0}")]
    HeaderParseError(String),

    #[error("Metadata parsing failed: {0}")]
    MetadataParseError(String),

    #[error("Tensor info parsing failed: {0}")]
    TensorInfoParseError(String),

    #[error("Hash computation failed: {0}")]
    HashError(String),

    #[error("UTF-8 decoding failed: {0}")]
    Utf8Error(#[from] std::string::FromUtf8Error),

    #[error("End of file reached unexpectedly")]
    UnexpectedEof,
}

impl RawGgufError {
    /// Check if this error is a malformed file error (for test categorization).
    pub fn is_malformed(&self) -> bool {
        matches!(
            self,
            RawGgufError::InvalidMagic(_)
                | RawGgufError::UnsupportedVersion(_)
                | RawGgufError::InvalidAlignment(_)
                | RawGgufError::MetadataCountExceeded(_, _)
                | RawGgufError::TensorCountExceeded(_, _)
                | RawGgufError::TensorNameTooLong(_, _)
                | RawGgufError::TooManyDimensions(_, _)
                | RawGgufError::InvalidDimension(_)
                | RawGgufError::UnknownGgmlType(_)
                | RawGgufError::TensorOffsetOutOfBounds(_, _, _)
                | RawGgufError::OverlappingTensors(_, _)
                | RawGgufError::TensorPastEndOfFile(_, _, _)
                | RawGgufError::DataStartNotAligned(_, _)
                | RawGgufError::HeaderParseError(_)
                | RawGgufError::MetadataParseError(_)
                | RawGgufError::TensorInfoParseError(_)
                | RawGgufError::UnexpectedEof
        )
    }
}
