//! Model identity types for GGUF files — RAW evidence only.
//!
//! These types capture exactly what is encoded in the GGUF file without
//! deriving quant layout or payload sizes. Those belong in CONF-5.

use crate::raw_gguf::error::RawGgufResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Read, Seek};

/// Complete model identity derived from a GGUF file — raw evidence only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelIdentity {
    /// GGUF header version.
    pub version: u32,

    /// Alignment value from header metadata (or default 32).
    /// Must be a multiple of 8 per GGUF spec.
    pub alignment: u64,

    /// Full-file SHA-256 hash (hex).
    pub file_hash: String,

    /// Tensor directory SHA-256 hash (hex) — hashes raw directory bytes only.
    pub tensor_directory_hash: String,

    /// All metadata key-value pairs from the GGUF file.
    pub metadata: HashMap<String, GgufValue>,

    /// Raw tensor descriptors (directory order preserved).
    /// Contains only what GGUF encodes: name, raw type ID, shape, offset, storage_upper_bound.
    pub tensors: Vec<TensorDescriptor>,

    /// Total file size in bytes.
    pub file_size: u64,

    /// Data section start offset (aligned).
    pub data_start_offset: u64,
}

/// Raw tensor descriptor — only what GGUF directory encodes.
/// NO payload_size, row_size, element_size — those are CONF-5 (quant layout).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorDescriptor {
    /// Tensor name (from directory).
    pub name: String,

    /// Raw GGML quantization type (from directory).
    /// Unknown types are preserved as opaque values.
    pub ggml_type: u32,

    /// Tensor shape (dimensions from directory).
    pub shape: Vec<u64>,

    /// Absolute byte offset in the file (data_start_offset + relative_offset).
    pub offset: u64,

    /// Storage upper bound in bytes (next tensor offset or file end).
    /// NOT payload_size — quant layout is CONF-5.
    pub storage_upper_bound: u64,
}

/// GGUF metadata value types (for exact round-trip).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value")]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl ModelIdentity {
    /// Compute the full-file SHA-256 hash.
    pub fn compute_file_hash(file_path: &std::path::Path) -> RawGgufResult<String> {
        let mut file = std::fs::File::open(file_path)?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Compute the tensor directory hash from RAW directory bytes.
    /// This hashes the exact directory bytes as they appear in the file,
    /// NOT reconstructed/sorted descriptors.
    pub fn compute_tensor_directory_hash(
        file_path: &std::path::Path,
        dir_start: u64,
        dir_len: u64,
    ) -> RawGgufResult<String> {
        let mut file = std::fs::File::open(file_path)?;
        file.seek(std::io::SeekFrom::Start(dir_start))?;
        let mut hasher = Sha256::new();
        let mut remaining = dir_len;
        let mut buf = vec![0u8; 8192];
        while remaining > 0 {
            let to_read = std::cmp::min(remaining as usize, buf.len());
            file.read_exact(&mut buf[..to_read])?;
            hasher.update(&buf[..to_read]);
            remaining -= to_read as u64;
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}
