//! Independent raw GGUF reader — conformance crate's own GGUF parser.
//!
//! This module provides an Airframe-owned view of raw GGUF bytes so model
//! identity, metadata, tensor offsets, and tensor shapes are not inherited from
//! production loader logic. It MUST NOT import any production Airframe loader code.
//!
//! Per the architecture spike, this reader owns RAW evidence only:
//! - GGUF header (magic, version, tensor_count, metadata_kv_count)
//! - All metadata key-value pairs (exact bytes)
//! - Raw tensor directory entries (name, n_dims, dims, ggml_type, offset)
//! - tensor_data_start (aligned)
//! - File length and byte-exact hashes
//! - storage_upper_bound per tensor (next offset or file end)
//!
//! It does NOT compute:
//! - payload_size / row_size / element_size (CONF-5)
//! - quant layout (block_elems, block_bytes from quant_formula - CONF-5)

pub mod error;
pub mod model_identity;
pub mod reader;
pub mod tensor_view;
#[cfg(test)]
mod valid_tests;

pub use error::{RawGgufError, RawGgufResult};
pub use model_identity::{ModelIdentity, TensorDescriptor};
pub use reader::{RawGgufReader, RawGgufReaderBuilder};
pub use tensor_view::TensorView;
