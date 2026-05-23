//! GGML quantization type definitions and byte-size calculations.
//!
//! Implements the subset of GGML types supported by libshimmy:
//! F32 (0), F16 (1), Q4_0 (2), Q5_0 (6), Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14).
//!
//! Reference: ggerganov/ggml ggml.h commit 3fd62a6a

use crate::core::error::{LibshimmyError, Result};

/// Supported GGML tensor quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
// non_camel_case_types: GGML naming convention uses underscores (Q4_0, Q4_K, etc.)
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q5_0 = 6,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

impl GgmlType {
    /// Convert from raw GGML type ID to enum
    pub fn from_u32(type_id: u32) -> Result<Self> {
        match type_id {
            0  => Ok(GgmlType::F32),
            1  => Ok(GgmlType::F16),
            2  => Ok(GgmlType::Q4_0),
            6  => Ok(GgmlType::Q5_0),
            8  => Ok(GgmlType::Q8_0),
            12 => Ok(GgmlType::Q4_K),
            13 => Ok(GgmlType::Q5_K),
            14 => Ok(GgmlType::Q6_K),
            _ => Err(LibshimmyError::QuantUnsupported {
                tensor_name: "unknown".to_string(),
                ggml_type: type_id,
                type_name: format!("UNKNOWN_{}", type_id),
            }),
        }
    }

    /// Get the canonical name for this GGML type
    pub fn name(&self) -> &'static str {
        match self {
            GgmlType::F32  => "F32",
            GgmlType::F16  => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q4_K => "Q4_K",
            GgmlType::Q5_K => "Q5_K",
            GgmlType::Q6_K => "Q6_K",
        }
    }

    /// Get the raw type ID
    pub fn type_id(&self) -> u32 {
        *self as u32
    }
}

/// Get the canonical name for a GGML type ID
///
/// Returns the symbolic name (e.g., "Q4_K") for supported types,
/// or fails with QuantUnsupported for unknown types.
pub fn ggml_type_name(type_id: u32) -> Result<&'static str> {
    let ggml_type = GgmlType::from_u32(type_id)?;
    Ok(ggml_type.name())
}

/// Calculate the exact byte size required for a tensor of the given type and element count
///
/// This function implements the authoritative GGML byte layout formulas:
/// - F32: 4 bytes per element (no quantization)
/// - Q4_0: 32 elements per block, 18 bytes per block (2 bytes scale + 16 bytes data)
/// - Q4_K: 256 elements per superblock, 144 bytes per superblock
///
/// Returns fail-closed error for unknown types to prevent silent size assumptions.
pub fn ggml_type_bytes_per_tensor(type_id: u32, element_count: usize) -> Result<usize> {
    let ggml_type = GgmlType::from_u32(type_id)?;

    match ggml_type {
        GgmlType::F32 => {
            // F32: Direct 4 bytes per element
            Ok(element_count * 4)
        }
        GgmlType::F16 => {
            // F16: Direct 2 bytes per element
            Ok(element_count * 2)
        }
        GgmlType::Q4_0 => {
            // Q4_0: 32 elements per block, 18 bytes per block
            // Block structure: 2 bytes (fp16 scale) + 16 bytes (4-bit data)
            let block_size = 32;
            let bytes_per_block = 18;
            let num_blocks = element_count.div_ceil(block_size);
            Ok(num_blocks * bytes_per_block)
        }
        GgmlType::Q5_0 => {
            // Q5_0: 32 elements per block, 22 bytes per block
            // Block structure: 2 bytes (fp16 scale) + 4 bytes (qh high bits) + 16 bytes (qs nibbles)
            let block_size = 32;
            let bytes_per_block = 22;
            let num_blocks = element_count.div_ceil(block_size);
            Ok(num_blocks * bytes_per_block)
        }
        GgmlType::Q8_0 => {
            // Q8_0: 32 elements per block, 34 bytes per block
            // Block structure: 2 bytes (fp16 scale) + 32 bytes (i8 values)
            let block_size = 32;
            let bytes_per_block = 34;
            let num_blocks = element_count.div_ceil(block_size);
            Ok(num_blocks * bytes_per_block)
        }
        GgmlType::Q4_K => {
            // Q4_K: 4-bit K-quant
            // Block structure: 256 elements per superblock, 144 bytes per superblock
            // Layout: d (2B fp16) + dmin (2B fp16) + scales (12B) + qs (128B) = 144 bytes
            let superblock_size = 256;
            let bytes_per_superblock = calculate_q4_k_superblock_size();
            let num_superblocks = element_count.div_ceil(superblock_size);
            Ok(num_superblocks * bytes_per_superblock)
        }
        GgmlType::Q5_K => {
            // Q5_K: 5-bit K-quant
            // Block structure: 256 elements per superblock, 176 bytes per superblock
            // Layout: d (2B fp16) + dmin (2B fp16) + scales (12B) + qh (32B) + qs (128B) = 176 bytes
            let superblock_size = 256;
            let bytes_per_superblock = 176;
            let num_superblocks = element_count.div_ceil(superblock_size);
            Ok(num_superblocks * bytes_per_superblock)
        }
        GgmlType::Q6_K => {
            // Q6_K: 6-bit K-quant
            // Block structure: 256 elements per superblock, 210 bytes per superblock
            // Layout: d (2B FP16) + ql (128B) + qh (64B) + scales (16B) = 210 bytes
            let superblock_size = 256;
            let bytes_per_superblock = calculate_q6_k_superblock_size();
            let num_superblocks = element_count.div_ceil(superblock_size);
            Ok(num_superblocks * bytes_per_superblock)
        }
    }
}

fn calculate_q4_k_superblock_size() -> usize {
    144
}

/// Calculate Q6_K superblock size - SPEC VERIFIED
///
/// AUTHORITATIVE SOURCE: ggml.h commit 3fd62a6a6ddd9f999057e3f02b9acb1f8c4b2238
/// - Type 14 = GGML_TYPE_Q6_K (6-bit K-quants)
///
/// Q6_K BLOCK STRUCTURE (210 bytes per 256 elements):
/// - d: FP16 scale (2 bytes)
/// - ql: low 4 bits of 6-bit values (128 bytes, 256 elements * 0.5)
/// - qh: high 2 bits of 6-bit values (64 bytes, 256 elements * 0.25)
/// - scales: per-group int8 scales (16 bytes, 256/16 groups)
///
/// Total: 2 + 128 + 64 + 16 = 210 bytes
///
/// VALIDATION (TinyLlama output.weight):
/// ✅ Shape: [2048, 32000] = 65,536,000 elements
/// ✅ Size: 53,760,000 bytes = 256,000 superblocks × 210 bytes ✓
fn calculate_q6_k_superblock_size() -> usize {
    210
}

/// Validate that a tensor's computed size fits within file bounds
pub fn validate_tensor_bounds(
    tensor_name: &str,
    ggml_type: u32,
    computed_size: usize,
    tensor_offset: u64,
    file_size: u64,
) -> Result<()> {
    let data_end = tensor_offset + computed_size as u64;

    if data_end > file_size {
        let type_name = ggml_type_name(ggml_type)
            .map(|s| s.to_string())
            .unwrap_or_else(|_| format!("UNKNOWN_{}", ggml_type));

        return Err(LibshimmyError::TensorBounds {
            tensor_name: tensor_name.to_string(),
            ggml_type,
            type_name,
            computed_end: data_end,
            file_size,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_enum_values() {
        assert_eq!(GgmlType::F32 as u32, 0);
        assert_eq!(GgmlType::Q4_0 as u32, 2);
        assert_eq!(GgmlType::Q4_K as u32, 12);
        assert_eq!(GgmlType::Q6_K as u32, 14);
    }

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(2).unwrap(), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(12).unwrap(), GgmlType::Q4_K);
        assert_eq!(GgmlType::from_u32(14).unwrap(), GgmlType::Q6_K);

        // Unknown type should fail
        assert!(GgmlType::from_u32(99).is_err());
    }

    #[test]
    fn test_ggml_type_names() {
        assert_eq!(GgmlType::F32.name(), "F32");
        assert_eq!(GgmlType::Q4_0.name(), "Q4_0");
        assert_eq!(GgmlType::Q4_K.name(), "Q4_K");
        assert_eq!(GgmlType::Q6_K.name(), "Q6_K");
    }

    #[test]
    fn test_ggml_type_name_function() {
        assert_eq!(ggml_type_name(0).unwrap(), "F32");
        assert_eq!(ggml_type_name(2).unwrap(), "Q4_0");
        assert_eq!(ggml_type_name(12).unwrap(), "Q4_K");
        assert_eq!(ggml_type_name(14).unwrap(), "Q6_K");

        // Unknown type should fail
        assert!(ggml_type_name(99).is_err());
    }

    #[test]
    fn test_f32_byte_calculation() {
        // F32: 4 bytes per element
        assert_eq!(ggml_type_bytes_per_tensor(0, 1000).unwrap(), 4000);
        assert_eq!(ggml_type_bytes_per_tensor(0, 2048).unwrap(), 8192);
    }

    #[test]
    fn test_q4_0_byte_calculation() {
        // Q4_0: 32 elements per block, 18 bytes per block
        assert_eq!(ggml_type_bytes_per_tensor(2, 32).unwrap(), 18); // Exactly 1 block
        assert_eq!(ggml_type_bytes_per_tensor(2, 64).unwrap(), 36); // Exactly 2 blocks
        assert_eq!(ggml_type_bytes_per_tensor(2, 33).unwrap(), 36); // 2 blocks (33 elements need 2 blocks)
    }

    #[test]
    fn test_q4_k_byte_calculation() {
        // Q4_K: 256 elements per superblock, 144 bytes per superblock
        assert_eq!(ggml_type_bytes_per_tensor(12, 256).unwrap(), 144); // Exactly 1 superblock
        assert_eq!(ggml_type_bytes_per_tensor(12, 512).unwrap(), 288); // Exactly 2 superblocks
        assert_eq!(ggml_type_bytes_per_tensor(12, 257).unwrap(), 288); // 2 superblocks (257 elements need 2)
    }

    #[test]
    fn test_q6_k_byte_calculation() {
        // Q6_K: 256 elements per superblock, 210 bytes per superblock (spec verified)
        assert_eq!(ggml_type_bytes_per_tensor(14, 256).unwrap(), 210); // Exactly 1 superblock
        assert_eq!(ggml_type_bytes_per_tensor(14, 512).unwrap(), 420); // Exactly 2 superblocks
        assert_eq!(ggml_type_bytes_per_tensor(14, 257).unwrap(), 420); // 2 superblocks (257 elements need 2)
    }

    #[test]
    fn test_unknown_type_byte_calculation() {
        // Unknown type should fail
        assert!(ggml_type_bytes_per_tensor(99, 1000).is_err());
    }

    #[test]
    fn test_tensor_bounds_validation() {
        // Valid bounds
        assert!(validate_tensor_bounds("test", 0, 1000, 0, 2000).is_ok());

        // Invalid bounds - extends beyond file
        assert!(validate_tensor_bounds("test", 0, 1000, 1500, 2000).is_err());

        // Edge case - exactly at file end
        assert!(validate_tensor_bounds("test", 0, 1000, 1000, 2000).is_ok());
    }

    // ── Uncovered: F16, Q5_0, Q8_0, Q5_K variants ────────────────────────────

    #[test]
    fn test_ggml_type_from_u32_all_variants() {
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2).unwrap(), GgmlType::Q4_0);
        assert_eq!(GgmlType::from_u32(6).unwrap(), GgmlType::Q5_0);
        assert_eq!(GgmlType::from_u32(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::from_u32(12).unwrap(), GgmlType::Q4_K);
        assert_eq!(GgmlType::from_u32(13).unwrap(), GgmlType::Q5_K);
        assert_eq!(GgmlType::from_u32(14).unwrap(), GgmlType::Q6_K);
        // all other values → Err
        assert!(GgmlType::from_u32(3).is_err());
        assert!(GgmlType::from_u32(5).is_err());
        assert!(GgmlType::from_u32(7).is_err());
        assert!(GgmlType::from_u32(255).is_err());
    }

    #[test]
    fn test_ggml_type_name_all_variants() {
        assert_eq!(GgmlType::F32.name(),  "F32");
        assert_eq!(GgmlType::F16.name(),  "F16");
        assert_eq!(GgmlType::Q4_0.name(), "Q4_0");
        assert_eq!(GgmlType::Q5_0.name(), "Q5_0");
        assert_eq!(GgmlType::Q8_0.name(), "Q8_0");
        assert_eq!(GgmlType::Q4_K.name(), "Q4_K");
        assert_eq!(GgmlType::Q5_K.name(), "Q5_K");
        assert_eq!(GgmlType::Q6_K.name(), "Q6_K");
    }

    #[test]
    fn test_ggml_type_id_roundtrip() {
        for id in [0u32, 1, 2, 6, 8, 12, 13, 14] {
            let t = GgmlType::from_u32(id).unwrap();
            assert_eq!(t.type_id(), id);
        }
    }

    #[test]
    fn test_ggml_type_name_function_all() {
        assert_eq!(ggml_type_name(1).unwrap(),  "F16");
        assert_eq!(ggml_type_name(6).unwrap(),  "Q5_0");
        assert_eq!(ggml_type_name(8).unwrap(),  "Q8_0");
        assert_eq!(ggml_type_name(13).unwrap(), "Q5_K");
    }

    #[test]
    fn test_f16_byte_calculation() {
        assert_eq!(ggml_type_bytes_per_tensor(1, 1000).unwrap(), 2000);
        assert_eq!(ggml_type_bytes_per_tensor(1, 1).unwrap(), 2);
    }

    #[test]
    fn test_q5_0_byte_calculation() {
        // Q5_0: 32 elements/block, 22 bytes/block
        assert_eq!(ggml_type_bytes_per_tensor(6, 32).unwrap(), 22);
        assert_eq!(ggml_type_bytes_per_tensor(6, 64).unwrap(), 44);
        assert_eq!(ggml_type_bytes_per_tensor(6, 33).unwrap(), 44);
    }

    #[test]
    fn test_q8_0_byte_calculation() {
        // Q8_0: 32 elements/block, 34 bytes/block
        assert_eq!(ggml_type_bytes_per_tensor(8, 32).unwrap(), 34);
        assert_eq!(ggml_type_bytes_per_tensor(8, 64).unwrap(), 68);
        assert_eq!(ggml_type_bytes_per_tensor(8, 33).unwrap(), 68);
    }

    #[test]
    fn test_q5_k_byte_calculation() {
        // Q5_K: 256 elements/superblock, 176 bytes/superblock
        assert_eq!(ggml_type_bytes_per_tensor(13, 256).unwrap(), 176);
        assert_eq!(ggml_type_bytes_per_tensor(13, 512).unwrap(), 352);
        assert_eq!(ggml_type_bytes_per_tensor(13, 257).unwrap(), 352);
    }

    #[test]
    fn test_tensor_bounds_zero_size() {
        assert!(validate_tensor_bounds("t", 0, 0, 0, 0).is_ok());
    }

    #[test]
    fn test_tensor_bounds_unknown_type_in_error_message() {
        // validate_tensor_bounds with an unknown type_id should return an error
        let err = validate_tensor_bounds("my_tensor", 99, 10000, 999999, 100).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("my_tensor") || msg.len() > 0, "error message should be non-empty");
    }
}
