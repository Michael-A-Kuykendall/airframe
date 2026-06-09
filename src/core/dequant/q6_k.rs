//! Q6_K dequantization (6-bit K-quant, 210-byte superblocks).
//!
//! Exact port of llama.cpp `dequantize_row_q6_K`.

use crate::core::{
    error::{LibshimmyError, Result},
    f16::f16_bits_to_f32,
    model::GgufTensorInfo,
    tensor::Tensor,
};
use memmap2::Mmap;

/// Dequantize Q6_K tensor to FP32.
pub fn dequantize_q6_k(
    tensor_info: &GgufTensorInfo,
    mmap: &Mmap,
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    let total_elements: usize = tensor_info.dimensions.iter().product();

    // Q6_K format: 256 elements per superblock, 210 bytes per superblock
    let superblock_size = 256;
    let bytes_per_superblock = 210;
    let num_superblocks = total_elements.div_ceil(superblock_size);

    // Tensor offset is relative to the aligned tensor data section start
    let data_start = (tensor_data_base_offset + tensor_info.offset) as usize;
    let data_end = data_start + num_superblocks * bytes_per_superblock;

    // Comprehensive bounds checking
    if data_end > mmap.len() {
        let type_name = crate::core::ggml_types::ggml_type_name(tensor_info.ggml_type)
            .map(|s| s.to_string())
            .unwrap_or_else(|_| format!("UNKNOWN_{}", tensor_info.ggml_type));
        return Err(LibshimmyError::TensorBounds {
            tensor_name: tensor_info.name.clone(),
            ggml_type: tensor_info.ggml_type,
            type_name,
            computed_end: data_end as u64,
            file_size: mmap.len() as u64,
        });
    }

    let mut fp32_data = Vec::with_capacity(total_elements);

    for superblock_idx in 0..num_superblocks {
        let superblock_start = data_start + superblock_idx * bytes_per_superblock;
        let superblock_data = &mmap[superblock_start..superblock_start + bytes_per_superblock];

        // Dequantize one superblock (256 elements)
        let superblock_fp32 = dequantize_q6_k_superblock(superblock_data);

        // Add elements to output, but don't exceed total_elements
        let elements_to_add = std::cmp::min(superblock_size, total_elements - fp32_data.len());
        fp32_data.extend_from_slice(&superblock_fp32[..elements_to_add]);
    }

    // Create tensor with shape from tensor_info
    let shape = tensor_info.dimensions.to_vec();
    Tensor::new(fp32_data, shape)
}

/// Dequantize a single Q6_K superblock (256 elements from 210 bytes)
///
/// EXACT PORT of llama.cpp dequantize_row_q6_K from ggml-quants.c
///
/// Memory layout:
/// - Bytes 0-127: ql (low 4 bits of quantized values)
/// - Bytes 128-191: qh (high 2 bits of quantized values)
/// - Bytes 192-207: scales (16 int8 per-group scales)
/// - Bytes 208-209: d (FP16 global scale)
fn dequantize_q6_k_superblock(data: &[u8]) -> [f32; 256] {
    let mut output = [0.0f32; 256];

    // Parse block structure
    let ql = &data[0..128]; // low 4 bits
    let qh = &data[128..192]; // high 2 bits
    let scales = &data[192..208]; // int8 scales
    let d_bytes = [data[208], data[209]];

    // FP16 to FP32 conversion for global scale
    let d = f16_bits_to_f32(u16::from_le_bytes(d_bytes));    // The Q6_K dequantization processes 256 elements in two 128-element chunks
    // Each chunk uses 64 bytes of ql, 32 bytes of qh, and 8 scales

    // Process two 128-element halves
    for n in 0..2 {
        let y_offset = n * 128;
        let ql_offset = n * 64;
        let qh_offset = n * 32;
        let sc_offset = n * 8;

        // Each 128-element chunk is processed as 4 groups of 32 elements
        for l in 0..32 {
            // Determine which scale to use (groups of 16 elements)
            let is = l / 16;

            // Extract the 6-bit quantized values
            // Each ql byte contains low 4 bits for 2 elements
            // Each qh byte contains high 2 bits for 4 elements

            let ql_idx = ql_offset + l;
            let qh_idx = qh_offset + l;

            // q1: elements at positions l+0, using ql low nibble and qh bits 0-1
            let q1 = ((ql[ql_idx] & 0x0F) | ((qh[qh_idx] & 0x03) << 4)) as i8 - 32;

            // q2: elements at positions l+32, using ql+32 low nibble and qh bits 2-3
            let q2 = ((ql[ql_idx + 32] & 0x0F) | (((qh[qh_idx] >> 2) & 0x03) << 4)) as i8 - 32;

            // q3: elements at positions l+64, using ql high nibble and qh bits 4-5
            let q3 = ((ql[ql_idx] >> 4) | (((qh[qh_idx] >> 4) & 0x03) << 4)) as i8 - 32;

            // q4: elements at positions l+96, using ql+32 high nibble and qh bits 6-7
            let q4 = ((ql[ql_idx + 32] >> 4) | (((qh[qh_idx] >> 6) & 0x03) << 4)) as i8 - 32;

            // Get the scales for this position
            let sc0 = scales[sc_offset + is] as i8; // for q1
            let sc2 = scales[sc_offset + is + 2] as i8; // for q2
            let sc4 = scales[sc_offset + is + 4] as i8; // for q3
            let sc6 = scales[sc_offset + is + 6] as i8; // for q4

            // Dequantize: value = d * scale * quantized_value
            output[y_offset + l] = d * (sc0 as f32) * (q1 as f32);
            output[y_offset + l + 32] = d * (sc2 as f32) * (q2 as f32);
            output[y_offset + l + 64] = d * (sc4 as f32) * (q3 as f32);
            output[y_offset + l + 96] = d * (sc6 as f32) * (q4 as f32);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q6_k_superblock_zero() {
        // A block with all zeros should produce zeros
        // d = 0.0 means all outputs are zero regardless of quantized values
        let data = [0u8; 210];
        // Set d (FP16) to 0 - already done by initialization

        let output = dequantize_q6_k_superblock(&data);

        for &val in &output {
            assert_eq!(val, 0.0, "All outputs should be zero when d=0");
        }
    }

    // ── Tensor-level (mmap) property tests ───────────────────────────────────

    use crate::core::model::GgufTensorInfo;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn info_q6k(dims: Vec<usize>) -> GgufTensorInfo {
        GgufTensorInfo {
            name: "t".to_string(),
            dimensions: dims,
            ggml_type: 14,
            offset: 0,
        }
    }

    #[test]
    fn test_q6_k_tensor_zero_superblock_produces_correct_count() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 210]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q6_k(&info_q6k(vec![256]), &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 256);
        assert!(tensor.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_q6_k_tensor_partial_superblock_truncated() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 210]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q6_k(&info_q6k(vec![128]), &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 128);
    }

    #[test]
    fn test_q6_k_tensor_oob_returns_error() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 10]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        assert!(dequantize_q6_k(&info_q6k(vec![256]), &mmap, 0).is_err());
    }

    #[test]
    fn test_q6_k_output_shape_matches_dimensions() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 210 * 2]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q6_k(&info_q6k(vec![64, 8]), &mmap, 0).unwrap();
        assert_eq!(tensor.shape, vec![64, 8]);
        assert_eq!(tensor.data.len(), 512);
    }

    #[test]
    fn test_dequantize_q6_k_superblock_structure() {
        // Test that the block parsing is correct by setting known values
        let mut data = [0u8; 210];

        // Set d to 1.0 (FP16: 0x3C00)
        data[208] = 0x00;
        data[209] = 0x3C;

        // Set all scales to 1 (at bytes 192-207)
        for byte in &mut data[192..208] {
            *byte = 1;
        }

        // Set first ql byte to encode value 32 (which becomes 0 after -32 offset)
        // For q1: (ql & 0x0F) | ((qh & 0x03) << 4) - 32
        // To get 0: we need the combined 6-bit value to be 32
        // 32 = 0b100000 = (0 & 0x0F) | ((2 & 0x03) << 4) = 0 | (2 << 4) = 32
        data[0] = 0x00; // ql[0] = 0
        data[128] = 0x02; // qh[0] bits 0-1 = 2

        let output = dequantize_q6_k_superblock(&data);

        // First element should be d * scale * (32 - 32) = 1.0 * 1 * 0 = 0
        assert_eq!(
            output[0], 0.0,
            "First element should be 0 when quantized value is 32"
        );
    }
}
