//! Q4_K dequantization (4-bit K-quant, 144-byte superblocks).
//!
//! Exact port of llama.cpp `dequantize_row_q4_K`.

use crate::core::{
    error::{LibshimmyError, Result},
    model::GgufTensorInfo,
    tensor::Tensor,
};
use memmap2::Mmap;

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;
const BYTES_PER_BLOCK: usize = 2 * 2 + K_SCALE_SIZE + (QK_K / 2); // 144

/// Dequantize Q4_K tensor to FP32.
pub fn dequantize_q4_k(
    tensor_info: &GgufTensorInfo,
    mmap: &Mmap,
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    let total_elements: usize = tensor_info.dimensions.iter().product();
    let num_blocks = total_elements.div_ceil(QK_K);

    // Tensor offset is relative to the aligned tensor data section start
    let data_start = (tensor_data_base_offset + tensor_info.offset) as usize;
    let data_end = data_start + num_blocks * BYTES_PER_BLOCK;

    if data_end > mmap.len() {
        return Err(LibshimmyError::TensorBounds {
            tensor_name: tensor_info.name.clone(),
            ggml_type: tensor_info.ggml_type,
            type_name: "Q4_K".to_string(),
            computed_end: data_end as u64,
            file_size: mmap.len() as u64,
        });
    }

    let mut fp32_data = Vec::with_capacity(total_elements);

    for block_idx in 0..num_blocks {
        let block_start = data_start + block_idx * BYTES_PER_BLOCK;
        let block = &mmap[block_start..block_start + BYTES_PER_BLOCK];

        let block_fp32 =
            dequantize_q4_k_block(block).map_err(|e| LibshimmyError::DequantizationError {
                tensor_name: tensor_info.name.clone(),
                ggml_type: tensor_info.ggml_type,
                type_name: "Q4_K".to_string(),
                reason: format!("block_idx={}: {}", block_idx, e),
            })?;

        let elements_to_add = std::cmp::min(QK_K, total_elements - fp32_data.len());
        fp32_data.extend_from_slice(&block_fp32[..elements_to_add]);
    }

    // Fail-closed: do not mask NaN/Inf.
    if let Some((idx, val)) = fp32_data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(LibshimmyError::DequantizationError {
            tensor_name: tensor_info.name.clone(),
            ggml_type: tensor_info.ggml_type,
            type_name: "Q4_K".to_string(),
            reason: format!("non-finite value at element_idx={} value={}", idx, val),
        });
    }

    Tensor::new(fp32_data, tensor_info.dimensions.clone())
}

fn dequantize_q4_k_block(block: &[u8]) -> std::result::Result<[f32; QK_K], String> {
    if block.len() != BYTES_PER_BLOCK {
        return Err(format!(
            "Q4_K block should be {} bytes, got {}",
            BYTES_PER_BLOCK,
            block.len()
        ));
    }

    let d = crate::core::f16::f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = crate::core::f16::f16_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
    if !d.is_finite() || !dmin.is_finite() {
        return Err(format!("non-finite d/dmin: d={} dmin={}", d, dmin));
    }

    let scales: &[u8] = &block[4..16];
    let mut q: &[u8] = &block[16..144];

    let mut out = [0.0f32; QK_K];
    let mut out_idx = 0usize;

    let mut is = 0;
    for _j in (0..QK_K).step_by(64) {
        let (sc0, m0) = get_scale_min_k4(is, scales);
        let d1 = d * (sc0 as f32);
        let m1 = dmin * (m0 as f32);

        let (sc1, m1u) = get_scale_min_k4(is + 1, scales);
        let d2 = d * (sc1 as f32);
        let m2 = dmin * (m1u as f32);

        // Exact llama.cpp ordering:
        // - 32 outputs from low nibbles with (d1,m1)
        // - 32 outputs from high nibbles with (d2,m2)
        for &qb in q.iter().take(32) {
            out[out_idx] = d1 * ((qb & 0x0F) as f32) - m1;
            out_idx += 1;
        }
        for &qb in q.iter().take(32) {
            out[out_idx] = d2 * ((qb >> 4) as f32) - m2;
            out_idx += 1;
        }

        q = &q[32..];
        is += 2;
    }

    Ok(out)
}

// Exact llama.cpp helper:
// static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m)
//
// if (j < 4) {
//     *d = q[j] & 63; *m = q[j + 4] & 63;
// } else {
//     *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
//     *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
// }
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    debug_assert_eq!(q.len(), K_SCALE_SIZE);
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | (((q[j - 4] >> 6) & 0x03) << 4);
        let m = ((q[j + 4] >> 4) & 0x0F) | (((q[j] >> 6) & 0x03) << 4);
        (d, m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_k_block_size_constant() {
        assert_eq!(BYTES_PER_BLOCK, 144);
    }

    #[test]
    fn test_get_scale_min_k4_matches_reference_logic() {
        // Construct a deterministic 12-byte scale array.
        let scales: [u8; 12] = [
            0b1100_0011, // j=0 low bits + upper bits for j>=4
            0b0100_0010, // j=1
            0b1000_0001, // j=2
            0b0000_0000, // j=3
            0b0011_1111, // j=4
            0b0000_1111, // j=5
            0b1111_0000, // j=6
            0b1010_1010, // j=7
            0b0101_0101, // j=8
            0b1001_1001, // j=9
            0b1111_1111, // j=10
            0b0000_0001, // j=11
        ];

        // For j < 4
        for j in 0..4 {
            let (d, m) = get_scale_min_k4(j, &scales);
            assert_eq!(d, scales[j] & 63);
            assert_eq!(m, scales[j + 4] & 63);
        }

        // Spot-check j >= 4 against the same formula used in llama.cpp
        for j in 4..8 {
            let (d, m) = get_scale_min_k4(j, &scales);
            let d_ref = (scales[j + 4] & 0x0F) | (((scales[j - 4] >> 6) & 0x03) << 4);
            let m_ref = ((scales[j + 4] >> 4) & 0x0F) | (((scales[j] >> 6) & 0x03) << 4);
            assert_eq!(d, d_ref);
            assert_eq!(m, m_ref);
        }
    }

    #[test]
    fn test_dequantize_q4_k_block_zeroes_is_finite() {
        let block = [0u8; BYTES_PER_BLOCK];
        let out = dequantize_q4_k_block(&block).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Tensor-level (mmap) property tests ───────────────────────────────────

    use crate::core::model::GgufTensorInfo;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn info_q4k(dims: Vec<usize>) -> GgufTensorInfo {
        GgufTensorInfo { name: "t".to_string(), dimensions: dims, ggml_type: 12, offset: 0 }
    }

    #[test]
    fn test_q4_k_tensor_zero_superblock_produces_correct_count() {
        // One zero superblock (144 bytes) → 256 elements, all finite
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; BYTES_PER_BLOCK]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q4_k(&info_q4k(vec![256]), &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 256);
        assert!(tensor.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_q4_k_tensor_partial_superblock_truncated() {
        // Request 128 elements (half a superblock); mmap has one full superblock
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; BYTES_PER_BLOCK]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q4_k(&info_q4k(vec![128]), &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 128);
    }

    #[test]
    fn test_q4_k_tensor_oob_returns_error() {
        // 10-byte file is too small for a 144-byte superblock
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 10]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        assert!(dequantize_q4_k(&info_q4k(vec![256]), &mmap, 0).is_err());
    }

    #[test]
    fn test_q4_k_output_shape_matches_dimensions() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; BYTES_PER_BLOCK * 2]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q4_k(&info_q4k(vec![64, 8]), &mmap, 0).unwrap();
        assert_eq!(tensor.shape, vec![64, 8]);
        assert_eq!(tensor.data.len(), 512);
    }
}
