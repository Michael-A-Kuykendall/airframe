//! Q5_K dequantization (5-bit K-quant, 176-byte superblocks).
//!
//! Exact port of llama.cpp `dequantize_row_q5_K`.

use crate::core::{
    error::{LibshimmyError, Result},
    model::GgufTensorInfo,
    tensor::Tensor,
};
use memmap2::Mmap;

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;
/// 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs) = 176
const BYTES_PER_BLOCK: usize = 176;

/// Dequantize Q5_K tensor to FP32.
pub fn dequantize_q5_k(
    tensor_info: &GgufTensorInfo,
    mmap: &Mmap,
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    let total_elements: usize = tensor_info.dimensions.iter().product();
    let num_blocks = total_elements.div_ceil(QK_K);

    let data_start = (tensor_data_base_offset + tensor_info.offset) as usize;
    let data_end = data_start + num_blocks * BYTES_PER_BLOCK;

    if data_end > mmap.len() {
        return Err(LibshimmyError::TensorBounds {
            tensor_name: tensor_info.name.clone(),
            ggml_type: tensor_info.ggml_type,
            type_name: "Q5_K".to_string(),
            computed_end: data_end as u64,
            file_size: mmap.len() as u64,
        });
    }

    let mut fp32_data = Vec::with_capacity(total_elements);

    for block_idx in 0..num_blocks {
        let block_start = data_start + block_idx * BYTES_PER_BLOCK;
        let block = &mmap[block_start..block_start + BYTES_PER_BLOCK];

        let block_fp32 =
            dequantize_q5_k_block(block).map_err(|e| LibshimmyError::DequantizationError {
                tensor_name: tensor_info.name.clone(),
                ggml_type: tensor_info.ggml_type,
                type_name: "Q5_K".to_string(),
                reason: format!("block_idx={}: {}", block_idx, e),
            })?;

        let elements_to_add = std::cmp::min(QK_K, total_elements - fp32_data.len());
        fp32_data.extend_from_slice(&block_fp32[..elements_to_add]);
    }

    if let Some((idx, val)) = fp32_data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(LibshimmyError::DequantizationError {
            tensor_name: tensor_info.name.clone(),
            ggml_type: tensor_info.ggml_type,
            type_name: "Q5_K".to_string(),
            reason: format!("non-finite value at element_idx={} value={}", idx, val),
        });
    }

    Tensor::new(fp32_data, tensor_info.dimensions.clone())
}

fn dequantize_q5_k_block(block: &[u8]) -> std::result::Result<[f32; QK_K], String> {
    if block.len() != BYTES_PER_BLOCK {
        return Err(format!(
            "Q5_K block should be {} bytes, got {}",
            BYTES_PER_BLOCK,
            block.len()
        ));
    }

    // Block layout:
    //   [0..1]   d       (fp16)
    //   [2..3]   dmin    (fp16)
    //   [4..15]  scales  (12 bytes, same 6-bit packed format as Q4_K)
    //   [16..47] qh      (32 bytes: for element i, high_bit = (qh[i%32] >> (i/32)) & 1)
    //   [48..175] qs     (128 bytes: low 4 bits per element)

    let d = crate::core::f16::f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = crate::core::f16::f16_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
    if !d.is_finite() || !dmin.is_finite() {
        return Err(format!("non-finite d/dmin: d={} dmin={}", d, dmin));
    }

    let scales: &[u8] = &block[4..16];
    let qh: &[u8] = &block[16..48];
    let qs: &[u8] = &block[48..176];

    let mut out = [0.0f32; QK_K];
    let mut is = 0;

    // 4 groups of 64 elements.  Each group has two 32-element sub-blocks.
    // Within each sub-block l (0..32):
    //   - ql index : group*32 + l  (same row of qs for both sub-blocks)
    //   - low nibble : qs[group*32+l] & 0x0F  (sub 0)
    //   - high nibble: qs[group*32+l] >> 4     (sub 1)
    //   - high bit   : (qh[l] >> (group*2 + sub)) & 1
    //   - val        : d * sc * q5 - dmin * m  where q5 = nibble | (high_bit << 4)
    for group in 0usize..4 {
        let (sc0, m0) = get_scale_min_k4(is, scales);
        let (sc1, m1u) = get_scale_min_k4(is + 1, scales);
        let d1 = d * (sc0 as f32);
        let m1 = dmin * (m0 as f32);
        let d2 = d * (sc1 as f32);
        let m2 = dmin * (m1u as f32);

        let bit0 = (group * 2) as u32;     // bit position for sub 0
        let bit1 = (group * 2 + 1) as u32; // bit position for sub 1

        // Sub-block 0: elements [group*64 .. group*64+32)
        for l in 0usize..32 {
            let ql = qs[group * 32 + l];
            let nibble = (ql & 0x0F) as f32;
            let high_bit = ((qh[l] >> bit0) & 1) as f32;
            let q5 = nibble + high_bit * 16.0;
            out[group * 64 + l] = d1 * q5 - m1;
        }

        // Sub-block 1: elements [group*64+32 .. group*64+64)
        for l in 0usize..32 {
            let ql = qs[group * 32 + l];
            let nibble = (ql >> 4) as f32;
            let high_bit = ((qh[l] >> bit1) & 1) as f32;
            let q5 = nibble + high_bit * 16.0;
            out[group * 64 + 32 + l] = d2 * q5 - m2;
        }

        is += 2;
    }

    Ok(out)
}

/// Exact llama.cpp `get_scale_min_k4` helper (shared with Q4_K).
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
    fn test_q5_k_block_size_constant() {
        assert_eq!(BYTES_PER_BLOCK, 176);
    }

    #[test]
    fn test_dequantize_q5_k_block_zeroes_is_finite() {
        let block = [0u8; BYTES_PER_BLOCK];
        let out = dequantize_q5_k_block(&block).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Tensor-level (mmap) property tests ───────────────────────────────────

    use crate::core::model::GgufTensorInfo;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn info_q5k(dims: Vec<usize>) -> GgufTensorInfo {
        GgufTensorInfo { name: "t".to_string(), dimensions: dims, ggml_type: 13, offset: 0 }
    }

    #[test]
    fn test_q5_k_tensor_zero_superblock_produces_correct_count() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; BYTES_PER_BLOCK]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q5_k(&info_q5k(vec![256]), &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 256);
        assert!(tensor.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_q5_k_tensor_partial_superblock_truncated() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; BYTES_PER_BLOCK]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q5_k(&info_q5k(vec![128]), &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 128);
    }

    #[test]
    fn test_q5_k_tensor_oob_returns_error() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 10]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        assert!(dequantize_q5_k(&info_q5k(vec![256]), &mmap, 0).is_err());
    }

    #[test]
    fn test_q5_k_output_shape_matches_dimensions() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; BYTES_PER_BLOCK * 2]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let tensor = dequantize_q5_k(&info_q5k(vec![64, 8]), &mmap, 0).unwrap();
        assert_eq!(tensor.shape, vec![64, 8]);
        assert_eq!(tensor.data.len(), 512);
    }

    #[test]
    fn test_high_bit_formula() {
        // Verify: for element i, high_bit = (qh[i%32] >> (i/32)) & 1
        // element 0:  qh[0] bit 0
        // element 32: qh[0] bit 1
        // element 64: qh[0] bit 2
        // element 255: qh[31] bit 7
        let mut qh = [0u8; 32];
        qh[0] = 0b10101010;   // bits 1,3,5,7 set  → elements 32,96,160,224 get high_bit=1
        qh[31] = 0b00000001;  // bit 0 set          → element 31 gets high_bit=1

        let check = |i: usize| -> u8 { (qh[i % 32] >> (i / 32)) & 1 };
        assert_eq!(check(0), 0);   // qh[0] bit 0 = 0
        assert_eq!(check(32), 1);  // qh[0] bit 1 = 1
        assert_eq!(check(64), 0);  // qh[0] bit 2 = 0
        assert_eq!(check(96), 1);  // qh[0] bit 3 = 1
        assert_eq!(check(31), 1);  // qh[31] bit 0 = 1
        assert_eq!(check(255), 0); // qh[31] bit 7 = 0
    }
}
