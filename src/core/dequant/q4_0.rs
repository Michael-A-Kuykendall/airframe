use crate::core::error::{LibshimmyError, Result};
use crate::core::model::GgufTensorInfo;
use crate::core::tensor::Tensor;
use memmap2::Mmap;

/// Dequantize Q4_0 tensor to FP32.
pub fn dequantize_q4_0(
    tensor_info: &GgufTensorInfo,
    mmap: &Mmap,
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    let total_elements: usize = tensor_info.dimensions.iter().product();

    let block_size = 32;
    let bytes_per_block = 18;
    let num_blocks = total_elements.div_ceil(block_size);

    let data_start = (tensor_data_base_offset + tensor_info.offset) as usize;
    let data_end = data_start + num_blocks * bytes_per_block;

    if data_end > mmap.len() {
        return Err(LibshimmyError::FixtureError {
            msg: "Tensor data extends beyond file".to_string(),
        });
    }

    let mut fp32_data = Vec::with_capacity(total_elements);

    for block_idx in 0..num_blocks {
        let block_start = data_start + block_idx * bytes_per_block;

        let scale_bytes = [mmap[block_start], mmap[block_start + 1]];
        let scale = crate::core::f16::f16_bits_to_f32(u16::from_le_bytes(scale_bytes));

        let mut block_values = [0.0f32; 32];
        for byte_idx in 0..16 {
            let byte_offset = block_start + 2 + byte_idx;
            let byte_val = mmap[byte_offset];

            let val_low = (byte_val & 0x0F) as i8 - 8;
            let val_high = ((byte_val >> 4) & 0x0F) as i8 - 8;

            block_values[byte_idx] = val_low as f32 * scale;
            block_values[byte_idx + 16] = val_high as f32 * scale;
        }

        let block_base = block_idx * block_size;
        for (i, &val) in block_values.iter().enumerate() {
            if block_base + i < total_elements {
                fp32_data.push(val);
            }
        }
    }

    fp32_data.truncate(total_elements);

    Tensor::new(fp32_data, tensor_info.dimensions.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::model::GgufTensorInfo;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_q4_0_mmap(blocks: &[(u16, [u8; 16])]) -> (NamedTempFile, memmap2::Mmap) {
        let mut f = NamedTempFile::new().unwrap();
        for (scale_bits, nibbles) in blocks {
            f.write_all(&scale_bits.to_le_bytes()).unwrap();
            f.write_all(nibbles).unwrap();
        }
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        (f, mmap)
    }

    fn tensor_info(dims: Vec<usize>) -> GgufTensorInfo {
        GgufTensorInfo {
            name: "test".to_string(),
            dimensions: dims,
            ggml_type: 2,
            offset: 0,
        }
    }

    // ── Basic dequant: all-zero nibbles ───────────────────────────────────────

    #[test]
    fn test_q4_0_all_zeros() {
        // Nibble 0x8 - 8 = 0 for both halves → all outputs zero regardless of scale
        let scale_bits = 0x3C00u16; // f16 = 1.0
        let nibbles = [0x88u8; 16]; // all nibbles = 8, value = 8-8 = 0
        let (_f, mmap) = make_q4_0_mmap(&[(scale_bits, nibbles)]);
        let info = tensor_info(vec![32]);
        let tensor = dequantize_q4_0(&info, &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 32);
        for &v in &tensor.data {
            assert_eq!(v, 0.0, "all nibbles=8 → all zeros");
        }
    }

    // ── Basic dequant: max positive nibble (0xF) ──────────────────────────────

    #[test]
    fn test_q4_0_max_positive() {
        // Nibble 0xF = 15 - 8 = 7, scale=1.0 → all outputs = 7.0
        let scale_bits = 0x3C00u16; // f16 = 1.0
        let nibbles = [0xFFu8; 16]; // low nibble=F=15-8=7, high nibble=F=15-8=7
        let (_f, mmap) = make_q4_0_mmap(&[(scale_bits, nibbles)]);
        let info = tensor_info(vec![32]);
        let tensor = dequantize_q4_0(&info, &mmap, 0).unwrap();
        for &v in &tensor.data {
            assert!((v - 7.0).abs() < 1e-4, "expected 7.0, got {v}");
        }
    }

    // ── Basic dequant: max negative nibble (0x0) ──────────────────────────────

    #[test]
    fn test_q4_0_max_negative() {
        // Nibble 0x0 = 0 - 8 = -8, scale=1.0 → all outputs = -8.0
        let scale_bits = 0x3C00u16;
        let nibbles = [0x00u8; 16];
        let (_f, mmap) = make_q4_0_mmap(&[(scale_bits, nibbles)]);
        let info = tensor_info(vec![32]);
        let tensor = dequantize_q4_0(&info, &mmap, 0).unwrap();
        for &v in &tensor.data {
            assert!((v - (-8.0)).abs() < 1e-4, "expected -8.0, got {v}");
        }
    }

    // ── Scale factor is applied ────────────────────────────────────────────────

    #[test]
    fn test_q4_0_scale_applied() {
        // scale=2.0 (f16 bits = 0x4000), nibbles=0xF → values should be 7 * 2.0 = 14.0
        let scale_bits = 0x4000u16; // f16 = 2.0
        let nibbles = [0xFFu8; 16];
        let (_f, mmap) = make_q4_0_mmap(&[(scale_bits, nibbles)]);
        let info = tensor_info(vec![32]);
        let tensor = dequantize_q4_0(&info, &mmap, 0).unwrap();
        for &v in &tensor.data {
            assert!((v - 14.0).abs() < 0.01, "expected 14.0, got {v}");
        }
    }

    // ── Output shape matches dimensions ───────────────────────────────────────

    #[test]
    fn test_q4_0_output_shape() {
        let scale_bits = 0x3C00u16;
        let nibbles = [0x88u8; 16];
        let (_f, mmap) = make_q4_0_mmap(&[(scale_bits, nibbles)]);
        let info = tensor_info(vec![32]);
        let tensor = dequantize_q4_0(&info, &mmap, 0).unwrap();
        assert_eq!(tensor.shape, vec![32]);
    }

    // ── Two blocks ────────────────────────────────────────────────────────────

    #[test]
    fn test_q4_0_two_blocks() {
        let scale1 = 0x3C00u16; // 1.0
        let scale2 = 0x4000u16; // 2.0
        let nibbles1 = [0xFFu8; 16]; // all 7
        let nibbles2 = [0x00u8; 16]; // all -8
        let (_f, mmap) = make_q4_0_mmap(&[(scale1, nibbles1), (scale2, nibbles2)]);
        let info = tensor_info(vec![64]);
        let tensor = dequantize_q4_0(&info, &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 64);
        // First 32: scale=1.0, nibble=7 → 7.0
        for &v in &tensor.data[..32] {
            assert!((v - 7.0).abs() < 0.01, "block1 expected 7.0, got {v}");
        }
        // Second 32: scale=2.0, nibble=-8 → -16.0
        for &v in &tensor.data[32..] {
            assert!((v - (-16.0)).abs() < 0.01, "block2 expected -16.0, got {v}");
        }
    }

    // ── Partial block (dimensions not multiple of 32) ─────────────────────────

    #[test]
    fn test_q4_0_partial_block_truncated() {
        // Request only 16 elements (half a block of 32)
        let scale_bits = 0x3C00u16;
        let nibbles = [0xFFu8; 16];
        let (_f, mmap) = make_q4_0_mmap(&[(scale_bits, nibbles)]);
        let info = tensor_info(vec![16]); // only 16 elements
        let tensor = dequantize_q4_0(&info, &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 16, "should truncate to 16 elements");
    }

    // ── Bounds check: tensor beyond file end ──────────────────────────────────

    #[test]
    fn test_q4_0_out_of_bounds_returns_error() {
        // Create a file with only 10 bytes — not enough for a Q4_0 block (18 bytes)
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 10]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let info = tensor_info(vec![32]); // needs 18 bytes, only 10 available
        let result = dequantize_q4_0(&info, &mmap, 0);
        assert!(result.is_err(), "should fail when data extends beyond mmap");
    }
}
