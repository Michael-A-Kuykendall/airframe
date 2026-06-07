use crate::core::error::{LibshimmyError, Result};
use crate::core::model::GgufTensorInfo;
use crate::core::tensor::Tensor;
use memmap2::Mmap;

/// Dequantize Q8_0 tensor to FP32.
///
/// Q8_0 block structure (34 bytes per 32 elements):
///   - d:  f16 scale   (2 bytes)
///   - qs: [i8; 32]    (32 bytes)
///
/// Dequant: w[i] = qs[i] as f32 * d
///
/// Reference: ggerganov/ggml ggml.h, GGML_TYPE_Q8_0 = 8
pub fn dequantize_q8_0(
    tensor_info: &GgufTensorInfo,
    mmap: &Mmap,
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    let total_elements: usize = tensor_info.dimensions.iter().product();

    let block_size = 32usize;
    let bytes_per_block = 34usize; // 2 (f16 scale) + 32 (i8 values)
    let num_blocks = total_elements.div_ceil(block_size);

    let data_start = (tensor_data_base_offset + tensor_info.offset) as usize;
    let data_end = data_start + num_blocks * bytes_per_block;

    if data_end > mmap.len() {
        return Err(LibshimmyError::FixtureError {
            msg: "Q8_0 tensor data extends beyond file".to_string(),
        });
    }

    let mut fp32_data = Vec::with_capacity(total_elements);

    for block_idx in 0..num_blocks {
        let block_start = data_start + block_idx * bytes_per_block;

        // Bytes 0-1: f16 scale
        let scale_bytes = [mmap[block_start], mmap[block_start + 1]];
        let scale = crate::core::f16::f16_bits_to_f32(u16::from_le_bytes(scale_bytes));

        // Bytes 2-33: 32 × i8 quantized values
        let block_base = block_idx * block_size;
        for i in 0..block_size {
            if block_base + i >= total_elements {
                break;
            }
            let q = mmap[block_start + 2 + i] as i8;
            fp32_data.push(q as f32 * scale);
        }
    }

    fp32_data.truncate(total_elements);
    Tensor::new(fp32_data, tensor_info.dimensions.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_block_size() {
        // Q8_0: 34 bytes per 32 elements
        let bytes_per_block: usize = 2 + 32; // f16 + 32 × i8
        assert_eq!(bytes_per_block, 34);
    }

    #[test]
    fn test_q8_0_dequant_formula() {
        // Verify the dequant formula: w = i8 * scale
        // scale = 0.5 (f16 repr), qs = [2i8, -4i8, ...] → [1.0, -2.0, ...]
        let scale: f32 = 0.5;
        let q: i8 = 2;
        let expected = 1.0f32;
        let result = q as f32 * scale;
        assert!((result - expected).abs() < 1e-6, "Got {}", result);
    }

    // ── dequantize_q8_0 with synthetic mmap data ──────────────────────────────

    use crate::core::model::GgufTensorInfo;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_q8_0_mmap(blocks: &[(u16, Vec<i8>)]) -> (NamedTempFile, memmap2::Mmap) {
        let mut f = NamedTempFile::new().unwrap();
        for (scale_bits, qs) in blocks {
            f.write_all(&scale_bits.to_le_bytes()).unwrap();
            let bytes: Vec<u8> = qs.iter().map(|&v| v as u8).collect();
            f.write_all(&bytes).unwrap();
        }
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        (f, mmap)
    }

    fn info(dims: Vec<usize>) -> GgufTensorInfo {
        GgufTensorInfo {
            name: "t".to_string(),
            dimensions: dims,
            ggml_type: 8,
            offset: 0,
        }
    }

    #[test]
    fn test_q8_0_all_zeros() {
        // scale=1.0, all qs=0 → all outputs 0.0
        let scale = 0x3C00u16; // f16 1.0
        let qs = vec![0i8; 32];
        let (_f, mmap) = make_q8_0_mmap(&[(scale, qs)]);
        let tensor = dequantize_q8_0(&info(vec![32]), &mmap, 0).unwrap();
        for &v in &tensor.data {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q8_0_known_values() {
        // scale=1.0, qs[0]=10, qs[1]=-5 → [10.0, -5.0, ...]
        let scale = 0x3C00u16;
        let mut qs = vec![0i8; 32];
        qs[0] = 10;
        qs[1] = -5;
        let (_f, mmap) = make_q8_0_mmap(&[(scale, qs)]);
        let tensor = dequantize_q8_0(&info(vec![32]), &mmap, 0).unwrap();
        assert!((tensor.data[0] - 10.0).abs() < 1e-4);
        assert!((tensor.data[1] - (-5.0)).abs() < 1e-4);
    }

    #[test]
    fn test_q8_0_scale_multiplied() {
        // scale=2.0, qs[0]=7 → 14.0
        let scale = 0x4000u16; // f16 2.0
        let mut qs = vec![0i8; 32];
        qs[0] = 7;
        let (_f, mmap) = make_q8_0_mmap(&[(scale, qs)]);
        let tensor = dequantize_q8_0(&info(vec![32]), &mmap, 0).unwrap();
        assert!(
            (tensor.data[0] - 14.0).abs() < 0.01,
            "expected 14.0, got {}",
            tensor.data[0]
        );
    }

    #[test]
    fn test_q8_0_bounds_error() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0u8; 10]).unwrap();
        f.flush().unwrap();
        let mmap = unsafe { memmap2::Mmap::map(f.as_file()) }.unwrap();
        let result = dequantize_q8_0(&info(vec![32]), &mmap, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_q8_0_two_blocks() {
        let scale1 = 0x3C00u16; // 1.0
        let scale2 = 0x4000u16; // 2.0
        let qs1 = vec![1i8; 32];
        let qs2 = vec![3i8; 32];
        let (_f, mmap) = make_q8_0_mmap(&[(scale1, qs1), (scale2, qs2)]);
        let tensor = dequantize_q8_0(&info(vec![64]), &mmap, 0).unwrap();
        assert_eq!(tensor.data.len(), 64);
        for &v in &tensor.data[..32] {
            assert!((v - 1.0).abs() < 0.01, "block1 expected 1.0, got {v}");
        }
        for &v in &tensor.data[32..] {
            assert!((v - 6.0).abs() < 0.01, "block2 expected 6.0, got {v}");
        }
    }
}
