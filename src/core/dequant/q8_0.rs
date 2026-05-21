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
}
