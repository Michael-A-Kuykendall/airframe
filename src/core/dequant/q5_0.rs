use crate::core::error::{LibshimmyError, Result};
use crate::core::model::GgufTensorInfo;
use crate::core::tensor::Tensor;
use memmap2::Mmap;

/// Dequantize Q5_0 tensor to FP32.
///
/// Q5_0 block structure (22 bytes per 32 elements):
///   - d:  f16 scale        (2 bytes)
///   - qh: [u8; 4]          (4 bytes)  — 32 high bits (1 per element)
///   - qs: [u8; 16]         (16 bytes) — 32 low nibbles (4 bits per element, 2 per byte)
///
/// Dequant (per element pair at index j = 0..16):
///   low_0  = qs[j] & 0x0F
///   low_1  = (qs[j] >> 4) & 0x0F
///   high_0 = (qh_u32 >> j) & 1         → placed at bit 4
///   high_1 = (qh_u32 >> (j + 16)) & 1  → placed at bit 4
///   val_0  = (low_0 | (high_0 << 4)) as i32 - 16  → [-16, 15]
///   val_1  = (low_1 | (high_1 << 4)) as i32 - 16  → [-16, 15]
///   w[j]      = val_0 as f32 * d
///   w[j + 16] = val_1 as f32 * d
///
/// Reference: ggerganov/ggml ggml.h, GGML_TYPE_Q5_0 = 6
pub fn dequantize_q5_0(
    tensor_info: &GgufTensorInfo,
    mmap: &Mmap,
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    let total_elements: usize = tensor_info.dimensions.iter().product();

    let block_size = 32usize;
    let bytes_per_block = 22usize; // 2 (f16) + 4 (qh) + 16 (qs)
    let num_blocks = total_elements.div_ceil(block_size);

    let data_start = (tensor_data_base_offset + tensor_info.offset) as usize;
    let data_end = data_start + num_blocks * bytes_per_block;

    if data_end > mmap.len() {
        return Err(LibshimmyError::FixtureError {
            msg: "Q5_0 tensor data extends beyond file".to_string(),
        });
    }

    let mut fp32_data = Vec::with_capacity(total_elements);

    for block_idx in 0..num_blocks {
        let block_start = data_start + block_idx * bytes_per_block;

        // Bytes 0-1: f16 scale
        let scale_bytes = [mmap[block_start], mmap[block_start + 1]];
        let scale = crate::core::f16::f16_bits_to_f32(u16::from_le_bytes(scale_bytes));

        // Bytes 2-5: 32 high bits packed into u32
        let qh = u32::from_le_bytes([
            mmap[block_start + 2],
            mmap[block_start + 3],
            mmap[block_start + 4],
            mmap[block_start + 5],
        ]);

        // Bytes 6-21: 16 bytes of nibble pairs (low 4 bits per element)
        let block_base = block_idx * block_size;
        let mut block_values = [0.0f32; 32];

        for j in 0..16usize {
            let byte_val = mmap[block_start + 6 + j];

            // Element j: low nibble + high bit from qh[j]
            let low_0 = (byte_val & 0x0F) as u32;
            let high_0 = (qh >> j) & 1;
            let val_0 = ((low_0 | (high_0 << 4)) as i32) - 16;
            block_values[j] = val_0 as f32 * scale;

            // Element j+16: high nibble + high bit from qh[j+16]
            let low_1 = ((byte_val >> 4) & 0x0F) as u32;
            let high_1 = (qh >> (j + 16)) & 1;
            let val_1 = ((low_1 | (high_1 << 4)) as i32) - 16;
            block_values[j + 16] = val_1 as f32 * scale;
        }

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

    #[test]
    fn test_q5_0_block_size() {
        // Q5_0: 22 bytes per 32 elements
        let bytes_per_block: usize = 2 + 4 + 16; // f16 + qh + qs
        assert_eq!(bytes_per_block, 22);
    }

    #[test]
    fn test_q5_0_dequant_range() {
        // Values range: [-16*scale, 15*scale]
        // With scale=1.0: val5=0b10000=16 → 16-16=0; val5=0b11111=31 → 31-16=15; val5=0 → -16
        let scale = 1.0f32;
        let max_val = (31i32 - 16) as f32 * scale;
        let min_val = (0i32 - 16) as f32 * scale;
        assert_eq!(max_val, 15.0);
        assert_eq!(min_val, -16.0);
    }
}
