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