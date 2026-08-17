//! IQ4_XS dequantization (4-bit integer quantization, Extra Small blocks).
//!
//! Exact port of GGML `dequantize_row_iq4_xs` from ggml-quants.c.
//! Type 30: 128 elements per superblock, 128 bytes per superblock.
//! Layout: [0..64] scales (32 fp16 = 64 bytes); [64..128] qs (128 4-bit = 64 bytes).

use crate::core::error::{LibshimmyError, Result};
use crate::core::model::GgufTensorInfo;
use crate::core::tensor::Tensor;

/// Dequantize IQ4_XS tensor to FP32.
pub fn dequantize_iq4_xs(
    tensor_info: &GgufTensorInfo,
    mmap: &[u8],
    tensor_data_base_offset: u64,
) -> Result<Tensor> {
    let total_elements: usize = tensor_info.dimensions.iter().product();

    // IQ4_XS: 128 elements per superblock, 128 bytes per superblock
    let superblock_size = 128;
    let bytes_per_superblock = 128;
    let num_superblocks = total_elements.div_ceil(superblock_size);

    // Tensor offset is relative to the aligned tensor data section start
    let data_start = (tensor_data_base_offset + tensor_info.offset) as usize;
    let data_end = data_start + num_superblocks * bytes_per_superblock;

    if data_end > mmap.len() {
        return Err(LibshimmyError::FixtureError {
            msg: "IQ4_XS tensor data extends beyond file".to_string(),
        });
    }

    let mut fp32_data = Vec::with_capacity(total_elements);

    for sb_idx in 0..num_superblocks {
        let sb_start = data_start + sb_idx * bytes_per_superblock;
        let scales = &mmap[sb_start..sb_start + 64]; // 32 fp16 scales
        let qs = &mmap[sb_start + 64..sb_start + 128]; // 128 4-bit values (64 bytes)

        let elements_in_this_sb = superblock_size.min(total_elements - sb_idx * superblock_size);

        for elem_in_sb in 0..elements_in_this_sb {
            let scale_idx = elem_in_sb / 4;
            let scale = f16_to_f32(u16::from_le_bytes([
                scales[scale_idx * 2],
                scales[scale_idx * 2 + 1],
            ]));

            let byte_idx = elem_in_sb / 2;
            let qs_byte = qs[byte_idx];
            let nibble = if elem_in_sb % 2 == 0 {
                qs_byte & 0x0f
            } else {
                qs_byte >> 4
            };
            let val = (nibble as f32 - 8.0) * scale;
            fp32_data.push(val);
        }
    }

    Ok(Tensor {
        data: fp32_data,
        shape: tensor_info.dimensions.clone(),
    })
}

/// Minimal IEEE 754 fp16 → fp32 conversion (local to avoid dependency).
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;

    let sign_bit = (sign as u32) << 31;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign_bit)
        } else {
            // subnormal
            let val = (mant as f32) * 2f32.powi(-24);
            if sign == 1 {
                -val
            } else {
                val
            }
        }
    } else if exp == 0x1f {
        // inf / nan
        let m = (mant as u32) << 13;
        f32::from_bits(sign_bit | 0x7f80_0000 | m)
    } else {
        let e = (exp as i32 + (127 - 15)) as u32;
        let m = (mant as u32) << 13;
        f32::from_bits(sign_bit | (e << 23) | m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::model::GgufTensorInfo;

    #[test]
    fn test_iq4_xs_dequant_basic() {
        // Create a minimal valid IQ4_XS block:
        // - 32 fp16 scales = 1.0 (0x3C00)
        // - 64 bytes of qs = all 0x00 (nibbles = 0 -> signed -8)
        // Expected: val = (-8) * 1.0 = -8.0 for all 128 elements
        let mut block = vec![0u8; 128];
        for i in (0..64).step_by(2) {
            block[i] = 0x00;
            block[i + 1] = 0x3c; // fp16 1.0
        }
        // qs all zero -> nibbles = 0 -> signed -8
        // scales all 1.0

        let tensor_info = GgufTensorInfo {
            name: "test".to_string(),
            dimensions: vec![128],
            ggml_type: 30,
            offset: 0,
        };

        let mmap = block;
        let result = dequantize_iq4_xs(&tensor_info, &mmap, 0).expect("dequant failed");
        assert_eq!(result.data.len(), 128);
        for &v in &result.data {
            assert!((v + 8.0).abs() < 1e-6, "got {}", v);
        }
    }

    #[test]
    fn test_iq4_xs_dequant_mixed() {
        // Scale 0 = 2.0, others = 1.0
        // qs: first byte = 0x0A (low nibble 10 -> 2, high nibble 0 -> -8)
        let mut block = vec![0u8; 128];
        for i in (0..64).step_by(2) {
            block[i] = 0x00;
            block[i + 1] = 0x3c; // 1.0
        }
        block[0] = 0x00;
        block[1] = 0x40; // 2.0 (fp16)
        block[64] = 0x0a; // qs[0] = 0x0A -> low=10(2), high=0(-8)

        let tensor_info = GgufTensorInfo {
            name: "test".to_string(),
            dimensions: vec![128],
            ggml_type: 30,
            offset: 0,
        };

        let mmap = block;
        let result = dequantize_iq4_xs(&tensor_info, &mmap, 0).expect("dequant failed");
        // Scale layout: 32 fp16 scales cover 4 elements each.
        // elem 0: scale[0]=2.0, nibble=10 -> (10-8)*2 = 4.0
        assert!((result.data[0] - 4.0).abs() < 1e-6);
        // elem 1: scale[0]=2.0, nibble=0 -> (0-8)*2 = -16.0
        assert!((result.data[1] + 16.0).abs() < 1e-6);
        // elem 2: scale[0]=2.0, nibble=0 -> (0-8)*2 = -16.0
        assert!((result.data[2] + 16.0).abs() < 1e-6);
    }
}
