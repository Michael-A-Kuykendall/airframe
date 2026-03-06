//! RMSNorm (Root Mean Square Layer Normalization).
//!
//! Matches llama.cpp `ggml_compute_forward_rms_norm_f32` exactly.

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

/// Apply RMSNorm: `output = (input / rms) * weight`.
pub fn rmsnorm_f32(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // Validate input is 1D or 2D
    if input.ndim() == 0 || input.ndim() > 2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "rmsnorm_input".to_string(),
            expected: vec![1, 2], // 1D or 2D
            got: vec![input.ndim()],
        });
    }

    // Weight must be 1D and match last dimension of input
    if weight.ndim() != 1 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "rmsnorm_weight".to_string(),
            expected: vec![1],
            got: vec![weight.ndim()],
        });
    }

    let last_dim = input.shape[input.ndim() - 1];
    if weight.shape[0] != last_dim {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "rmsnorm_weight_size".to_string(),
            expected: vec![last_dim],
            got: vec![weight.shape[0]],
        });
    }

    let mut output = input.data.clone();

    match input.ndim() {
        1 => {
            // Single vector normalization
            rmsnorm_vector(&mut output, &weight.data, eps);
        }
        2 => {
            // Batch normalization - normalize each row independently
            let batch_size = input.shape[0];
            let hidden_size = input.shape[1];

            for i in 0..batch_size {
                let start = i * hidden_size;
                let end = start + hidden_size;
                rmsnorm_vector(&mut output[start..end], &weight.data, eps);
            }
        }
        _ => unreachable!(), // Already validated above
    }

    Tensor::new(output, input.shape.clone())
}

/// Apply RMSNorm to a single vector in-place
///
/// LLAMA.CPP PARITY: Matches ggml_compute_forward_rms_norm_f32 exactly:
/// 1. Sum of squares in f32
/// 2. Mean = sum/n in f32
/// 3. Scale = 1.0f / sqrtf(mean + eps) - compute reciprocal, not RMS directly
/// 4. Multiply by scale (not divide by RMS) - different rounding behavior
fn rmsnorm_vector(data: &mut [f32], weight: &[f32], eps: f32) {
    let n = data.len();

    // Step 1: Sum of squares in f64 (matches llama.cpp double precision)
    let sum: f64 = data.iter().map(|&x| x as f64 * x as f64).sum();

    // Step 2: Mean in f64
    let mean = sum / n as f64;

    // Step 3: Scale = 1/sqrt(mean+eps) (matches: const float scale = 1.0f/sqrtf(mean + eps))
    // NOTE: This is different from computing rms then dividing!
    let scale = 1.0f32 / ((mean as f32) + eps).sqrt();

    // Step 4: Multiply by scale (matches: ggml_vec_scale_f32)
    // Then multiply by weight
    for (i, value) in data.iter_mut().enumerate() {
        *value = *value * scale * weight[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_simple_vector() {
        // Input: [1, 2, 3, 4]
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        // Weight: all ones (no scaling)
        let weight = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let eps = 1e-5;

        let output = rmsnorm_f32(&input, &weight, eps).unwrap();

        // Expected RMS = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps) ≈ 2.739
        let expected_rms = (7.5f32 + eps).sqrt();
        let expected = [
            1.0 / expected_rms,
            2.0 / expected_rms,
            3.0 / expected_rms,
            4.0 / expected_rms,
        ];

        assert_eq!(output.shape, vec![4]);
        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((output.data[i] - expected_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rmsnorm_with_scaling() {
        // Input: [2, 4]
        let input = Tensor::new(vec![2.0, 4.0], vec![2]).unwrap();
        // Weight: [0.5, 2.0] (scale factors)
        let weight = Tensor::new(vec![0.5, 2.0], vec![2]).unwrap();
        let eps = 1e-8;

        let output = rmsnorm_f32(&input, &weight, eps).unwrap();

        // RMS = sqrt((4+16)/2 + eps) = sqrt(10 + eps) ≈ 3.162
        let expected_rms = (10.0f32 + eps).sqrt();
        let expected = [
            (2.0 / expected_rms) * 0.5, // First element scaled by 0.5
            (4.0 / expected_rms) * 2.0, // Second element scaled by 2.0
        ];

        assert_eq!(output.shape, vec![2]);
        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((output.data[i] - expected_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rmsnorm_batch() {
        // Batch of 2 vectors, each of size 3
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let eps = 1e-5;

        let output = rmsnorm_f32(&input, &weight, eps).unwrap();

        assert_eq!(output.shape, vec![2, 3]);

        // First row: [1, 2, 3] -> RMS = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.16
        let rms1 = ((1.0 + 4.0 + 9.0) / 3.0 + eps).sqrt();
        assert!((output.data[0] - 1.0 / rms1).abs() < 1e-6);
        assert!((output.data[1] - 2.0 / rms1).abs() < 1e-6);
        assert!((output.data[2] - 3.0 / rms1).abs() < 1e-6);

        // Second row: [4, 5, 6] -> RMS = sqrt((16+25+36)/3) = sqrt(77/3) ≈ 5.07
        let rms2 = ((16.0 + 25.0 + 36.0) / 3.0 + eps).sqrt();
        assert!((output.data[3] - 4.0 / rms2).abs() < 1e-6);
        assert!((output.data[4] - 5.0 / rms2).abs() < 1e-6);
        assert!((output.data[5] - 6.0 / rms2).abs() < 1e-6);
    }

    #[test]
    fn test_rmsnorm_zero_vector() {
        // Edge case: all zeros
        let input = Tensor::new(vec![0.0, 0.0, 0.0], vec![3]).unwrap();
        let weight = Tensor::new(vec![1.0, 1.0, 1.0], vec![3]).unwrap();
        let eps = 1e-5;

        let output = rmsnorm_f32(&input, &weight, eps).unwrap();

        // RMS = sqrt(0 + eps) = sqrt(eps)
        // Output should be [0, 0, 0] since 0 / sqrt(eps) * 1 = 0
        assert_eq!(output.data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rmsnorm_dimension_mismatch() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let weight = Tensor::new(vec![1.0, 1.0], vec![2]).unwrap(); // Wrong size

        let result = rmsnorm_f32(&input, &weight, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_rmsnorm_known_values() {
        // Test with known values for regression testing
        let input = Tensor::new(vec![0.5, -0.5, 1.0, -1.0], vec![4]).unwrap();
        let weight = Tensor::new(vec![2.0, 2.0, 2.0, 2.0], vec![4]).unwrap();
        let eps = 1e-6;

        let output = rmsnorm_f32(&input, &weight, eps).unwrap();

        // RMS = sqrt((0.25 + 0.25 + 1.0 + 1.0)/4) = sqrt(0.625) ≈ 0.7906
        let expected_rms = 0.625f32.sqrt();

        // Each element should be (input[i] / rms) * 2.0
        let expected = [
            (0.5 / expected_rms) * 2.0,
            (-0.5 / expected_rms) * 2.0,
            (1.0 / expected_rms) * 2.0,
            (-1.0 / expected_rms) * 2.0,
        ];

        for (i, &expected_val) in expected.iter().enumerate() {
            assert!((output.data[i] - expected_val).abs() < 1e-5);
        }
    }
}
