//! Activation functions and element-wise operations.

use crate::core::{error::Result, tensor::Tensor};

/// SiLU (Swish): `x / (1 + exp(-x))`.
pub fn silu_f32(input: &Tensor) -> Result<Tensor> {
    let output_data: Vec<f32> = input.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    Tensor::new(output_data, input.shape.clone())
}

/// Element-wise multiplication of two tensors
pub fn multiply_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.validate_shape_eq(b)?;

    let output_data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x * y)
        .collect();

    Tensor::new(output_data, a.shape.clone())
}

/// Element-wise addition of two tensors (for residual connections)
pub fn add_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.validate_shape_eq(b)?;

    let output_data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&x, &y)| x + y)
        .collect();

    Tensor::new(output_data, a.shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_known_values() {
        // Test SiLU with known values
        let input = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
        let output = silu_f32(&input).unwrap();

        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!((output.data[0] - 0.0).abs() < 1e-6);

        // SiLU(1) = 1 * sigmoid(1) = 1 * (1/(1+e^-1)) ≈ 0.731
        let expected_1 = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((output.data[1] - expected_1).abs() < 1e-6);

        // SiLU(-1) = -1 * sigmoid(-1) = -1 * (1/(1+e^1)) ≈ -0.269
        let expected_neg1 = -1.0 / (1.0 + 1.0f32.exp());
        assert!((output.data[2] - expected_neg1).abs() < 1e-6);

        // SiLU(2) = 2 * sigmoid(2) ≈ 1.761
        let expected_2 = 2.0 / (1.0 + (-2.0f32).exp());
        assert!((output.data[3] - expected_2).abs() < 1e-6);
    }

    #[test]
    fn test_silu_shape_preservation() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = silu_f32(&input).unwrap();

        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_multiply_basic() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).unwrap();

        let result = multiply_f32(&a, &b).unwrap();

        assert_eq!(result.data, vec![2.0, 6.0, 12.0]);
        assert_eq!(result.shape, vec![3]);
    }

    #[test]
    fn test_multiply_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        let result = multiply_f32(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiply_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 1.0, 0.5, 2.0], vec![2, 2]).unwrap();

        let result = multiply_f32(&a, &b).unwrap();

        assert_eq!(result.data, vec![2.0, 2.0, 1.5, 8.0]);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[test]
    fn test_add_basic() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let result = add_f32(&a, &b).unwrap();

        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
        assert_eq!(result.shape, vec![3]);
    }

    #[test]
    fn test_add_shape_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        let result = add_f32(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2]).unwrap();

        let result = add_f32(&a, &b).unwrap();

        assert_eq!(result.data, vec![1.5, 3.5, 5.5, 7.5]);
        assert_eq!(result.shape, vec![2, 2]);
    }
}
