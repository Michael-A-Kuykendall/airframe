//! Matrix multiplication kernels.
//!
//! Two variants: `matmul_f32` for GGML column-major weights,
//! `matmul_row_major_f32` for row-major activations (e.g., Q @ K^T).

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

/// C = A @ B where B is GGML column-major.
///
/// A: `[M, K]` row-major, B: `[K, N]` GGML layout -> C: `[M, N]`
pub fn matmul_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Validate dimensions
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "matmul".to_string(),
            expected: vec![2, 2], // Both should be 2D
            got: vec![a.ndim(), b.ndim()],
        });
    }

    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];

    if k != k2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "matmul_inner_dim".to_string(),
            expected: vec![k],
            got: vec![k2],
        });
    }

    // Allocate result
    let mut result = vec![0.0; m * n];

    // Naive O(n³) implementation - reference correctness over performance
    // A is row-major: A[i, k] = a.data[i * K + k]
    // B is GGML column-major: B[k, j] = b.data[j * K + k]
    // NOTE: Use f64 accumulator to match llama.cpp's ggml_float (double) behavior
    for i in 0..m {
        for j in 0..n {
            let mut sum: f64 = 0.0;
            for k_idx in 0..k {
                let a_val = a.data[i * k + k_idx];
                // GGML column-major: B[k_idx, j] = b.data[j * k + k_idx]
                let b_val = b.data[j * k + k_idx];
                sum += a_val as f64 * b_val as f64;
            }
            result[i * n + j] = sum as f32;
        }
    }

    Tensor::new(result, vec![m, n])
}

/// C = A @ B where both are row-major.
///
/// A: `[M, K]`, B: `[K, N]` -> C: `[M, N]`
pub fn matmul_row_major_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Validate dimensions
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "matmul_row_major".to_string(),
            expected: vec![2, 2], // Both should be 2D
            got: vec![a.ndim(), b.ndim()],
        });
    }

    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];

    if k != k2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "matmul_row_major_inner_dim".to_string(),
            expected: vec![k],
            got: vec![k2],
        });
    }

    // Allocate result
    let mut result = vec![0.0; m * n];

    // Standard row-major matmul
    // A[i, k] = a.data[i * K + k]
    // B[k, j] = b.data[k * N + j]
    // NOTE: Use f64 accumulator to match llama.cpp's ggml_float (double) behavior
    for i in 0..m {
        for j in 0..n {
            let mut sum: f64 = 0.0;
            for k_idx in 0..k {
                let a_val = a.data[i * k + k_idx];
                // Standard row-major: B[k_idx, j] = b.data[k_idx * N + j]
                let b_val = b.data[k_idx * n + j];
                sum += (a_val * b_val) as f64;
            }
            result[i * n + j] = sum as f32;
        }
    }

    Tensor::new(result, vec![m, n])
}

/// Reference matrix-vector multiplication: y = A @ x
/// A: [M, N], x: [N] -> y: [M]
pub fn matvec_f32(a: &Tensor, x: &Tensor) -> Result<Tensor> {
    // Validate dimensions
    if a.ndim() != 2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "matvec_matrix".to_string(),
            expected: vec![2],
            got: vec![a.ndim()],
        });
    }

    if x.ndim() != 1 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "matvec_vector".to_string(),
            expected: vec![1],
            got: vec![x.ndim()],
        });
    }

    let m = a.shape[0];
    let n = a.shape[1];
    let n2 = x.shape[0];

    if n != n2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "matvec_inner_dim".to_string(),
            expected: vec![n],
            got: vec![n2],
        });
    }

    // Allocate result
    let mut result = vec![0.0f32; m];

    // Matrix-vector multiplication
    // NOTE: Use f64 accumulator to match llama.cpp's ggml_float (double) behavior
    for (i, result_elem) in result.iter_mut().enumerate() {
        let mut sum: f64 = 0.0;
        for j in 0..n {
            sum += (a.data[i * n + j] * x.data[j]) as f64;
        }
        *result_elem = sum as f32;
    }

    Tensor::new(result, vec![m])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_row_major_2x2() {
        // A = [[1, 2], [3, 4]] in row-major
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        // B = [[5, 6], [7, 8]] in row-major
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        let c = matmul_row_major_f32(&a, &b).unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_ggml_weight() {
        // This test verifies GGML column-major weight convention
        // A = [[1, 2], [3, 4]] in row-major (activation)
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        // W = [[5, 7], [6, 8]] in row-major storage
        // GGML interprets this as column-major: col0=[5,6], col1=[7,8]
        // So logical W is [[5,7],[6,8]]
        let w = Tensor::new(vec![5.0, 7.0, 6.0, 8.0], vec![2, 2]).unwrap();

        let c = matmul_f32(&a, &w).unwrap();

        // With GGML indexing: W[k,j] = data[j*K + k]
        // W[0,0]=5, W[1,0]=7, W[0,1]=6, W[1,1]=8
        // C[0,0] = A[0,0]*W[0,0] + A[0,1]*W[1,0] = 1*5 + 2*7 = 19
        // C[0,1] = A[0,0]*W[0,1] + A[0,1]*W[1,1] = 1*6 + 2*8 = 22
        // C[1,0] = A[1,0]*W[0,0] + A[1,1]*W[1,0] = 3*5 + 4*7 = 43
        // C[1,1] = A[1,0]*W[0,1] + A[1,1]*W[1,1] = 3*6 + 4*8 = 50
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_row_major_3x2_2x4() {
        // A: 3x2, B: 2x4 -> C: 3x4
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]).unwrap();

        let c = matmul_row_major_f32(&a, &b).unwrap();

        assert_eq!(c.shape, vec![3, 4]);
        assert_eq!(c.len(), 12);

        // Verify a few key values
        assert_eq!(c.get_nd(&[0, 0]).unwrap(), 11.0); // 1*1 + 2*5 = 11
                                                      // Row 2: [5, 6] @ Col 3: [4, 8] = 5*4 + 6*8 = 20 + 48 = 68
        assert_eq!(c.get_nd(&[2, 3]).unwrap(), 68.0);
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![3, 1]).unwrap(); // Wrong inner dimension

        let result = matmul_f32(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_matvec_basic() {
        // A = [[1, 2, 3], [4, 5, 6]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // x = [1, 2, 3]
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();

        let y = matvec_f32(&a, &x).unwrap();

        // Expected: [14, 32] (1*1+2*2+3*3=14, 4*1+5*2+6*3=32)
        assert_eq!(y.shape, vec![2]);
        assert_eq!(y.data, vec![14.0, 32.0]);
    }

    #[test]
    fn test_matvec_dimension_mismatch() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let x = Tensor::new(vec![3.0, 4.0, 5.0], vec![3]).unwrap(); // Wrong dimension

        let result = matvec_f32(&a, &x);
        assert!(result.is_err());
    }

    #[test]
    fn test_identity_matrix() {
        // Identity matrix
        let identity = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let x = Tensor::new(vec![3.0, 4.0], vec![2]).unwrap();

        let result = matvec_f32(&identity, &x).unwrap();
        assert_eq!(result.data, vec![3.0, 4.0]); // Should be unchanged
    }
}
