//! Softmax with optional causal masking.
//!
//! Supports 2D/3D/4D attention tensors with llama.cpp-compatible
//! numerical precision (f64 accumulator, multiply by reciprocal).

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

/// Softmax with optional causal mask.
///
/// # Shapes
/// - 2D: `[query_len, key_len]`
/// - 3D: `[n_head, query_len, key_len]`
/// - 4D: `[batch, n_head, query_len, key_len]`
pub fn softmax_f32(input: &Tensor, causal_mask: bool) -> Result<Tensor> {
    let mut output = input.data.clone();

    match input.ndim() {
        2 => {
            // Single attention matrix [query_len, key_len]
            let query_len = input.shape[0];
            let key_len = input.shape[1];

            // For causal attention, query_len <= key_len
            // During decode: query_len=1, key_len=full_seq_len
            // During prefill: query_len=key_len=prompt_len

            softmax_2d_non_square(&mut output, query_len, key_len, causal_mask);
        }
        3 => {
            // Multi-head attention [n_head, query_len, key_len]
            let n_head = input.shape[0];
            let query_len = input.shape[1];
            let key_len = input.shape[2];

            let matrix_size = query_len * key_len;
            for h in 0..n_head {
                let head_offset = h * matrix_size;
                softmax_2d_non_square(
                    &mut output[head_offset..head_offset + matrix_size],
                    query_len,
                    key_len,
                    causal_mask,
                );
            }
        }
        4 => {
            // Batched multi-head attention [batch, n_head, query_len, key_len]
            let batch_size = input.shape[0];
            let n_head = input.shape[1];
            let query_len = input.shape[2];
            let key_len = input.shape[3];

            let matrix_size = query_len * key_len;
            for b in 0..batch_size {
                for h in 0..n_head {
                    let offset = (b * n_head + h) * matrix_size;
                    softmax_2d_non_square(
                        &mut output[offset..offset + matrix_size],
                        query_len,
                        key_len,
                        causal_mask,
                    );
                }
            }
        }
        _ => {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "softmax_input".to_string(),
                expected: vec![2, 3, 4], // 2D, 3D, or 4D
                got: vec![input.ndim()],
            });
        }
    }

    Tensor::new(output, input.shape.clone())
}

/// Apply softmax to a 2D attention matrix in-place (supports non-square)
///
/// For causal masking with KV cache:
/// - query_pos is the absolute position of each query token in the sequence
/// - For decode: query_len=1, the single query is at position (key_len - 1)
/// - For prefill: query_len=key_len, query position i is at absolute position i
///
/// With KV cache, the attention pattern is:
/// - Query at absolute position q can attend to key positions 0..=q
/// - For decode (query_len=1): the query is at position (key_len-1), can attend to all keys
fn softmax_2d_non_square(data: &mut [f32], query_len: usize, key_len: usize, causal_mask: bool) {
    for q in 0..query_len {
        let row_start = q * key_len;
        let row_end = row_start + key_len;
        let row = &mut data[row_start..row_end];

        // Apply causal mask: each query can only attend to positions up to its own position
        // For KV cache: query at local index q corresponds to absolute position (key_len - query_len + q)
        // This is because the cached keys are at positions 0..(key_len-query_len) and
        // new queries are at positions (key_len-query_len)..(key_len)
        if causal_mask {
            let query_abs_pos = key_len - query_len + q;
            for elem in row.iter_mut().skip(query_abs_pos + 1) {
                *elem = f32::NEG_INFINITY;
            }
        }

        // Find max for numerical stability
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) and sum
        // LLAMA.CPP PARITY: Use f64 accumulator (ggml_float)
        let mut sum: f64 = 0.0;
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val as f64;
        }

        // Normalize by sum
        // LLAMA.CPP PARITY: Compute reciprocal in f64, cast to f32, then MULTIPLY
        // This matches: sum = 1.0/sum; ggml_vec_scale_f32(ne00, dp, sum);
        // Using multiplication instead of division has different rounding behavior
        if sum > 0.0 {
            let inv_sum = (1.0 / sum) as f32; // reciprocal in f64, then cast to f32
            for val in row.iter_mut() {
                *val *= inv_sum; // multiply, not divide
            }
        }
    }
}

/// Create a causal mask tensor for attention
/// Returns a boolean tensor where true indicates positions to mask
pub fn create_causal_mask(seq_len: usize) -> Vec<Vec<bool>> {
    let mut mask = vec![vec![false; seq_len]; seq_len];
    for (i, row) in mask.iter_mut().enumerate() {
        for elem in row.iter_mut().skip(i + 1) {
            *elem = true; // Mask future positions
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_no_mask() {
        // Simple 2x2 matrix without masking
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let output = softmax_f32(&input, false).unwrap();

        assert_eq!(output.shape, vec![2, 2]);

        // Check row sums are ~1
        let row1_sum = output.data[0] + output.data[1];
        let row2_sum = output.data[2] + output.data[3];
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);

        // Check values are positive
        for &val in &output.data {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_softmax_causal_mask() {
        // 3x3 matrix with causal masking
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let output = softmax_f32(&input, true).unwrap();

        assert_eq!(output.shape, vec![3, 3]);

        // Check causal mask: upper triangular should be ~0
        assert!((output.data[1] - 0.0).abs() < 1e-6); // [0,1]
        assert!((output.data[2] - 0.0).abs() < 1e-6); // [0,2]
        assert!((output.data[5] - 0.0).abs() < 1e-6); // [1,2]

        // Check row sums are ~1
        for i in 0..3 {
            let row_sum: f32 = (0..3).map(|j| output.data[i * 3 + j]).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }

        // Check diagonal and lower triangle are positive
        assert!(output.data[0] > 0.0); // [0,0]
        assert!(output.data[3] > 0.0); // [1,0]
        assert!(output.data[4] > 0.0); // [1,1]
        assert!(output.data[6] > 0.0); // [2,0]
        assert!(output.data[7] > 0.0); // [2,1]
        assert!(output.data[8] > 0.0); // [2,2]
    }

    #[test]
    fn test_softmax_3d_multihead() {
        // 2 heads, 2x2 each
        let input =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]).unwrap();
        let output = softmax_f32(&input, false).unwrap();

        assert_eq!(output.shape, vec![2, 2, 2]);

        // Check each head's rows sum to 1
        for h in 0..2 {
            for i in 0..2 {
                let row_start = h * 4 + i * 2;
                let row_sum = output.data[row_start] + output.data[row_start + 1];
                assert!((row_sum - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_softmax_4d_batch() {
        // 1 batch, 1 head, 2x2
        let input = Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![1, 1, 2, 2]).unwrap();
        let output = softmax_f32(&input, true).unwrap();

        assert_eq!(output.shape, vec![1, 1, 2, 2]);

        // Check causal masking
        assert!((output.data[1] - 0.0).abs() < 1e-6); // [0,1] masked
        assert!(output.data[0] > 0.0); // [0,0] not masked
        assert!(output.data[2] > 0.0); // [1,0] not masked
        assert!(output.data[3] > 0.0); // [1,1] not masked

        // Check row sums
        let row1_sum = output.data[0] + output.data[1];
        let row2_sum = output.data[2] + output.data[3];
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that could cause overflow without max subtraction
        let input = Tensor::new(vec![100.0, 101.0, 99.0, 102.0], vec![2, 2]).unwrap();
        let output = softmax_f32(&input, false).unwrap();

        // Should not contain NaN or inf
        for &val in &output.data {
            assert!(val.is_finite());
            assert!(val >= 0.0);
        }

        // Row sums should still be 1
        let row1_sum = output.data[0] + output.data[1];
        let row2_sum = output.data[2] + output.data[3];
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_create_causal_mask() {
        let mask = create_causal_mask(3);

        // Check shape
        assert_eq!(mask.len(), 3);
        assert_eq!(mask[0].len(), 3);

        // Check pattern: upper triangular should be true
        assert!(!mask[0][0]); // diagonal
        assert!(mask[0][1]); // future
        assert!(mask[0][2]); // future
        assert!(!mask[1][0]); // past
        assert!(!mask[1][1]); // diagonal
        assert!(mask[1][2]); // future
        assert!(!mask[2][0]); // past
        assert!(!mask[2][1]); // past
        assert!(!mask[2][2]); // diagonal
    }

    #[test]
    fn test_softmax_1d_fails() {
        // 1D tensor should fail
        let input = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap();
        let result = softmax_f32(&input, false);
        assert!(result.is_err());
    }
}
