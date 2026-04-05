//! Rotary Position Embedding (RoPE).
//!
//! Standard Llama RoPE with configurable base frequency and rope_dim.

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

/// Apply RoPE to query/key tensors.
///
/// Input: `[seq_len, n_head, head_dim]` or `[batch, seq_len, n_head, head_dim]`
pub fn apply_rope_f32(
    tensor: &Tensor,
    position_ids: &[usize],
    rope_base: f32,
    rope_dim: usize,
) -> Result<Tensor> {
    apply_rope_scaled_f32(tensor, position_ids, rope_base, rope_dim, 1.0)
}

/// Apply RoPE with linear position scaling.
///
/// `rope_scale=1.0` is standard RoPE. Values below 1.0 stretch the usable
/// context by compressing the effective angles.
pub fn apply_rope_scaled_f32(
    tensor: &Tensor,
    position_ids: &[usize],
    rope_base: f32,
    rope_dim: usize,
    rope_scale: f32,
) -> Result<Tensor> {
    if rope_base <= 0.0 {
        return Err(LibshimmyError::Unsupported(
            "RoPE base must be positive".to_string(),
        ));
    }

    if rope_scale <= 0.0 {
        return Err(LibshimmyError::Unsupported(format!(
            "RoPE scale must be positive (got {rope_scale})"
        )));
    }

    crate::ensure!(rope_dim > 0, "rope_dim must be > 0");
    crate::ensure!(rope_dim % 2 == 0, "rope_dim must be even (got {rope_dim})");

    let mut output = tensor.data.clone();

    match tensor.ndim() {
        3 => {
            // Shape: [seq_len, n_head, head_dim]
            let seq_len = tensor.shape[0];
            let n_head = tensor.shape[1];
            let head_dim = tensor.shape[2];

            crate::ensure!(
                rope_dim <= head_dim,
                "rope_dim ({rope_dim}) must be <= head_dim ({head_dim})"
            );

            if position_ids.len() != seq_len {
                return Err(LibshimmyError::ShapeMismatch {
                    tensor: "rope_position_ids".to_string(),
                    expected: vec![seq_len],
                    got: vec![position_ids.len()],
                });
            }

            apply_rope_3d(
                &mut output,
                seq_len,
                n_head,
                head_dim,
                position_ids,
                rope_base,
                rope_dim,
                rope_scale,
            )?;
        }
        4 => {
            // Shape: [batch, seq_len, n_head, head_dim]
            let batch_size = tensor.shape[0];
            let seq_len = tensor.shape[1];
            let n_head = tensor.shape[2];
            let head_dim = tensor.shape[3];

            crate::ensure!(
                rope_dim <= head_dim,
                "rope_dim ({rope_dim}) must be <= head_dim ({head_dim})"
            );

            if position_ids.len() != seq_len {
                return Err(LibshimmyError::ShapeMismatch {
                    tensor: "rope_position_ids".to_string(),
                    expected: vec![seq_len],
                    got: vec![position_ids.len()],
                });
            }

            for b in 0..batch_size {
                let batch_offset = b * seq_len * n_head * head_dim;
                apply_rope_3d(
                    &mut output[batch_offset..batch_offset + seq_len * n_head * head_dim],
                    seq_len,
                    n_head,
                    head_dim,
                    position_ids,
                    rope_base,
                    rope_dim,
                    rope_scale,
                )?;
            }
        }
        _ => {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "rope_input".to_string(),
                expected: vec![3, 4], // 3D or 4D
                got: vec![tensor.ndim()],
            });
        }
    }

    Tensor::new(output, tensor.shape.clone())
}

/// Apply RoPE to 3D tensor data in-place
fn apply_rope_3d(
    data: &mut [f32],
    _seq_len: usize,
    n_head: usize,
    head_dim: usize,
    position_ids: &[usize],
    rope_base: f32,
    rope_dim: usize,
    rope_scale: f32,
) -> Result<()> {
    crate::ensure!(
        rope_dim <= head_dim,
        "rope_dim ({rope_dim}) must be <= head_dim ({head_dim})"
    );

    let actual_rope_dim = rope_dim;

    // Precompute frequency coefficients
    let mut freqs = Vec::with_capacity(actual_rope_dim / 2);
    for i in 0..(actual_rope_dim / 2) {
        let freq = 1.0 / rope_base.powf(2.0 * i as f32 / actual_rope_dim as f32);
        freqs.push(freq);
    }

    for (seq_idx, &pos) in position_ids.iter().enumerate() {
        for head in 0..n_head {
            let head_offset = seq_idx * n_head * head_dim + head * head_dim;

            // Apply rotation to pairs of dimensions
            for (i, &freq) in freqs.iter().enumerate() {
                let idx1 = head_offset + 2 * i;
                let idx2 = head_offset + 2 * i + 1;

                if idx2 < data.len() {
                    let angle = pos as f32 * rope_scale * freq;
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();

                    let x = data[idx1];
                    let y = data[idx2];

                    // Rotation matrix: [cos -sin; sin cos]
                    data[idx1] = x * cos_val - y * sin_val;
                    data[idx2] = x * sin_val + y * cos_val;
                }
            }
        }
    }

    Ok(())
}

/// Create standard position IDs for a sequence
pub fn create_position_ids(seq_len: usize, start_pos: usize) -> Vec<usize> {
    (start_pos..start_pos + seq_len).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::LibshimmyError;

    #[test]
    fn test_rope_simple_rotation() {
        // Simple 2D rotation test: single head, 2 dimensions
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 1, 2]).unwrap();
        let position_ids = vec![0];

        let output = apply_rope_f32(&input, &position_ids, 10000.0, 2).unwrap();

        // At position 0, rotation should be identity (angle = 0)
        assert_eq!(output.shape, vec![1, 1, 2]);
        assert!((output.data[0] - 1.0).abs() < 1e-6);
        assert!((output.data[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rope_90_degree_rotation() {
        // Test rotation at position where we get ~90 degrees
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 1, 2]).unwrap();

        // Choose position and base to get π/2 rotation
        // freq = 1/base^0 = 1, so angle = pos * 1
        // For π/2, we need pos = π/2 ≈ 1.57
        let position_ids = vec![1]; // This will give angle = 1 radian ≈ 57.3°

        let output = apply_rope_f32(&input, &position_ids, 1.0, 2).unwrap();

        // Expected: [cos(1), sin(1)] ≈ [0.540, 0.841]
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();

        assert!((output.data[0] - expected_cos).abs() < 1e-6);
        assert!((output.data[1] - expected_sin).abs() < 1e-6);
    }

    #[test]
    fn test_rope_multiple_heads() {
        // Test with multiple heads
        let input = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            vec![1, 2, 4], // 1 seq, 2 heads, 4 dims each
        )
        .unwrap();
        let position_ids = vec![0];

        let output = apply_rope_f32(&input, &position_ids, 10000.0, 4).unwrap();

        // At position 0, should be unchanged
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn test_rope_sequence() {
        // Test with sequence of positions
        let input = Tensor::new(
            vec![1.0, 0.0, 1.0, 0.0],
            vec![2, 1, 2], // 2 seq, 1 head, 2 dims
        )
        .unwrap();
        let position_ids = vec![0, 1];

        let output = apply_rope_f32(&input, &position_ids, 10000.0, 2).unwrap();

        // First position (0) should be unchanged
        assert!((output.data[0] - 1.0).abs() < 1e-6);
        assert!((output.data[1] - 0.0).abs() < 1e-6);

        // Second position should be rotated
        // freq = 1/10000^0 = 1, angle = 1 * 1 = 1
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!((output.data[2] - expected_cos).abs() < 1e-6);
        assert!((output.data[3] - expected_sin).abs() < 1e-6);
    }

    #[test]
    fn test_rope_partial_dimensions() {
        // Test when rope_dim < head_dim (only rotate first rope_dim dimensions)
        let input = Tensor::new(
            vec![1.0, 0.0, 5.0, 7.0],
            vec![1, 1, 4], // 1 seq, 1 head, 4 dims
        )
        .unwrap();
        let position_ids = vec![1];

        // Only rotate first 2 dimensions
        let output = apply_rope_f32(&input, &position_ids, 1.0, 2).unwrap();

        // First 2 dims rotated
        let expected_cos = 1.0f32.cos();
        let expected_sin = 1.0f32.sin();
        assert!((output.data[0] - expected_cos).abs() < 1e-6);
        assert!((output.data[1] - expected_sin).abs() < 1e-6);

        // Last 2 dims unchanged
        assert_eq!(output.data[2], 5.0);
        assert_eq!(output.data[3], 7.0);
    }

    #[test]
    fn test_rope_batch() {
        // Test 4D tensor (batched)
        let input = Tensor::new(
            vec![1.0, 0.0, 2.0, 0.0],
            vec![2, 1, 1, 2], // 2 batch, 1 seq, 1 head, 2 dims
        )
        .unwrap();
        let position_ids = vec![0];

        let output = apply_rope_f32(&input, &position_ids, 10000.0, 2).unwrap();

        // Both batches should be unchanged at position 0
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn test_rope_dim_exceeds_head_dim_fails_closed() {
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 1, 2]).unwrap();
        let position_ids = vec![0];

        let err = apply_rope_f32(&input, &position_ids, 10000.0, 4).unwrap_err();
        match err {
            LibshimmyError::InvariantViolation { .. } => {}
            other => panic!("expected InvariantViolation, got {other:?}"),
        }
    }

    #[test]
    fn test_rope_dim_must_be_even() {
        let input = Tensor::new(vec![1.0, 0.0, 2.0], vec![1, 1, 3]).unwrap();
        let position_ids = vec![0];

        let err = apply_rope_f32(&input, &position_ids, 10000.0, 3).unwrap_err();
        match err {
            LibshimmyError::InvariantViolation { .. } => {}
            other => panic!("expected InvariantViolation, got {other:?}"),
        }
    }

    #[test]
    fn test_create_position_ids() {
        assert_eq!(create_position_ids(3, 0), vec![0, 1, 2]);
        assert_eq!(create_position_ids(2, 5), vec![5, 6]);
        assert_eq!(create_position_ids(0, 10), Vec::<usize>::new());
    }

    #[test]
    fn test_rope_dimension_mismatch() {
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 1, 2]).unwrap();
        let position_ids = vec![0, 1]; // Wrong length

        let result = apply_rope_f32(&input, &position_ids, 10000.0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_invalid_base() {
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 1, 2]).unwrap();
        let position_ids = vec![0];

        let result = apply_rope_f32(&input, &position_ids, -1.0, 2);
        assert!(result.is_err());
    }
}
