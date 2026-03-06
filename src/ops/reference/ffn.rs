//! SwiGLU feed-forward network.
//!
//! `FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down`

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};
use crate::ops::reference::{activations, matmul};

/// SwiGLU FFN forward pass.
pub fn ffn_swiglu_f32(
    input: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
) -> Result<Tensor> {
    match input.ndim() {
        2 => {
            // Single sequence: [seq_len, hidden_size]
            ffn_swiglu_2d(input, gate_weight, up_weight, down_weight)
        }
        3 => {
            // Batched: [batch, seq_len, hidden_size]
            ffn_swiglu_3d(input, gate_weight, up_weight, down_weight)
        }
        _ => {
            Err(LibshimmyError::ShapeMismatch {
                tensor: "ffn_input".to_string(),
                expected: vec![2, 3], // 2D or 3D
                got: vec![input.ndim()],
            })
        }
    }
}

/// FFN for 2D input [seq_len, hidden_size]
fn ffn_swiglu_2d(
    input: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
) -> Result<Tensor> {
    let _seq_len = input.shape[0];
    let hidden_size = input.shape[1];

    // Validate weight shapes
    validate_ffn_weights(gate_weight, up_weight, down_weight, hidden_size)?;

    // 1. Gate projection: x @ W_gate -> [seq_len, ff_dim]
    let gate_proj = matmul::matmul_f32(input, gate_weight)?;

    // 2. Up projection: x @ W_up -> [seq_len, ff_dim]
    let up_proj = matmul::matmul_f32(input, up_weight)?;

    // 3. Apply SiLU to gate projection
    let gate_activated = activations::silu_f32(&gate_proj)?;

    // 4. Element-wise multiply: SiLU(gate) ⊙ up -> [seq_len, ff_dim]
    let gated = activations::multiply_f32(&gate_activated, &up_proj)?;

    // 5. Down projection: gated @ W_down -> [seq_len, hidden_size]
    matmul::matmul_f32(&gated, down_weight)
}

/// FFN for 3D input [batch, seq_len, hidden_size]
fn ffn_swiglu_3d(
    input: &Tensor,
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
) -> Result<Tensor> {
    let batch_size = input.shape[0];
    let seq_len = input.shape[1];
    let hidden_size = input.shape[2];

    let mut batch_outputs = Vec::new();

    for b in 0..batch_size {
        // Extract batch: [seq_len, hidden_size]
        let batch_input = extract_batch(input, b, seq_len, hidden_size)?;

        // Process single batch
        let batch_output = ffn_swiglu_2d(&batch_input, gate_weight, up_weight, down_weight)?;

        batch_outputs.push(batch_output);
    }

    // Concatenate batches
    concatenate_batches(&batch_outputs, batch_size, seq_len, hidden_size)
}

/// Validate FFN weight shapes
fn validate_ffn_weights(
    gate_weight: &Tensor,
    up_weight: &Tensor,
    down_weight: &Tensor,
    hidden_size: usize,
) -> Result<()> {
    // Gate and up weights should have same shape: [hidden_size, ff_dim]
    if gate_weight.ndim() != 2 || gate_weight.shape[0] != hidden_size {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "gate_weight".to_string(),
            expected: vec![hidden_size, gate_weight.shape.get(1).copied().unwrap_or(0)],
            got: gate_weight.shape.clone(),
        });
    }

    let ff_dim = gate_weight.shape[1];

    if up_weight.shape != gate_weight.shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "up_weight".to_string(),
            expected: gate_weight.shape.clone(),
            got: up_weight.shape.clone(),
        });
    }

    // Down weight should be [ff_dim, hidden_size]
    let expected_down_shape = vec![ff_dim, hidden_size];
    if down_weight.shape != expected_down_shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "down_weight".to_string(),
            expected: expected_down_shape,
            got: down_weight.shape.clone(),
        });
    }

    Ok(())
}

/// Extract batch from 3D tensor
fn extract_batch(
    tensor: &Tensor,
    batch_idx: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Result<Tensor> {
    let start = batch_idx * seq_len * hidden_size;
    let end = start + seq_len * hidden_size;
    let batch_data = tensor.data[start..end].to_vec();
    Tensor::new(batch_data, vec![seq_len, hidden_size])
}

/// Concatenate batches into 3D tensor
fn concatenate_batches(
    batches: &[Tensor],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
) -> Result<Tensor> {
    let mut concat_data = Vec::with_capacity(batch_size * seq_len * hidden_size);

    for batch in batches {
        concat_data.extend_from_slice(&batch.data);
    }

    Tensor::new(concat_data, vec![batch_size, seq_len, hidden_size])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_shape_invariants() {
        let seq_len = 2;
        let hidden_size = 4;
        let ff_dim = 6;

        let input =
            Tensor::new(vec![1.0; seq_len * hidden_size], vec![seq_len, hidden_size]).unwrap();
        let gate_weight =
            Tensor::new(vec![0.1; hidden_size * ff_dim], vec![hidden_size, ff_dim]).unwrap();
        let up_weight =
            Tensor::new(vec![0.2; hidden_size * ff_dim], vec![hidden_size, ff_dim]).unwrap();
        let down_weight =
            Tensor::new(vec![0.3; ff_dim * hidden_size], vec![ff_dim, hidden_size]).unwrap();

        let output = ffn_swiglu_f32(&input, &gate_weight, &up_weight, &down_weight).unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape, vec![seq_len, hidden_size]);

        // Should not contain NaN
        for &val in &output.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ffn_known_values() {
        // Simple test with known values
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();

        // Identity-like weights for predictable output
        let gate_weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let up_weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let down_weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let output = ffn_swiglu_f32(&input, &gate_weight, &up_weight, &down_weight).unwrap();

        assert_eq!(output.shape, vec![1, 2]);

        // With input [1, 0]:
        // gate_proj = [1, 0] @ I = [1, 0]
        // up_proj = [1, 0] @ I = [1, 0]
        // gate_activated = SiLU([1, 0]) = [SiLU(1), SiLU(0)] = [~0.731, 0]
        // gated = [~0.731, 0] ⊙ [1, 0] = [~0.731, 0]
        // output = [~0.731, 0] @ I = [~0.731, 0]

        let expected_silu_1 = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((output.data[0] - expected_silu_1).abs() < 1e-6);
        assert!((output.data[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_ffn_batched() {
        let batch_size = 2;
        let seq_len = 1;
        let hidden_size = 2;
        let ff_dim = 2;

        let input = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![batch_size, seq_len, hidden_size],
        )
        .unwrap();
        let gate_weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![hidden_size, ff_dim]).unwrap();
        let up_weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![hidden_size, ff_dim]).unwrap();
        let down_weight = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![ff_dim, hidden_size]).unwrap();

        let output = ffn_swiglu_f32(&input, &gate_weight, &up_weight, &down_weight).unwrap();

        assert_eq!(output.shape, vec![batch_size, seq_len, hidden_size]);

        // Should not contain NaN
        for &val in &output.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_ffn_weight_validation() {
        let input = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let gate_weight = Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap(); // Correct
        let up_weight = Tensor::new(vec![1.0, 0.0, 0.0], vec![3, 1]).unwrap(); // Wrong shape
        let down_weight = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();

        let result = ffn_swiglu_f32(&input, &gate_weight, &up_weight, &down_weight);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffn_deterministic() {
        let input = Tensor::new(vec![0.5, -0.5], vec![1, 2]).unwrap();
        let gate_weight = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]).unwrap();
        let up_weight = Tensor::new(vec![0.5, 0.6, 0.7, 0.8], vec![2, 2]).unwrap();
        let down_weight = Tensor::new(vec![0.9, 1.0, 1.1, 1.2], vec![2, 2]).unwrap();

        let output1 = ffn_swiglu_f32(&input, &gate_weight, &up_weight, &down_weight).unwrap();
        let output2 = ffn_swiglu_f32(&input, &gate_weight, &up_weight, &down_weight).unwrap();

        // Should be deterministic
        assert_eq!(output1.data, output2.data);
    }

    #[test]
    fn test_ffn_invalid_input_dims() {
        let input = Tensor::new(vec![1.0, 2.0], vec![2]).unwrap(); // 1D input
        let gate_weight = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let up_weight = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let down_weight = Tensor::new(vec![1.0], vec![1, 1]).unwrap();

        let result = ffn_swiglu_f32(&input, &gate_weight, &up_weight, &down_weight);
        assert!(result.is_err());
    }
}
