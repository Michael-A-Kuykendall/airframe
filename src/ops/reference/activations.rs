//! Activation functions and element-wise operations.

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

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

/// Element-wise addition with NumPy-style broadcasting.
///
/// Supports the ViT use-case: `[1, N, D] + [N, D] → [N, D]` and the
/// symmetric `[N, D] + [1, N, D] → [N, D]`.  Both tensors must have the
/// same number of elements once leading size-1 dimensions are squeezed.
pub fn add_broadcast_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Squeeze leading size-1 dims from both sides then fall through to add_f32.
    let a_sq = squeeze_leading_ones(a);
    let b_sq = squeeze_leading_ones(b);
    a_sq.validate_shape_eq(&b_sq)?;

    let output_data: Vec<f32> = a_sq
        .data
        .iter()
        .zip(b_sq.data.iter())
        .map(|(&x, &y)| x + y)
        .collect();

    Tensor::new(output_data, a_sq.shape.clone())
}

/// Add a 1-D bias to every row of a 2-D tensor.
///
/// `input`: `[rows, d]`, `bias`: `[d]` → returns `[rows, d]`.
/// Also handles the degenerate 1-D case `[d] + [d]`.
pub fn add_bias_f32(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let d = *input.shape.last().unwrap();
    if bias.shape != vec![d] {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "add_bias".to_string(),
            expected: vec![d],
            got: bias.shape.clone(),
        });
    }
    let mut out = input.data.clone();
    for row_start in (0..out.len()).step_by(d) {
        for j in 0..d {
            out[row_start + j] += bias.data[j];
        }
    }
    Tensor::new(out, input.shape.clone())
}

/// Layer Normalization: `(x - mean) / sqrt(var + eps) * weight + bias`.
///
/// Normalizes over the last dimension.  Supports 1-D `[D]` and 2-D `[N, D]`
/// inputs; bias is optional (pass `None` to skip the shift).
pub fn layernorm_f32(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let ndim = input.ndim();
    if ndim == 0 || ndim > 2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "layernorm_input".to_string(),
            expected: vec![1, 2],
            got: vec![ndim],
        });
    }

    let d = input.shape[ndim - 1];

    if weight.ndim() != 1 || weight.shape[0] != d {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "layernorm_weight".to_string(),
            expected: vec![d],
            got: weight.shape.clone(),
        });
    }
    if let Some(b) = bias {
        if b.ndim() != 1 || b.shape[0] != d {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "layernorm_bias".to_string(),
                expected: vec![d],
                got: b.shape.clone(),
            });
        }
    }

    let rows = if ndim == 1 { 1 } else { input.shape[0] };
    let mut out = Vec::with_capacity(input.data.len());

    for r in 0..rows {
        let row = &input.data[r * d..(r + 1) * d];

        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();

        for (i, &x) in row.iter().enumerate() {
            let normalized = (x - mean) * inv_std;
            let scaled = normalized * weight.data[i];
            let shifted = match bias {
                Some(b) => scaled + b.data[i],
                None => scaled,
            };
            out.push(shifted);
        }
    }

    // Output shape matches input shape
    Tensor::new(out, input.shape.clone())
}

/// GELU activation (tanh approximation matching PyTorch default).
///
/// Formula: `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// Used in ViT FFN blocks (SigLIP uses GELU, not SwiGLU).
pub fn gelu_f32(input: &Tensor) -> Result<Tensor> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
    const COEFF: f32 = 0.044_715;

    let output_data: Vec<f32> = input
        .data
        .iter()
        .map(|&x| {
            let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
            x * 0.5 * (1.0 + inner.tanh())
        })
        .collect();

    Tensor::new(output_data, input.shape.clone())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return a view (clone with squeezed shape) with all leading size-1 dims removed.
fn squeeze_leading_ones(t: &Tensor) -> Tensor {
    let first_non_one = t
        .shape
        .iter()
        .position(|&d| d != 1)
        .unwrap_or(t.shape.len());
    let new_shape = if first_non_one == t.shape.len() {
        // All dims are 1 — keep a single scalar dim
        vec![1]
    } else {
        t.shape[first_non_one..].to_vec()
    };
    Tensor {
        data: t.data.clone(),
        shape: new_shape,
    }
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

    // ── add_broadcast ─────────────────────────────────────────────────────────

    #[test]
    fn test_add_broadcast_leading_one() {
        // [1, 3] + [3] → [3]
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let result = add_broadcast_f32(&a, &b).unwrap();
        assert_eq!(result.data, vec![11.0, 22.0, 33.0]);
        assert_eq!(result.shape, vec![3]);
    }

    #[test]
    fn test_add_broadcast_vit_shape() {
        // Mimics [1, 4, 2] pos_embed + [4, 2] patch_features → [4, 2]
        let pos = Tensor::new(vec![1.0; 8], vec![1, 4, 2]).unwrap();
        let feat = Tensor::new(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], vec![4, 2]).unwrap();
        let result = add_broadcast_f32(&pos, &feat).unwrap();
        assert_eq!(result.shape, vec![4, 2]);
        assert!((result.data[0] - 1.0).abs() < 1e-6);
        assert!((result.data[7] - 4.5).abs() < 1e-6);
    }

    // ── layernorm ─────────────────────────────────────────────────────────────

    #[test]
    fn test_layernorm_1d_no_bias() {
        // Hand-computed: input=[1,2,3,4], mean=2.5, var=1.25, eps=1e-5
        // inv_std = 1/sqrt(1.25+1e-5) ≈ 0.8944
        // normalized ≈ [-1.3416, -0.4472, 0.4472, 1.3416]
        // weight=ones → same
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let weight = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let result = layernorm_f32(&input, &weight, None, 1e-5).unwrap();
        assert_eq!(result.shape, vec![4]);
        // Sum of normalized values should be ~0
        let sum: f32 = result.data.iter().sum();
        assert!(sum.abs() < 1e-4, "sum={sum}");
        // Variance of output should be ~1
        let mean = sum / 4.0;
        let var: f32 = result.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-4, "var={var}");
    }

    #[test]
    fn test_layernorm_2d_with_bias() {
        let input = Tensor::new(
            vec![
                1.0, 3.0, // row 0: mean=2, var=1
                5.0, 7.0,
            ], // row 1: mean=6, var=1
            vec![2, 2],
        )
        .unwrap();
        let weight = Tensor::new(vec![2.0, 2.0], vec![2]).unwrap();
        let bias = Tensor::new(vec![1.0, 1.0], vec![2]).unwrap();
        // row 0: normalized=[-1,1], *weight=[−2,2], +bias=[−1,3]
        // row 1: normalized=[-1,1], *weight=[−2,2], +bias=[−1,3]
        let result = layernorm_f32(&input, &weight, Some(&bias), 1e-5).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
        assert!((result.data[0] - (-1.0)).abs() < 1e-4);
        assert!((result.data[1] - 3.0).abs() < 1e-4);
        assert!((result.data[2] - (-1.0)).abs() < 1e-4);
        assert!((result.data[3] - 3.0).abs() < 1e-4);
    }

    // ── gelu ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_gelu_known_values() {
        // PyTorch reference values (tanh approx):
        //   gelu(0.0)  = 0.0
        //   gelu(1.0)  ≈ 0.8413
        //   gelu(-1.0) ≈ -0.1587
        //   gelu(2.0)  ≈ 1.9546
        let input = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]).unwrap();
        let result = gelu_f32(&input).unwrap();
        assert!(result.data[0].abs() < 1e-5, "gelu(0) should be 0");
        assert!(
            (result.data[1] - 0.8413).abs() < 1e-3,
            "gelu(1)={}",
            result.data[1]
        );
        assert!(
            (result.data[2] - (-0.1587)).abs() < 1e-3,
            "gelu(-1)={}",
            result.data[2]
        );
        assert!(
            (result.data[3] - 1.9546).abs() < 1e-3,
            "gelu(2)={}",
            result.data[3]
        );
    }

    #[test]
    fn test_gelu_shape_preservation() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = gelu_f32(&input).unwrap();
        assert_eq!(result.shape, vec![2, 2]);
    }
}
