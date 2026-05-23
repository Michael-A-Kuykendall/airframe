//! Operation dispatcher for all tensor math.
//!
//! Enforces "all math lives in ops/" by providing a single entry point
//! for matrix operations, normalization, attention, and activations.

use crate::core::{error::Result, tensor::Tensor};
use crate::ops::reference::{activations, attention, ffn, matmul, rmsnorm, rope, softmax};
use crate::runtime::kvcache::KvCache;

/// Dispatcher for all tensor operations.
pub struct OpDispatcher;

impl OpDispatcher {
    pub fn new() -> Self {
        Self
    }

    /// Matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N] -> C: [M, N]
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        matmul::matmul_f32(a, b)
    }

    /// Matrix-vector multiplication: y = A @ x
    /// A: [M, N], x: [N] -> y: [M]
    pub fn matvec(&self, a: &Tensor, x: &Tensor) -> Result<Tensor> {
        matmul::matvec_f32(a, x)
    }

    /// RMSNorm: Root Mean Square Layer Normalization
    /// Normalizes input by RMS and applies learned scale
    pub fn rmsnorm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        rmsnorm::rmsnorm_f32(input, weight, eps)
    }

    /// RoPE: Rotary Position Embedding
    /// Standard Llama RoPE with configurable base and rope_dim
    pub fn rope(
        &self,
        tensor: &Tensor,
        position_ids: &[usize],
        rope_base: f32,
        rope_dim: usize,
    ) -> Result<Tensor> {
        rope::apply_rope_f32(tensor, position_ids, rope_base, rope_dim)
    }

    /// Softmax with optional causal masking
    /// Supports 2D, 3D, and 4D attention matrices
    pub fn softmax(&self, input: &Tensor, causal_mask: bool) -> Result<Tensor> {
        softmax::softmax_f32(input, causal_mask)
    }

    /// Multi-head attention with GQA support
    /// Supports both standard MHA and Grouped Query Attention
    // too_many_arguments: attention requires all tensor weights and rope params; no logical grouping available
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        input: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        v_weight: &Tensor,
        o_weight: &Tensor,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        position_ids: &[usize],
        rope_base: f32,
        rope_dim: usize,
        rope_scale: f32,
        causal_mask: bool,
    ) -> Result<Tensor> {
        attention::attention_f32(
            input,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            n_head,
            n_head_kv,
            head_dim,
            position_ids,
            rope_base,
            rope_dim,
            rope_scale,
            causal_mask,
        )
    }

    /// Multi-head attention WITH KV cache support
    ///
    /// This version properly handles prefill and decode phases:
    /// - Prefill: Stores K, V in cache, attends to all input tokens
    /// - Decode: Stores new K, V, attends to ALL cached tokens + new token
    // too_many_arguments: cache-aware attention requires all tensor weights, cache ref, and rope params
    #[allow(clippy::too_many_arguments)]
    pub fn attention_with_cache(
        &self,
        input: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        v_weight: &Tensor,
        o_weight: &Tensor,
        n_head: usize,
        n_head_kv: usize,
        head_dim: usize,
        position_ids: &[usize],
        rope_base: f32,
        rope_dim: usize,
        rope_scale: f32,
        layer_idx: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        attention::attention_with_cache_f32(
            input,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            n_head,
            n_head_kv,
            head_dim,
            position_ids,
            rope_base,
            rope_dim,
            rope_scale,
            layer_idx,
            kv_cache,
        )
    }

    /// SwiGLU Feed-Forward Network
    /// Combines gating and up projections with SiLU activation
    pub fn ffn_swiglu(
        &self,
        input: &Tensor,
        gate_weight: &Tensor,
        up_weight: &Tensor,
        down_weight: &Tensor,
    ) -> Result<Tensor> {
        ffn::ffn_swiglu_f32(input, gate_weight, up_weight, down_weight)
    }

    /// SiLU activation function
    pub fn silu(&self, input: &Tensor) -> Result<Tensor> {
        activations::silu_f32(input)
    }

    /// Element-wise multiplication
    pub fn multiply(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        activations::multiply_f32(a, b)
    }

    /// Element-wise addition (for residual connections)
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        activations::add_f32(a, b)
    }
}

impl Default for OpDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::Tensor;
    use crate::runtime::kvcache::KvCache;

    fn t(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::new(data, shape).unwrap()
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_and_default_are_equivalent() {
        let _a = OpDispatcher::new();
        let _b = OpDispatcher::default();
        // no panic → pass
    }

    // ── matmul ────────────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_2x2() {
        let ops = OpDispatcher::new();
        // [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = t(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let c = ops.matmul(&a, &b).unwrap();
        assert_eq!(c.data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_1x2_by_2x1() {
        let ops = OpDispatcher::new();
        // [[2, 3]] @ [[4], [5]] = [[23]]
        let a = t(vec![2.0, 3.0], vec![1, 2]);
        let b = t(vec![4.0, 5.0], vec![2, 1]);
        let c = ops.matmul(&a, &b).unwrap();
        assert_eq!(c.data, &[23.0]);
    }

    // ── matvec ────────────────────────────────────────────────────────────────

    #[test]
    fn test_matvec_2x2() {
        let ops = OpDispatcher::new();
        // [[1,0],[0,1]] @ [3,4] = [3,4]
        let a = t(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let x = t(vec![3.0, 4.0], vec![2]);
        let y = ops.matvec(&a, &x).unwrap();
        assert_eq!(y.data, &[3.0, 4.0]);
    }

    #[test]
    fn test_matvec_scaling() {
        let ops = OpDispatcher::new();
        // [[2,0],[0,3]] @ [5,7] = [10,21]
        let a = t(vec![2.0, 0.0, 0.0, 3.0], vec![2, 2]);
        let x = t(vec![5.0, 7.0], vec![2]);
        let y = ops.matvec(&a, &x).unwrap();
        let data = &y.data;
        assert!((data[0] - 10.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 21.0).abs() < 1e-5, "got {}", data[1]);
    }

    // ── rmsnorm ───────────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_unit_weights() {
        let ops = OpDispatcher::new();
        // With weight=1, output should be normalized version of input
        let input = t(vec![3.0, 4.0], vec![2]);
        let weight = t(vec![1.0, 1.0], vec![2]);
        let out = ops.rmsnorm(&input, &weight, 1e-6).unwrap();
        // RMS of [3,4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        // normalized: [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
        let data = &out.data;
        assert!((data[0] - 0.8485).abs() < 0.001, "got {}", data[0]);
        assert!((data[1] - 1.1314).abs() < 0.001, "got {}", data[1]);
    }

    #[test]
    fn test_rmsnorm_zero_weight_yields_zero() {
        let ops = OpDispatcher::new();
        let input = t(vec![1.0, 2.0, 3.0], vec![3]);
        let weight = t(vec![0.0, 0.0, 0.0], vec![3]);
        let out = ops.rmsnorm(&input, &weight, 1e-6).unwrap();
        for &v in &out.data {
            assert!((v).abs() < 1e-5, "expected zero, got {v}");
        }
    }

    // ── softmax ───────────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sum_to_one() {
        let ops = OpDispatcher::new();
        let input = t(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out = ops.softmax(&input, false).unwrap();
        let sum: f32 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
    }

    #[test]
    fn test_softmax_uniform_input() {
        let ops = OpDispatcher::new();
        let input = t(vec![0.0, 0.0, 0.0, 0.0], vec![1, 4]);
        let out = ops.softmax(&input, false).unwrap();
        for &v in &out.data {
            assert!((v - 0.25).abs() < 1e-5, "expected 0.25, got {v}");
        }
    }

    // ── Softmax properties: all values in (0,1], sum==1.0 ────────────────────

    #[test]
    fn test_softmax_all_values_in_open_0_1() {
        let ops = OpDispatcher::new();
        let cases: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[-10.0, 0.0, 10.0],
            &[100.0, 100.0, 100.0],
            &[0.5],               // single element: result is exactly 1.0, which is in (0,1]
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        for vals in cases {
            let n = vals.len();
            let input = t(vals.to_vec(), vec![1, n]);
            let out = ops.softmax(&input, false).unwrap();
            for &v in &out.data {
                // softmax values are strictly positive and at most 1.0
                assert!(v > 0.0 && v <= 1.0, "softmax value {v} not in (0,1] for input {:?}", vals);
            }
        }
    }

    #[test]
    fn test_softmax_sum_invariant_across_inputs() {
        let ops = OpDispatcher::new();
        let cases: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[-5.0, 0.0, 5.0],
            &[1000.0, -1000.0],
            &[0.1, 0.2, 0.3, 0.4],
        ];
        for vals in cases {
            let n = vals.len();
            let input = t(vals.to_vec(), vec![1, n]);
            let out = ops.softmax(&input, false).unwrap();
            let sum: f32 = out.data.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "sum={sum} != 1.0 for input {:?}", vals
            );
        }
    }



    #[test]
    fn test_silu_at_zero_is_zero() {
        let ops = OpDispatcher::new();
        let input = t(vec![0.0], vec![1]);
        let out = ops.silu(&input).unwrap();
        assert!((out.data[0]).abs() < 1e-5, "silu(0) should be ~0, got {}", out.data[0]);
    }

    #[test]
    fn test_silu_positive_value() {
        let ops = OpDispatcher::new();
        // silu(1) = 1 * sigmoid(1) ≈ 0.7311
        let input = t(vec![1.0], vec![1]);
        let out = ops.silu(&input).unwrap();
        assert!((out.data[0] - 0.7311).abs() < 0.001, "silu(1) ≈ 0.7311, got {}", out.data[0]);
    }

    // ── multiply ─────────────────────────────────────────────────────────────

    #[test]
    fn test_multiply_elementwise() {
        let ops = OpDispatcher::new();
        let a = t(vec![1.0, 2.0, 3.0], vec![3]);
        let b = t(vec![4.0, 5.0, 6.0], vec![3]);
        let c = ops.multiply(&a, &b).unwrap();
        assert_eq!(c.data, &[4.0, 10.0, 18.0]);
    }

    // ── add ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_add_elementwise() {
        let ops = OpDispatcher::new();
        let a = t(vec![1.0, 2.0], vec![2]);
        let b = t(vec![10.0, 20.0], vec![2]);
        let c = ops.add(&a, &b).unwrap();
        assert_eq!(c.data, &[11.0, 22.0]);
    }

    // ── rope ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_does_not_panic_on_small_input() {
        let ops = OpDispatcher::new();
        // Shape: [seq_len=1, n_head=1, head_dim=4], rope_dim=4
        let input = t(vec![1.0, 0.0, 1.0, 0.0], vec![1, 1, 4]);
        let pos_ids = vec![0usize];
        let out = ops.rope(&input, &pos_ids, 10000.0, 4);
        // At position 0, RoPE rotates by angle 0 → output should equal input
        match out {
            Ok(result) => assert_eq!(result.data.len(), 4),
            Err(e) => panic!("rope failed: {e}"),
        }
    }

    // ── ffn_swiglu ────────────────────────────────────────────────────────────

    #[test]
    fn test_ffn_swiglu_shape() {
        let ops = OpDispatcher::new();
        // dim=2, ff_dim=3: gate=[2,3], up=[2,3], down=[3,2]
        // input: [1, 2], gate_weight: [dim, ff_dim] = [2, 3]
        let input = t(vec![1.0, 2.0], vec![1, 2]);
        let gate = t(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 3]);
        let up   = t(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 3]);
        let down = t(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], vec![3, 2]);
        let out = ops.ffn_swiglu(&input, &gate, &up, &down).unwrap();
        assert_eq!(out.data.len(), 2, "FFN output dim should match input dim");
    }
}
