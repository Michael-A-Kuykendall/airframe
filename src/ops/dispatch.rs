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
