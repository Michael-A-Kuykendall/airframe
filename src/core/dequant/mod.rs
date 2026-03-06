//! Dequantization kernels for GGML quantized tensors.
//!
//! Converts Q4_K and Q6_K superblocks to FP32.

pub mod q4_k;
pub mod q6_k;

pub use q4_k::dequantize_q4_k;
pub use q6_k::dequantize_q6_k;
