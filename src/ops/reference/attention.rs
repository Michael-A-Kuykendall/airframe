//! Multi-head attention with Grouped Query Attention (GQA) support.
//!
//! Implements both with and without KV cache for prefill/decode phases.
#![allow(clippy::too_many_arguments)]

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};
use crate::ops::reference::{matmul, rope, softmax};
use crate::runtime::kvcache::KvCache;

/// Multi-head attention with GQA.
///
/// GQA: `n_head_kv < n_head`, multiple query heads share K/V heads.
pub fn attention_f32(
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
    // Validate GQA configuration
    if !n_head.is_multiple_of(n_head_kv) {
        return Err(LibshimmyError::Unsupported(format!(
            "n_head ({}) must be divisible by n_head_kv ({})",
            n_head, n_head_kv
        )));
    }

    let group_size = n_head / n_head_kv;

    match input.ndim() {
        2 => {
            // Single sequence: [seq_len, hidden_size]
            attention_2d(
                input,
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                n_head,
                n_head_kv,
                head_dim,
                group_size,
                position_ids,
                rope_base,
                rope_dim,
                rope_scale,
                causal_mask,
            )
        }
        3 => {
            // Batched: [batch, seq_len, hidden_size]
            attention_3d(
                input,
                q_weight,
                k_weight,
                v_weight,
                o_weight,
                n_head,
                n_head_kv,
                head_dim,
                group_size,
                position_ids,
                rope_base,
                rope_dim,
                rope_scale,
                causal_mask,
            )
        }
        _ => {
            Err(LibshimmyError::ShapeMismatch {
                tensor: "attention_input".to_string(),
                expected: vec![2, 3], // 2D or 3D
                got: vec![input.ndim()],
            })
        }
    }
}

/// Apply per-head RMSNorm to a heads tensor [seq_len, n_heads, head_dim].
/// Used for Qwen3 QK norm (applied after projection, before RoPE).
fn apply_qk_norm(heads: &Tensor, norm_weight: &Tensor, eps: f32) -> Result<Tensor> {
    let seq_len = heads.shape[0];
    let n_heads = heads.shape[1];
    let head_dim = heads.shape[2];
    let mut out = heads.data.clone();
    for t in 0..seq_len {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            let slice = &heads.data[base..base + head_dim];
            let rms_sq: f32 = slice.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let scale = 1.0 / (rms_sq + eps).sqrt();
            for d in 0..head_dim {
                out[base + d] = slice[d] * scale * norm_weight.data[d];
            }
        }
    }
    Tensor::new(out, vec![seq_len, n_heads, head_dim])
}

/// Multi-head attention with KV cache.
///
/// Handles prefill (store all) and decode (append one) phases.
// too_many_arguments: attention requires full tensor set, cache, and rope params; no logical grouping
#[allow(clippy::too_many_arguments)]
pub fn attention_with_cache_f32(
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
    qk_norm: Option<(&Tensor, &Tensor)>, // (q_norm_weight, k_norm_weight) for Qwen3
    attention_scale: Option<&Tensor>,    // per-head scale [n_head] for Qwen3
) -> Result<Tensor> {
    // Validate GQA configuration
    if !n_head.is_multiple_of(n_head_kv) {
        return Err(LibshimmyError::Unsupported(format!(
            "n_head ({}) must be divisible by n_head_kv ({})",
            n_head, n_head_kv
        )));
    }

    // Only 2D input supported with cache
    if input.ndim() != 2 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "attention_input".to_string(),
            expected: vec![2],
            got: vec![input.ndim()],
        });
    }

    let seq_len = input.shape[0];
    let hidden_size = input.shape[1];
    let group_size = n_head / n_head_kv;
    let cache_len = kv_cache.len();

    // Validate weight shapes
    validate_weight_shapes(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        hidden_size,
        n_head,
        n_head_kv,
        head_dim,
    )?;

    // 1. Compute Q, K, V projections for new tokens
    let q = matmul::matmul_f32(input, q_weight)?; // [seq_len, n_head * head_dim]
    if layer_idx == 0 {
        println!(
            "Q result (first 10): {:?}",
            q.data.iter().take(10).collect::<Vec<_>>()
        );
    }
    let k = matmul::matmul_f32(input, k_weight)?; // [seq_len, n_head_kv * head_dim]
    let v = matmul::matmul_f32(input, v_weight)?; // [seq_len, n_head_kv * head_dim]

    // 2. Reshape to multi-head format
    let q_heads = reshape_to_heads(&q, seq_len, n_head, head_dim)?; // [seq_len, n_head, head_dim]
    let k_heads = reshape_to_heads(&k, seq_len, n_head_kv, head_dim)?; // [seq_len, n_head_kv, head_dim]
    let v_heads = reshape_to_heads(&v, seq_len, n_head_kv, head_dim)?; // [seq_len, n_head_kv, head_dim]

    // 2.5. QK Norm (Qwen3): per-head RMSNorm on Q and K before RoPE
    let rms_eps = 1e-6_f32;
    let (q_heads, k_heads) = if let Some((q_nw, k_nw)) = qk_norm {
        (
            apply_qk_norm(&q_heads, q_nw, rms_eps)?,
            apply_qk_norm(&k_heads, k_nw, rms_eps)?,
        )
    } else {
        (q_heads, k_heads)
    };

    // 3. Apply RoPE to Q and K (using position_ids for absolute positions)
    let q_rope =
        rope::apply_rope_scaled_f32(&q_heads, position_ids, rope_base, rope_dim, rope_scale)?;
    let k_rope =
        rope::apply_rope_scaled_f32(&k_heads, position_ids, rope_base, rope_dim, rope_scale)?;

    // 4. Store new K, V in cache (before retrieving full cache)
    // Note: We store AFTER RoPE for K, but V is unmodified
    if seq_len == 1 {
        // Decode: append single token
        kv_cache.append_layer(layer_idx, &k_rope, &v_heads)?;
    } else {
        // Prefill: store all tokens
        kv_cache.prefill_layer(layer_idx, &k_rope, &v_heads)?;
    }

    // 5. Get FULL K, V from cache (including historical tokens)
    // For prefill: this includes what we just stored
    // For decode: this includes all previous + new token
    let (k_cached, v_cached) =
        get_kv_for_attention(kv_cache, layer_idx, &k_rope, &v_heads, cache_len, seq_len)?;
    let total_len = cache_len + seq_len;

    // DIAGNOSTIC: Check if tracing is enabled
    let trace_attention = std::env::var("LIBSHIMMY_TRACE_ATTENTION").is_ok();
    let trace_layer = std::env::var("LIBSHIMMY_TRACE_ATTENTION_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let should_trace_layer = trace_attention && layer_idx == trace_layer;

    if should_trace_layer {
        eprintln!(
            "🔍 Attention layer {}: seq_len={}, cache_len={}, total_len={}",
            layer_idx, seq_len, cache_len, total_len
        );
    }

    // 6. Compute attention for each query head
    let mut attn_outputs = Vec::new();

    for h in 0..n_head {
        let kv_head = h / group_size; // Which K/V head this Q head uses

        // Extract Q head: [seq_len, head_dim] (only query new tokens)
        let q_head = extract_head(&q_rope, h, seq_len, head_dim)?;

        // Extract K, V heads: [total_len, head_dim] (full sequence including cache)
        let k_head = extract_head(&k_cached, kv_head, total_len, head_dim)?;
        let v_head = extract_head(&v_cached, kv_head, total_len, head_dim)?;

        // Compute attention: Q @ K^T
        // Q: [seq_len, head_dim], K^T: [head_dim, total_len]
        // Result: [seq_len, total_len]
        let k_head_t = transpose_2d(&k_head)?; // [head_dim, total_len]
        let scores = matmul::matmul_row_major_f32(&q_head, &k_head_t)?; // [seq_len, total_len]

        // DIAGNOSTIC: Dump UNSCALED kq scores for head 0 (to match oracle L0.7)
        if should_trace_layer && h == 0 {
            let rms: f32 =
                (scores.data.iter().map(|x| x * x).sum::<f32>() / scores.data.len() as f32).sqrt();
            eprintln!("🔍 RUNTIME L{}.7 kq UNSCALED (head 0):", layer_idx);
            eprintln!("   shape: {:?}", scores.shape);
            eprintln!("   RMS: {:.8}", rms);
            eprintln!(
                "   first 20: {:?}",
                &scores.data[..20.min(scores.data.len())]
            );
        }

        // Scale by sqrt(head_dim), then optionally multiply by per-head attention.scale (Qwen3)
        let base_scale = 1.0 / (head_dim as f32).sqrt();
        let head_scale = attention_scale
            .and_then(|s| s.data.get(h))
            .copied()
            .unwrap_or(1.0);
        let scaled_scores = scale_tensor(&scores, base_scale * head_scale)?;

        // Apply softmax with causal mask
        // For causal mask with cache: each query position can only attend to
        // positions up to and including itself in the full sequence
        let attn_weights = softmax::softmax_f32(&scaled_scores, true)?;

        // DIAGNOSTIC: Dump softmax output for head 0
        if should_trace_layer && h == 0 {
            let rms: f32 = (attn_weights.data.iter().map(|x| x * x).sum::<f32>()
                / attn_weights.data.len() as f32)
                .sqrt();
            eprintln!("🔍 RUNTIME L{}.8 kq_soft_max (head 0):", layer_idx);
            eprintln!("   shape: {:?}", attn_weights.shape);
            eprintln!("   RMS: {:.8}", rms);
            eprintln!(
                "   first 20: {:?}",
                &attn_weights.data[..20.min(attn_weights.data.len())]
            );
        }

        // Apply attention: weights @ V
        // attn_weights: [seq_len, total_len], V: [total_len, head_dim]
        // Result: [seq_len, head_dim]
        let attn_out = matmul::matmul_row_major_f32(&attn_weights, &v_head)?;

        if should_trace_layer && h == 0 && seq_len == 1 && total_len >= 2 {
            let dot0 = scores.data[0];
            let dot1 = scores.data[1];
            let s0 = scaled_scores.data[0];
            let s1 = scaled_scores.data[1];
            let w0 = attn_weights.data[0];
            let w1 = attn_weights.data[1];
            let v0 = v_head.data[0];
            let v1 = v_head.data[head_dim];
            let context0 = attn_out.data[0];
            eprintln!(
                "🔍 CPU-ATTN-SCALAR L{} h0 d0 | dot=({:.8},{:.8}) scaled=({:.8},{:.8}) w=({:.8},{:.8}) v=({:.8},{:.8}) ctx={:.8}",
                layer_idx, dot0, dot1, s0, s1, w0, w1, v0, v1, context0
            );
        }
        attn_outputs.push(attn_out);
    }

    // 7. Concatenate all heads
    let concat_output = concatenate_heads(&attn_outputs, seq_len, n_head, head_dim)?;

    // 8. Final output projection
    matmul::matmul_f32(&concat_output, o_weight)
}

/// Get K, V for attention computation
///
/// For prefill (cache_len == 0): Return the new K, V directly
/// For decode: Concatenate cached K, V with new K, V
fn get_kv_for_attention(
    kv_cache: &KvCache,
    layer_idx: usize,
    k_new: &Tensor,
    v_new: &Tensor,
    cache_len: usize,
    new_len: usize,
) -> Result<(Tensor, Tensor)> {
    if cache_len == 0 {
        // Prefill: just use new K, V
        return Ok((k_new.clone(), v_new.clone()));
    }

    // Decode: need to get cached K, V and concatenate with new
    let n_head_kv = k_new.shape[1];
    let head_dim = k_new.shape[2];

    // Get cached data (before we stored the new token)
    // Note: We stored the new token already, but kv_cache.len() hasn't been updated yet
    // So we read cache_len tokens from cache and concatenate with new
    let total_len = cache_len + new_len;

    // Extract cached K, V (positions 0..cache_len)
    let k_cache = &kv_cache.key_cache[layer_idx];
    let v_cache = &kv_cache.value_cache[layer_idx];

    // Concatenate: [cache_len, n_head_kv, head_dim] + [new_len, n_head_kv, head_dim]
    // -> [total_len, n_head_kv, head_dim]
    let mut k_data = Vec::with_capacity(total_len * n_head_kv * head_dim);
    let mut v_data = Vec::with_capacity(total_len * n_head_kv * head_dim);

    // Copy cached tokens
    let cache_elements = cache_len * n_head_kv * head_dim;
    k_data.extend_from_slice(&k_cache.data[..cache_elements]);
    v_data.extend_from_slice(&v_cache.data[..cache_elements]);

    // Copy new tokens
    k_data.extend_from_slice(&k_new.data);
    v_data.extend_from_slice(&v_new.data);

    let k_full = Tensor::new(k_data, vec![total_len, n_head_kv, head_dim])?;
    let v_full = Tensor::new(v_data, vec![total_len, n_head_kv, head_dim])?;

    Ok((k_full, v_full))
}

/// Attention for 2D input [seq_len, hidden_size]
fn attention_2d(
    input: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    o_weight: &Tensor,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    group_size: usize,
    position_ids: &[usize],
    rope_base: f32,
    rope_dim: usize,
    rope_scale: f32,
    causal_mask: bool,
) -> Result<Tensor> {
    let seq_len = input.shape[0];
    let hidden_size = input.shape[1];

    // Validate weight shapes
    validate_weight_shapes(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        hidden_size,
        n_head,
        n_head_kv,
        head_dim,
    )?;

    // 1. Compute Q, K, V projections
    let q = matmul::matmul_f32(input, q_weight)?; // [seq_len, n_head * head_dim]
    let k = matmul::matmul_f32(input, k_weight)?; // [seq_len, n_head_kv * head_dim]
    let v = matmul::matmul_f32(input, v_weight)?; // [seq_len, n_head_kv * head_dim]

    // 2. Reshape to multi-head format
    let q_heads = reshape_to_heads(&q, seq_len, n_head, head_dim)?; // [seq_len, n_head, head_dim]
    let k_heads = reshape_to_heads(&k, seq_len, n_head_kv, head_dim)?; // [seq_len, n_head_kv, head_dim]
    let v_heads = reshape_to_heads(&v, seq_len, n_head_kv, head_dim)?; // [seq_len, n_head_kv, head_dim]

    // 3. Apply RoPE to Q and K
    let q_rope =
        rope::apply_rope_scaled_f32(&q_heads, position_ids, rope_base, rope_dim, rope_scale)?;
    let k_rope =
        rope::apply_rope_scaled_f32(&k_heads, position_ids, rope_base, rope_dim, rope_scale)?;

    // DIAGNOSTIC: Check if tracing is enabled
    let trace_attention = std::env::var("LIBSHIMMY_TRACE_ATTENTION").is_ok();

    // 4. Compute attention for each query head
    let mut attn_outputs = Vec::new();

    for h in 0..n_head {
        let kv_head = h / group_size; // Which K/V head this Q head uses

        // Extract Q head: [seq_len, head_dim]
        let q_head = extract_head(&q_rope, h, seq_len, head_dim)?;

        // Extract K, V heads: [seq_len, head_dim]
        let k_head = extract_head(&k_rope, kv_head, seq_len, head_dim)?;
        let v_head = extract_head(&v_heads, kv_head, seq_len, head_dim)?;

        // Compute attention: Q @ K^T
        // NOTE: K^T is a row-major transposed activation, NOT a GGML weight
        // So we must use row-major matmul
        let k_head_t = transpose_2d(&k_head)?; // [head_dim, seq_len]
        let scores = matmul::matmul_row_major_f32(&q_head, &k_head_t)?; // [seq_len, seq_len]

        // DIAGNOSTIC: Dump UNSCALED kq scores for head 0 (to match oracle L0.7)
        if trace_attention && h == 0 {
            let rms: f32 =
                (scores.data.iter().map(|x| x * x).sum::<f32>() / scores.data.len() as f32).sqrt();
            eprintln!("🔍 RUNTIME L0.7 kq UNSCALED (head 0):");
            eprintln!("   shape: {:?}", scores.shape);
            eprintln!("   RMS: {:.8}", rms);
            eprintln!(
                "   first 20: {:?}",
                &scores.data[..20.min(scores.data.len())]
            );
        }

        // Scale by sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scaled_scores = scale_tensor(&scores, scale)?;

        // Apply softmax with causal mask
        let attn_weights = softmax::softmax_f32(&scaled_scores, causal_mask)?;

        // DIAGNOSTIC: Dump softmax output for head 0
        if trace_attention && h == 0 {
            let rms: f32 = (attn_weights.data.iter().map(|x| x * x).sum::<f32>()
                / attn_weights.data.len() as f32)
                .sqrt();
            eprintln!("🔍 RUNTIME L0.8 kq_soft_max (head 0):");
            eprintln!("   shape: {:?}", attn_weights.shape);
            eprintln!("   RMS: {:.8}", rms);
            eprintln!(
                "   first 20: {:?}",
                &attn_weights.data[..20.min(attn_weights.data.len())]
            );
        }

        // Apply attention: weights @ V
        // NOTE: Both attn_weights and v_head are row-major activations
        let attn_out = matmul::matmul_row_major_f32(&attn_weights, &v_head)?; // [seq_len, head_dim]
        attn_outputs.push(attn_out);
    }

    // 5. Concatenate all heads
    let concat_output = concatenate_heads(&attn_outputs, seq_len, n_head, head_dim)?;

    // 6. Final output projection
    matmul::matmul_f32(&concat_output, o_weight)
}

/// Attention for 3D input [batch, seq_len, hidden_size]
fn attention_3d(
    input: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    o_weight: &Tensor,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    group_size: usize,
    position_ids: &[usize],
    rope_base: f32,
    rope_dim: usize,
    rope_scale: f32,
    causal_mask: bool,
) -> Result<Tensor> {
    let batch_size = input.shape[0];
    let seq_len = input.shape[1];
    let hidden_size = input.shape[2];

    let mut batch_outputs = Vec::new();

    for b in 0..batch_size {
        // Extract batch: [seq_len, hidden_size]
        let batch_input = extract_batch(input, b, seq_len, hidden_size)?;

        // Process single batch
        let batch_output = attention_2d(
            &batch_input,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            n_head,
            n_head_kv,
            head_dim,
            group_size,
            position_ids,
            rope_base,
            rope_dim,
            rope_scale,
            causal_mask,
        )?;

        batch_outputs.push(batch_output);
    }

    // Concatenate batches
    concatenate_batches(&batch_outputs, batch_size, seq_len, hidden_size)
}

// Helper functions

fn validate_weight_shapes(
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    o_weight: &Tensor,
    hidden_size: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
) -> Result<()> {
    let expected_q_shape = vec![hidden_size, n_head * head_dim];
    let expected_kv_shape = vec![hidden_size, n_head_kv * head_dim];
    let expected_o_shape = vec![n_head * head_dim, hidden_size];

    if q_weight.shape != expected_q_shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "q_weight".to_string(),
            expected: expected_q_shape,
            got: q_weight.shape.clone(),
        });
    }

    if k_weight.shape != expected_kv_shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "k_weight".to_string(),
            expected: expected_kv_shape.clone(),
            got: k_weight.shape.clone(),
        });
    }

    if v_weight.shape != expected_kv_shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "v_weight".to_string(),
            expected: expected_kv_shape,
            got: v_weight.shape.clone(),
        });
    }

    if o_weight.shape != expected_o_shape {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "o_weight".to_string(),
            expected: expected_o_shape,
            got: o_weight.shape.clone(),
        });
    }

    Ok(())
}

fn reshape_to_heads(
    tensor: &Tensor,
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<Tensor> {
    // Reshape [seq_len, n_head * head_dim] -> [seq_len, n_head, head_dim]
    Tensor::new(tensor.data.clone(), vec![seq_len, n_head, head_dim])
}

fn extract_head(
    tensor: &Tensor,
    head_idx: usize,
    seq_len: usize,
    head_dim: usize,
) -> Result<Tensor> {
    // Extract single head from [seq_len, n_head, head_dim] -> [seq_len, head_dim]
    let mut head_data = Vec::with_capacity(seq_len * head_dim);

    for s in 0..seq_len {
        for d in 0..head_dim {
            let idx = s * tensor.shape[1] * tensor.shape[2] + head_idx * head_dim + d;
            head_data.push(tensor.data[idx]);
        }
    }

    Tensor::new(head_data, vec![seq_len, head_dim])
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    // Transpose [M, N] -> [N, M]
    let m = tensor.shape[0];
    let n = tensor.shape[1];
    let mut transposed = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            transposed[j * m + i] = tensor.data[i * n + j];
        }
    }

    Tensor::new(transposed, vec![n, m])
}

fn scale_tensor(tensor: &Tensor, scale: f32) -> Result<Tensor> {
    let scaled_data: Vec<f32> = tensor.data.iter().map(|&x| x * scale).collect();
    Tensor::new(scaled_data, tensor.shape.clone())
}

fn concatenate_heads(
    heads: &[Tensor],
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
) -> Result<Tensor> {
    // Concatenate [seq_len, head_dim] * n_head -> [seq_len, n_head * head_dim]
    let mut concat_data = Vec::with_capacity(seq_len * n_head * head_dim);

    for s in 0..seq_len {
        for head in heads.iter().take(n_head) {
            for d in 0..head_dim {
                concat_data.push(head.data[s * head_dim + d]);
            }
        }
    }

    Tensor::new(concat_data, vec![seq_len, n_head * head_dim])
}

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
    fn test_attention_shape_invariants() {
        let seq_len = 2;
        let hidden_size = 4;
        let n_head = 2;
        let n_head_kv = 1; // GQA: 2 query heads share 1 key/value head
        let head_dim = 2;

        // Create input and weights
        let input = Tensor::zeros(vec![seq_len, hidden_size]);
        let q_weight = Tensor::zeros(vec![hidden_size, n_head * head_dim]);
        let k_weight = Tensor::zeros(vec![hidden_size, n_head_kv * head_dim]);
        let v_weight = Tensor::zeros(vec![hidden_size, n_head_kv * head_dim]);
        let o_weight = Tensor::zeros(vec![n_head * head_dim, hidden_size]);

        let position_ids = vec![0, 1];

        let output = attention_f32(
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            n_head,
            n_head_kv,
            head_dim,
            &position_ids,
            10000.0,
            head_dim,
            1.0,
            true,
        )
        .unwrap();

        // Output should have same shape as input
        assert_eq!(output.shape, vec![seq_len, hidden_size]);

        // Should not contain NaN
        for &val in &output.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_attention_deterministic() {
        let seq_len = 1;
        let hidden_size = 2;
        let n_head = 1;
        let n_head_kv = 1;
        let head_dim = 2;

        let input = Tensor::new(vec![1.0, 0.0], vec![seq_len, hidden_size]).unwrap();
        let q_weight = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![hidden_size, n_head * head_dim],
        )
        .unwrap();
        let k_weight = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![hidden_size, n_head_kv * head_dim],
        )
        .unwrap();
        let v_weight = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![hidden_size, n_head_kv * head_dim],
        )
        .unwrap();
        let o_weight = Tensor::new(
            vec![1.0, 0.0, 0.0, 1.0],
            vec![n_head * head_dim, hidden_size],
        )
        .unwrap();

        let position_ids = vec![0];

        let output1 = attention_f32(
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            n_head,
            n_head_kv,
            head_dim,
            &position_ids,
            10000.0,
            head_dim,
            1.0,
            false,
        )
        .unwrap();

        let output2 = attention_f32(
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            n_head,
            n_head_kv,
            head_dim,
            &position_ids,
            10000.0,
            head_dim,
            1.0,
            false,
        )
        .unwrap();

        // Should be deterministic
        assert_eq!(output1.data, output2.data);
    }

    #[test]
    fn test_attention_gqa_validation() {
        let seq_len = 1;
        let hidden_size = 2;
        let n_head = 3; // Not divisible by n_head_kv
        let n_head_kv = 2;
        let head_dim = 2;

        let input = Tensor::zeros(vec![seq_len, hidden_size]);
        let q_weight = Tensor::zeros(vec![hidden_size, n_head * head_dim]);
        let k_weight = Tensor::zeros(vec![hidden_size, n_head_kv * head_dim]);
        let v_weight = Tensor::zeros(vec![hidden_size, n_head_kv * head_dim]);
        let o_weight = Tensor::zeros(vec![n_head * head_dim, hidden_size]);

        let position_ids = vec![0];

        let result = attention_f32(
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            n_head,
            n_head_kv,
            head_dim,
            &position_ids,
            10000.0,
            head_dim,
            1.0,
            false,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_attention_weight_shape_validation() {
        let seq_len = 1;
        let hidden_size = 4;
        let n_head = 2;
        let n_head_kv = 1;
        let head_dim = 2;

        let input = Tensor::zeros(vec![seq_len, hidden_size]);
        let q_weight = Tensor::zeros(vec![hidden_size, 3]); // Wrong shape
        let k_weight = Tensor::zeros(vec![hidden_size, n_head_kv * head_dim]);
        let v_weight = Tensor::zeros(vec![hidden_size, n_head_kv * head_dim]);
        let o_weight = Tensor::zeros(vec![n_head * head_dim, hidden_size]);

        let position_ids = vec![0];

        let result = attention_f32(
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            n_head,
            n_head_kv,
            head_dim,
            &position_ids,
            10000.0,
            head_dim,
            1.0,
            false,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_attention_batched() {
        let batch_size = 2;
        let seq_len = 1;
        let hidden_size = 2;
        let n_head = 1;
        let n_head_kv = 1;
        let head_dim = 2;

        let input = Tensor::zeros(vec![batch_size, seq_len, hidden_size]);
        let q_weight = Tensor::zeros(vec![hidden_size, n_head * head_dim]);
        let k_weight = Tensor::zeros(vec![hidden_size, n_head_kv * head_dim]);
        let v_weight = Tensor::zeros(vec![hidden_size, n_head_kv * head_dim]);
        let o_weight = Tensor::zeros(vec![n_head * head_dim, hidden_size]);

        let position_ids = vec![0];

        let output = attention_f32(
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            n_head,
            n_head_kv,
            head_dim,
            &position_ids,
            10000.0,
            head_dim,
            1.0,
            false,
        )
        .unwrap();

        assert_eq!(output.shape, vec![batch_size, seq_len, hidden_size]);
    }

    /// T-1.4: Bidirectional attention (causal_mask=false) correctness.
    ///
    /// With causal masking OFF every token must attend to every other token.
    /// We verify this by checking that token-0's output changes when a later
    /// token's value changes — which would be impossible under causal masking.
    #[test]
    fn test_bidirectional_attention_all_tokens_visible() {
        // 4-token sequence, 1 head, head_dim=4, hidden=4
        let seq_len = 4;
        let hidden = 4;
        let n_head = 1;
        let n_head_kv = 1;
        let head_dim = 4;

        // Identity projections so Q/K/V = input directly
        let eye4: Vec<f32> = vec![
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let q_w = Tensor::new(eye4.clone(), vec![hidden, n_head * head_dim]).unwrap();
        let k_w = Tensor::new(eye4.clone(), vec![hidden, n_head_kv * head_dim]).unwrap();
        let v_w = Tensor::new(eye4.clone(), vec![hidden, n_head_kv * head_dim]).unwrap();
        let o_w = Tensor::new(eye4.clone(), vec![n_head * head_dim, hidden]).unwrap();

        // Input where token-3 has a distinctive value
        let input_a = Tensor::new(
            vec![
                1., 0., 0., 0., // tok 0
                0., 1., 0., 0., // tok 1
                0., 0., 1., 0., // tok 2
                0., 0., 0., 9., // tok 3 — large distinctive value
            ],
            vec![seq_len, hidden],
        )
        .unwrap();

        // Same but token-3 is zeroed
        let input_b = Tensor::new(
            vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
                0., // tok 3 now zero
            ],
            vec![seq_len, hidden],
        )
        .unwrap();

        let pos_ids: Vec<usize> = (0..seq_len).collect();

        let out_a = attention_f32(
            &input_a, &q_w, &k_w, &v_w, &o_w, n_head, n_head_kv, head_dim, &pos_ids, 10000.0,
            head_dim, 1.0, false, // bidirectional
        )
        .unwrap();

        let out_b = attention_f32(
            &input_b, &q_w, &k_w, &v_w, &o_w, n_head, n_head_kv, head_dim, &pos_ids, 10000.0,
            head_dim, 1.0, false,
        )
        .unwrap();

        // Token-0's output must differ between input_a and input_b because
        // token-0 can attend to token-3 (bidirectional).
        let tok0_a: f32 = out_a.data[..hidden].iter().map(|x| x.abs()).sum();
        let tok0_b: f32 = out_b.data[..hidden].iter().map(|x| x.abs()).sum();
        assert!(
            (tok0_a - tok0_b).abs() > 1e-4,
            "tok0 output unchanged despite tok3 change — causal mask may be stuck ON: a={tok0_a} b={tok0_b}"
        );
    }
}
