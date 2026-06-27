//! Llama model architecture implementation.
//!
//! Defines `LlamaBlock` (single transformer layer) and `LlamaModel`
//! (full stack) with optional L0 tracing for parity validation.
// TODO: migrate eprintln!/println! calls to tracing::{debug!, info!} (post-v2.0 telemetry cleanup)

use crate::core::{error::Result, spec::ModelSpec, tensor::Tensor, weight_id::WeightId};
use crate::ops::dispatch::OpDispatcher;
use crate::runtime::kvcache::KvCache;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to enable verbose diagnostics (expensive!)
/// Set SHIMMY_VERBOSE=1 to enable
static VERBOSE_DIAGNOSTICS: AtomicBool = AtomicBool::new(false);

/// Global flag to enable L0 tracing (expensive!)
/// Set SHIMMY_L0_TRACE_PATH=/path/to/output.csv to enable
static L0_TRACING_ENABLED: AtomicBool = AtomicBool::new(false);
static L0_TRACE_PATH: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

pub fn init_verbose_diagnostics() {
    if std::env::var("SHIMMY_VERBOSE").is_ok() {
        VERBOSE_DIAGNOSTICS.store(true, Ordering::SeqCst);
    }
    if let Ok(path) = std::env::var("SHIMMY_L0_TRACE_PATH") {
        *L0_TRACE_PATH.lock().unwrap() = Some(path);
        L0_TRACING_ENABLED.store(true, Ordering::SeqCst);
    }
}

fn is_verbose() -> bool {
    VERBOSE_DIAGNOSTICS.load(Ordering::Relaxed)
}

fn is_l0_tracing() -> bool {
    L0_TRACING_ENABLED.load(Ordering::Relaxed)
}

fn should_trace_post_attn(layer: usize, cache_len: usize) -> bool {
    if std::env::var("LIBSHIMMY_TRACE_POST_ATTN").is_err() {
        return false;
    }
    let trace_layer = std::env::var("LIBSHIMMY_TRACE_POST_ATTN_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let trace_cache_len = std::env::var("LIBSHIMMY_TRACE_POST_ATTN_CACHELEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);

    layer == trace_layer && cache_len == trace_cache_len
}

fn should_trace_ffn_out(layer: usize, cache_len: usize) -> bool {
    if std::env::var("LIBSHIMMY_TRACE_FFN_OUT").is_err() {
        return false;
    }
    let trace_layer = std::env::var("LIBSHIMMY_TRACE_FFN_OUT_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let trace_cache_len = std::env::var("LIBSHIMMY_TRACE_FFN_OUT_CACHELEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);

    layer == trace_layer && cache_len == trace_cache_len
}

fn should_trace_layer_out(layer: usize, cache_len: usize) -> bool {
    if std::env::var("LIBSHIMMY_TRACE_LAYER_OUT").is_err() {
        return false;
    }
    let trace_layer = std::env::var("LIBSHIMMY_TRACE_LAYER_OUT_LAYER")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let trace_cache_len = std::env::var("LIBSHIMMY_TRACE_LAYER_OUT_CACHELEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    trace_layer.is_none_or(|l| l == layer) && trace_cache_len.is_none_or(|c| c == cache_len)
}

/// Emit L0 checkpoint to trace file
/// Format: id,layer_idx,name,statistic,first20_values
fn emit_l0_checkpoint(id: &str, layer_idx: usize, name: &str, tensor: &Tensor) {
    if !is_l0_tracing() {
        return;
    }

    println!(
        "DEBUG: Emitting L0 checkpoint: {} layer {} {}",
        id, layer_idx, name
    );

    let sq_sum: f32 = tensor.data.iter().map(|x| x * x).sum();
    let rms = (sq_sum / tensor.data.len() as f32).sqrt();

    let first20: Vec<String> = tensor
        .data
        .iter()
        .take(20)
        .map(|x| format!("{:.8}", x))
        .collect();
    let first20_str = first20.join("|");

    let line = format!("{},{},{},{:.8},{}\n", id, layer_idx, name, rms, first20_str);

    if let Some(path) = L0_TRACE_PATH.lock().unwrap().as_ref() {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = file.write_all(line.as_bytes());
        }
    }
}

/// Diagnostic: Dump tensor statistics for debugging NaN issues
/// Returns (non_finite_count, min, max, mean, rms)
/// ONLY runs if SHIMMY_VERBOSE=1 is set
// dead_code: diagnostic helper gated on SHIMMY_VERBOSE; retained for NaN debugging sessions
#[allow(dead_code)]
fn dump_tensor_stats(name: &str, tensor: &Tensor) -> (usize, f32, f32, f32, f32) {
    if !is_verbose() {
        return (0, 0.0, 0.0, 0.0, 0.0); // Skip expensive computation
    }

    let non_finite = tensor.data.iter().filter(|x| !x.is_finite()).count();
    let min = tensor.data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = tensor
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = tensor.data.iter().sum();
    let mean = sum / tensor.data.len() as f32;
    let sq_sum: f32 = tensor.data.iter().map(|x| x * x).sum();
    let rms = (sq_sum / tensor.data.len() as f32).sqrt();

    eprintln!("📊 {} stats:", name);
    eprintln!("   Elements: {}", tensor.data.len());
    eprintln!("   Non-finite: {}", non_finite);
    eprintln!("   Min: {:.6e}, Max: {:.6e}", min, max);
    eprintln!("   Mean: {:.6e}, RMS: {:.6e}", mean, rms);

    // Flag extreme magnitudes (typical weights are in -10 to 10 range)
    if max.abs() > 1e3 || min.abs() > 1e3 {
        eprintln!("   ⚠️  EXTREME MAGNITUDES DETECTED - likely dequant error");
    }
    if non_finite > 0 {
        eprintln!("   🔴 NON-FINITE VALUES - dequant is broken");
    }

    (non_finite, min, max, mean, rms)
}

/// Llama transformer block execution plan
///
/// Declarative specification of how to execute a single Llama layer
/// given the model specification and weights
#[derive(Debug, Clone)]
pub struct LlamaBlock {
    pub layer_idx: usize,
    pub spec: ModelSpec,
}

impl LlamaBlock {
    pub fn new(layer_idx: usize, spec: ModelSpec) -> Self {
        Self { layer_idx, spec }
    }

    /// Execute a single Llama layer
    ///
    /// Input: [seq_len, hidden_size] or [batch, seq_len, hidden_size]
    /// Returns: same shape as input
    pub fn forward(
        &self,
        input: &Tensor,
        weights: &std::collections::HashMap<WeightId, Tensor>,
        kv_cache: &mut KvCache,
        position_ids: &[usize],
        ops: &OpDispatcher,
    ) -> Result<Tensor> {
        let layer = self.layer_idx;

        // Get required weights for this layer
        let attn_norm_weight = weights.get(&WeightId::AttnNorm { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("attn_norm_{}", layer),
            }
        })?;

        let q_weight = weights.get(&WeightId::AttnQ { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("attn_q_{}", layer),
            }
        })?;

        let k_weight = weights.get(&WeightId::AttnK { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("attn_k_{}", layer),
            }
        })?;

        let v_weight = weights.get(&WeightId::AttnV { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("attn_v_{}", layer),
            }
        })?;

        let o_weight = weights.get(&WeightId::AttnO { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("attn_o_{}", layer),
            }
        })?;

        let ffn_norm_weight = weights.get(&WeightId::FfnNorm { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("ffn_norm_{}", layer),
            }
        })?;

        let gate_weight = weights.get(&WeightId::FfnGate { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("ffn_gate_{}", layer),
            }
        })?;

        let up_weight = weights.get(&WeightId::FfnUp { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("ffn_up_{}", layer),
            }
        })?;

        let down_weight = weights.get(&WeightId::FfnDown { layer }).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: format!("ffn_down_{}", layer),
            }
        })?;

        // Optional per-head QK norm weights (Qwen3)
        let q_norm_weight = weights.get(&WeightId::AttnQNorm { layer });
        let k_norm_weight = weights.get(&WeightId::AttnKNorm { layer });
        let qk_norm = match (q_norm_weight, k_norm_weight) {
            (Some(qn), Some(kn)) => Some((qn, kn)),
            _ => None,
        };

        // 1. Attention block with residual connection
        let attn_input = ops.rmsnorm(input, attn_norm_weight, self.spec.rms_eps)?;

        if layer == 0 {
            emit_l0_checkpoint("L0.2", layer, "attn_input", &attn_input);
        }

        // Use attention WITH KV cache support
        let attn_output = ops.attention_with_cache(
            &attn_input,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            self.spec.n_head,
            self.spec.n_head_kv,
            self.spec.n_embd / self.spec.n_head, // head_dim
            position_ids,
            self.spec.rope_base,
            self.spec.rope_dim,
            self.spec.rope_scale,
            layer, // layer_idx for KV cache
            kv_cache,
            qk_norm,
        )?;

        if layer == 0 {
            emit_l0_checkpoint("L0.3", layer, "attn_output", &attn_output);
        }

        // Residual connection: input + attn_output
        let post_attn = ops.add(input, &attn_output)?;

        if should_trace_post_attn(layer, kv_cache.len()) {
            let sq_sum: f32 = post_attn.data.iter().map(|x| x * x).sum();
            let rms = (sq_sum / post_attn.data.len() as f32).sqrt();
            eprintln!(
                "CPU-POST-ATTN-L{}: cache_len={} RMS {:.8}, first10: {:?}",
                layer,
                kv_cache.len(),
                rms,
                &post_attn.data[..10.min(post_attn.data.len())]
            );
        }

        if layer == 0 {
            emit_l0_checkpoint("L0.4", layer, "post_attn", &post_attn);
        }

        // 2. FFN block with residual connection
        let ffn_input = ops.rmsnorm(&post_attn, ffn_norm_weight, self.spec.rms_eps)?;

        if layer == 0 {
            emit_l0_checkpoint("L0.5", layer, "ffn_input", &ffn_input);
        }

        let ffn_output = ops.ffn_swiglu(&ffn_input, gate_weight, up_weight, down_weight)?;

        if should_trace_ffn_out(layer, kv_cache.len()) {
            let sq_sum: f32 = ffn_output.data.iter().map(|x| x * x).sum();
            let rms = (sq_sum / ffn_output.data.len() as f32).sqrt();
            eprintln!(
                "CPU-FFN-OUT-L{}: cache_len={} RMS {:.8}, first10: {:?}",
                layer,
                kv_cache.len(),
                rms,
                &ffn_output.data[..10.min(ffn_output.data.len())]
            );
        }

        if layer == 0 {
            emit_l0_checkpoint("L0.6", layer, "ffn_output", &ffn_output);
        }

        // Residual connection: post_attn + ffn_output
        let output = ops.add(&post_attn, &ffn_output)?;

        if should_trace_layer_out(layer, kv_cache.len()) {
            let sq_sum: f32 = output.data.iter().map(|x| x * x).sum();
            let rms = (sq_sum / output.data.len() as f32).sqrt();
            eprintln!(
                "CPU-LAYER-OUT-L{}: cache_len={} RMS {:.8}, first10: {:?}",
                layer,
                kv_cache.len(),
                rms,
                &output.data[..10.min(output.data.len())]
            );
        }

        Ok(output)
    }
}

/// Llama model execution plan
///
/// Orchestrates execution of all layers in sequence
#[derive(Debug, Clone)]
pub struct LlamaModel {
    pub spec: ModelSpec,
    pub layers: Vec<LlamaBlock>,
}

impl LlamaModel {
    /// Create execution plan from model specification
    pub fn from_spec(spec: ModelSpec) -> Self {
        let layers = (0..spec.n_layer)
            .map(|i| LlamaBlock::new(i, spec.clone()))
            .collect();

        Self { spec, layers }
    }

    /// Execute full model forward pass
    ///
    /// Input: [seq_len, hidden_size] or [batch, seq_len, hidden_size]
    /// Returns: [seq_len, n_vocab] or [batch, seq_len, n_vocab]
    pub fn forward(
        &self,
        input_ids: &[usize],
        weights: &std::collections::HashMap<WeightId, Tensor>,
        kv_cache: &mut KvCache,
        ops: &OpDispatcher,
    ) -> Result<Tensor> {
        // 1. Token embedding
        let token_embed_weight = weights.get(&WeightId::TokenEmbed).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: "token_embed".to_string(),
            }
        })?;

        let mut hidden_states = self.embed_tokens(input_ids, token_embed_weight)?;

        // L0 TRACING: Emit embedding checkpoint
        emit_l0_checkpoint("L0.1", 0, "inp_embd", &hidden_states);

        // DIAGNOSTIC CHECKPOINT: Check embeddings
        let (emb_nf, _, _, _, _) = dump_tensor_stats("embeddings (after lookup)", &hidden_states);
        if emb_nf > 0 {
            return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                "NaN in embeddings: {} non-finite",
                emb_nf
            )));
        }

        // 2. Generate position IDs
        let position_ids: Vec<usize> = (kv_cache.len()..kv_cache.len() + input_ids.len()).collect();

        // 3. Execute all layers
        for (layer_idx, layer_block) in self.layers.iter().enumerate() {
            hidden_states =
                layer_block.forward(&hidden_states, weights, kv_cache, &position_ids, ops)?;

            if layer_idx == 0 {
                emit_l0_checkpoint("L0.21", layer_idx, "l_out", &hidden_states);
            }

            // DIAGNOSTIC CHECKPOINT: Check after each layer (only first and last for brevity)
            if layer_idx == 0 || layer_idx == self.layers.len() - 1 {
                let (layer_nf, _, _, _, _) =
                    dump_tensor_stats(&format!("layer_{}_output", layer_idx), &hidden_states);
                if layer_nf > 0 {
                    return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                        "NaN after layer {}: {} non-finite",
                        layer_idx, layer_nf
                    )));
                }
            }
        }

        // 4. Final layer norm (output_norm.weight)
        let output_norm_weight = weights.get(&WeightId::OutputNorm).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: "output_norm".to_string(),
            }
        })?;

        let normalized = ops.rmsnorm(&hidden_states, output_norm_weight, self.spec.rms_eps)?;
        // DIAGNOSTIC CHECKPOINT: Check normalized hidden states before output projection
        let (norm_nf, _, _, _, _) = dump_tensor_stats("normalized_hidden_states", &normalized);
        if norm_nf > 0 {
            return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                "NaN in normalized hidden states BEFORE output projection: {} non-finite",
                norm_nf
            )));
        }

        // 5. Output projection to vocabulary
        let output_proj_weight = weights.get(&WeightId::OutputProj).ok_or_else(|| {
            crate::core::error::LibshimmyError::MissingTensor {
                name: "output_proj".to_string(),
            }
        })?;

        // DIAGNOSTIC CHECKPOINT: Check output.weight (Q6_K tensor) stats
        let (proj_nf, proj_min, proj_max, _, _) =
            dump_tensor_stats("output_proj_weight (Q6_K)", output_proj_weight);
        if proj_nf > 0 {
            return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                "NaN in output.weight (Q6_K dequant broken): {} non-finite",
                proj_nf
            )));
        }
        if proj_max.abs() > 1e3 || proj_min.abs() > 1e3 {
            eprintln!(
                "⚠️  output.weight has EXTREME values [{:.2e}, {:.2e}] - likely Q6_K dequant error",
                proj_min, proj_max
            );
        }

        let logits = ops.matmul(&normalized, output_proj_weight)?;

        // L0 TRACING: Emit logits checkpoint
        emit_l0_checkpoint("L0.22", 0, "logits", &logits);

        // DIAGNOSTIC CHECKPOINT: Check logits after matmul
        let (logits_nf, _, _, _, _) = dump_tensor_stats("logits (after matmul)", &logits);
        if logits_nf > 0 {
            return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                "NaN in logits after matmul: {} non-finite (but inputs were finite!)",
                logits_nf
            )));
        }

        Ok(logits)
    }

    /// Convert token IDs to embeddings
    fn embed_tokens(&self, input_ids: &[usize], embed_weight: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let hidden_size = self.spec.n_embd;

        // Validate embedding weight shape
        // GGUF format: [hidden_size, n_vocab] where ne[0]=hidden_size, ne[1]=n_vocab
        // Both shapes use the same row-major extraction: token i at data[i*hidden_size : (i+1)*hidden_size]
        let valid_shape = embed_weight.shape == vec![hidden_size, self.spec.n_vocab]
            || embed_weight.shape == vec![self.spec.n_vocab, hidden_size];

        if !valid_shape {
            return Err(crate::core::error::LibshimmyError::ShapeMismatch {
                tensor: "token_embed".to_string(),
                expected: vec![self.spec.n_vocab, hidden_size],
                got: embed_weight.shape.clone(),
            });
        }

        let actual_vocab_size = self.spec.n_vocab;

        let mut embed_data = Vec::with_capacity(seq_len * hidden_size);

        for &token_id in input_ids {
            if token_id >= actual_vocab_size {
                return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                    "Token ID {} >= vocab size {}",
                    token_id, actual_vocab_size
                )));
            }

            // Extract embedding for this token
            // GGML shape [hidden_size, n_vocab] means:
            //   ne[0] = hidden_size (columns)
            //   ne[1] = n_vocab (rows)
            // Data layout: row-major, so token i's embedding is at data[i*hidden_size : (i+1)*hidden_size]
            // This is true regardless of whether shape looks "transposed" - GGML is row-major.
            let start_idx = token_id * hidden_size;
            let end_idx = start_idx + hidden_size;
            embed_data.extend_from_slice(&embed_weight.data[start_idx..end_idx]);
        }

        let embeddings = Tensor::new(embed_data, vec![seq_len, hidden_size])?;

        // FAIL-CLOSED INVARIANT: Check for corrupted embeddings
        // Typical embedding RMS should be in reasonable range (e.g., < 100)
        // If RMS is huge, it indicates weight corruption or mis-indexing
        let sq_sum: f32 = embeddings.data.iter().map(|x| x * x).sum();
        let rms = (sq_sum / embeddings.data.len() as f32).sqrt();
        if rms > 100.0 {
            return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                "Embeddings RMS too high: {:.2e} (expected < 100.0) - weight corruption detected",
                rms
            )));
        }
        if !embeddings.data.iter().all(|x| x.is_finite()) {
            return Err(crate::core::error::LibshimmyError::Unsupported(
                "Non-finite values in embeddings - weight corruption detected".to_string(),
            ));
        }

        Ok(embeddings)
    }

    /// Get list of all required weights for this model
    pub fn required_weights(&self) -> Vec<WeightId> {
        WeightId::all_for_layers(self.spec.n_layer)
    }

    /// Validate that all required weights are present
    pub fn validate_weights(
        &self,
        weights: &std::collections::HashMap<WeightId, Tensor>,
    ) -> Result<()> {
        for weight_id in self.required_weights() {
            if !weights.contains_key(&weight_id) {
                return Err(crate::core::error::LibshimmyError::MissingTensor {
                    name: format!("{:?}", weight_id),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::Tensor;
    use std::collections::HashMap;

    fn create_tiny_spec() -> ModelSpec {
        ModelSpec {
            chat_template: None,
            n_vocab: 100,
            n_embd: 8,
            n_layer: 2,
            n_head: 2,
            n_head_kv: 1, // GQA
            ff_dim: 16,
            rms_eps: 1e-5,
            rope_base: 10000.0,
            rope_scale: 1.0,
            rope_dim: 4, // head_dim
            yarn_alpha: 0.0,
            yarn_beta: 0.0,
            n_ctx: 32,
            head_dim: 0,
            gqa_ratio: 0,
            kv_dim: 0,
            arch: crate::core::spec::ModelArch::Llama,
            file_type: crate::core::spec::GgufFileType::F32,
            model_name: "test-tiny".to_string(),
            temp_buffer_size: 0,
            kv_cache_size_per_layer: 0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            has_qk_norm: false,
        }
        .compute_derived()
    }

    fn create_dummy_weights(spec: &ModelSpec) -> HashMap<WeightId, Tensor> {
        let mut weights = HashMap::new();

        // Token embedding
        weights.insert(
            WeightId::TokenEmbed,
            Tensor::zeros(vec![spec.n_vocab, spec.n_embd]),
        );

        // Layer weights
        for layer in 0..spec.n_layer {
            let head_dim = spec.n_embd / spec.n_head;

            weights.insert(
                WeightId::AttnNorm { layer },
                Tensor::ones(vec![spec.n_embd]),
            );
            weights.insert(
                WeightId::AttnQ { layer },
                Tensor::zeros(vec![spec.n_embd, spec.n_head * head_dim]),
            );
            weights.insert(
                WeightId::AttnK { layer },
                Tensor::zeros(vec![spec.n_embd, spec.n_head_kv * head_dim]),
            );
            weights.insert(
                WeightId::AttnV { layer },
                Tensor::zeros(vec![spec.n_embd, spec.n_head_kv * head_dim]),
            );
            weights.insert(
                WeightId::AttnO { layer },
                Tensor::zeros(vec![spec.n_head * head_dim, spec.n_embd]),
            );

            weights.insert(WeightId::FfnNorm { layer }, Tensor::ones(vec![spec.n_embd]));
            weights.insert(
                WeightId::FfnGate { layer },
                Tensor::zeros(vec![spec.n_embd, spec.ff_dim]),
            );
            weights.insert(
                WeightId::FfnUp { layer },
                Tensor::zeros(vec![spec.n_embd, spec.ff_dim]),
            );
            weights.insert(
                WeightId::FfnDown { layer },
                Tensor::zeros(vec![spec.ff_dim, spec.n_embd]),
            );
        }

        // Output weights
        weights.insert(WeightId::OutputNorm, Tensor::ones(vec![spec.n_embd]));
        weights.insert(
            WeightId::OutputProj,
            Tensor::zeros(vec![spec.n_embd, spec.n_vocab]),
        );

        weights
    }

    #[test]
    fn test_llama_model_creation() {
        let spec = create_tiny_spec();
        let model = LlamaModel::from_spec(spec.clone());

        assert_eq!(model.spec.n_layer, 2);
        assert_eq!(model.layers.len(), 2);

        // Check layer indices
        assert_eq!(model.layers[0].layer_idx, 0);
        assert_eq!(model.layers[1].layer_idx, 1);
    }

    #[test]
    fn test_required_weights() {
        let spec = create_tiny_spec();
        let model = LlamaModel::from_spec(spec);

        let required = model.required_weights();

        // Should have: 1 token_embed + 2*9 layer weights + 2 output weights = 21 total
        assert_eq!(required.len(), 1 + 2 * 9 + 2); // token + 2 layers + output + output_norm

        // Check some specific weights
        assert!(required.contains(&WeightId::TokenEmbed));
        assert!(required.contains(&WeightId::AttnQ { layer: 0 }));
        assert!(required.contains(&WeightId::AttnQ { layer: 1 }));
        assert!(required.contains(&WeightId::OutputNorm));
        assert!(required.contains(&WeightId::OutputProj));
    }

    #[test]
    fn test_weight_validation() {
        let spec = create_tiny_spec();
        let model = LlamaModel::from_spec(spec.clone());

        // Complete weights should validate
        let complete_weights = create_dummy_weights(&spec);
        assert!(model.validate_weights(&complete_weights).is_ok());

        // Missing weights should fail
        let mut incomplete_weights = complete_weights.clone();
        incomplete_weights.remove(&WeightId::TokenEmbed);
        assert!(model.validate_weights(&incomplete_weights).is_err());
    }

    #[test]
    fn test_weight_validation_missing_output_norm_fails() {
        let spec = create_tiny_spec();
        let model = LlamaModel::from_spec(spec.clone());

        let mut weights = create_dummy_weights(&spec);
        weights.remove(&WeightId::OutputNorm);

        assert!(model.validate_weights(&weights).is_err());
    }

    #[test]
    fn test_token_embedding() {
        let spec = create_tiny_spec();
        let model = LlamaModel::from_spec(spec.clone());

        // Create embedding weight with known pattern
        let mut embed_data = vec![0.0; spec.n_vocab * spec.n_embd];
        for i in 0..spec.n_vocab {
            for j in 0..spec.n_embd {
                embed_data[i * spec.n_embd + j] = (i * 10 + j) as f32;
            }
        }
        let embed_weight = Tensor::new(embed_data, vec![spec.n_vocab, spec.n_embd]).unwrap();

        // Test embedding lookup
        let input_ids = vec![0, 1, 2];
        let embeddings = model.embed_tokens(&input_ids, &embed_weight).unwrap();

        assert_eq!(embeddings.shape, vec![3, spec.n_embd]);

        // Check first token embedding (token 0)
        for j in 0..spec.n_embd {
            assert_eq!(embeddings.data[j], j as f32);
        }

        // Check second token embedding (token 1)
        for j in 0..spec.n_embd {
            assert_eq!(embeddings.data[spec.n_embd + j], (10 + j) as f32);
        }
    }

    #[test]
    fn test_token_embedding_bounds() {
        let spec = create_tiny_spec();
        let model = LlamaModel::from_spec(spec.clone());

        let embed_weight = Tensor::zeros(vec![spec.n_vocab, spec.n_embd]);

        // Valid token IDs should work
        let valid_ids = vec![0, spec.n_vocab - 1];
        assert!(model.embed_tokens(&valid_ids, &embed_weight).is_ok());

        // Out of bounds token ID should fail
        let invalid_ids = vec![spec.n_vocab];
        assert!(model.embed_tokens(&invalid_ids, &embed_weight).is_err());
    }
}
