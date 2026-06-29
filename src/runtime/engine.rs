//! Inference engine orchestrating prefill/decode phases.
//!
//! Manages model execution and KV cache state for autoregressive generation.

use crate::control::{ControlDecision, InferenceControl, InferenceEvent, NoopControl};
use crate::core::weight_id::WeightId;
use crate::core::{error::Result, tensor::Tensor};
use crate::family::ModelFamily;
use crate::ops::dispatch::OpDispatcher;
use crate::runtime::kvcache::KvCache;
use std::collections::HashMap;

/// Transformer inference engine with KV cache.
pub struct Engine {
    pub model: Box<dyn ModelFamily>,
    pub kv_cache: KvCache,
    pub ops: OpDispatcher,
}

impl Engine {
    /// Create new engine with model and cache
    pub fn new(model: Box<dyn ModelFamily>) -> Self {
        let spec = model.spec();
        let kv_cache = KvCache::new(
            spec.n_ctx,
            spec.n_layer,
            spec.n_head_kv,
            spec.n_embd / spec.n_head, // head_dim
        );

        Self {
            model,
            kv_cache,
            ops: OpDispatcher::new(),
        }
    }

    /// Reset engine state (clear KV cache)
    pub fn reset(&mut self) {
        self.kv_cache.reset();
    }

    /// Get current sequence length
    pub fn current_len(&self) -> usize {
        self.kv_cache.len()
    }

    /// Check if engine is at capacity
    pub fn is_full(&self) -> bool {
        self.kv_cache.is_full()
    }

    /// Prefill phase: process multiple tokens at once
    ///
    /// Input: token IDs for the prompt
    /// Returns: logits for the last token [n_vocab]
    pub fn prefill(
        &mut self,
        input_ids: &[usize],
        weights: &HashMap<WeightId, Tensor>,
    ) -> Result<Tensor> {
        if input_ids.is_empty() {
            return Err(crate::core::error::LibshimmyError::Unsupported(
                "Cannot prefill with empty input".to_string(),
            ));
        }

        if self.kv_cache.len() + input_ids.len() > self.kv_cache.max_seq_len {
            return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                "Prefill would exceed max sequence length: {} + {} > {}",
                self.kv_cache.len(),
                input_ids.len(),
                self.kv_cache.max_seq_len
            )));
        }

        // Run model forward pass
        let logits = self
            .model
            .forward(input_ids, weights, &mut self.kv_cache, &self.ops)?;

        // Update KV cache length
        self.kv_cache.complete_prefill(input_ids.len())?;

        // Extract logits for last token
        self.extract_last_token_logits(&logits, input_ids.len())
    }

    /// Decode phase: process single token
    ///
    /// Input: single token ID
    /// Returns: logits for next token [n_vocab]
    pub fn decode(
        &mut self,
        token_id: usize,
        weights: &HashMap<WeightId, Tensor>,
    ) -> Result<Tensor> {
        if self.is_full() {
            return Err(crate::core::error::LibshimmyError::Unsupported(
                "Cannot decode: sequence is at maximum length".to_string(),
            ));
        }

        // INVARIANT: Record cache length before decode
        let cache_len_before = self.kv_cache.len();

        // Run model forward pass with single token
        let input_ids = vec![token_id];
        let logits = self
            .model
            .forward(&input_ids, weights, &mut self.kv_cache, &self.ops)?;

        // Update KV cache length
        self.kv_cache.complete_decode()?;

        // INVARIANT CHECK: cache_len increments exactly once per decode call
        let cache_len_after = self.kv_cache.len();
        assert_eq!(
            cache_len_after,
            cache_len_before + 1,
            "INVARIANT VIOLATION: cache_len did not increment by exactly 1. Before: {}, After: {}",
            cache_len_before,
            cache_len_after
        );

        // Extract logits (should be [1, n_vocab])
        self.extract_last_token_logits(&logits, 1)
    }

    /// Generate sequence using greedy decoding
    ///
    /// Input: prompt token IDs and maximum new tokens to generate
    /// Returns: complete sequence (prompt + generated tokens)
    pub fn generate(
        &mut self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        weights: &HashMap<WeightId, Tensor>,
    ) -> Result<Vec<usize>> {
        let control = NoopControl;
        self.generate_with_control(prompt_ids, max_new_tokens, weights, &control, None)
    }

    /// Generate sequence using greedy decoding with an injected control hook.
    ///
    /// The hook is invoked after selecting the candidate token, before appending/decoding.
    pub fn generate_with_control<C: InferenceControl>(
        &mut self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        weights: &HashMap<WeightId, Tensor>,
        control: &C,
        decoder: Option<&dyn crate::control::TokenDecoder>,
    ) -> Result<Vec<usize>> {
        // Reset state
        self.reset();

        // Prefill with prompt
        let mut last_logits = self.prefill(prompt_ids, weights)?;

        // Start with prompt tokens
        let mut generated_ids = prompt_ids.to_vec();
        let mut text_buffer = String::new();

        // Generate tokens one by one
        for step in 0..max_new_tokens {
            if self.is_full() {
                break;
            }

            // Greedy sampling: select token with highest logit
            let next_token = self.greedy_sample(&last_logits)?;

            // Control hook (observational): decide whether to accept this token.
            let event = InferenceEvent {
                tokens: &generated_ids,
                candidate_token: next_token,
                step,
                kv: self.kv_cache.snapshot(),
                text: &text_buffer,
            };

            match control.intervene(&event) {
                ControlDecision::Allow => {
                    // If allowed and decoder available, accumulate text
                    if let Some(dec) = decoder {
                        text_buffer.push_str(&dec.decode_single(next_token));
                    }
                }
                ControlDecision::ForceToken(forced) => {
                    if let Some(dec) = decoder {
                        text_buffer.push_str(&dec.decode_single(forced));
                    }
                    generated_ids.push(forced);
                    last_logits = self.decode(forced, weights)?;
                    continue;
                }
                ControlDecision::EarlyExit => break,
                ControlDecision::BlockAndTerminate(reason) => {
                    return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                        "Blocked by control hook at step {}: {}",
                        step, reason
                    )));
                }
            }

            generated_ids.push(next_token);

            // Decode next token
            last_logits = self.decode(next_token, weights)?;
        }

        Ok(generated_ids)
    }

    /// Simple greedy sampling: select token with highest logit
    fn greedy_sample(&self, logits: &Tensor) -> Result<usize> {
        if logits.ndim() != 1 {
            return Err(crate::core::error::LibshimmyError::ShapeMismatch {
                tensor: "greedy_sample_logits".to_string(),
                expected: vec![1],
                got: vec![logits.ndim()],
            });
        }

        let mut max_idx = 0;
        let mut max_val = logits.data[0];

        for (i, &val) in logits.data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx)
    }

    /// Extract logits for the last token from model output
    fn extract_last_token_logits(&self, logits: &Tensor, seq_len: usize) -> Result<Tensor> {
        match logits.ndim() {
            2 => {
                // Shape: [seq_len, n_vocab]
                if logits.shape[0] != seq_len {
                    return Err(crate::core::error::LibshimmyError::ShapeMismatch {
                        tensor: "logits_seq_len".to_string(),
                        expected: vec![seq_len, logits.shape[1]],
                        got: logits.shape.clone(),
                    });
                }

                let n_vocab = logits.shape[1];
                let last_token_start = (seq_len - 1) * n_vocab;
                let last_token_end = last_token_start + n_vocab;

                let last_logits = logits.data[last_token_start..last_token_end].to_vec();
                Tensor::new(last_logits, vec![n_vocab])
            }
            3 => {
                crate::ensure!(
                    false,
                    "3D logits are unsupported until batching is implemented; got logits shape {:?}",
                    logits.shape
                );
                unreachable!();
            }
            _ => Err(crate::core::error::LibshimmyError::ShapeMismatch {
                tensor: "logits_ndim".to_string(),
                expected: vec![2],
                got: vec![logits.ndim()],
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::LibshimmyError;
    use crate::core::spec::ModelSpec;
    use crate::core::tensor::Tensor;
    use crate::family::llama::LlamaModel;
    use std::collections::HashMap;

    fn create_toy_spec() -> ModelSpec {
        ModelSpec {
            n_vocab: 10,
            n_embd: 4,
            n_layer: 1,
            n_head: 1,
            n_head_kv: 1,
            ff_dim: 8,
            rms_eps: 1e-5,
            rope_base: 10000.0,
            rope_scale: 1.0,
            rope_dim: 4,
            yarn_alpha: 0.0,
            yarn_beta: 0.0,
            n_ctx: 8,
            head_dim: 0,
            gqa_ratio: 0,
            kv_dim: 0,
            arch: crate::core::spec::ModelArch::Llama,
            file_type: crate::core::spec::GgufFileType::F32,
            model_name: "test-toy".to_string(),
            chat_template: None,
            temp_buffer_size: 0,
            kv_cache_size_per_layer: 0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            has_qk_norm: false,
            post_norm_enabled: false,
        }
        .compute_derived()
    }

    fn create_toy_weights(spec: &ModelSpec) -> HashMap<WeightId, Tensor> {
        let mut weights = HashMap::new();

        // Token embedding: small values to avoid overflow
        let mut embed_data = vec![0.1; spec.n_vocab * spec.n_embd];
        for i in 0..spec.n_vocab {
            for j in 0..spec.n_embd {
                embed_data[i * spec.n_embd + j] = 0.1 * (i + j) as f32;
            }
        }
        weights.insert(
            WeightId::TokenEmbed,
            Tensor::new(embed_data, vec![spec.n_vocab, spec.n_embd]).unwrap(),
        );

        // Layer 0 weights - small values
        let head_dim = spec.n_embd / spec.n_head;
        weights.insert(
            WeightId::AttnNorm { layer: 0 },
            Tensor::ones(vec![spec.n_embd]),
        );
        weights.insert(
            WeightId::AttnQ { layer: 0 },
            Tensor::new(
                vec![0.1; spec.n_embd * spec.n_head * head_dim],
                vec![spec.n_embd, spec.n_head * head_dim],
            )
            .unwrap(),
        );
        weights.insert(
            WeightId::AttnK { layer: 0 },
            Tensor::new(
                vec![0.1; spec.n_embd * spec.n_head_kv * head_dim],
                vec![spec.n_embd, spec.n_head_kv * head_dim],
            )
            .unwrap(),
        );
        weights.insert(
            WeightId::AttnV { layer: 0 },
            Tensor::new(
                vec![0.1; spec.n_embd * spec.n_head_kv * head_dim],
                vec![spec.n_embd, spec.n_head_kv * head_dim],
            )
            .unwrap(),
        );
        weights.insert(
            WeightId::AttnO { layer: 0 },
            Tensor::new(
                vec![0.1; spec.n_head * head_dim * spec.n_embd],
                vec![spec.n_head * head_dim, spec.n_embd],
            )
            .unwrap(),
        );

        weights.insert(
            WeightId::FfnNorm { layer: 0 },
            Tensor::ones(vec![spec.n_embd]),
        );
        weights.insert(
            WeightId::FfnGate { layer: 0 },
            Tensor::new(
                vec![0.1; spec.n_embd * spec.ff_dim],
                vec![spec.n_embd, spec.ff_dim],
            )
            .unwrap(),
        );
        weights.insert(
            WeightId::FfnUp { layer: 0 },
            Tensor::new(
                vec![0.1; spec.n_embd * spec.ff_dim],
                vec![spec.n_embd, spec.ff_dim],
            )
            .unwrap(),
        );
        weights.insert(
            WeightId::FfnDown { layer: 0 },
            Tensor::new(
                vec![0.1; spec.ff_dim * spec.n_embd],
                vec![spec.ff_dim, spec.n_embd],
            )
            .unwrap(),
        );

        // Output weights
        weights.insert(WeightId::OutputNorm, Tensor::ones(vec![spec.n_embd]));
        weights.insert(
            WeightId::OutputProj,
            Tensor::new(
                vec![0.1; spec.n_embd * spec.n_vocab],
                vec![spec.n_embd, spec.n_vocab],
            )
            .unwrap(),
        );

        weights
    }

    #[test]
    fn test_extract_last_token_logits_rejects_3d_logits() {
        let model = Box::new(LlamaModel::from_spec(create_toy_spec()));
        let engine = Engine::new(model);

        let logits = Tensor::zeros(vec![1, 2, 10]);
        let err = engine.extract_last_token_logits(&logits, 2).unwrap_err();

        match err {
            LibshimmyError::InvariantViolation { .. } => {}
            other => panic!("expected InvariantViolation, got {other:?}"),
        }
    }

    #[test]
    fn test_engine_creation() {
        let spec = create_toy_spec();
        let model = Box::new(LlamaModel::from_spec(spec.clone()));
        let engine = Engine::new(model);

        assert_eq!(engine.current_len(), 0);
        assert!(!engine.is_full());
        assert_eq!(engine.kv_cache.max_seq_len, spec.n_ctx);
    }

    #[test]
    fn test_engine_prefill() {
        let spec = create_toy_spec();
        let model = Box::new(LlamaModel::from_spec(spec.clone()));
        let mut engine = Engine::new(model);
        let weights = create_toy_weights(&spec);

        let prompt_ids = vec![0, 1, 2];
        let logits = engine.prefill(&prompt_ids, &weights).unwrap();

        // Should return logits for vocabulary
        assert_eq!(logits.shape, vec![spec.n_vocab]);
        assert_eq!(engine.current_len(), 3);

        // Should not contain NaN
        for &val in &logits.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_engine_decode() {
        let spec = create_toy_spec();
        let model = Box::new(LlamaModel::from_spec(spec.clone()));
        let mut engine = Engine::new(model);
        let weights = create_toy_weights(&spec);

        // Prefill first
        let prompt_ids = vec![0, 1];
        let _ = engine.prefill(&prompt_ids, &weights).unwrap();

        // Then decode
        let logits = engine.decode(2, &weights).unwrap();

        assert_eq!(logits.shape, vec![spec.n_vocab]);
        assert_eq!(engine.current_len(), 3);

        // Should not contain NaN
        for &val in &logits.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_engine_generate() {
        let spec = create_toy_spec();
        let model = Box::new(LlamaModel::from_spec(spec.clone()));
        let mut engine = Engine::new(model);
        let weights = create_toy_weights(&spec);

        let prompt_ids = vec![0, 1];
        let generated = engine.generate(&prompt_ids, 2, &weights).unwrap();

        // Should have prompt + 2 new tokens
        assert_eq!(generated.len(), 4);
        assert_eq!(&generated[..2], &prompt_ids);

        // All token IDs should be valid
        for &token_id in &generated {
            assert!(token_id < spec.n_vocab);
        }
    }

    #[test]
    fn test_engine_capacity_limits() {
        let spec = create_toy_spec();
        let model = Box::new(LlamaModel::from_spec(spec.clone()));
        let mut engine = Engine::new(model);
        let weights = create_toy_weights(&spec);

        // Fill to capacity (n_ctx = 8)
        let long_prompt = vec![0; spec.n_ctx];
        let _ = engine.prefill(&long_prompt, &weights).unwrap();

        assert!(engine.is_full());

        // Should fail to decode more
        let result = engine.decode(1, &weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_reset() {
        let spec = create_toy_spec();
        let model = Box::new(LlamaModel::from_spec(spec.clone()));
        let mut engine = Engine::new(model);
        let weights = create_toy_weights(&spec);

        // Add some tokens
        let _ = engine.prefill(&[0, 1, 2], &weights).unwrap();
        assert_eq!(engine.current_len(), 3);

        // Reset should clear
        engine.reset();
        assert_eq!(engine.current_len(), 0);
        assert!(!engine.is_full());
    }

    #[test]
    fn test_greedy_sampling() {
        let spec = create_toy_spec();
        let model = Box::new(LlamaModel::from_spec(spec.clone()));
        let engine = Engine::new(model);

        // Create logits with clear maximum
        let logits = Tensor::new(vec![0.1, 0.5, 0.2, 0.8, 0.3], vec![5]).unwrap();
        let selected = engine.greedy_sample(&logits).unwrap();

        // Should select index 3 (highest value 0.8)
        assert_eq!(selected, 3);
    }
}
