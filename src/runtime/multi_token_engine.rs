use crate::control::{ControlDecision, InferenceControl, InferenceEvent, NoopControl};
use crate::core::weight_id::WeightId;
use crate::core::{error::Result, spec::ModelSpec, tensor::Tensor};
use crate::family::llama::LlamaModel;
use crate::fixtures::PromptFixture;
use crate::runtime::{engine::Engine, sampling::Sampler};
use crate::validation::{DeterminismProof, LogitInfo};
use std::collections::HashMap;

/// Multi-token engine for V2.0 Slice 01 - Deterministic 16-token decode
///
/// Focuses on greedy decode with deterministic behavior and validation
pub struct MultiTokenEngine {
    engine: Engine,
    sampler: Sampler,
}

impl MultiTokenEngine {
    /// Create new multi-token engine with greedy sampling
    pub fn new(llama_model: LlamaModel) -> Self {
        let engine = Engine::new(llama_model);
        let sampler = Sampler::greedy(); // GreedyOnly for V2

        Self { engine, sampler }
    }

    /// Create new multi-token engine from model spec
    pub fn from_spec(spec: ModelSpec) -> Self {
        let llama_model = LlamaModel::from_spec(spec);
        Self::new(llama_model)
    }

    /// Reset engine state for new sequence
    pub fn reset(&mut self) {
        self.engine.reset();
    }

    /// Decode sequence of exactly 16 tokens using greedy selection
    ///
    /// This is the core method for V2.0 Slice 01 requirements
    pub fn decode_sequence(
        &mut self,
        prompt_tokens: &[u32],
        weights: &HashMap<WeightId, Tensor>,
    ) -> Result<DecodeSequenceResult> {
        let control = NoopControl;
        self.decode_sequence_with_control(prompt_tokens, weights, &control, None)
    }

    /// Decode sequence using greedy selection with an injected control hook.
    ///
    /// The hook is invoked after selecting the candidate token, before updating KV.
    pub fn decode_sequence_with_control<C: InferenceControl>(
        &mut self,
        prompt_tokens: &[u32],
        weights: &HashMap<WeightId, Tensor>,
        control: &C,
        decoder: Option<&dyn crate::control::TokenDecoder>,
    ) -> Result<DecodeSequenceResult> {
        // Reset state for deterministic behavior
        self.reset();

        // Convert u32 tokens to usize for engine
        let prompt_ids: Vec<usize> = prompt_tokens.iter().map(|&t| t as usize).collect();

        // Track per-step logits for validation
        let mut per_step_logits = Vec::new();
        let mut generated_tokens = Vec::new();

        // Prefill with prompt
        let mut current_logits = self.engine.prefill(&prompt_ids, weights)?;

        // We mirror Engine::generate hook semantics: event.tokens excludes candidate token.
        let mut sequence_so_far: Vec<usize> = prompt_ids.clone();
        let mut text_buffer = String::new();

        // Generate exactly 16 tokens
        for step in 0..16 {
            // Validate logits are finite
            let finite = current_logits.data.iter().all(|&x| x.is_finite());
            if !finite {
                return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                    "Non-finite logits at step {}",
                    step
                )));
            }

            // Greedy sampling: select token with highest logit
            let next_token = self.sampler.sample(&current_logits)?;

            // Control hook
            let event = InferenceEvent {
                tokens: &sequence_so_far,
                candidate_token: next_token,
                step,
                kv: self.engine.kv_cache.snapshot(),
                text: &text_buffer,
            };
            match control.intervene(&event) {
                ControlDecision::Allow => {
                    if let Some(dec) = decoder {
                        text_buffer.push_str(&dec.decode_single(next_token));
                    }
                }
                ControlDecision::EarlyExit => break,
                ControlDecision::BlockAndTerminate(reason) => {
                    return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                        "Blocked by control hook at step {}: {}",
                        step, reason
                    )));
                }
            }

            // Record logit info for validation
            let max_logit_value = current_logits
                .data
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied();

            per_step_logits.push(LogitInfo {
                step,
                max_logit_index: next_token as u32,
                max_logit_value,
                finite,
            });

            generated_tokens.push(next_token as u32);

            // Check if we can continue (not at capacity)
            if self.engine.is_full() {
                return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                    "Engine at capacity after {} tokens",
                    step + 1
                )));
            }

            sequence_so_far.push(next_token);

            // Decode next token to update KV cache
            current_logits = self.engine.decode(next_token, weights)?;
        }

        Ok(DecodeSequenceResult {
            prompt_tokens: prompt_tokens.to_vec(),
            generated_tokens,
            per_step_logits,
        })
    }

    /// Test determinism by running decode_sequence twice and comparing results
    pub fn test_determinism(
        &mut self,
        prompt_tokens: &[u32],
        weights: &HashMap<WeightId, Tensor>,
    ) -> Result<DeterminismTestResult> {
        // First run
        let result1 = self.decode_sequence(prompt_tokens, weights)?;

        // Second run
        let result2 = self.decode_sequence(prompt_tokens, weights)?;

        // Compare results
        let identical = result1.generated_tokens == result2.generated_tokens;

        let determinism_proof = DeterminismProof {
            run1_tokens: result1.generated_tokens.clone(),
            run2_tokens: result2.generated_tokens.clone(),
            identical,
        };

        Ok(DeterminismTestResult {
            first_run: result1,
            determinism_proof,
        })
    }

    /// Decode sequence from a prompt fixture
    pub fn decode_from_fixture(
        &mut self,
        fixture: &PromptFixture,
        weights: &HashMap<WeightId, Tensor>,
    ) -> Result<DecodeSequenceResult> {
        self.decode_sequence(&fixture.token_ids, weights)
    }

    /// Get current KV cache length (for validation)
    pub fn current_cache_len(&self) -> usize {
        self.engine.current_len()
    }

    /// Check if engine is at capacity
    pub fn is_at_capacity(&self) -> bool {
        self.engine.is_full()
    }
}

/// Result of decode_sequence operation
#[derive(Debug, Clone)]
pub struct DecodeSequenceResult {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub per_step_logits: Vec<LogitInfo>,
}

impl DecodeSequenceResult {
    /// Get total sequence (prompt + generated)
    pub fn full_sequence(&self) -> Vec<u32> {
        let mut full = self.prompt_tokens.clone();
        full.extend(&self.generated_tokens);
        full
    }

    /// Validate that exactly 16 tokens were generated
    pub fn validate_token_count(&self) -> Result<()> {
        if self.generated_tokens.len() != 16 {
            return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                "Expected 16 tokens, got {}",
                self.generated_tokens.len()
            )));
        }
        Ok(())
    }

    /// Validate that all logits were finite
    pub fn validate_finite_logits(&self) -> Result<()> {
        for logit_info in &self.per_step_logits {
            if !logit_info.finite {
                return Err(crate::core::error::LibshimmyError::Unsupported(format!(
                    "Non-finite logits at step {}",
                    logit_info.step
                )));
            }
        }
        Ok(())
    }
}

/// Result of determinism test
#[derive(Debug, Clone)]
pub struct DeterminismTestResult {
    pub first_run: DecodeSequenceResult,
    pub determinism_proof: DeterminismProof,
}

impl DeterminismTestResult {
    /// Check if the test passed (identical results)
    pub fn is_deterministic(&self) -> bool {
        self.determinism_proof.identical
    }

    /// Get the first divergence step if not deterministic
    pub fn first_divergence_step(&self) -> Option<usize> {
        if self.determinism_proof.identical {
            return None;
        }

        for (i, (a, b)) in self
            .determinism_proof
            .run1_tokens
            .iter()
            .zip(&self.determinism_proof.run2_tokens)
            .enumerate()
        {
            if a != b {
                return Some(i);
            }
        }

        // Different lengths
        if self.determinism_proof.run1_tokens.len() != self.determinism_proof.run2_tokens.len() {
            return Some(
                self.determinism_proof
                    .run1_tokens
                    .len()
                    .min(self.determinism_proof.run2_tokens.len()),
            );
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::spec::ModelSpec;
    use crate::fixtures::PromptFixture;
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
            n_ctx: 32, // Larger context for 16-token generation
            head_dim: 0,
            gqa_ratio: 0,
            kv_dim: 0,
            arch: crate::core::spec::ModelArch::Llama,
            file_type: crate::core::spec::GgufFileType::F32,
            model_name: "test-toy".to_string(),
            temp_buffer_size: 0,
            kv_cache_size_per_layer: 0,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
        }
        .compute_derived()
    }

    fn create_toy_weights(spec: &ModelSpec) -> HashMap<WeightId, Tensor> {
        let mut weights = HashMap::new();

        // Token embedding
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

        // Layer 0 weights
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

    fn create_test_fixture() -> PromptFixture {
        PromptFixture {
            prompt_name: "test".to_string(),
            prompt_text: "Hello".to_string(),
            token_ids: vec![1, 2, 3],
            description: "Test fixture".to_string(),
        }
    }

    #[test]
    fn test_multi_token_engine_creation() {
        let spec = create_toy_spec();
        let engine = MultiTokenEngine::from_spec(spec);

        assert_eq!(engine.current_cache_len(), 0);
        assert!(!engine.is_at_capacity());
    }

    #[test]
    fn test_decode_sequence_16_tokens() {
        let spec = create_toy_spec();
        let mut engine = MultiTokenEngine::from_spec(spec.clone());
        let weights = create_toy_weights(&spec);

        let prompt_tokens = vec![1, 2, 3];
        let result = engine.decode_sequence(&prompt_tokens, &weights).unwrap();

        // Should generate exactly 16 tokens
        assert_eq!(result.generated_tokens.len(), 16);
        assert_eq!(result.per_step_logits.len(), 16);

        // Validate token count
        assert!(result.validate_token_count().is_ok());

        // Validate finite logits
        assert!(result.validate_finite_logits().is_ok());

        // All generated tokens should be valid
        for &token in &result.generated_tokens {
            assert!(token < spec.n_vocab as u32);
        }
    }

    #[test]
    fn test_determinism() {
        let spec = create_toy_spec();
        let mut engine = MultiTokenEngine::from_spec(spec.clone());
        let weights = create_toy_weights(&spec);

        let prompt_tokens = vec![1, 2];
        let result = engine.test_determinism(&prompt_tokens, &weights).unwrap();

        // Should be deterministic (greedy sampling)
        assert!(result.is_deterministic());
        assert_eq!(result.first_divergence_step(), None);

        // Both runs should have same length
        assert_eq!(result.determinism_proof.run1_tokens.len(), 16);
        assert_eq!(result.determinism_proof.run2_tokens.len(), 16);
    }

    #[test]
    fn test_decode_from_fixture() {
        let spec = create_toy_spec();
        let mut engine = MultiTokenEngine::from_spec(spec.clone());
        let weights = create_toy_weights(&spec);

        let fixture = create_test_fixture();
        let result = engine.decode_from_fixture(&fixture, &weights).unwrap();

        // Should use fixture tokens as prompt
        assert_eq!(result.prompt_tokens, fixture.token_ids);
        assert_eq!(result.generated_tokens.len(), 16);
    }

    #[test]
    fn test_full_sequence() {
        let spec = create_toy_spec();
        let mut engine = MultiTokenEngine::from_spec(spec.clone());
        let weights = create_toy_weights(&spec);

        let prompt_tokens = vec![1, 2];
        let result = engine.decode_sequence(&prompt_tokens, &weights).unwrap();

        let full_seq = result.full_sequence();

        // Should be prompt + 16 generated tokens
        assert_eq!(full_seq.len(), 2 + 16);
        assert_eq!(&full_seq[..2], &prompt_tokens);
        assert_eq!(&full_seq[2..], &result.generated_tokens);
    }

    #[test]
    fn test_reset_behavior() {
        let spec = create_toy_spec();
        let mut engine = MultiTokenEngine::from_spec(spec.clone());
        let weights = create_toy_weights(&spec);

        // Generate some tokens
        let prompt_tokens = vec![1, 2, 3];
        engine.decode_sequence(&prompt_tokens, &weights).unwrap();

        // Cache should have some length
        assert!(engine.current_cache_len() > 0);

        // Reset should clear
        engine.reset();
        assert_eq!(engine.current_cache_len(), 0);
    }

    #[test]
    fn test_logit_info_validation() {
        let spec = create_toy_spec();
        let mut engine = MultiTokenEngine::from_spec(spec.clone());
        let weights = create_toy_weights(&spec);

        let prompt_tokens = vec![1];
        let result = engine.decode_sequence(&prompt_tokens, &weights).unwrap();

        // Check logit info structure
        for (i, logit_info) in result.per_step_logits.iter().enumerate() {
            assert_eq!(logit_info.step, i);
            assert!(logit_info.finite);
            assert!(logit_info.max_logit_index < spec.n_vocab as u32);
            assert!(logit_info.max_logit_value.is_some());
        }
    }
}
