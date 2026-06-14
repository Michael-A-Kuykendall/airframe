//! Inference Saturation Fabric (ISF)
//!
//! Replaces the imperative generate() loop in GpuRuntime with a D0 reactive
//! graph. The loop disappears — replaced by fact assertion + run_to_fixpoint().
//!
//! Architecture:
//! - Tier 1 facts: PromptToken, DecodeStep — asserted by caller
//! - Tier 2 facts: EmbeddingReady, PrefillBatchReady, PrefillComplete — derived by rules
//! - Tier 3 consequents: DecodeLogitsReady, GenerationHalt — drive external actions
//!
//! FSE invariant: ∂runtime / ∂rules ≈ 0 for shared selectors.
//! Adding TDR monitoring, vault logging, streaming = register a rule, zero cost.
//!
//! Patent Notice: Implements FSE + D0 Saturation Fabric architecture.
//! Pending patent by Michael A. Kuykendall. All rights reserved.

use crate::facts::{
    alpha_key_of, HaltReason, InferenceFact,
    KEY_DECODE_STEP, KEY_EMBEDDING_READY, KEY_PREFILL_BATCH_READY,
    KEY_PREFILL_COMPLETE, KEY_PROMPT_TOKEN,
};
use d0_engine::{AlphaKey, ClosureProgram, FactStore, RunBudget, SaturationFabric};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

/// Re-export RunBudget::default() so callers don't need d0_engine directly.
pub fn d0_run_budget() -> RunBudget {
    RunBudget::default()
}

/// Output from one generate() call.
#[derive(Debug)]
pub struct GenerateOutput {
    pub text: String,
    pub tokens_generated: usize,
    pub halt_reason: HaltReason,
}

/// Shared mutable state threaded through rule closures via Arc<Mutex<>>.
/// This is the "working memory" of the ISF session — rules read and write it.
pub struct ISFState {
    /// Collected embeddings: position → flat f32 vec (dim elements)
    pub embeddings: Vec<Option<Vec<f32>>>,
    /// Number of prompt tokens expected
    pub prompt_len: u32,
    /// Generated text so far
    pub generated_text: String,
    /// Logits from last forward pass — read by decode step rules
    pub logits: Vec<f32>,
    /// Halt flag — set by rules when EOS or max_tokens reached
    pub halt: Option<HaltReason>,
    /// Step counter for decode
    pub decode_step: u32,
    /// Max tokens allowed
    pub max_tokens: u32,
    /// EOS token ID
    pub eos_token: u32,
    /// Extra stop token IDs
    pub extra_stop_ids: Vec<u32>,
    /// Streaming callback — called with each decoded token piece
    pub on_token: Option<Box<dyn FnMut(&str) + Send>>,
}

impl ISFState {
    pub fn new(
        prompt_len: u32,
        max_tokens: u32,
        eos_token: u32,
        extra_stop_ids: Vec<u32>,
        on_token: Option<Box<dyn FnMut(&str) + Send>>,
    ) -> Self {
        Self {
            embeddings: vec![None; prompt_len as usize],
            prompt_len,
            generated_text: String::new(),
            logits: Vec::new(),
            halt: None,
            decode_step: 0,
            max_tokens,
            eos_token,
            extra_stop_ids,
            on_token,
        }
    }

    pub fn all_embeddings_ready(&self) -> bool {
        self.embeddings.iter().all(|e| e.is_some())
    }

    pub fn batched_embeddings(&self) -> Vec<f32> {
        self.embeddings
            .iter()
            .flat_map(|e| e.as_ref().unwrap().iter().cloned())
            .collect()
    }
}

/// The Inference Saturation Fabric.
///
/// Owns the SaturationFabric<InferenceFact> and shared ISFState.
/// Rules are registered at construction time; generate() asserts facts and
/// runs to fixpoint.
pub struct InferenceSaturationFabric {
    pub fabric: SaturationFabric<InferenceFact>,
    pub state: Arc<Mutex<ISFState>>,
}

impl InferenceSaturationFabric {
    /// Create a new ISF session.
    ///
    /// `dequant_fn`: closure that takes (token_id: u32, dim: u32) and returns
    ///   the embedding as Vec<f32>. Called once per unique token_id.
    ///
    /// `prefill_fn`: closure that takes (batched_embd: Vec<f32>, prompt_len: u32)
    ///   and returns (hidden: Vec<f32>, logits: Vec<f32>).
    ///
    /// `forward_fn`: closure that takes (token_embd: Vec<f32>, current_pos: u32)
    ///   and returns (hidden: Vec<f32>, logits: Vec<f32>).
    ///
    /// `sample_fn`: closure that takes (logits: &mut Vec<f32>) and returns token_id: u32.
    ///
    /// `decode_fn`: closure that takes (token_id: u32) and returns the text piece: String.
    pub fn new(
        state: Arc<Mutex<ISFState>>,
        dequant_fn: Arc<dyn Fn(u32, u32) -> Vec<f32> + Send + Sync>,
        prefill_fn: Arc<dyn Fn(Vec<f32>, u32) -> (Vec<f32>, Vec<f32>) + Send + Sync>,
        forward_fn: Arc<dyn Fn(Vec<f32>, u32) -> (Vec<f32>, Vec<f32>) + Send + Sync>,
        sample_fn: Arc<dyn Fn(&mut Vec<f32>) -> u32 + Send + Sync>,
        decode_fn: Arc<dyn Fn(u32) -> String + Send + Sync>,
        kv_increment_fn: Arc<dyn Fn() + Send + Sync>,
        dim: u32,
    ) -> Self {
        let mut program = ClosureProgram::new();

        // ── Rule 1: PromptToken → EmbeddingReady ─────────────────────────
        // Selector: PromptToken. Deduplicated by token_id via FactStore dedup.
        // Each unique token_id triggers one GPU dequant; result broadcast to
        // all positions sharing that token_id.
        {
            let state_ref = state.clone();
            let dequant = dequant_fn.clone();
            program.register(AlphaKey(KEY_PROMPT_TOKEN), move |fact, _store| {
                if let InferenceFact::PromptToken { position, token_id } = fact {
                    let embedding = dequant(*token_id, dim);
                    {
                        let mut s = state_ref.lock().unwrap();
                        let pos = *position as usize;
                        if pos < s.embeddings.len() {
                            s.embeddings[pos] = Some(embedding);
                        }
                    }
                    vec![InferenceFact::EmbeddingReady {
                        position: *position,
                        token_id: *token_id,
                    }]
                } else {
                    vec![]
                }
            });
        }

        // ── Rule 2: EmbeddingReady → PrefillBatchReady (when all collected) ─
        // Selector: EmbeddingReady. Each fires; last one triggers batch.
        {
            let state_ref = state.clone();
            program.register(AlphaKey(KEY_EMBEDDING_READY), move |_fact, _store| {
                let s = state_ref.lock().unwrap();
                if s.all_embeddings_ready() {
                    vec![InferenceFact::PrefillBatchReady {
                        token_count: s.prompt_len,
                    }]
                } else {
                    vec![]
                }
            });
        }

        // ── Rule 3: PrefillBatchReady → PrefillComplete ───────────────────
        // Fires the actual GPU prefill dispatch.
        {
            let state_ref = state.clone();
            let prefill = prefill_fn.clone();
            let kv_inc = kv_increment_fn.clone();
            program.register(AlphaKey(KEY_PREFILL_BATCH_READY), move |fact, _store| {
                if let InferenceFact::PrefillBatchReady { token_count } = fact {
                    let batched = {
                        let s = state_ref.lock().unwrap();
                        s.batched_embeddings()
                    };
                    let (_hidden, logits) = prefill(batched, *token_count);
                    // Increment KV cache once per prompt token
                    for _ in 0..*token_count {
                        kv_inc();
                    }
                    {
                        let mut s = state_ref.lock().unwrap();
                        s.logits = logits;
                    }
                    vec![InferenceFact::PrefillComplete { position: *token_count }]
                } else {
                    vec![]
                }
            });
        }

        // ── Rule 4: PrefillComplete → DecodeStep { step=0 } ──────────────
        // Bridges prefill to decode. Sample first token from prefill logits.
        {
            let state_ref = state.clone();
            let sample = sample_fn.clone();
            let decode = decode_fn.clone();
            program.register(AlphaKey(KEY_PREFILL_COMPLETE), move |_fact, _store| {
                let (token_id, halt) = {
                    let mut s = state_ref.lock().unwrap();
                    let token_id = sample(&mut s.logits);
                    let halt = token_id == s.eos_token
                        || s.extra_stop_ids.contains(&token_id);
                    (token_id, halt)
                };

                if halt {
                    return vec![InferenceFact::GenerationHalt {
                        reason: HaltReason::EosToken,
                    }];
                }

                let piece = decode(token_id);
                {
                    let mut s = state_ref.lock().unwrap();
                    s.generated_text.push_str(&piece);
                    if let Some(cb) = s.on_token.as_mut() {
                        cb(&piece);
                    }
                    s.decode_step = 1;
                }

                vec![InferenceFact::DecodeStep {
                    step: 0,
                    token_id,
                }]
            });
        }

        // ── Rule 5: DecodeStep → next DecodeStep (or Halt) ───────────────
        // The reactive inversion: each decode token self-asserts the next step.
        // step field guarantees uniqueness → FactStore dedup won't block it.
        {
            let state_ref = state.clone();
            let forward = forward_fn.clone();
            let sample = sample_fn.clone();
            let decode = decode_fn.clone();
            let kv_inc = kv_increment_fn.clone();
            program.register(AlphaKey(KEY_DECODE_STEP), move |fact, _store| {
                if let InferenceFact::DecodeStep { step, token_id } = fact {
                    // Get current KV position
                    let current_pos = {
                        let s = state_ref.lock().unwrap();
                        s.prompt_len + *step
                    };

                    // Dequant embedding for this token — forward pass
                    // (In full ISF this also goes through EmbeddingReady,
                    //  but for decode we call forward directly for now)
                    let (_hidden, mut logits) = forward(vec![*token_id as f32], current_pos);
                    kv_inc();

                    // Sample next token
                    let next_token = sample(&mut logits);

                    // Check halt conditions
                    let (halt, halt_reason) = {
                        let s = state_ref.lock().unwrap();
                        let is_eos = next_token == s.eos_token;
                        let is_stop = s.extra_stop_ids.contains(&next_token);
                        let is_max = (*step + 1) >= s.max_tokens as u32;
                        if is_eos || is_stop {
                            (true, HaltReason::EosToken)
                        } else if is_max {
                            (true, HaltReason::MaxTokensReached)
                        } else {
                            (false, HaltReason::EosToken) // unused
                        }
                    };

                    if halt {
                        return vec![InferenceFact::GenerationHalt { reason: halt_reason }];
                    }

                    // Decode and emit
                    let piece = decode(next_token);
                    {
                        let mut s = state_ref.lock().unwrap();
                        s.generated_text.push_str(&piece);
                        s.logits = logits;
                        if let Some(cb) = s.on_token.as_mut() {
                            cb(&piece);
                        }
                        s.decode_step = *step + 2;
                    }

                    // Self-assert next decode step — the D0 reactive inversion
                    vec![InferenceFact::DecodeStep {
                        step: step + 1,
                        token_id: next_token,
                    }]
                } else {
                    vec![]
                }
            });
        }

        let fabric = SaturationFabric::new(
            program,
            alpha_key_of,
            |_consequent, _store: &mut FactStore<InferenceFact>| vec![],
        );

        Self { fabric, state }
    }

    /// Assert all prompt tokens and run to fixpoint.
    /// Returns the complete generated text.
    pub fn generate(&mut self, token_ids: &[u32]) -> GenerateOutput {
        // Assert all prompt tokens — Tier 1 structural facts
        for (pos, &id) in token_ids.iter().enumerate() {
            self.fabric.assert(InferenceFact::PromptToken {
                position: pos as u32,
                token_id: id,
            });
        }

        // Run to fixpoint — the fabric drives everything from here
        self.fabric.run_to_fixpoint(RunBudget::default());

        // Extract results
        let state = self.state.lock().unwrap();
        let halt_reason = state.halt.clone().unwrap_or(HaltReason::MaxTokensReached);
        let tokens_generated = state.decode_step as usize;

        GenerateOutput {
            text: state.generated_text.clone(),
            tokens_generated,
            halt_reason,
        }
    }
}
