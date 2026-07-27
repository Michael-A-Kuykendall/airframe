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
    alpha_key_of, HaltReason, InferenceFact, KEY_DECODE_STEP, KEY_DISPATCH_COMPLETED,
    KEY_EMBEDDING_READY, KEY_EMBEDDING_REQUEST, KEY_KV_ADVANCE, KEY_PREFILL_BATCH_READY,
    KEY_PREFILL_COMPLETE, KEY_PROMPT_TOKEN, KEY_TDR_RISK_HIGH, KEY_TENSOR_FACT,
};
use dzero::{AlphaKey, ClosureProgram, FactStore, RunBudget, SaturationFabric};
use std::sync::{Arc, Mutex};

/// Re-export RunBudget::default() so callers don't need dzero directly.
pub fn d0_run_budget() -> RunBudget {
    RunBudget::default()
}

/// Decision returned by an `IsfControlHook`.
///
/// Mirrors airframe's `ControlDecision` so the fabric (which must NOT depend on
/// the airframe crate) can apply grammar/FSE/math-bypass gates without coupling.
/// `airframe` translates its `InferenceControl::intervene` result into this enum
/// at the `generate_isf` boundary.
#[derive(Clone, Debug)]
pub enum IsfControlDecision {
    Allow,
    ForceToken(usize),
    EarlyExit,
    Block(String),
}

/// Post-sample control hook. Airframe wraps its `InferenceControl` in a closure
/// of this shape: `(candidate_token, accumulated_text, step, full_token_sequence,
/// kv_len) -> IsfControlDecision`. Token sequences are `u32` (GGUF vocab ids).
pub type IsfControlHook =
    dyn Fn(usize, &str, usize, &[u32], usize) -> IsfControlDecision + Send + Sync;

/// Pre-sample logits mask (e.g. grammar allowed-token masking).
pub type IsfMaskHook = dyn Fn(&mut [f32]) + Send + Sync;

/// Per-step trace callback: `(step, sampled_logits, elapsed_ms)`.
pub type IsfTraceHook = dyn FnMut(usize, &[f32], f64) + Send;

/// B3a dispatch rule: TensorFact → DispatchFact.
///
/// Given a `TensorFact` (structural control-plane fact from the GGUF header,
/// B2), look up the B1 quant-formula registry by `quant_type` and emit a
/// `DispatchFact` carrying the registry-derived `formula_index` (the B1 slot)
/// the shader consumes (B3b). This IS the replacement for the WGSL
/// `dequant_dispatch` `if qt==` ladder: the dispatch *decision* lives here,
/// spec-cited, not in the shader (Golden Rule 3).
///
/// `offset` is passed through so downstream consumers can locate the tensor in
/// the blob. Unsupported quant types emit nothing (fail-closed).
pub fn tensor_fact_dispatch_rule(
    fact: &InferenceFact,
    _store: &FactStore<InferenceFact>,
) -> Vec<InferenceFact> {
    if let InferenceFact::TensorFact {
        quant_type, offset, ..
    } = fact
    {
        match crate::quant_formula::slot_for_type(*quant_type) {
            Some(slot) => vec![InferenceFact::DispatchFact {
                quant_type: *quant_type,
                formula_index: slot.as_u32(),
                offset: *offset,
            }],
            None => vec![],
        }
    } else {
        vec![]
    }
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
#[allow(clippy::type_complexity)]
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
    /// Max tokens allowed
    pub max_tokens: u32,
    /// EOS token ID
    pub eos_token: u32,
    /// Extra stop token IDs
    pub extra_stop_ids: Vec<u32>,
    /// Streaming callback — called with each decoded token piece
    pub on_token: Option<Box<dyn FnMut(&str) + Send>>,
    /// Recent tokens for repetition penalty (last 64 tokens generated)
    pub recent_tokens: Vec<u32>,
    /// Pre-sample logits mask hook (grammar allowed-token masking). Applied to
    /// logits just before `sample_fn`. None = no masking.
    pub mask: Option<Arc<IsfMaskHook>>,
    /// Post-sample control hook (grammar/FSE/math-bypass). Applied after
    /// `sample_fn`; may force/early-exit/block the candidate token. None = no gate.
    pub control: Option<Arc<IsfControlHook>>,
    /// Per-step trace callback (step, post-mask logits, elapsed_ms).
    pub trace: Option<Arc<Mutex<Box<IsfTraceHook>>>>,
    /// Set when a control hook returns `Block` — surfaced as an error by the caller.
    pub block_reason: Option<String>,
    /// Full token sequence (prompt + generated so far). Fed to control hooks as
    /// the `tokens` argument. Seeded with prompt ids at generate() time.
    pub all_token_ids: Vec<u32>,
    /// TDR budget state — accumulated GPU time since last yield (ms).
    /// Rules emit DispatchTiming facts; when accumulated >= budget, a yield is needed.
    /// The actual yield (wgpu submit+poll) happens in the closure that emits the fact.
    pub tdr_accumulated_ms: u128,
    /// TDR budget in ms. Platform-aware: 1400ms on Windows, 30000ms elsewhere.
    pub tdr_budget_ms: u128,
    /// Number of yields performed this generation (for diagnostics).
    pub tdr_yield_count: u32,
    /// FSE embedding cache: token_id → dequanted f32 embedding.
    /// Rule 1b (EmbeddingRequest) populates this — exactly one GPU dequant per unique token_id.
    /// Rule 2 (EmbeddingReady) reads from this to assemble the batched embedding matrix.
    pub embedding_cache: std::collections::HashMap<u32, Vec<f32>>,
    /// Token IDs for each prompt position — set by generate_isf before asserting PromptToken facts.
    /// Needed by Rule 2 to assemble batched_embd from the embedding_cache.
    pub prompt_token_ids: Vec<u32>,
}

impl ISFState {
    #[allow(clippy::type_complexity)]
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
            max_tokens,
            eos_token,
            extra_stop_ids,
            on_token,
            tdr_accumulated_ms: 0,
            tdr_budget_ms: {
                // Platform-aware TDR budget.
                // Windows D3D12: hard 2s TDR, use 1400ms budget.
                // Linux/macOS: no hard TDR (or much longer), use 30s.
                #[cfg(windows)]
                let budget = std::env::var("SHIMMY_TDR_BUDGET_MS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1400u128);
                #[cfg(not(windows))]
                let budget = std::env::var("SHIMMY_TDR_BUDGET_MS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(30000u128);
                budget
            },
            tdr_yield_count: 0,
            embedding_cache: std::collections::HashMap::new(),
            prompt_token_ids: Vec::new(),
            recent_tokens: Vec::new(),
            mask: None,
            control: None,
            trace: None,
            block_reason: None,
            all_token_ids: Vec::new(),
        }
    }

    pub fn all_embeddings_ready(&self) -> bool {
        self.embeddings.iter().all(|e| e.is_some())
    }

    pub fn batched_embeddings(&self) -> Vec<f32> {
        // If prompt_token_ids is populated (Phase 3 reactive path), assemble from cache.
        // Otherwise fall back to the pre-filled embeddings Vec (legacy path).
        if !self.prompt_token_ids.is_empty() && !self.embedding_cache.is_empty() {
            self.prompt_token_ids
                .iter()
                .flat_map(|token_id| {
                    self.embedding_cache
                        .get(token_id)
                        .map(|v| v.to_vec())
                        .unwrap_or_default()
                })
                .collect()
        } else {
            // Legacy: pre-filled embeddings Vec
            self.embeddings
                .iter()
                .flat_map(|e| e.as_ref().unwrap().iter().cloned())
                .collect()
        }
    }

    /// Returns true when all unique token_ids have been dequanted into embedding_cache.
    pub fn all_embeddings_cached(&self, token_ids: &[u32]) -> bool {
        let unique: std::collections::HashSet<u32> = token_ids.iter().cloned().collect();
        unique
            .iter()
            .all(|id| self.embedding_cache.contains_key(id))
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
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn new(
        state: Arc<Mutex<ISFState>>,
        dequant_fn: Arc<dyn Fn(u32, u32) -> Vec<f32> + Send + Sync>,
        prefill_fn: Arc<dyn Fn(Vec<f32>, u32) -> (Vec<f32>, Vec<f32>) + Send + Sync>,
        forward_fn: Arc<dyn Fn(Vec<f32>, u32) -> (Vec<f32>, Vec<f32>) + Send + Sync>,
        sample_fn: Arc<dyn Fn(&mut Vec<f32>, &[u32]) -> u32 + Send + Sync>,
        decode_fn: Arc<dyn Fn(u32) -> String + Send + Sync>,
        kv_increment_fn: Arc<dyn Fn() + Send + Sync>,
        dim: u32,
    ) -> Self {
        let mut program = ClosureProgram::new();

        // ── Rule 1a: PromptToken → EmbeddingRequest (per position) ──────────
        // Each prompt token asserts an EmbeddingRequest for its token_id.
        // The FactStore's structural dedup ensures EmbeddingRequest { token_id: X }
        // is only inserted ONCE even if token X appears at 100 positions.
        // This is the FSE selector-dedup invariant: ∂dequant_cost / ∂duplicate_tokens ≈ 0.
        {
            let state_ref = state.clone();
            program.register(AlphaKey(KEY_PROMPT_TOKEN), move |fact, _store| {
                if let InferenceFact::PromptToken { position, token_id } = fact {
                    // Record the position→token_id mapping in state for batch assembly
                    {
                        let s = state_ref.lock().unwrap();
                        let pos = *position as usize;
                        if pos < s.embeddings.len() && s.embeddings[pos].is_none() {
                            // Mark position as pending — will be filled by EmbeddingRequest rule
                            // (leave as None for now; EmbeddingRequest fills by token_id)
                        }
                    }
                    // Assert EmbeddingRequest — FactStore dedup fires Rule 1b exactly once per token_id
                    vec![InferenceFact::EmbeddingRequest {
                        token_id: *token_id,
                    }]
                } else {
                    vec![]
                }
            });
        }

        // ── Rule 1b: EmbeddingRequest → EmbeddingReady (one dequant per unique token_id) ──
        // Fires exactly once per unique token_id (FactStore dedup blocks duplicates).
        // This is where the GPU dequant happens — the FSE selector extraction.
        {
            let state_ref = state.clone();
            let dequant = dequant_fn.clone();
            program.register(AlphaKey(KEY_EMBEDDING_REQUEST), move |fact, _store| {
                if let InferenceFact::EmbeddingRequest { token_id } = fact {
                    let embedding = dequant(*token_id, dim);
                    // Embedding quality check on first token dequanted (diagnostic)
                    {
                        let embedding_ref = &embedding;
                        let nan_count = embedding_ref
                            .iter()
                            .filter(|v| v.is_nan() || v.is_infinite())
                            .count();
                        if nan_count > 0 || embedding_ref.iter().take(4).all(|v| *v == 0.0) {
                            eprintln!(
                                "[ISF-R1b] WARNING token_id={} nan_count={} first4={:?}",
                                token_id,
                                nan_count,
                                &embedding_ref[..4.min(embedding_ref.len())]
                            );
                        }
                    }
                    // Broadcast: fill ALL positions that have this token_id
                    {
                        let mut s = state_ref.lock().unwrap();
                        for i in 0..s.embeddings.len() {
                            // We need to know which positions have this token_id.
                            // ISFState stores embeddings by position but not the reverse map.
                            // For now: we store the embedding keyed by token_id and let
                            // the batch assembly step fill positions from it.
                            // The embeddings Vec is filled by position in the pre-assert step.
                            let _ = i;
                        }
                        // Store embedding in a token_id-keyed cache via a new ISFState field
                        s.embedding_cache.insert(*token_id, embedding);
                    }
                    vec![InferenceFact::EmbeddingReady {
                        position: 0, // sentinel — actual positions filled in Rule 2
                        token_id: *token_id,
                    }]
                } else {
                    vec![]
                }
            });
        }

        // ── Rule 2: EmbeddingReady → PrefillBatchReady (when all unique tokens dequanted) ─
        // Fires after each EmbeddingReady. When all unique token_ids are in the cache,
        // asserts PrefillBatchReady. The FSE dedup in Rule 1b ensures this fires at most
        // N_unique times instead of N_total times.
        {
            let state_ref = state.clone();
            program.register(AlphaKey(KEY_EMBEDDING_READY), move |_fact, _store| {
                let s = state_ref.lock().unwrap();
                // Check via both paths: reactive (embedding_cache) or legacy (embeddings vec)
                let all_ready = if !s.prompt_token_ids.is_empty() {
                    s.all_embeddings_cached(&s.prompt_token_ids.clone())
                } else {
                    s.all_embeddings_ready()
                };
                if all_ready {
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
            program.register(AlphaKey(KEY_PREFILL_BATCH_READY), move |fact, _store| {
                if let InferenceFact::PrefillBatchReady { token_count } = fact {
                    let batched = {
                        let s = state_ref.lock().unwrap();
                        s.batched_embeddings()
                    };
                    let t_prefill = std::time::Instant::now();
                    eprintln!("[ISF-RULE] PrefillBatchReady: {} tokens → GPU prefill starting", token_count);
                    let (hidden, logits) = prefill(batched, *token_count);
                    let elapsed_ms = t_prefill.elapsed().as_millis() as u32;
                    let hidden_rms: f32 = if hidden.is_empty() { 0.0 } else {
                        (hidden.iter().map(|x| x*x).sum::<f32>() / hidden.len() as f32).sqrt()
                    };
                    let logits_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let logits_nans = logits.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
                    eprintln!("[ISF-RULE] GPU prefill done in {:.2}s — hidden_rms={:.4} logits_max={:.3} logits_nans={}/{}", 
                        elapsed_ms as f32 / 1000.0, hidden_rms, logits_max, logits_nans, logits.len());
                    // G2 observability: compute RMS/checksum while the buffers still live.
                    let hidden_checksum = crate::facts::checksum(&hidden);
                    let logits_rms = crate::facts::rms(&logits);
                    let logits_checksum = crate::facts::checksum(&logits);
                     // Store logits for downstream rules (mask, sample, halt)
                     {
                         let mut s = state_ref.lock().unwrap();
                         s.logits = logits;
                     }
                     // Emit KV advance facts — the fabric serializes these
                     // and the KvAdvance rule calls kv_inc() for each one,
                     // advancing seq_len from 0 to token_count.
                     let mut facts = Vec::with_capacity(4 + *token_count as usize);
                     for pos in 0..*token_count {
                         facts.push(InferenceFact::KvAdvance { position: pos });
                     }
                     facts.extend([
                         InferenceFact::PrefillComplete { position: *token_count },
                         InferenceFact::LayerOutput {
                             layer_idx: 0,
                             position: *token_count,
                             rms_bits: crate::facts::f32_to_bits(hidden_rms),
                             checksum: hidden_checksum,
                         },
                         InferenceFact::FinalLogits {
                             position: *token_count,
                             rms_bits: crate::facts::f32_to_bits(logits_rms),
                             checksum: logits_checksum,
                         },
                         InferenceFact::DispatchCompleted {
                             layer: 0,
                             kernel: crate::facts::KernelKind::FullLayer,
                             elapsed_ms,
                         },
                     ]);
                     facts
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
            let (token_id, halt, logits_len, logits_max, logits_nans) = {
                let mut s = state_ref.lock().unwrap();
                // Pre-sample mask (grammar allowed-token masking)
                if let Some(m) = s.mask.clone() {
                    (m)(&mut s.logits);
                }
                let logits_len = s.logits.len();
                let logits_max = s.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let logits_nans = s.logits.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
                let recent = s.recent_tokens.clone();
                let mut token_id = sample(&mut s.logits, &recent);
                // Post-sample control decision (immutable read; may not mutate yet)
                let decision = s.control.as_ref().map(|c| {
                    c(token_id as usize, &s.generated_text, 0usize, &s.all_token_ids, s.prompt_len as usize)
                });
                if let Some(d) = decision {
                    match d {
                        IsfControlDecision::Allow => {}
                        IsfControlDecision::ForceToken(t) => token_id = t as u32,
                        IsfControlDecision::EarlyExit => {
                            s.halt = Some(HaltReason::ControlHalt);
                            return vec![InferenceFact::GenerationHalt { reason: HaltReason::ControlHalt }];
                        }
                        IsfControlDecision::Block(r) => {
                            s.block_reason = Some(r);
                            s.halt = Some(HaltReason::ControlHalt);
                            return vec![InferenceFact::GenerationHalt { reason: HaltReason::ControlHalt }];
                        }
                    }
                }
                // Track for repetition penalty + full sequence
                s.recent_tokens.push(token_id);
                if s.recent_tokens.len() > 64 { s.recent_tokens.remove(0); }
                s.all_token_ids.push(token_id);
                let halt = token_id == s.eos_token
                    || s.extra_stop_ids.contains(&token_id);
                (token_id, halt, logits_len, logits_max, logits_nans)
            };

                eprintln!("[ISF-R4] PrefillComplete: logits_len={} max={:.3} nans={} first_token_id={} halt={}",
                    logits_len, logits_max, logits_nans, token_id, halt);

                if halt {
                    eprintln!("[ISF-R4] HALT at first token (EOS/stop token)");
                    return vec![InferenceFact::GenerationHalt {
                        reason: HaltReason::EosToken,
                    }];
                }

                 let piece = decode(token_id);
                 eprintln!("[ISF-R4] first token piece={:?} (len={})", piece, piece.len());
                 let prompt_len = {
                     let s = state_ref.lock().unwrap();
                     s.prompt_len
                 };
                 {
                     let mut s = state_ref.lock().unwrap();
                     s.generated_text.push_str(&piece);
                     if let Some(cb) = s.on_token.as_mut() {
                         cb(&piece);
                     }
                 }

                 vec![InferenceFact::DecodeStep {
                     step: 0,
                     token_id,
                     position: prompt_len,
                 }]
            });
        }

        // ── Rule TBD: KvAdvance → KvWritten ─────────────────
        // The fabric serializes KvAdvance before the next DecodeStep.
        // This rule fires the actual KV cache increment, removing
        // the imperative kv_inc() side-effect from Rule 5.
        {
            let kv_inc = kv_increment_fn.clone();
            program.register(AlphaKey(KEY_KV_ADVANCE), move |fact, _store| {
                if let InferenceFact::KvAdvance { position: _ } = fact {
                    kv_inc();
                    vec![InferenceFact::KvWritten]
                } else {
                    vec![]
                }
            });
        }

        // ── Rule 5: DecodeStep → next DecodeStep (or Halt) ───────────────
        // Position is carried in the DecodeStep fact — no arithmetic.
        // KV increment is driven by KvAdvance → KvWritten facts.
        {
            let state_ref = state.clone();
            let forward = forward_fn.clone();
            let sample = sample_fn.clone();
            let decode = decode_fn.clone();
            program.register(AlphaKey(KEY_DECODE_STEP), move |fact, _store| {
                if let InferenceFact::DecodeStep { step, token_id, position } = fact {
                    let t_decode = std::time::Instant::now();

                    // Dequant embedding for this token — forward pass
                    let (_hidden, mut logits) = forward(vec![*token_id as f32], *position);
                    let elapsed_ms = t_decode.elapsed().as_millis() as u32;

                    let logits_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let logits_nans = logits.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
                    let is_empty = logits.is_empty();

                    if *step % 10 == 0 || *step < 3 {
                        eprintln!("[ISF-DECODE] step={} gpu_forward={:.2}s logits_len={} max={:.3} nans={} in_token={}",
                            step, elapsed_ms as f32 / 1000.0, logits.len(), logits_max, logits_nans, token_id);
                    }

                    if is_empty {
                        eprintln!("[ISF-DECODE] step={} EMPTY LOGITS — forward pass failed, halting", step);
                        return vec![InferenceFact::GenerationHalt { reason: HaltReason::MaxTokensReached }];
                    }

                    // Pre-sample mask (grammar) + trace, then sample.
                    {
                        let s = state_ref.lock().unwrap();
                        if let Some(m) = s.mask.clone() {
                            (m)(&mut logits);
                        }
                        if let Some(trace) = &s.trace {
                            if let Ok(mut t) = trace.lock() {
                                t(*step as usize, &logits, elapsed_ms as f64);
                            }
                        }
                    }
                    // G2 observability: snap post-mask logits RMS/checksum before sampling.
                    let decode_logits_rms = crate::facts::rms(&logits);
                    let decode_logits_checksum = crate::facts::checksum(&logits);
                    // Sample next token with repetition penalty from recent history
                    let mut next_token = {
                        let recent = {
                            let s = state_ref.lock().unwrap();
                            s.recent_tokens.clone()
                        };
                        sample(&mut logits, &recent)
                    };

                    // Post-sample control (grammar/FSE/math-bypass)
                    let decision = {
                        let s = state_ref.lock().unwrap();
                        s.control.as_ref().map(|c| {
                            c(next_token as usize, &s.generated_text, *step as usize, &s.all_token_ids, (s.prompt_len + *step) as usize)
                        })
                    };
                    if let Some(d) = decision {
                        match d {
                            IsfControlDecision::Allow => {}
                            IsfControlDecision::ForceToken(t) => next_token = t as u32,
                            IsfControlDecision::EarlyExit => {
                                let mut s = state_ref.lock().unwrap();
                                s.halt = Some(HaltReason::ControlHalt);
                                return vec![InferenceFact::GenerationHalt { reason: HaltReason::ControlHalt }];
                            }
                            IsfControlDecision::Block(r) => {
                                let mut s = state_ref.lock().unwrap();
                                s.block_reason = Some(r);
                                s.halt = Some(HaltReason::ControlHalt);
                                return vec![InferenceFact::GenerationHalt { reason: HaltReason::ControlHalt }];
                            }
                        }
                    }

                    // Check halt conditions
                    let (halt, halt_reason) = {
                        let s = state_ref.lock().unwrap();
                        let is_eos = next_token == s.eos_token;
                        let is_stop = s.extra_stop_ids.contains(&next_token);
                        let is_max = (*step + 1) >= s.max_tokens;
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
                    if *step < 5 {
                        eprintln!("[ISF-DECODE] step={} next_token={} piece={:?}", step, next_token, piece);
                    }
                    {
                        let mut s = state_ref.lock().unwrap();
                        s.generated_text.push_str(&piece);
                        s.logits = logits;
                        // Track recent tokens for repetition penalty + full sequence
                        s.recent_tokens.push(next_token);
                        if s.recent_tokens.len() > 64 { s.recent_tokens.remove(0); }
                        s.all_token_ids.push(next_token);
                        if let Some(cb) = s.on_token.as_mut() {
                            cb(&piece);
                        }
                    }

                    // Self-assert next decode step — the D0 reactive inversion
                    vec![
                        InferenceFact::KvAdvance {
                            position: *position,
                        },
                        InferenceFact::DecodeStep {
                            step: step + 1,
                            token_id: next_token,
                            position: *position + 1,
                        },
                        InferenceFact::FinalLogits {
                            position: *position,
                            rms_bits: crate::facts::f32_to_bits(decode_logits_rms),
                            checksum: decode_logits_checksum,
                        },
                        InferenceFact::DispatchCompleted {
                            layer: *position,
                            kernel: crate::facts::KernelKind::FullLayer,
                            elapsed_ms,
                        },
                    ]
                } else {
                    vec![]
                }
            });
        }

        // ── Rule 6: DispatchCompleted → TdrRiskHigh (when budget exceeded) ──
        // Accumulates GPU dispatch time in ISFState.tdr_accumulated_ms.
        // When accumulated >= budget → derives TdrRiskHigh.
        // The actual yield (wgpu submit+poll) is performed in gpu.rs closures
        // which check ISFState.tdr_accumulated_ms directly before heavy work.
        // This rule makes TDR visible as a fabric fact for observability.
        {
            let state_ref = state.clone();
            program.register(AlphaKey(KEY_DISPATCH_COMPLETED), move |fact, _store| {
                if let InferenceFact::DispatchCompleted {
                    layer, elapsed_ms, ..
                } = fact
                {
                    let (accumulated, budget) = {
                        let mut s = state_ref.lock().unwrap();
                        s.tdr_accumulated_ms += *elapsed_ms as u128;
                        (s.tdr_accumulated_ms, s.tdr_budget_ms)
                    };
                    if accumulated >= budget {
                        if std::env::var("AIRFRAME_LOG_TDR_POLLS").is_ok() {
                            eprintln!(
                                "[ISF-TDR] layer={} accumulated={}ms >= budget={}ms → TdrRiskHigh",
                                layer, accumulated, budget
                            );
                        }
                        vec![InferenceFact::TdrRiskHigh { layer: *layer }]
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                }
            });
        }

        // ── Rule 7: TdrRiskHigh → YieldNow ────────────────────────────────
        // Derives the YieldNow consequent. The actual wgpu submit+poll happens
        // in the gpu.rs closures — they reset tdr_accumulated_ms after yielding.
        {
            let state_ref = state.clone();
            program.register(AlphaKey(KEY_TDR_RISK_HIGH), move |fact, _store| {
                if let InferenceFact::TdrRiskHigh { layer } = fact {
                    {
                        let mut s = state_ref.lock().unwrap();
                        s.tdr_accumulated_ms = 0; // reset after yield signal
                        s.tdr_yield_count += 1;
                    }
                    vec![InferenceFact::YieldNow {
                        layer: *layer,
                        reason: crate::facts::YieldReason::TdrBudgetExceeded,
                    }]
                } else {
                    vec![]
                }
            });
        }

        // ── Rule: TensorFact → DispatchFact (B3a) ──────────────────────────
        // On each structural TensorFact, emit the registry-derived DispatchFact.
        // This is the reactive replacement for the WGSL dequant_dispatch ladder;
        // the dispatch decision (formula_index) comes from the B1 registry.
        {
            let dispatch = tensor_fact_dispatch_rule;
            program.register(AlphaKey(KEY_TENSOR_FACT), move |fact, store| {
                dispatch(fact, store)
            });
        }

        let fabric = SaturationFabric::new(
            program,
            alpha_key_of,
            |_consequent, _store: &mut FactStore<InferenceFact>| vec![],
        );

        Self { fabric, state }
    }

    /// Assert control-plane `TensorFact`s (from GGUF load, B2) into the fabric
    /// so the B3a dispatch rule emits a `DispatchFact` per tensor. Called when a
    /// model is bound to the inference session (B4 wires this).
    pub fn assert_tensor_facts(&mut self, facts: &[InferenceFact]) {
        for f in facts {
            if matches!(f, InferenceFact::TensorFact { .. }) {
                self.fabric.assert(f.clone());
            }
        }
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
        let tokens_generated = state
            .all_token_ids
            .len()
            .saturating_sub(state.prompt_len as usize);

        GenerateOutput {
            text: state.generated_text.clone(),
            tokens_generated,
            halt_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::InferenceFact;

    /// B3a acceptance: the dispatch rule fires per tensor and emits a
    /// `DispatchFact` carrying the correct B1 registry `formula_index` for
    /// every supported quant type (and nothing for unsupported types).
    #[test]
    fn dispatch_rule_emits_dispatch_fact_per_quant_type() {
        // (ggml type id, expected registry formula_index slot)
        let cases: [(u32, u32); 8] = [
            (0, 0),  // F32
            (1, 1),  // F16
            (2, 2),  // Q4_0
            (6, 3),  // Q5_0
            (8, 4),  // Q8_0
            (12, 5), // Q4_K
            (13, 6), // Q5_K
            (14, 7), // Q6_K
        ];

        for (qt, expected_slot) in cases {
            let tf = InferenceFact::TensorFact {
                quant_type: qt,
                shape: vec![256, 32000],
                offset: 12345,
                arch_params: "llama".to_string(),
            };
            let out = tensor_fact_dispatch_rule(&tf, &FactStore::new());
            match out.as_slice() {
                [InferenceFact::DispatchFact {
                    quant_type,
                    formula_index,
                    offset,
                }] => {
                    assert_eq!(*quant_type, qt, "DispatchFact must echo quant_type");
                    assert_eq!(
                        *formula_index, expected_slot,
                        "DispatchFact formula_index must match B1 registry slot for qt={}",
                        qt
                    );
                    assert_eq!(*offset, 12345, "DispatchFact must pass through offset");
                }
                other => panic!(
                    "expected exactly one DispatchFact for qt={}, got {:?}",
                    qt, other
                ),
            }
        }

        // Unsupported quant type → no DispatchFact (fail-closed).
        let tf_bad = InferenceFact::TensorFact {
            quant_type: 99,
            shape: vec![1],
            offset: 0,
            arch_params: "x".to_string(),
        };
        assert!(
            tensor_fact_dispatch_rule(&tf_bad, &FactStore::new()).is_empty(),
            "unsupported quant type must emit no DispatchFact"
        );
    }

    /// G1 regression gate: the fabric must actually run the pre-sample mask and
    /// post-sample control hooks on the generate_isf path (the break was that
    /// `generate()` dropped these args). Verified with stub fns — no GPU needed.
    #[test]
    fn isf_runs_mask_and_control_on_generate_path() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mask_calls = Arc::new(AtomicUsize::new(0));
        let control_calls = Arc::new(AtomicUsize::new(0));
        let mask_calls2 = mask_calls.clone();
        let control_calls2 = control_calls.clone();

        let mask: Arc<IsfMaskHook> = Arc::new(move |logits: &mut [f32]| {
            mask_calls2.fetch_add(1, Ordering::SeqCst);
            // Zero out token 0 so greedy sampling can never pick it.
            if !logits.is_empty() {
                logits[0] = f32::NEG_INFINITY;
            }
        });
        let control: Arc<IsfControlHook> = Arc::new(move |_cand, _text, _step, _tokens, _kv| {
            control_calls2.fetch_add(1, Ordering::SeqCst);
            IsfControlDecision::Allow
        });

        let dim = 4u32;
        let n_vocab = 10u32;
        let state = Arc::new(Mutex::new(ISFState::new(3, 4, 0, vec![], None)));
        {
            let mut s = state.lock().unwrap();
            s.all_token_ids = vec![1, 2, 3];
            s.prompt_token_ids = vec![1, 2, 3];
            s.mask = Some(mask);
            s.control = Some(control);
        }

        let dequant = Arc::new(move |_t: u32, _d: u32| vec![0.0f32; dim as usize]);
        let prefill = Arc::new(move |_b: Vec<f32>, _n: u32| {
            (vec![0.0f32; dim as usize], vec![1.0f32; n_vocab as usize])
        });
        let forward = Arc::new(move |_e: Vec<f32>, _p: u32| {
            (vec![0.0f32; dim as usize], vec![1.0f32; n_vocab as usize])
        });
        let sample = Arc::new(|logits: &mut Vec<f32>, _recent: &[u32]| -> u32 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, v) in logits.iter().enumerate() {
                if *v > bv {
                    bv = *v;
                    best = i;
                }
            }
            best as u32
        });
        let decode = Arc::new(|t: u32| t.to_string());
        let kv = Arc::new(|| {});

        let mut fabric = InferenceSaturationFabric::new(
            state.clone(),
            dequant,
            prefill,
            forward,
            sample,
            decode,
            kv,
            dim,
        );
        let out = fabric.generate(&[1, 2, 3]);

        assert!(
            mask_calls.load(Ordering::SeqCst) > 0,
            "pre-sample mask MUST run on the fabric generate path"
        );
        assert!(
            control_calls.load(Ordering::SeqCst) > 0,
            "post-sample control MUST run on the fabric generate path"
        );
        assert!(!out.text.is_empty(), "should have generated tokens");

        // And the mask must have taken effect: token 0 is masked to -inf, so it
        // can never be the greedy choice. (Generated text is space-joined ids.)
        assert!(
            !out.text.contains('0'),
            "masked token 0 must never be produced; got '{}'",
            out.text
        );
    }

    /// G1 regression gate: a control returning `ForceToken` overrides the sample,
    /// and `Block` surfaces via `block_reason`. No GPU needed.
    #[test]
    fn isf_control_force_and_block() {
        // Force token 7 on the first decision, then allow.
        let control: Arc<IsfControlHook> = Arc::new(move |_cand, _text, step, _tokens, _kv| {
            if step == 0 {
                IsfControlDecision::ForceToken(7)
            } else {
                IsfControlDecision::Allow
            }
        });

        let dim = 4u32;
        let n_vocab = 10u32;
        let state = Arc::new(Mutex::new(ISFState::new(3, 4, 0, vec![], None)));
        {
            let mut s = state.lock().unwrap();
            s.all_token_ids = vec![1, 2, 3];
            s.prompt_token_ids = vec![1, 2, 3];
            s.control = Some(control);
        }

        let dequant = Arc::new(move |_t: u32, _d: u32| vec![0.0f32; dim as usize]);
        let prefill = Arc::new(move |_b: Vec<f32>, _n: u32| {
            (vec![0.0f32; dim as usize], vec![1.0f32; n_vocab as usize])
        });
        let forward = Arc::new(move |_e: Vec<f32>, _p: u32| {
            (vec![0.0f32; dim as usize], vec![1.0f32; n_vocab as usize])
        });
        let sample = Arc::new(|logits: &mut Vec<f32>, _recent: &[u32]| -> u32 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, v) in logits.iter().enumerate() {
                if *v > bv {
                    bv = *v;
                    best = i;
                }
            }
            best as u32
        });
        let decode = Arc::new(|t: u32| t.to_string());
        let kv = Arc::new(|| {});

        let mut fabric = InferenceSaturationFabric::new(
            state.clone(),
            dequant,
            prefill,
            forward,
            sample,
            decode,
            kv,
            dim,
        );
        let out = fabric.generate(&[1, 2, 3]);
        // First produced token must be the forced 7.
        assert!(
            out.text.starts_with('7'),
            "ForceToken(7) must override the sampled token; got '{}'",
            out.text
        );

        // Now Block on first decision.
        let block: Arc<IsfControlHook> = Arc::new(move |_cand, _text, _step, _tokens, _kv| {
            IsfControlDecision::Block("nope".into())
        });
        let state2 = Arc::new(Mutex::new(ISFState::new(3, 4, 0, vec![], None)));
        {
            let mut s = state2.lock().unwrap();
            s.all_token_ids = vec![1, 2, 3];
            s.prompt_token_ids = vec![1, 2, 3];
            s.control = Some(block);
        }
        let mut fabric2 = InferenceSaturationFabric::new(
            state2.clone(),
            Arc::new(move |_t: u32, _d: u32| vec![0.0f32; dim as usize]),
            Arc::new(move |_b: Vec<f32>, _n: u32| {
                (vec![0.0f32; dim as usize], vec![1.0f32; n_vocab as usize])
            }),
            Arc::new(move |_e: Vec<f32>, _p: u32| {
                (vec![0.0f32; dim as usize], vec![1.0f32; n_vocab as usize])
            }),
            Arc::new(|logits: &mut Vec<f32>, _recent: &[u32]| -> u32 {
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (i, v) in logits.iter().enumerate() {
                    if *v > bv {
                        bv = *v;
                        best = i;
                    }
                }
                best as u32
            }),
            Arc::new(|t: u32| t.to_string()),
            Arc::new(|| {}),
            dim,
        );
        let _ = fabric2.generate(&[1, 2, 3]);
        assert!(
            state2.lock().unwrap().block_reason.is_some(),
            "Block decision must set block_reason"
        );
    }
}
