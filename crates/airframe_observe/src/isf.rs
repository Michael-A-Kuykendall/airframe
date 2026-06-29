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
    KEY_EMBEDDING_READY, KEY_EMBEDDING_REQUEST, KEY_PREFILL_BATCH_READY, KEY_PREFILL_COMPLETE,
    KEY_PROMPT_TOKEN, KEY_TDR_RISK_HIGH,
};
use dzero::{AlphaKey, ClosureProgram, FactStore, RunBudget, SaturationFabric};
use std::sync::{Arc, Mutex};

/// Re-export RunBudget::default() so callers don't need dzero directly.
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
    /// Recent tokens for repetition penalty (last 64 tokens generated)
    pub recent_tokens: Vec<u32>,
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
            decode_step: 0,
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
            let kv_inc = kv_increment_fn.clone();
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
                    // Increment KV cache once per prompt token
                    for _ in 0..*token_count {
                        kv_inc();
                    }
                    if std::env::var("AIRFRAME_LOG_TDR_POLLS").is_ok() {
                        eprintln!("[DIAG] after prefill kv_inc called {} times", token_count);
                    }
                    {
                        let mut s = state_ref.lock().unwrap();
                        s.logits = logits;
                    }
                    // Emit DispatchCompleted fact for TDR accounting (Rule 6 picks it up)
                    vec![
                        InferenceFact::PrefillComplete { position: *token_count },
                        InferenceFact::DispatchCompleted {
                            layer: 0, // prefill spans all layers — use 0 as sentinel
                            kernel: crate::facts::KernelKind::FullLayer,
                            elapsed_ms,
                        },
                    ]
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
                    let logits_len = s.logits.len();
                    let logits_max = s.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let logits_nans = s.logits.iter().filter(|v| v.is_nan() || v.is_infinite()).count();
                    let recent = s.recent_tokens.clone();
                    let token_id = sample(&mut s.logits, &recent);
                    // Track for repetition penalty
                    s.recent_tokens.push(token_id);
                    if s.recent_tokens.len() > 64 { s.recent_tokens.remove(0); }
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
                    let t_decode = std::time::Instant::now();
                    // Get current KV position
                    let current_pos = {
                        let s = state_ref.lock().unwrap();
                        s.prompt_len + *step
                    };

                    // Dequant embedding for this token — forward pass
                    let (_hidden, mut logits) = forward(vec![*token_id as f32], current_pos);
                    kv_inc();
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

                    // Sample next token with repetition penalty from recent history
                    let next_token = {
                        let recent = {
                            let s = state_ref.lock().unwrap();
                            s.recent_tokens.clone()
                        };
                        sample(&mut logits, &recent)
                    };

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
                        // Track recent tokens for repetition penalty
                        s.recent_tokens.push(next_token);
                        if s.recent_tokens.len() > 64 { s.recent_tokens.remove(0); }
                        if let Some(cb) = s.on_token.as_mut() {
                            cb(&piece);
                        }
                        s.decode_step = *step + 2;
                    }

                    // Self-assert next decode step — the D0 reactive inversion
                    vec![
                        InferenceFact::DecodeStep {
                            step: step + 1,
                            token_id: next_token,
                        },
                        InferenceFact::DispatchCompleted {
                            layer: current_pos, // decode position as layer proxy
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
