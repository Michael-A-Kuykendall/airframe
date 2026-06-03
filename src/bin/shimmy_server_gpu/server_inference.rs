//! Inference execution sub-module for `shimmy_server_gpu`.
//! Contains token sampling, grammar masking, trace capture, and the main
//! `run_inference_completion` / `process_inference_job` pipeline.
//!
//! All types shared with the parent module are accessible via `use super::*;`.
use super::*;
use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{BindlessPipeline, LayerParams, RMSNormParams};
use airframe::core::spec::ModelSpec;
use airframe::debug_trace::{
    topk_from_logits, InferenceTracePackage, LayerTrace, TensorTrace, TokenTrace,
};
use libfse::metrics::{
    logit_l2_norm, logit_variance, max_probability_from_logits, shannon_entropy_from_logits,
};
use schoolmarm::{Grammar, GrammarState};
use shimmytok::{EncodeOptions, Tokenizer};
use std::sync::{Arc, Mutex};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let val = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (val >> 40) as f32 / 16777216.0 // 24-bit mantissa → [0, 1)
    }
}

/// Sample a token from logits with temperature, top-p, and repetition penalty.
fn sample_token(
    logits: &mut [f32],
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    recent_tokens: &[u32],
    rng: &mut Rng,
) -> u32 {
    // 1. Repetition penalty: discount tokens we've already generated
    if repetition_penalty != 1.0 {
        for &tok in recent_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= repetition_penalty;
                } else {
                    logits[idx] *= repetition_penalty;
                }
            }
        }
    }

    // 2. Temperature = 0 → greedy
    if temperature < 1e-7 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
    }

    // 3. Apply temperature
    let inv_t = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }

    // 4. Softmax (numerically stable)
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // 5. Top-p (nucleus) filtering
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    // Re-normalize the nucleus
    let nucleus = &indexed[..cutoff];
    let nsum: f32 = nucleus.iter().map(|(_, p)| p).sum();
    let r = rng.next_f32() * nsum;

    let mut acc = 0.0f32;
    for &(idx, p) in nucleus {
        acc += p;
        if acc >= r {
            return idx as u32;
        }
    }
    nucleus.last().unwrap().0 as u32
}

#[derive(Clone)]
struct TraceConfig {
    path: String,
    include_values: bool,
    include_prefill: bool,
    start_step: usize,
    max_steps: Option<usize>,
    top_k: usize,
}

impl TraceConfig {
    fn from_request(req: &InferenceRequest) -> Option<Self> {
        let path = req
            .debug_trace_path
            .clone()
            .or_else(|| std::env::var("SHIMMY_INFERENCE_TRACE_PATH").ok())?;
        Some(Self {
            path,
            include_values: req
                .debug_trace_full
                .unwrap_or_else(|| env_flag("SHIMMY_INFERENCE_TRACE_FULL")),
            include_prefill: req
                .debug_trace_include_prefill
                .unwrap_or_else(|| !matches!(std::env::var("SHIMMY_INFERENCE_TRACE_INCLUDE_PREFILL"), Ok(v) if v == "0" || v.eq_ignore_ascii_case("false"))),
            start_step: req
                .debug_trace_start_step
                .or_else(|| env_usize("SHIMMY_INFERENCE_TRACE_START_STEP"))
                .unwrap_or(0),
            max_steps: req
                .debug_trace_max_steps
                .or_else(|| env_usize("SHIMMY_INFERENCE_TRACE_MAX_STEPS")),
            top_k: env_usize("SHIMMY_INFERENCE_TRACE_TOPK").unwrap_or(8),
        })
    }

    fn should_capture_step(&self, step_index: usize) -> bool {
        if step_index < self.start_step {
            return false;
        }
        self.max_steps.map_or(true, |limit| {
            step_index.saturating_sub(self.start_step) < limit
        })
    }
}

fn build_layer_trace(
    layer_idx: usize,
    current_pos: u32,
    seq_len: u32,
    logical_pos_base: u32,
    include_values: bool,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    post_attn: &[f32],
    ffn_out: &[f32],
    output: &[f32],
) -> LayerTrace {
    LayerTrace {
        layer_idx,
        current_pos,
        seq_len,
        logical_pos_base,
        q: TensorTrace::from_slice(q, include_values),
        k: TensorTrace::from_slice(k, include_values),
        v: TensorTrace::from_slice(v, include_values),
        post_attn: TensorTrace::from_slice(post_attn, include_values),
        ffn_out: TensorTrace::from_slice(ffn_out, include_values),
        output: TensorTrace::from_slice(output, include_values),
    }
}

fn write_trace_package(path: &str, trace: &InferenceTracePackage) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(trace)?;
    fs::write(path, json)?;
    Ok(())
}

fn developer_mode_grammar() -> &'static str {
    r#"
root ::= start body end
start ::= "fn " | "use " | "struct " | "enum " | "impl "
body ::= [\x09\x0A\x0D\x20-\x7E]*
end ::= "// END_RUST_FILE"
"#
}

fn apply_grammar_mask(
    logits: &mut [f32],
    grammar_state: &GrammarState,
    vocab_texts: &[String],
    eos_token: u32,
    im_end_token: Option<u32>,
) {
    let vocab_refs: Vec<&str> = vocab_texts.iter().map(|s| s.as_str()).collect();
    let allowed = grammar_state.allowed_tokens(&vocab_refs);

    for (idx, logit) in logits.iter_mut().enumerate() {
        if idx >= allowed.len() || !allowed[idx] {
            *logit = f32::NEG_INFINITY;
        }
    }

    if grammar_state.is_accepting() {
        let eos_idx = eos_token as usize;
        if eos_idx < logits.len() {
            logits[eos_idx] = 0.0;
        }
        if let Some(im_end) = im_end_token {
            let im_end_idx = im_end as usize;
            if im_end_idx < logits.len() {
                logits[im_end_idx] = 0.0;
            }
        }
    }
}

fn rust_compile_check(source: &str) -> Result<(), String> {
    let temp_dir = std::env::temp_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    let pid = std::process::id();

    let src_path = temp_dir.join(format!("shimmy_compile_check_{}_{}.rs", pid, stamp));
    let out_path = temp_dir.join(format!("shimmy_compile_check_{}_{}", pid, stamp));

    std::fs::write(&src_path, source).map_err(|e| format!("write_temp_failed:{}", e))?;

    let output = std::process::Command::new("rustc")
        .arg("--edition")
        .arg("2021")
        .arg(&src_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .map_err(|e| format!("rustc_spawn_failed:{}", e))?;

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_path);

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let first = stderr.lines().next().unwrap_or("compile_failed");
        Err(first.to_string())
    }
}

/// OpenAI-compatible chat completions request body.
// When the output head buffer exceeds max_storage_buffer_binding_size, fall back to a plain
// CPU dot-product against the pre-dequantized embedding table that is already in RAM.
/// MiniCPM-V-2.6 image placeholder token ID.
const IMAGE_TOKEN_ID: u32 = 151_646;

/// Build the flat embedding sequence for prefill, splicing visual token embeddings
/// in place of any `IMAGE_TOKEN_ID` placeholder.
///
/// When `visual_tokens` is `Some`, the first occurrence of token 151646 in
/// `prompt_tokens` is replaced with `N_tiles × 64` visual embedding vectors
/// (each `dim`-dimensional).  All other tokens use the normal CPU embedding table
/// lookup.  The KV-cache advance count (returned as the second element) may
/// therefore differ from `prompt_tokens.len()`.
fn build_prefill_embeddings(
    prompt_tokens: &[u32],
    embd_table_cpu: &[f32],
    emb_scale: f32,
    dim: usize,
    visual_tokens: Option<&[f32]>, // flat [N_tiles × 64 × dim] f32
    visual_seq_len: usize,          // N_tiles × 64 (number of visual positions)
) -> (Vec<f32>, usize) {
    let mut out: Vec<f32> = Vec::with_capacity(prompt_tokens.len() * dim);
    let mut kv_advance = 0usize;
    let mut image_injected = false;

    for &token_id in prompt_tokens {
        if token_id == IMAGE_TOKEN_ID && !image_injected {
            if let Some(vt) = visual_tokens {
                // Splice all visual token embeddings (already in LLM space, dim-dimensional)
                for chunk in vt.chunks(dim) {
                    out.extend(chunk.iter().map(|x| x * emb_scale));
                }
                kv_advance += visual_seq_len;
                image_injected = true;
                continue;
            }
        }
        let emb_start = token_id as usize * dim;
        out.extend(embd_table_cpu[emb_start..emb_start + dim].iter().map(|x| x * emb_scale));
        kv_advance += 1;
    }
    (out, kv_advance)
}

// run_inference_completion takes many args by design — GPU pipeline requires all context inline.
// TODO: consider grouping device/queue/model/pipeline into a GpuContext struct to reduce arg count.
#[allow(clippy::too_many_arguments)]
fn run_inference_completion(
    job_id: Option<&str>,
    states: Option<&Arc<Mutex<std::collections::HashMap<String, JobState>>>>,
    stream_tx: Option<&tokio::sync::broadcast::Sender<String>>,
    req: &InferenceRequest,
    prior_tokens: &[u32],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &BindlessModel,
    pipeline: &BindlessPipeline,
    tokenizer: &Tokenizer,
    spec: &ModelSpec,
    kv_cache: Arc<Mutex<KVCache>>,
    embd_table_cpu: &[f32], // Pre-dequantized CPU embedding table
) -> Result<(InferenceResponse, String), Box<dyn std::error::Error>> {
    let max_new_tokens = req.max_tokens.unwrap_or(64);
    let temperature = req.temperature.unwrap_or(0.8);
    let top_p = req.top_p.unwrap_or(0.95);
    let rep_penalty = req.repetition_penalty.unwrap_or(1.1);
    let use_stream = req.stream.unwrap_or(false) && stream_tx.is_some();
    let mut rng = Rng::new(req.seed.unwrap_or(42));
    // SHIMMY_PREFILL_CHUNK: token batch size for chunked prefill. Default 64 is conservative;
    // with the AttnOut encoder isolation (A2 TDR fix) larger values should also be safe on Windows.
    let prefill_chunk: u32 = std::env::var("SHIMMY_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    eprintln!(
        "[GPU Server] Sampling: temp={:.2}, top_p={:.2}, rep_penalty={:.2}, stream={}",
        temperature, top_p, rep_penalty, use_stream
    );

    let disable_im_end_stop = env_flag("SHIMMY_DISABLE_IM_END_STOP");

    let prompt_mode = req.prompt_mode.as_deref().unwrap_or("creative").to_string();
    let user_prompt = req.prompt.as_deref().ok_or("missing prompt")?;
    let templated_prompt = build_templated_prompt(&prompt_mode, user_prompt)?;
    let mut prompt_tokens = tokenizer.encode_with_options(
        &templated_prompt,
        &EncodeOptions::with_parse_special(true, true),
    )?;
    let trace_config = TraceConfig::from_request(req);
    if !prior_tokens.is_empty() {
        prompt_tokens = prior_tokens
            .iter()
            .copied()
            .chain(prompt_tokens.into_iter())
            .collect();
    }

    let (visual_embedding_flat, visual_seq_len): (Option<Vec<f32>>, usize) = (None, 0);

    // Math bypass: pre-compute arithmetic answers and force their tokens.
    // Set SHIMMY_MATH_BYPASS_DISABLE=1 to observe raw model behavior for diagnostics.
    let bypass_disabled = std::env::var("SHIMMY_MATH_BYPASS_DISABLE").as_deref() == Ok("1");
    let math_bypass_tokens = if bypass_disabled {
        vec![]
    } else {
        airframe::math_bypass_control::compute_bypass_tokens(user_prompt, tokenizer)
    };
    let math_bypass_was_active = !math_bypass_tokens.is_empty();
    let mut math_bypass_queue: std::collections::VecDeque<u32> =
        math_bypass_tokens.into_iter().collect();
    if math_bypass_was_active {
        eprintln!(
            "[MathBypass] Detected arithmetic in prompt — forcing {} token(s) instead of sampling",
            math_bypass_queue.len()
        );
    }

    let eos = tokenizer.eos_token();
    let im_end_token: Option<u32> = tokenizer.encode("<|im_end|>", false).ok().and_then(|v| {
        if v.len() == 1 {
            Some(v[0])
        } else {
            None
        }
    });
    // Gemma-2 chat stop token (<end_of_turn> = token 107)
    let end_of_turn_token: Option<u32> = tokenizer
        .encode_with_options("<end_of_turn>", &EncodeOptions::with_parse_special(false, true))
        .ok()
        .and_then(|v| if v.len() == 1 { Some(v[0]) } else { None });

    let grammar_text = if prompt_mode == "developer" {
        Some(developer_mode_grammar())
    } else {
        None
    };

    let mut grammar_state: Option<GrammarState> = if let Some(grammar_text) = grammar_text {
        let grammar = Grammar::new(grammar_text)
            .map_err(|e| format!("developer grammar parse failed: {}", e))?;
        Some(
            GrammarState::new(grammar)
                .map_err(|e| format!("developer grammar init failed: {}", e))?,
        )
    } else {
        None
    };

    let vocab_texts: Option<Vec<String>> = if grammar_state.is_some() {
        Some(
            (0..spec.n_vocab)
                .map(|tid| tokenizer.decode_single(tid as u32, true).unwrap_or_default())
                .collect(),
        )
    } else {
        None
    };

    let mut recent_tokens: Vec<u32> = Vec::new();

    eprintln!(
        "=== Request: '{}' → {} tokens (templated)",
        user_prompt,
        prompt_tokens.len()
    );

    // Guard: reject requests that would overflow the KV cache.
    // prompt_tokens must leave at least 1 slot for decode; n_ctx is the hard limit.
    if prompt_tokens.len() >= spec.n_ctx as usize {
        return Err(format!(
            "prompt too long: {} tokens >= max context {} \
             (reduce prompt or increase SHIMMY_MAX_CTX)",
            prompt_tokens.len(),
            spec.n_ctx
        )
        .into());
    }

    let dim = spec.n_embd as u32;
    // Gemma / Gemma-2 scales input embeddings by sqrt(hidden_size) before the first layer.
    // Other architectures (LLaMA, etc.) do not apply this scale.
    let emb_scale: f32 = if spec.arch_string().contains("gemma") {
        (spec.n_embd as f32).sqrt()
    } else {
        1.0
    };
    // Detect per-tensor quantization types.  Q4_K_M models use Q6_K for attn_v and
    // ffn_down while everything else is Q4_K.  Pack into one u32:
    //   bits  0-7  = main qt (Q/K/attn_out/gate/up)
    //   bits  8-15 = V qt
    //   bits 16-23 = ffn_down qt
    let qt_main = model
        .metadata
        .get_tensor_type("blk.0.attn_q.weight")
        .unwrap_or(2);
    let qt_v = model
        .metadata
        .get_tensor_type("blk.0.attn_v.weight")
        .unwrap_or(qt_main);
    let qt_ffn_down = model
        .metadata
        .get_tensor_type("blk.0.ffn_down.weight")
        .unwrap_or(qt_main);
    let packed_quant_type = qt_main | (qt_v << 8) | (qt_ffn_down << 16);
    let layer_params = LayerParams {
        dim,
        head_count: spec.n_head as u32,
        head_count_kv: spec.n_head_kv as u32,
        head_dim: spec.head_dim as u32,
        rope_dim: spec.rope_dim as u32,
        rms_eps: spec.rms_eps,
        ffn_dim: spec.ff_dim as u32,
        temp_stride: spec.temp_buffer_size as u32,
        quant_type: packed_quant_type,
        attn_logit_softcap: spec.attn_logit_softcap,
        post_norm_enabled: if spec.arch_string().contains("gemma") { 1 } else { 0 },
        qk_norm_enabled: if spec.has_qk_norm { 1 } else { 0 },
    };

    let mut generated_text = String::new();
    let mut ppl_window: std::collections::VecDeque<f32> =
        std::collections::VecDeque::with_capacity(10);
    let mut stop_reason = "max_tokens";
    let mut metrics_violation = None;
    let mut generated_count: usize = 0;
    let mut trace_package = trace_config.as_ref().map(|_| InferenceTracePackage {
        schema_version: 1,
        model_arch: spec.arch_string().to_string(),
        prompt_mode: prompt_mode.clone(),
        seed: req.seed.unwrap_or(42),
        max_tokens: max_new_tokens,
        temperature,
        top_p,
        repetition_penalty: rep_penalty,
        prompt_token_count: prompt_tokens.len(),
        templated_prompt: templated_prompt.clone(),
        prefill_steps: Vec::new(),
        decode_steps: Vec::new(),
        final_stop_reason: String::new(),
        final_tokens_generated: 0,
        final_text: String::new(),
    });

    {
        let mut cache = kv_cache.lock().unwrap();
        cache.reset();
    }

    // Dynamic RoPE selection: use native frequencies when the prompt fits the
    // model training window; switch to YaRN-extended only when the total
    // sequence exceeds the training context. Prevents rope_scale distortion
    // on short prompts from an 8192-context server.
    if spec.rope_scale < 1.0 {
        if let Some(preflight) = model.preflight.as_ref() {
            let l_train = (spec.n_ctx as f32 * spec.rope_scale).round() as usize;
            let total_seq = prompt_tokens.len() + max_new_tokens;
            let (rope_data, mode_label) = if total_seq <= l_train {
                (&preflight.rope_data_native, "native")
            } else {
                (&preflight.rope_data_ext, "extended")
            };
            queue.write_buffer(&preflight.rope_cache_buffer, 0, bytemuck::cast_slice(rope_data));
            eprintln!(
                "[GPU Server] RoPE: {} (seq_needed={}, l_train={})",
                mode_label, total_seq, l_train
            );
        }
    }

    eprintln!(
        "[GPU Server] Prefill phase: processing {} prompt tokens...",
        prompt_tokens.len()
    );

    let norm_weight_offset = model
        .metadata
        .get_tensor_offset("output_norm.weight")
        .expect("output_norm.weight not found");
    let norm_params = RMSNormParams {
        count: dim,
        weights_offset: (norm_weight_offset / 4) as u32, // word index (byte_offset / 4) — matches sh_rmsnorm.wgsl read_blob() convention
        eps: spec.rms_eps,
        padding: 0,
    };

    let prefill_logits_f32 = if let Some(cfg) = trace_config.as_ref() {
        if cfg.include_prefill {
            let mut last_logits = vec![0.0; spec.n_vocab];
            // Track position within the visual embedding flat buffer for image token expansion.
            let mut visual_offset: usize = 0;
            let visual_token_dim = dim as usize;
            for (prefill_step, &token_id) in prompt_tokens.iter().enumerate() {
                // For the image placeholder, iterate over each visual token embedding individually.
                let visual_slice: Option<&[f32]> = if token_id == IMAGE_TOKEN_ID {
                    visual_embedding_flat.as_deref()
                } else {
                    None
                };
                let n_positions: usize = if visual_slice.is_some() { visual_seq_len } else { 1 };

                for vis_pos in 0..n_positions {
                    let layer_output_init: Vec<f32> = if let Some(vt) = visual_slice {
                        let start = (visual_offset + vis_pos) * visual_token_dim;
                        vt[start..start + visual_token_dim].iter().map(|x| x * emb_scale).collect()
                    } else {
                        let emb_start = token_id as usize * dim as usize;
                        embd_table_cpu[emb_start..emb_start + dim as usize].iter().map(|x| x * emb_scale).collect()
                    };
                let mut layer_output = layer_output_init;
                let (cache_len_before, window_base_before) = {
                    let cache = kv_cache.lock().unwrap();
                    (cache.get_seq_len(), cache.get_window_base())
                };
                let mut layer_traces = Vec::new();

                for layer_idx in 0..spec.n_layer {
                    let compiled = &model.metadata.compiled_layers[layer_idx];
                    let layer_params_l = LayerParams { quant_type: compiled.quant_type_packed, ..layer_params };

                    let (gpu_output, gpu_post_attn, gpu_ffn_out, gpu_q, gpu_k, gpu_v) = {
                        let mut cache = kv_cache.lock().unwrap();
                        pipeline.run_layer_with_cache_debug(
                            device,
                            queue,
                            model,
                            &mut cache,
                            layer_idx,
                            &layer_output,
                            compiled.offsets,
                            layer_params_l,
                        )
                    };

                    if cfg.should_capture_step(prefill_step) {
                        layer_traces.push(build_layer_trace(
                            layer_idx,
                            cache_len_before,
                            cache_len_before + 1,
                            window_base_before,
                            cfg.include_values,
                            &gpu_q,
                            &gpu_k,
                            &gpu_v,
                            &gpu_post_attn,
                            &gpu_ffn_out,
                            &gpu_output,
                        ));
                    }

                    layer_output = gpu_output;
                }

                let normed_output =
                    pipeline.run_rmsnorm_test(device, queue, model, &layer_output, norm_params);
                {
                    let head_tensor = if model.metadata.get_tensor_type("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
                    let head_off = (model.metadata.get_tensor_offset(head_tensor).unwrap_or(0) / 4) as u32;
                    let head_qt  = model.metadata.get_tensor_type(head_tensor).unwrap_or(2);
                    last_logits = pipeline.run_lm_head_blob(device, queue, model, &normed_output, spec.n_vocab as u32, dim, head_off, head_qt, spec.final_logit_softcap);
                }

                let (cache_len_after, window_base_after) = {
                    let mut cache = kv_cache.lock().unwrap();
                    cache.increment().map_err(|e| e)?;
                    (cache.get_seq_len(), cache.get_window_base())
                };

                if cfg.should_capture_step(prefill_step) {
                    if let Some(trace) = trace_package.as_mut() {
                        trace.prefill_steps.push(TokenTrace {
                            phase: "prefill".to_string(),
                            step_index: prefill_step,
                            token_id,
                            token_text: tokenizer.decode_single(token_id, true).unwrap_or_default(),
                            cache_len_before,
                            cache_len_after,
                            window_base_before,
                            window_base_after,
                            logits_topk: topk_from_logits(&last_logits, cfg.top_k, Some(tokenizer)),
                            layers: layer_traces,
                        });
                    }
                }
                } // end vis_pos loop
                if visual_slice.is_some() {
                    visual_offset += visual_seq_len;
                }
            }
            last_logits
        } else {
            let (batched_embd, kv_advance_a) = build_prefill_embeddings(
                &prompt_tokens,
                embd_table_cpu,
                emb_scale,
                dim as usize,
                visual_embedding_flat.as_deref(),
                visual_seq_len,
            );

            let (_pre_norm_a, l21_a, gpu_logits_a) = {
                let cache_guard = kv_cache.lock().unwrap();
                // FSE: process the full prompt in one shot. Per-layer sync removed from
                // inference.rs — all 28 layers execute in one command buffer. chunk_tokens
                // only exists as a safety valve for extremely long prompts (>512 tokens).
                // Layer weights are traversed once per chunk, not once per layer per chunk.
                pipeline.run_full_model_prefill_chunked_with_cache_state(
                    device,
                    queue,
                    model,
                    &batched_embd,
                    None, // blob head — quantized weights read directly from GGUF blob
                    0,
                    Some((cache_guard.get_k_buffers(), cache_guard.get_v_buffers())),
                    spec,
                    prefill_chunk,
                )?
            };

            let logits_a = gpu_logits_a;

            {
                let mut cache = kv_cache.lock().unwrap();
                for _ in 0..kv_advance_a {
                    cache.increment().map_err(|e| e)?;
                }
                if cache.is_int4() {
                    pipeline.requantize_all_kv_int4(
                        device, queue, &cache,
                        spec.n_head_kv as u32, spec.head_dim as u32,
                        prompt_tokens.len() as u32, spec.n_layer,
                    );
                    device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU device lost during requantize poll (A)");
                }
            }
            logits_a
        }
    } else {
        let (batched_embd, kv_advance_b) = build_prefill_embeddings(
            &prompt_tokens,
            embd_table_cpu,
            emb_scale,
            dim as usize,
            visual_embedding_flat.as_deref(),
            visual_seq_len,
        );

        let (pre_norm_b, l21_b, gpu_logits_b) = {
            let cache_guard = kv_cache.lock().unwrap();
            pipeline.run_full_model_prefill_chunked_with_cache_state(
                device,
                queue,
                model,
                &batched_embd,
                None, // blob head — quantized weights read directly from GGUF blob
                0,
                Some((cache_guard.get_k_buffers(), cache_guard.get_v_buffers())),
                spec,
                prefill_chunk,
            )?
        };

        let logits_b = gpu_logits_b;

        {
            let mut cache = kv_cache.lock().unwrap();
            for _ in 0..kv_advance_b {
                cache.increment().map_err(|e| e)?;
            }
            if cache.is_int4() {
                pipeline.requantize_all_kv_int4(
                    device, queue, &cache,
                    spec.n_head_kv as u32, spec.head_dim as u32,
                    prompt_tokens.len() as u32, spec.n_layer,
                );
                device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU device lost during requantize poll (B)");
            }
        }
        logits_b
    };



    let mut logits_vec = prefill_logits_f32;

    // === Prefill Sanity Gate ===
    // Log a structured [PREFILL_SANITY] block immediately after prefill.
    // PPL > 500 at step 0 = almost always VRAM pressure (garbage numerics), NOT a template issue.
    // Grep [PREFILL_SANITY] in server output when debugging new model onboarding.
    {
        let raw: Vec<f32> = logits_vec.iter().filter(|&&x| x > -100.0).copied().collect();
        let entropy = shannon_entropy_from_logits(&raw);
        let norm = logit_l2_norm(&raw);
        let max_p = max_probability_from_logits(&raw);
        let ppl_est = entropy.exp();
        let kv_mb = (spec.n_ctx as u64 * spec.n_head_kv as u64 * spec.head_dim as u64 * 4
            * spec.n_layer as u64 * 2)
            / 1_048_576;
        let status = if ppl_est < 50.0 {
            "OK"
        } else if ppl_est < 500.0 {
            "ELEVATED -- monitor first tokens"
        } else {
            "WARN:high_ppl -- likely VRAM pressure; try lower SHIMMY_MAX_CTX"
        };
        eprintln!(
            "[PREFILL_SANITY] arch={}  ctx={}  kv_cache={}MB  top1_prob={:.4}  ppl_est={:.2}  norm={:.2}  {}",
            spec.arch_string(), spec.n_ctx, kv_mb, max_p, ppl_est, norm, status
        );
    }

    let mut prev_step_ms: f64 = 0.0;
    for _step in 0..max_new_tokens {
        // Apply final logit softcap (Gemma-2 uses 30.0; 0.0 = disabled)
        if spec.final_logit_softcap > 0.0 {
            let cap = spec.final_logit_softcap;
            for l in logits_vec.iter_mut() {
                *l = (*l / cap).tanh() * cap;
            }
        }

        if let (Some(gs), Some(vocab)) = (grammar_state.as_ref(), vocab_texts.as_ref()) {
            apply_grammar_mask(&mut logits_vec, gs, vocab, eos, im_end_token);
        }

        let raw_logits: Vec<f32> = logits_vec
            .iter()
            .filter(|&&x| x > -100.0)
            .copied()
            .collect();
        let entropy = shannon_entropy_from_logits(&raw_logits);
        let _variance = logit_variance(&raw_logits);
        let max_prob = max_probability_from_logits(&raw_logits);
        let norm = logit_l2_norm(&raw_logits);

        if ppl_window.len() >= 10 {
            ppl_window.pop_front();
        }
        ppl_window.push_back(entropy.exp());
        let perplexity = ppl_window.iter().sum::<f32>() / ppl_window.len() as f32;

        let is_unsafe = perplexity > 500.0 || norm > 1e5;

        let next_token = if let Some(forced) = math_bypass_queue.pop_front() {
            eprintln!("[MathBypass] Step {}: forcing token {}", generated_count, forced);
            forced
        } else if is_unsafe {
            eprintln!(
                "[REDO] Metric Violation (PPL={:.2}, Norm={:.2}). Falling back to greedy.",
                perplexity, norm
            );
            if metrics_violation.is_none() {
                metrics_violation = Some(format!("Self-Healed PPL Spike: {:.2}", perplexity));
            }
            sample_token(&mut logits_vec, 0.0, 1.0, 1.0, &[], &mut rng)
        } else {
            sample_token(
                &mut logits_vec,
                temperature,
                top_p,
                rep_penalty,
                &recent_tokens,
                &mut rng,
            )
        };

        recent_tokens.push(next_token);
        if recent_tokens.len() > 64 {
            recent_tokens.remove(0);
        }

        if next_token == eos && !req.ignore_eos.unwrap_or(false) {
            stop_reason = "eos";
            break;
        }

        if let Some(eot) = end_of_turn_token {
            if next_token == eot && !req.ignore_eos.unwrap_or(false) {
                stop_reason = "end_of_turn";
                break;
            }
        }

        if !disable_im_end_stop {
            if let Some(im_end) = im_end_token {
                if next_token == im_end && !req.ignore_eos.unwrap_or(false) {
                    stop_reason = "im_end";
                    break;
                }
            }
        }

        let piece = tokenizer.decode_single(next_token, true)?;
        eprintln!(
            "[TOKEN] Step {}: id={}, text={:?}, entropy={:.3}, max_prob={:.3}",
            generated_count, next_token, piece, entropy, max_prob
        );
        generated_text.push_str(&piece);
        generated_count += 1;

        // Stream the piece before any early-exit break, so the HTTP collector
        // (which reads from the stream channel, not resp.text) always receives
        // every token that was appended to generated_text.
        if let Some(tx) = stream_tx {
            let _ = tx.send(piece.clone());
        }

        // Math bypass: stop cleanly once the forced-answer queue is drained.
        if math_bypass_was_active && math_bypass_queue.is_empty() {
            stop_reason = "math_bypass";
            break;
        }

        if let (Some(states_map), Some(job_id_ref)) = (states, job_id) {
            if let Ok(mut st) = states_map.lock() {
                if let Some(state) = st.get_mut(job_id_ref) {
                    state.partial_text = Some(generated_text.clone());
                }
            }
        }

        if let Some(gs) = grammar_state.as_mut() {
            if let Err(err) = gs.accept_token(&piece) {
                stop_reason = "grammar_reject";
                metrics_violation = Some(format!("grammar_reject: {}", err));
                break;
            }
            if gs.is_accepting() {
                stop_reason = "grammar_accept";
                break;
            }
        }

        if prompt_mode == "developer" && generated_text.contains("// END_RUST_FILE") {
            stop_reason = "end_marker";
            break;
        }

        let trace_step_index = generated_count.saturating_sub(1);
        let capture_trace_step = trace_config
            .as_ref()
            .map(|cfg| cfg.should_capture_step(trace_step_index))
            .unwrap_or(false);
        let (cache_len_before_step, window_base_before_step) = {
            let cache = kv_cache.lock().unwrap();
            (cache.get_seq_len(), cache.get_window_base())
        };

        {
            let cache = kv_cache.lock().unwrap();
            if cache.get_seq_len() >= cache.max_len() {
                stop_reason = "context_limit";
                metrics_violation = Some(format!("context_limit:{}", cache.max_len()));
                break;
            }
        }

        let step_t0 = std::time::Instant::now();

        let emb_start = next_token as usize * dim as usize;
        let mut layer_output: Vec<f32> =
            embd_table_cpu[emb_start..emb_start + dim as usize].iter().map(|x| x * emb_scale).collect();

        let mut step_layer_traces = Vec::new();
        if capture_trace_step {
            let cfg = trace_config.as_ref().unwrap();
            let (current_pos, logical_pos_base) = {
                let cache = kv_cache.lock().unwrap();
                (cache.get_seq_len(), cache.get_window_base())
            };
            for layer_idx in 0..spec.n_layer {
                let compiled = &model.metadata.compiled_layers[layer_idx];
                let layer_params_l = LayerParams { quant_type: compiled.quant_type_packed, ..layer_params };

                let (gpu_output, gpu_post_attn, gpu_ffn_out, gpu_q, gpu_k, gpu_v) = {
                    let mut cache = kv_cache.lock().unwrap();
                    pipeline.run_layer_with_cache_debug(
                        device,
                        queue,
                        model,
                        &mut cache,
                        layer_idx,
                        &layer_output,
                        compiled.offsets,
                        layer_params_l,
                    )
                };

                step_layer_traces.push(build_layer_trace(
                    layer_idx,
                    current_pos,
                    current_pos + 1,
                    logical_pos_base,
                    cfg.include_values,
                    &gpu_q,
                    &gpu_k,
                    &gpu_v,
                    &gpu_post_attn,
                    &gpu_ffn_out,
                    &gpu_output,
                ));

                layer_output = gpu_output;
            }
        } else {
            for layer_idx in 0..spec.n_layer {
                let compiled = &model.metadata.compiled_layers[layer_idx];
                let layer_params_l = LayerParams { quant_type: compiled.quant_type_packed, ..layer_params };

                {
                    let mut cache = kv_cache.lock().unwrap();
                    if cache.is_int4() {
                        layer_output = pipeline.run_layer_with_cache_int4(
                            device, queue, model, &mut cache, layer_idx, &layer_output,
                            compiled.offsets, layer_params_l,
                        );
                    } else {
                        layer_output = pipeline.run_layer_with_cache(
                            device, queue, model, &mut cache, layer_idx, &layer_output,
                            compiled.offsets, layer_params_l,
                        );
                    }
                }
            }
        }

        {
            let mut cache = kv_cache.lock().unwrap();
            cache.increment().map_err(|e| e)?;
        }

        let normed_output =
            pipeline.run_rmsnorm_test(device, queue, model, &layer_output, norm_params);

        {
            let head_tensor = if model.metadata.get_tensor_type("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
            let head_off = (model.metadata.get_tensor_offset(head_tensor).unwrap_or(0) / 4) as u32;
            let head_qt  = model.metadata.get_tensor_type(head_tensor).unwrap_or(2);
            logits_vec = pipeline.run_lm_head_blob(device, queue, model, &normed_output, spec.n_vocab as u32, dim, head_off, head_qt, spec.final_logit_softcap);
        }

        let step_ms = step_t0.elapsed().as_secs_f64() * 1000.0;
        prev_step_ms = step_ms;

        let (cache_len_after_step, window_base_after_step) = {
            let cache = kv_cache.lock().unwrap();
            (cache.get_seq_len(), cache.get_window_base())
        };

        if capture_trace_step {
            if let (Some(trace), Some(cfg)) = (trace_package.as_mut(), trace_config.as_ref()) {
                trace.decode_steps.push(TokenTrace {
                    phase: "decode".to_string(),
                    step_index: trace_step_index,
                    token_id: next_token,
                    token_text: piece,
                    cache_len_before: cache_len_before_step,
                    cache_len_after: cache_len_after_step,
                    window_base_before: window_base_before_step,
                    window_base_after: window_base_after_step,
                    logits_topk: topk_from_logits(&logits_vec, cfg.top_k, Some(tokenizer)),
                    layers: step_layer_traces,
                });
            }
        }
    }

    let include_debug_raw = std::env::var("SHIMMY_DEBUG_RAW")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let expose_candidate = req.expose_candidate.unwrap_or(false)
        || std::env::var("SHIMMY_EXPOSE_CANDIDATE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
        || include_debug_raw;

    let mut debug_raw_text: Option<String> = None;
    let mut debug_sanitizer_reason: Option<String> = None;
    let mut sanitizer_reason: Option<String> = None;
    let mut candidate_text: Option<String> = None;

    let mut final_text = if prompt_mode == "developer" {
        if include_debug_raw {
            debug_raw_text = Some(generated_text.clone());
        }
        let (sanitized, reason) = sanitize_developer_output_with_reason(&generated_text);
        sanitizer_reason = reason.clone();
        debug_sanitizer_reason = reason;
        if expose_candidate {
            candidate_text = Some(sanitized.clone());
        }
        sanitized
    } else {
        generated_text
    };

    if prompt_mode == "developer" && !final_text.is_empty() {
        if let Err(err) = rust_compile_check(&final_text) {
            sanitizer_reason = Some(format!("compile_check_failed:{}", err));
            final_text.clear();
        }
    }

    let (policy_status, policy_reason) = if prompt_mode == "developer" {
        let status = if final_text.is_empty() {
            "fail_closed"
        } else {
            "pass"
        };
        (
            Some(status.to_string()),
            Some(sanitizer_reason.unwrap_or_else(|| "unknown".to_string())),
        )
    } else {
        (None, None)
    };

    let resp = InferenceResponse {
        text: final_text,
        stop_reason: stop_reason.to_string(),
        tokens_generated: generated_count,
        metrics_violation,
        final_ppl: None,
        accuracy: None,
        policy_status,
        policy_reason,
        candidate_text,
        debug_raw_text,
        debug_sanitizer_reason: (include_debug_raw && prompt_mode == "developer")
            .then(|| debug_sanitizer_reason.unwrap_or_else(|| "(no_reason)".to_string())),
        debug_trace_path: trace_config.as_ref().map(|cfg| cfg.path.clone()),
    };

    if let (Some(mut trace), Some(cfg)) = (trace_package, trace_config.as_ref()) {
        trace.final_stop_reason = resp.stop_reason.clone();
        trace.final_tokens_generated = resp.tokens_generated;
        trace.final_text = resp.text.clone();
        write_trace_package(&cfg.path, &trace)?;
    }

    Ok((resp, templated_prompt))
}

pub(super) async fn process_inference_job(
    job_id: String,
    states: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, JobState>>>,
    session_states: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, SessionState>>>,
    stream_tx: Option<tokio::sync::broadcast::Sender<String>>,
    req: InferenceRequest,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &BindlessModel,
    pipeline: &BindlessPipeline,
    tokenizer: &Tokenizer,
    spec: &ModelSpec,
    kv_cache: Arc<Mutex<KVCache>>,
    embd_table_cpu: &[f32],         // Pre-dequantized F32 token embeddings (CPU)
) -> Result<InferenceResponse, Box<dyn std::error::Error>> {
    let session_window_tokens = spec.n_ctx;
    let disable_session_window = env_flag("SHIMMY_DISABLE_SESSION_WINDOW");
    let session_id = req.session_id.clone();

    let prior_tokens = if !disable_session_window {
        if let Some(session_id_ref) = session_id.as_deref() {
            let st = session_states.lock().unwrap();
            st.get(session_id_ref)
                .map(|state| {
                    state
                        .token_window
                        .iter()
                        .copied()
                        .rev()
                        .take(session_window_tokens)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let (resp, templated_prompt) = run_inference_completion(
        Some(&job_id),
        Some(&states),
        stream_tx.as_ref(),
        &req,
        &prior_tokens,
        device,
        queue,
        model,
        pipeline,
        tokenizer,
        spec,
        Arc::clone(&kv_cache),
        embd_table_cpu,
    )?;

    if !disable_session_window {
        if let Some(session_id_ref) = session_id.as_deref() {
            let response_suffix = if req.prompt_mode.as_deref().unwrap_or("creative") == "raw" {
                "</s>\n"
            } else {
                ""
            };
            let mut appended_tokens = tokenizer.encode_with_options(
                &templated_prompt,
                &EncodeOptions::with_parse_special(true, true),
            )?;
            if !resp.text.is_empty() {
                let assistant_tail = format!("{}{}", resp.text, response_suffix);
                appended_tokens.extend(tokenizer.encode(&assistant_tail, false)?);
            }

            let new_window = {
                let mut st = session_states.lock().unwrap();
                let session = st.entry(session_id_ref.to_string()).or_default();
                session.token_window.extend(appended_tokens);
                if session.token_window.len() > session_window_tokens {
                    let drop_count = session.token_window.len() - session_window_tokens;
                    session.token_window.drain(0..drop_count);
                }
                session.token_window.clone()
            };

            eprintln!("[SESSION] Stored {} tokens in rolling window", new_window.len());
        }
    }

    Ok(resp)
}

// HTTP error helper for future streaming error path — not yet called from active code
// Convenience wrapper — developer mode sanitization via reason-bearing variant
#[allow(dead_code)]
fn sanitize_developer_output(text: &str) -> String {
    sanitize_developer_output_with_reason(text).0
}

fn sanitize_developer_output_with_reason(text: &str) -> (String, Option<String>) {
    // The assistant turn was prefilled with "// BEGIN_RUST_FILE\n" before generation,
    // so generated text starts immediately with the Rust body.
    // The model is expected to end with "// END_RUST_FILE" (we stop generation on that marker).
    // FAIL-CLOSED: must look like a real Rust file or we return empty.
    let s = text.replace("\r\n", "\n");
    const END: &str = "// END_RUST_FILE";

    let candidate = if let Some(e) = s.find(END) {
        s[..e].trim().to_string()
    } else {
        // Model hit max_tokens before writing the end marker — use everything generated.
        s.trim().to_string()
    };

    if looks_like_rust_file(&candidate) {
        (candidate + "\n", Some("ok:prefill".to_string()))
    } else if candidate.is_empty() {
        (String::new(), Some("empty_output".to_string()))
    } else {
        (
            String::new(),
            Some(format!(
                "invalid_rust(first30={:?})",
                &candidate[..candidate.len().min(30)]
            )),
        )
    }
}

fn looks_like_rust_file(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s.contains("<|user|>")
        || s.contains("<|assistant|>")
        || s.contains("<|system|>")
        || s.contains("<|im_start|>")
        || s.contains("<|im_end|>")
    {
        return false;
    }
    let has_code_anchor = s.contains("fn main")
        || s.contains("use std::")
        || s.contains("struct ")
        || s.contains("enum ")
        || s.contains("impl ");
    let has_rust_punct = s.contains('{') && s.contains('}') && s.contains(';');
    has_code_anchor && has_rust_punct
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn developer_grammar_parses() {
        let grammar = Grammar::new(developer_mode_grammar());
        assert!(grammar.is_ok(), "developer grammar must parse");
    }

    #[test]
    fn developer_grammar_rejects_prose_prefix() {
        let grammar = Grammar::new(developer_mode_grammar()).unwrap();
        let state = GrammarState::new(grammar).unwrap();
        let vocab = vec!["To", "fn ", "use "];
        let allowed = state.allowed_tokens(&vocab);
        assert!(!allowed[0], "prose token should be disallowed at start");
        assert!(allowed[1] || allowed[2], "rust anchors should be allowed");
    }

    // ── build_prefill_embeddings tests ────────────────────────────────────────

    /// All normal tokens — no image placeholder.
    /// Expected: embeddings are a simple table lookup, kv_advance == n_tokens.
    #[test]
    fn bpe_no_image_token() {
        let dim = 4usize;
        // Vocab: 3 tokens, each 4-dimensional
        let table: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let tokens: Vec<u32> = vec![0, 1, 2];
        let (emb, kv) = build_prefill_embeddings(&tokens, &table, 1.0, dim, None, 0);
        assert_eq!(kv, 3, "kv_advance must equal token count when no image");
        assert_eq!(emb.len(), 3 * dim);
        // Token 1 → table[4..8]
        assert_eq!(&emb[4..8], &[4.0f32, 5.0, 6.0, 7.0]);
    }

    /// Image placeholder present but no visual tokens provided.
    /// The placeholder should fall back to a normal table lookup.
    #[test]
    fn bpe_image_token_no_visual_data() {
        let dim = 4usize;
        let table: Vec<f32> = vec![0.0f32; (IMAGE_TOKEN_ID as usize + 1) * dim];
        // write a sentinel value at the image-token row
        let mut t = table.clone();
        for i in 0..dim {
            t[IMAGE_TOKEN_ID as usize * dim + i] = 99.0;
        }
        let tokens = vec![0u32, IMAGE_TOKEN_ID, 0u32];
        let (emb, kv) = build_prefill_embeddings(&tokens, &t, 1.0, dim, None, 0);
        assert_eq!(kv, 3);
        assert_eq!(emb.len(), 3 * dim);
        // middle slot should be the sentinel embedding
        assert_eq!(&emb[dim..2 * dim], &vec![99.0f32; dim][..]);
    }

    /// Image placeholder with visual tokens provided.
    /// The placeholder should be expanded to `visual_seq_len` positions.
    #[test]
    fn bpe_image_token_with_visual_data() {
        let dim = 4usize;
        let visual_seq_len = 3usize; // pretend 3 visual tokens
        // visual_flat: 3 × dim values, each row = row_index cast to f32
        let visual_flat: Vec<f32> = (0..visual_seq_len)
            .flat_map(|i| vec![i as f32; dim])
            .collect();

        // emb table: zeroed (shouldn't be consulted for the image slot)
        let table: Vec<f32> = vec![0.0f32; (IMAGE_TOKEN_ID as usize + 1) * dim];
        let tokens = vec![0u32, IMAGE_TOKEN_ID, 0u32];
        let (emb, kv) = build_prefill_embeddings(
            &tokens,
            &table,
            1.0,
            dim,
            Some(&visual_flat),
            visual_seq_len,
        );
        // kv_advance = 1 (tok 0) + visual_seq_len (image) + 1 (tok 0)
        assert_eq!(kv, 2 + visual_seq_len);
        assert_eq!(emb.len(), (2 + visual_seq_len) * dim);
        // First `dim` values = tok 0 from table (all zeros)
        assert_eq!(&emb[..dim], &vec![0.0f32; dim][..]);
        // Next visual_seq_len × dim values = visual embeddings
        assert_eq!(&emb[dim..dim + visual_seq_len * dim], &visual_flat[..]);
        // Last `dim` values = tok 0 from table (all zeros)
        assert_eq!(&emb[dim + visual_seq_len * dim..], &vec![0.0f32; dim][..]);
    }

    /// Only the FIRST image placeholder is expanded; subsequent ones use the table.
    #[test]
    fn bpe_only_first_image_token_expanded() {
        let dim = 2usize;
        let visual_seq_len = 1usize;
        let visual_flat: Vec<f32> = vec![7.0f32; dim];
        let mut table = vec![0.0f32; (IMAGE_TOKEN_ID as usize + 1) * dim];
        // give the image-token row a distinct sentinel
        table[IMAGE_TOKEN_ID as usize * dim] = 55.0;
        table[IMAGE_TOKEN_ID as usize * dim + 1] = 55.0;

        let tokens = vec![IMAGE_TOKEN_ID, IMAGE_TOKEN_ID];
        let (emb, kv) = build_prefill_embeddings(
            &tokens,
            &table,
            1.0,
            dim,
            Some(&visual_flat),
            visual_seq_len,
        );
        // First: expanded (visual) → kv += 1; second: table lookup → kv += 1
        assert_eq!(kv, 1 + 1);
        assert_eq!(emb.len(), 2 * dim);
        assert_eq!(&emb[..dim], &[7.0f32, 7.0]);       // visual
        assert_eq!(&emb[dim..], &[55.0f32, 55.0]);     // table fallback
    }

    /// emb_scale is applied to every output value.
    #[test]
    fn bpe_emb_scale_applied() {
        let dim = 2usize;
        let table: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // tok 0 = [1,2], tok 1 = [3,4]
        let (emb, _) = build_prefill_embeddings(&[0, 1], &table, 2.0, dim, None, 0);
        assert_eq!(emb, vec![2.0f32, 4.0, 6.0, 8.0]);
    }
}
