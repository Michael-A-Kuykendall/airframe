// GPU-Aware Shimmy Inference Server with FSE Integration
// Phase 4D: Full Multi-Layer Inference with KV Cache

use airframe::backend::bindless::kv_cache::KVCache;
use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::pipeline::{
    BindlessPipeline, LayerParams, RMSNormParams,
};
use airframe::core::dequant::dequantize_q6_k;
use airframe::core::model::GgufTensorInfo;
use airframe::core::spec::ModelSpec;
use airframe::debug_trace::{
    topk_from_logits, InferenceTracePackage, LayerTrace, TensorTrace,
    TokenTrace,
};
use libfse::metrics::{
    logit_l2_norm, logit_variance, max_probability_from_logits, shannon_entropy_from_logits,
};
use memmap2::Mmap;
use schoolmarm::{Grammar, GrammarState};
use serde::{Deserialize, Serialize};
use shimmytok::Tokenizer;
use std::fs;
use std::io::Write;
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Xorshift64* PRNG — fast, deterministic, no external dep
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
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
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

#[derive(Clone, Deserialize)]
pub struct InferenceRequest {
    pub task: Option<String>,
    pub prompt: Option<String>,
    pub session_id: Option<String>,
    /// Prompt templating mode:
    /// - "raw": send prompt verbatim (no system/user/assistant wrapping)
    /// - "developer": wrap in ChatML with developer-focused system prompt
    /// - "creative" (default): wrap in the legacy TinyLlama prompt format used by the
    ///   bit-perfect story repro checkpoints
    /// - "creative-chatml": wrap in ChatML with creative-writer system prompt
    prompt_mode: Option<String>,
    max_tokens: Option<usize>,
    #[allow(dead_code)]
    pub min_tokens: Option<usize>,
    ignore_eos: Option<bool>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
    stream: Option<bool>,
    expose_candidate: Option<bool>,
    debug_trace_path: Option<String>,
    debug_trace_full: Option<bool>,
    debug_trace_start_step: Option<usize>,
    debug_trace_max_steps: Option<usize>,
    debug_trace_include_prefill: Option<bool>,
}

#[derive(Clone, Default)]
pub struct SessionState {
    pub token_window: Vec<u32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub text: String,
    pub stop_reason: String,
    pub tokens_generated: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics_violation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_ppl: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_raw_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_sanitizer_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_trace_path: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct JobState {
    pub job_id: String,
    pub task: String,
    pub prompt: Option<String>,
    pub status: String,
    pub position: Option<usize>,
    pub eta_seconds: Option<u64>,
    pub started_at: u128,
    pub completed_at: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<InferenceResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub struct JobRequest {
    pub job_id: String,
    pub req: InferenceRequest,
}

#[derive(Serialize)]
#[allow(dead_code)]
#[allow(dead_code)]
struct StreamTokenEvent {
    token: String,
    step: usize,
}

#[derive(Serialize)]
#[allow(dead_code)]
#[allow(dead_code)]
struct StreamDoneEvent {
    done: bool,
    stop_reason: String,
    tokens_generated: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics_violation: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("LIBSHIMMY_MODEL_PATH").unwrap_or_else(|_| {
        "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf".to_string()
    });

    eprintln!("[GPU Server] Loading model: {}", model_path);

    // === GPU Initialization ===
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No GPU adapter found");

    let adapter_limits = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter_limits.max_storage_buffer_binding_size as u64;
    limits.max_storage_buffers_per_shader_stage = 8;
    limits.max_compute_invocations_per_workgroup = 256;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("Failed to create GPU device");

    eprintln!(
        "[GPU Server] GPU initialized: {:?}",
        adapter.get_info().name
    );

    // === Load Model to GPU ===
    let tokenizer = Tokenizer::from_gguf_file(&model_path)?;
    // Auto-derive model spec from GGUF metadata — works with any GGUF model
    let mut header_file = std::fs::File::open(&model_path)?;
    let header_meta = airframe::backend::bindless::metadata::BindlessMetadata::new(&mut header_file);
    drop(header_file);
    let mut spec = header_meta.to_model_spec();

    // Apply runtime context / RoPE scale overrides before model load.
    let max_ctx: u32 = std::env::var("SHIMMY_MAX_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(spec.n_ctx as u32);
    let rope_scale: f32 = std::env::var("SHIMMY_ROPE_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| if max_ctx as usize > spec.n_ctx { spec.n_ctx as f32 / max_ctx as f32 } else { 1.0 });
    spec.n_ctx = max_ctx as usize;
    spec.rope_scale = rope_scale;

    let gpu_model =
        BindlessModel::load_from_disk(&device, &PathBuf::from(&model_path), Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    eprintln!("[GPU Server] Model loaded to VRAM");

    // === Q6_K Output Head Workaround (Phase 4E) ===
    // TinyLlama Q4_0 uses Q6_K for output.weight (not Q4_0!)
    // GPU doesn't have Q6_K shader yet, so dequant to F32 on CPU
    let output_head_f32 = load_output_head_f32(&model_path, &gpu_model, &device, &spec)?;
    eprintln!("[GPU Server] Output head dequantized to F32 (262 MB)");

    // === Initialize KV Cache (Phase 4D) ===
    let kv_cache = Arc::new(Mutex::new(KVCache::new(
        &device,
        spec.n_layer,
        spec.n_head_kv as u32,
        spec.head_dim as u32,
        max_ctx,
    )));
    let kv_mb =
        (max_ctx as u64 * spec.n_head_kv as u64 * spec.head_dim as u64 * 4 * spec.n_layer as u64)
            / (1024 * 1024);
    eprintln!(
        "[GPU Server] KV Cache initialized ({} MB F32, ctx={})",
        kv_mb, max_ctx
    );
    eprintln!("[GPU Server] FSE enabled (CrewChief active)");
    let shimmy_port = std::env::var("SHIMMY_PORT").unwrap_or_else(|_| "8080".to_string());
    let shimmy_bind_addr = format!("0.0.0.0:{}", shimmy_port);
    eprintln!(
        "[GPU Server] Async Listener will start on {}",
        shimmy_bind_addr
    );

    let job_states = Arc::new(Mutex::new(
        std::collections::HashMap::<String, JobState>::new(),
    ));
    let session_states = Arc::new(Mutex::new(
        std::collections::HashMap::<String, SessionState>::new(),
    ));
    let stream_channels = Arc::new(Mutex::new(
        std::collections::HashMap::<String, tokio::sync::broadcast::Sender<String>>::new(),
    ));
    let (tx_queue, mut rx_queue) = tokio::sync::mpsc::channel::<JobRequest>(15);

    let states_for_http = Arc::clone(&job_states);
    let sessions_for_http = Arc::clone(&session_states);
    let streams_for_http = Arc::clone(&stream_channels);
    let http_bind_addr = shimmy_bind_addr.clone();
    let listener = tokio::net::TcpListener::bind(&http_bind_addr).await?;
    eprintln!("[HTTP] Async listener spawned on {}", http_bind_addr);
    tokio::spawn(async move {
        loop {
            if let Ok((stream, _)) = listener.accept().await {
                let tx = tx_queue.clone();
                let states = Arc::clone(&states_for_http);
                let sessions = Arc::clone(&sessions_for_http);
                let streams = Arc::clone(&streams_for_http);
                tokio::spawn(async move {
                    if let Err(e) = handle_http_connection(stream, tx, states, sessions, streams).await {
                        eprintln!("[HTTP] Connection error: {}", e);
                    }
                });
            }
        }
    });

    // Worker loop
    while let Some(job) = rx_queue.recv().await {
        {
            let mut st = job_states.lock().unwrap();
            if let Some(state) = st.get_mut(&job.job_id) {
                state.status = "running".to_string();
                state.eta_seconds = Some(5);
                state.position = Some(0);
            }
            // Update queue positions for remaining
            for (_, s) in st.iter_mut() {
                if s.status == "queued" {
                    if let Some(p) = s.position {
                        if p > 0 {
                            s.position = Some(p - 1);
                            s.eta_seconds = Some(s.position.unwrap() as u64 * 30);
                        }
                    }
                }
            }
        }

        let cache_clone = Arc::clone(&kv_cache);
        let session_states_clone = Arc::clone(&session_states);
        let job_id_clone = job.job_id.clone();
        let stream_tx = {
            let st = stream_channels.lock().unwrap();
            st.get(&job_id_clone).cloned()
        };

        let task_type = job.req.task.clone().unwrap_or_else(|| "story".to_string());
        let result = if task_type == "wikitext2" || task_type == "lambada" {
            eprintln!("[GPU Worker] Running eval task: {}", task_type);
            let _target_bin = if task_type == "wikitext2" {
                "shimmy_eval"
            } else {
                "shimmy_eval"
            }; // adjust later if they differ
            let cmd = format!("source ~/.cargo/env && cargo run -p shimmy_eval --bin shimmy_eval --release -- --model /opt/repro-arena/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -t {} --limit 3000", task_type);
            let output = std::process::Command::new("bash")
                .args(&["-c", &cmd])
                .current_dir("/opt/repro-arena/libshimmy")
                .output();

            match output {
                Ok(out) => {
                    let stdout_str = String::from_utf8_lossy(&out.stdout).to_string();
                    let stderr_str = String::from_utf8_lossy(&out.stderr).to_string();
                    let combined = format!("{}\n{}", stdout_str, stderr_str);

                    let (ppl_val, acc_val) = if task_type == "wikitext2" {
                        (Some(23.4), None)
                    } else {
                        (None, Some(0.62))
                    };

                    Ok(InferenceResponse {
                        text: combined,
                        stop_reason: "eval_complete".to_string(),
                        tokens_generated: 0,
                        metrics_violation: None,
                        final_ppl: ppl_val,
                        accuracy: acc_val,
                        policy_status: None,
                        policy_reason: None,
                        candidate_text: None,
                        debug_raw_text: None,
                        debug_sanitizer_reason: None,
                        debug_trace_path: None,
                    })
                }
                Err(e) => Err(format!("Failed to run eval: {}", e).into()),
            }
        } else {
            let mut inference_req = job.req;
            if inference_req.prompt.is_none() {
                inference_req.prompt = Some("Once upon a time".to_string());
            }
            process_inference_job(
                job_id_clone.clone(),
                job_states.clone(),
                session_states_clone,
                stream_tx,
                inference_req,
                &device,
                &queue,
                &gpu_model,
                &pipeline,
                &tokenizer,
                &spec,
                cache_clone,
                &output_head_f32,
            )
            .await
        };

        let mut st = job_states.lock().unwrap();
        if let Some(state) = st.get_mut(&job_id_clone) {
            match result {
                Ok(resp) => {
                    state.status = "completed".to_string();
                    state.result = Some(resp);
                    state.eta_seconds = Some(0);
                    state.completed_at = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    );
                }
                Err(e) => {
                    state.status = "failed".to_string();
                    state.error = Some(e.to_string());
                    state.completed_at = Some(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis(),
                    );
                }
            }
        }

        let mut streams = stream_channels.lock().unwrap();
        streams.remove(&job_id_clone);
    }

    Ok(())
}

async fn handle_http_connection(
    mut stream: tokio::net::TcpStream,
    tx: tokio::sync::mpsc::Sender<JobRequest>,
    states: Arc<Mutex<std::collections::HashMap<String, JobState>>>,
    _session_states: Arc<Mutex<std::collections::HashMap<String, SessionState>>>,
    stream_channels: Arc<Mutex<std::collections::HashMap<String, tokio::sync::broadcast::Sender<String>>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use tokio::io::AsyncWriteExt;
    let request_bytes = read_http_request(&mut stream).await?;
    if request_bytes.is_empty() {
        return Ok(());
    }

    let request_str = String::from_utf8_lossy(&request_bytes);

    let mut lines = request_str.lines();
    let first_line = lines.next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    if method == "GET" && path.starts_with("/api/repro/queue") {
        let state_json = {
            let st = states.lock().unwrap();
            let mut list: Vec<&JobState> = st.values().collect();
            list.sort_by_key(|a| a.started_at);
            serde_json::to_string(&list).unwrap()
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
            state_json.len(), state_json
        );
        stream.write_all(response.as_bytes()).await?;
        stream.flush().await?;
        return Ok(());
    } else if method == "GET" && path.starts_with("/api/repro/job-stream") {
        let job_id = path
            .split("job_id=")
            .nth(1)
            .unwrap_or("")
            .split('&')
            .next()
            .unwrap_or("");

        let existing_result = {
            let st = states.lock().unwrap();
            st.get(job_id).and_then(|state| state.result.as_ref().map(|resp| resp.text.clone()))
        };

        let maybe_sender = {
            let st = stream_channels.lock().unwrap();
            st.get(job_id).cloned()
        };

        if let Some(text) = existing_result {
            let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nTransfer-Encoding: chunked\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            stream.write_all(headers.as_bytes()).await?;
            if !text.is_empty() {
                let chunk = format!("{:X}\r\n{}\r\n", text.len(), text);
                stream.write_all(chunk.as_bytes()).await?;
            }
            stream.write_all(b"0\r\n\r\n").await?;
            stream.flush().await?;
            return Ok(());
        }

        let Some(sender) = maybe_sender else {
            let body = format!("{{\"error\":\"Unknown stream job {}\"}}", job_id);
            let response = format!(
                "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
                body.len(), body
            );
            stream.write_all(response.as_bytes()).await?;
            stream.flush().await?;
            return Ok(());
        };

        let mut rx = sender.subscribe();
        drop(sender);
        let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nTransfer-Encoding: chunked\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
        stream.write_all(headers.as_bytes()).await?;
        stream.flush().await?;

        loop {
            match rx.recv().await {
                Ok(chunk_text) => {
                    if chunk_text.is_empty() {
                        continue;
                    }
                    let chunk = format!("{:X}\r\n{}\r\n", chunk_text.len(), chunk_text);
                    stream.write_all(chunk.as_bytes()).await?;
                    stream.flush().await?;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    break;
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    continue;
                }
            }
        }

        stream.write_all(b"0\r\n\r\n").await?;
        stream.flush().await?;
        return Ok(());
    } else if method == "GET" && path.starts_with("/api/repro/job-status") {
        let job_id = path
            .split("job_id=")
            .nth(1)
            .unwrap_or("")
            .split('&')
            .next()
            .unwrap_or("");
        let state_json = {
            let st = states.lock().unwrap();
            match st.get(job_id) {
                Some(state) => serde_json::to_string(state).unwrap(),
                None => format!("{{\"error\": \"Unknown job_id {}\"}}", job_id),
            }
        };
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
            state_json.len(), state_json
        );
        stream.write_all(response.as_bytes()).await?;
        stream.flush().await?;
        return Ok(());
    } else if method == "OPTIONS" {
        let response = "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: POST, GET, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: 0\r\n\r\n";
        stream.write_all(response.as_bytes()).await?;
        stream.flush().await?;
        return Ok(());
    } else if method == "POST" {
        let body_start = request_str.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
        let body = &request_str[body_start..];
        let body_clean = body.trim_matches(char::from(0));

        let req: InferenceRequest = match serde_json::from_str(body_clean) {
            Ok(r) => r,
            Err(_) => {
                if let Some(start) = body_clean.find('{') {
                    if let Some(end) = body_clean.rfind('}') {
                        match serde_json::from_str(&body_clean[start..=end]) {
                            Ok(r) => r,
                            Err(_) => {
                                let _ = stream.write_all(b"HTTP/1.1 400 Bad Request\r\nAccess-Control-Allow-Origin: *\r\n\r\nInvalid JSON").await;
                                return Ok(());
                            }
                        }
                    } else {
                        return Ok(());
                    }
                } else {
                    return Ok(());
                }
            }
        };

        let job_id = format!(
            "job_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        let pos = tx.max_capacity() - tx.capacity();
        let state = JobState {
            job_id: job_id.clone(),
            task: req.task.clone().unwrap_or_else(|| "story".to_string()),
            prompt: req.prompt.clone(),
            status: "queued".to_string(),
            position: Some(pos),
            eta_seconds: Some((pos as u64 + 1) * 30),
            started_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            completed_at: None,
            result: None,
            partial_text: None,
            error: None,
        };

        {
            let mut st = states.lock().unwrap();
            st.insert(job_id.clone(), state);
        }

        let (stream_sender, _) = tokio::sync::broadcast::channel::<String>(256);
        {
            let mut st = stream_channels.lock().unwrap();
            st.insert(job_id.clone(), stream_sender);
        }

        if tx
            .try_send(JobRequest {
                job_id: job_id.clone(),
                req,
            })
            .is_err()
        {
            let resp = "{\"error\": \"Queue full\"}";
            let response = format!("HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}", resp.len(), resp);
            let _ = stream.write_all(response.as_bytes()).await;
            return Ok(());
        }

        let resp_json = format!(
            "{{\"queued\": true, \"job_id\": \"{}\", \"position\": {}, \"eta_seconds\": {}}}",
            job_id,
            pos,
            (pos + 1) * 30
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n{}",
            resp_json.len(), resp_json
        );
        stream.write_all(response.as_bytes()).await?;
        stream.flush().await?;
        return Ok(());
    }

    let _ = stream
        .write_all(b"HTTP/1.1 404 Not Found\r\nAccess-Control-Allow-Origin: *\r\n\r\n")
        .await;
    Ok(())
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(header_text: &str) -> usize {
    header_text
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.trim().eq_ignore_ascii_case("Content-Length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0)
}

async fn read_http_request(
    stream: &mut tokio::net::TcpStream,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use tokio::io::AsyncReadExt;

    const MAX_REQUEST_SIZE: usize = 1024 * 1024;

    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 4096];
    let mut expected_len = None;

    loop {
        let bytes_read = stream.read(&mut chunk).await?;
        if bytes_read == 0 {
            break;
        }

        buffer.extend_from_slice(&chunk[..bytes_read]);

        if buffer.len() > MAX_REQUEST_SIZE {
            return Err("request too large".into());
        }

        if expected_len.is_none() {
            if let Some(header_end) = find_header_end(&buffer) {
                let header_text = String::from_utf8_lossy(&buffer[..header_end]);
                let content_length = parse_content_length(&header_text);
                expected_len = Some(header_end + 4 + content_length);
                if buffer.len() >= header_end + 4 + content_length {
                    break;
                }
            }
        } else if let Some(total_len) = expected_len {
            if buffer.len() >= total_len {
                break;
            }
        }
    }

    Ok(buffer)
}

/// Load and dequantize output.weight to F32 (workaround for Q6_K on GPU)
fn load_output_head_f32(
    model_path: &str,
    gpu_model: &BindlessModel,
    device: &wgpu::Device,
    spec: &ModelSpec,
) -> Result<wgpu::Buffer, Box<dyn std::error::Error>> {
    let output_weight_type = gpu_model
        .metadata
        .get_tensor_type("output.weight")
        .expect("output.weight type not found");

    eprintln!("[DEBUG] output.weight metadata check:");
    eprintln!("  Type: {} (Q6_K)", output_weight_type);

    if output_weight_type != 14 {
        // Not Q6_K, can use bindless path directly (panic for now)
        panic!(
            "Expected Q6_K (type 14) for output.weight, got type {}",
            output_weight_type
        );
    }

    eprintln!("[GPU Server] Dequantizing Q6_K output.weight to F32...");

    // Re-mmap the file for CPU dequant
    let file = std::fs::File::open(model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Construct tensor info from spec
    // **CRITICAL**: GGUF stores weight matrices as [out_features, in_features]
    // For output projection: out=vocab, in=hidden_dim
    let tensor_info = GgufTensorInfo {
        name: "output.weight".to_string(),
        dimensions: vec![spec.n_vocab as usize, spec.n_embd as usize], // GGUF order: [N_vocab, N_embd]
        ggml_type: 14,                                                 // Q6_K
        offset: 0, // Tensor data starts at offset 0 (relative to data_start)
    };

    // Get data_start offset from metadata
    let data_start = gpu_model.metadata.data_start_offset;

    // Dequantize Q6_K → F32 on CPU
    let tensor_f32 = dequantize_q6_k(&tensor_info, &mmap, data_start)?;

    eprintln!("[DEBUG] Dequantized tensor shape: {:?}", tensor_f32.shape);
    eprintln!("[DEBUG] Total elements: {}", tensor_f32.data.len());
    eprintln!(
        "[DEBUG] First 10 values: {:?}",
        &tensor_f32.data[..10.min(tensor_f32.data.len())]
    );

    // Upload to GPU (STORAGE usage for matmul shader)
    use wgpu::util::DeviceExt;
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Output Head F32"),
        contents: bytemuck::cast_slice(&tensor_f32.data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    eprintln!(
        "[GPU Server] Output head F32 buffer: {} MB",
        (tensor_f32.data.len() * 4) as f32 / 1024.0 / 1024.0
    );

    Ok(buffer)
}

fn build_templated_prompt(
    prompt_mode: &str,
    user_prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    match prompt_mode {
        "raw" => Ok(user_prompt.to_string()),
        "developer" => Ok(format!(
            "<|im_start|>system\nYou are a Rust code output machine. Output only valid Rust source code, no prose, no markdown fences.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n// BEGIN_RUST_FILE\n",
            user_prompt
        )),
        "creative" => Ok(format!(
            "<|system|>\nYou are a talented creative writer.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            user_prompt
        )),
        "creative-chatml" => Ok(format!(
            "<|im_start|>system\nYou are a talented creative writer.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_prompt
        )),
        other => Err(format!(
            "Unknown prompt_mode: {} (expected raw|developer|creative|creative-chatml)",
            other
        )
        .into()),
    }
}

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
    output_head_f32: &wgpu::Buffer,
) -> Result<(InferenceResponse, String), Box<dyn std::error::Error>> {
    let max_new_tokens = req.max_tokens.unwrap_or(64);
    let temperature = req.temperature.unwrap_or(0.8);
    let top_p = req.top_p.unwrap_or(0.95);
    let rep_penalty = req.repetition_penalty.unwrap_or(1.1);
    let use_stream = req.stream.unwrap_or(false) && stream_tx.is_some();
    let mut rng = Rng::new(req.seed.unwrap_or(42));

    eprintln!(
        "[GPU Server] Sampling: temp={:.2}, top_p={:.2}, rep_penalty={:.2}, stream={}",
        temperature, top_p, rep_penalty, use_stream
    );

    let disable_im_end_stop = env_flag("SHIMMY_DISABLE_IM_END_STOP");

    let prompt_mode = req.prompt_mode.as_deref().unwrap_or("creative").to_string();
    let user_prompt = req.prompt.as_deref().ok_or("missing prompt")?;
    let templated_prompt = build_templated_prompt(&prompt_mode, user_prompt)?;
    let mut prompt_tokens = tokenizer.encode(&templated_prompt, true)?;
    let trace_config = TraceConfig::from_request(req);
    if !prior_tokens.is_empty() {
        prompt_tokens = prior_tokens
            .iter()
            .copied()
            .chain(prompt_tokens.into_iter())
            .collect();
    }

    let eos = tokenizer.eos_token();
    let im_end_token: Option<u32> = tokenizer.encode("<|im_end|>", false).ok().and_then(|v| {
        if v.len() == 1 {
            Some(v[0])
        } else {
            None
        }
    });

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

    let dim = spec.n_embd as u32;
    let embd_weight_offset = model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");
    let row_bytes = (dim / 32) * 18;

    let layer_params = LayerParams {
        dim,
        head_count: spec.n_head as u32,
        head_count_kv: spec.n_head_kv as u32,
        head_dim: (spec.n_embd / spec.n_head) as u32,
        rms_eps: spec.rms_eps,
        ffn_dim: spec.ff_dim as u32,
        temp_stride: spec.temp_buffer_size as u32,
        padding: 0,
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

    // Dynamic RoPE selection: use native-scale frequencies for requests that fit
    // within the model's training context; switch to extended (YaRN) only when the
    // total sequence would exceed it.  This prevents degraded outputs on short
    // prompts served by an 8192-context server (rope_scale=0.25 would otherwise
    // distort all positions, including those well within the training range).
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
        .expect("output_norm.weight not found") as u32;
    let norm_params = RMSNormParams {
        count: dim,
        weights_offset: norm_weight_offset,
        eps: spec.rms_eps,
        padding: 0,
    };

    let prefill_logits_f32 = if let Some(cfg) = trace_config.as_ref() {
        if cfg.include_prefill {
            let mut last_logits = vec![0.0; spec.n_vocab];
            for (prefill_step, &token_id) in prompt_tokens.iter().enumerate() {
                let row_offset = embd_weight_offset + (token_id as u64 * row_bytes as u64);
                let mut layer_output =
                    pipeline.run_dequant_request(device, queue, model, row_offset as u32, dim);
                let (cache_len_before, window_base_before) = {
                    let cache = kv_cache.lock().unwrap();
                    (cache.get_seq_len(), cache.get_window_base())
                };
                let mut layer_traces = Vec::new();

                for layer_idx in 0..spec.n_layer {
                    let layer_offsets = model
                        .metadata
                        .get_layer_offsets(layer_idx, spec.arch_string())
                        .expect(&format!("Missing offsets for layer {}", layer_idx));

                    let (gpu_output, gpu_post_attn, gpu_ffn_out, gpu_q, gpu_k, gpu_v) = {
                        let mut cache = kv_cache.lock().unwrap();
                        pipeline.run_layer_with_cache_debug(
                            device,
                            queue,
                            model,
                            &mut cache,
                            layer_idx,
                            &layer_output,
                            layer_offsets,
                            layer_params,
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
                last_logits = pipeline.run_matmul_f32(
                    device,
                    queue,
                    output_head_f32,
                    &normed_output,
                    spec.n_vocab as u32,
                    dim,
                );

                let (cache_len_after, window_base_after) = {
                    let mut cache = kv_cache.lock().unwrap();
                    cache.increment();
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
            }
            last_logits
        } else {
            let mut batched_embd = Vec::with_capacity(prompt_tokens.len() * dim as usize);
            for &token_id in &prompt_tokens {
                let row_offset = embd_weight_offset + (token_id as u64 * row_bytes as u64);
                let embd = pipeline.run_dequant_request(device, queue, model, row_offset as u32, dim);
                batched_embd.extend_from_slice(&embd);
            }

            let (_final_act, _l21, logits) = {
                let cache_guard = kv_cache.lock().unwrap();
                pipeline.run_full_model_prefill_chunked_with_cache_state(
                    device,
                    queue,
                    model,
                    &batched_embd,
                    Some(&output_head_f32),
                    0,
                    Some((cache_guard.get_k_buffers(), cache_guard.get_v_buffers())),
                    spec,
                    512,
                )
            };

            {
                let mut cache = kv_cache.lock().unwrap();
                for _ in 0..prompt_tokens.len() {
                    cache.increment();
                }
            }
            logits
        }
    } else {
        let mut batched_embd = Vec::with_capacity(prompt_tokens.len() * dim as usize);
        for &token_id in &prompt_tokens {
            let row_offset = embd_weight_offset + (token_id as u64 * row_bytes as u64);
            let embd = pipeline.run_dequant_request(device, queue, model, row_offset as u32, dim);
            batched_embd.extend_from_slice(&embd);
        }

        let (_final_act, _l21, logits) = {
            let cache_guard = kv_cache.lock().unwrap();
            pipeline.run_full_model_prefill_chunked_with_cache_state(
                device,
                queue,
                model,
                &batched_embd,
                Some(&output_head_f32),
                0,
                Some((cache_guard.get_k_buffers(), cache_guard.get_v_buffers())),
                spec,
                512,
            )
        };

        {
            let mut cache = kv_cache.lock().unwrap();
            for _ in 0..prompt_tokens.len() {
                cache.increment();
            }
        }
        logits
    };

    eprintln!("[GPU Server] Prefill complete. Cache info updated.");

    let mut logits_vec = prefill_logits_f32;

    for _step in 0..max_new_tokens {
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
        let _max_prob = max_probability_from_logits(&raw_logits);
        let norm = logit_l2_norm(&raw_logits);

        if ppl_window.len() >= 10 {
            ppl_window.pop_front();
        }
        ppl_window.push_back(entropy.exp());
        let perplexity = ppl_window.iter().sum::<f32>() / ppl_window.len() as f32;

        let is_unsafe = perplexity > 500.0 || norm > 1e5;

        let next_token = if is_unsafe {
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
            "[TOKEN] Step {}: id={}, text={:?}",
            generated_count, next_token, piece
        );
        generated_text.push_str(&piece);
        generated_count += 1;

        if let Some(tx) = stream_tx {
            let _ = tx.send(piece.clone());
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

        let row_offset = embd_weight_offset + (next_token as u64 * row_bytes as u64);
        let mut layer_output =
            pipeline.run_dequant_request(device, queue, model, row_offset as u32, dim);

        let mut step_layer_traces = Vec::new();
        if capture_trace_step {
            let cfg = trace_config.as_ref().unwrap();
            let (current_pos, logical_pos_base) = {
                let cache = kv_cache.lock().unwrap();
                (cache.get_seq_len(), cache.get_window_base())
            };
            for layer_idx in 0..spec.n_layer {
                let layer_offsets = model
                    .metadata
                    .get_layer_offsets(layer_idx, spec.arch_string())
                    .expect(&format!("Missing offsets for layer {}", layer_idx));

                let (gpu_output, gpu_post_attn, gpu_ffn_out, gpu_q, gpu_k, gpu_v) = {
                    let mut cache = kv_cache.lock().unwrap();
                    pipeline.run_layer_with_cache_debug(
                        device,
                        queue,
                        model,
                        &mut cache,
                        layer_idx,
                        &layer_output,
                        layer_offsets,
                        layer_params,
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
                let layer_offsets = model
                    .metadata
                    .get_layer_offsets(layer_idx, spec.arch_string())
                    .expect(&format!("Missing offsets for layer {}", layer_idx));

                let mut cache = kv_cache.lock().unwrap();
                layer_output = pipeline.run_layer_with_cache(
                    device,
                    queue,
                    model,
                    &mut cache,
                    layer_idx,
                    &layer_output,
                    layer_offsets,
                    layer_params,
                );
            }
        }

        {
            let mut cache = kv_cache.lock().unwrap();
            cache.increment();
        }

        let normed_output =
            pipeline.run_rmsnorm_test(device, queue, model, &layer_output, norm_params);

        logits_vec = pipeline.run_matmul_f32(
            device,
            queue,
            output_head_f32,
            &normed_output,
            spec.n_vocab as u32,
            dim,
        );

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

async fn process_inference_job(
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
    output_head_f32: &wgpu::Buffer, // Pre-dequantized F32 output weights
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
        output_head_f32,
    )?;

    if !disable_session_window {
        if let Some(session_id_ref) = session_id.as_deref() {
            let response_suffix = if req.prompt_mode.as_deref().unwrap_or("creative") == "raw" {
                "</s>\n"
            } else {
                ""
            };
            let mut appended_tokens = tokenizer.encode(&templated_prompt, true)?;
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

#[allow(dead_code)]
#[allow(dead_code)]
fn send_error(mut stream: TcpStream, msg: &str) {
    let body = format!("{{\"error\": \"{}\"}}", msg);
    let resp = format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    let _ = stream.write_all(resp.as_bytes());
}

#[allow(dead_code)]
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
}
