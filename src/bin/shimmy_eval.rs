use airframe::backend::bindless::{
    loader::BindlessModel,
    pipeline::{BindlessPipeline, RMSNormParams},
};
use airframe::core::error::Result;
use airframe::core::model::Model as CpuModelContainer;
use airframe::core::spec::ModelSpec;
use airframe::core::tensor::Tensor;
use airframe::family::llama::LlamaModel;
use airframe::ops::dispatch::OpDispatcher;
use airframe::runtime::engine::Engine as CpuCore;
use airframe::runtime::kvcache::KvCache;

use airframe::core::dequant::dequantize_q6_k;
use airframe::core::model::GgufTensorInfo;
use airframe::core::weight_id::WeightId;
use chrono::Utc;
use clap::Parser;
use memmap2::Mmap;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use shimmytok::Tokenizer;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "cpu")]
    backend: String,

    #[arg(long, default_value = "wikitext")]
    task: String,

    #[arg(long)]
    wikitext_path: Option<PathBuf>,

    #[arg(long)]
    lambada_path: Option<PathBuf>,

    #[arg(long)]
    hellaswag_path: Option<PathBuf>,

    #[arg(long)]
    arc_path: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    arc_n_shot: usize,

    #[arg(long, default_value_t = 0)]
    arc_resume_from: usize,

    #[arg(long)]
    monitor: bool,

    #[arg(long, visible_alias = "limit")]
    max_eval_tokens: Option<usize>,

    #[arg(long, default_value_t = 0)]
    lambada_resume_from: usize,

    #[arg(long, default_value_t = 0)]
    hellaswag_resume_from: usize,

    #[arg(long)]
    max_chunks: Option<usize>,

    #[arg(long, default_value_t = 1)]
    l0probe_token: usize,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value_t = 1000)]
    det_runs: usize,

    #[arg(long, default_value_t = 3)]
    det_warmup_runs: usize,

    #[arg(long, default_value_t = 256)]
    det_max_tokens: usize,

    #[arg(long, default_value_t = 0.0)]
    det_temperature: f32,

    #[arg(long, default_value_t = 0.95)]
    det_top_p: f32,

    #[arg(long, default_value_t = 0)]
    det_top_k: usize,

    #[arg(long, default_value_t = 1.0)]
    det_repetition_penalty: f32,

    #[arg(long, default_value_t = 42)]
    det_seed: u64,

    #[arg(long, default_value_t = 2048)]
    det_context_window: usize,

    #[arg(long, default_value_t = 4)]
    det_attention_sink_token_count: usize,

    #[arg(long, default_value_t = 2048)]
    det_sliding_window_size: usize,

    #[arg(long)]
    det_output_dir: Option<PathBuf>,
}

struct Monitor {
    running: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Monitor {
    fn new(enabled: bool) -> Self {
        if !enabled {
            return Self {
                running: Arc::new(AtomicBool::new(false)),
                handle: None,
            };
        }

        println!("[Monitor] Starting nvidia-smi poller...");
        let running = Arc::new(AtomicBool::new(true));
        let r_clone = running.clone();

        let handle = std::thread::spawn(move || {
            let start = std::time::Instant::now();
            println!("Time(s), GPU_Util(%), Mem_Used(MB)");
            while r_clone.load(Ordering::Relaxed) {
                let output = Command::new("nvidia-smi")
                    .args(&[
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ])
                    .output();

                if let Ok(out) = output {
                    let s = String::from_utf8_lossy(&out.stdout);
                    let clean = s.trim();
                    println!("{:.2}, {}", start.elapsed().as_secs_f32(), clean);
                }

                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        });

        Self {
            running,
            handle: Some(handle),
        }
    }

    fn stop(self) {
        if let Some(h) = self.handle {
            self.running.store(false, Ordering::Relaxed);
            h.join().unwrap();
        }
    }
}

trait EvalEngine {
    fn name(&self) -> &str;
    fn process_prompt(&mut self, tokens: &[usize]) -> Result<Vec<f32>>;
    fn reset(&mut self);
    fn runtime_manifest(&self) -> serde_json::Value {
        json!({"engine": self.name()})
    }
}

struct CpuEvalEngine {
    inner: CpuCore,
    weights: HashMap<WeightId, Tensor>,
}

impl CpuEvalEngine {
    fn new(path: &Path) -> Result<Self> {
        println!("Loading CPU Model (dequantizing to F32)...");
        let container = CpuModelContainer::from_gguf(path)?;
        let spec = container.spec;
        let weights = container.weights;

        let llama_model = LlamaModel::from_spec(spec);
        let engine = CpuCore::new(llama_model);

        Ok(Self {
            inner: engine,
            weights,
        })
    }
}

impl EvalEngine for CpuEvalEngine {
    fn name(&self) -> &str {
        "CPU-Airframe"
    }

    fn process_prompt(&mut self, tokens: &[usize]) -> Result<Vec<f32>> {
        let logits = self.inner.prefill(tokens, &self.weights)?;
        Ok(logits.data.to_vec())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

struct GpuEvalEngine {
    pipeline: BindlessPipeline,
    model: BindlessModel,
    input_embd_table: Vec<f32>,
    vocab_size: usize,
    dim: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    weights_f32: Option<wgpu::Buffer>,
    kv_cache_k_layers: Vec<wgpu::Buffer>,
    kv_cache_v_layers: Vec<wgpu::Buffer>,
    spec: ModelSpec,
    adapter_name: String,
    adapter_backend: String,
    adapter_driver: String,
    adapter_driver_info: String,
    // State for incremental processing
    current_sequence: Vec<usize>,
    current_pos: usize,
}

fn load_output_head_f32_override(
    model_path: &Path,
    model: &BindlessModel,
    device: &wgpu::Device,
    spec: &ModelSpec,
) -> Result<Option<wgpu::Buffer>> {
    let output_weight_type = model
        .metadata
        .get_tensor_type("output.weight")
        .expect("output.weight type not found");

    if output_weight_type != 14 {
        return Ok(None);
    }

    println!("[GpuEvalEngine] output.weight is Q6_K; loading F32 head override...");

    let file = std::fs::File::open(model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let tensor_info = GgufTensorInfo {
        name: "output.weight".to_string(),
        dimensions: vec![spec.n_vocab, spec.n_embd],
        ggml_type: 14,
        offset: 0,
    };

    let tensor_f32 = dequantize_q6_k(&tensor_info, &mmap, model.metadata.data_start_offset)?;

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ShimmyEval Output Head F32"),
        contents: bytemuck::cast_slice(&tensor_f32.data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    println!(
        "[GpuEvalEngine] F32 head override ready: {:.2} MB",
        (tensor_f32.data.len() * 4) as f32 / 1024.0 / 1024.0
    );

    Ok(Some(buffer))
}

impl GpuEvalEngine {
    async fn new(path: &Path) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find wgpu adapter");
        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ShimmyEval"),
                required_features: wgpu::Features::empty() | wgpu::Features::TIMESTAMP_QUERY,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .unwrap();

        println!("Loading Embedding Table (CPU)...");
        let container = CpuModelContainer::from_gguf(path)?;
        let spec = container.spec;
        let emb = container
            .weights
            .get(&WeightId::TokenEmbed)
            .expect("Missing token_embd.weight");

        let model = BindlessModel::load_from_disk(&device, path, Some(&spec));
        let pipeline = BindlessPipeline::new(&device);
        let weights_f32 = load_output_head_f32_override(path, &model, &device, &spec)?;;
        let kv_size_per_buffer = spec.kv_cache_size_per_layer as u64;
        let mut kv_cache_k_layers = Vec::with_capacity(spec.n_layer);
        let mut kv_cache_v_layers = Vec::with_capacity(spec.n_layer);
        for i in 0..spec.n_layer {
            kv_cache_k_layers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ShimmyEval KV Cache K L{}", i)),
                size: kv_size_per_buffer,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            kv_cache_v_layers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ShimmyEval KV Cache V L{}", i)),
                size: kv_size_per_buffer,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }

        Ok(Self {
            pipeline,
            model,
            input_embd_table: emb.data.clone(),
            vocab_size: spec.n_vocab,
            dim: spec.n_embd,
            device,
            queue,
            weights_f32,
            kv_cache_k_layers,
            kv_cache_v_layers,
            spec,
            adapter_name: adapter_info.name,
            adapter_backend: format!("{:?}", adapter_info.backend),
            adapter_driver: adapter_info.driver,
            adapter_driver_info: adapter_info.driver_info,
            current_sequence: Vec::new(),
            current_pos: 0,
        })
    }
}

impl EvalEngine for GpuEvalEngine {
    fn name(&self) -> &str {
        "GPU-Bindless"
    }

    fn process_prompt(&mut self, tokens: &[usize]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(vec![0.0; self.vocab_size]);
        }

        // Stop at context limit
        if self.current_pos + tokens.len() > self.spec.n_ctx {
            return Err(airframe::core::error::LibshimmyError::Unsupported(
                format!("context limit {} exceeded", self.spec.n_ctx),
            ));
        }

        // Batch Prefill: Collect embeddings for entire prompt
        let batch_size = tokens.len();
        let mut batched_embd = Vec::with_capacity(batch_size * self.dim);

        for &token_id in tokens {
            self.current_sequence.push(token_id);
            let start = token_id * self.dim;
            let end = start + self.dim;
            if end > self.input_embd_table.len() {
                // Return zeros / error for OOV if not handled?
                // For now, push zeros if OOV (shouldn't happen with valid vocab)
                batched_embd.extend_from_slice(&vec![0.0; self.dim]);
            } else {
                batched_embd.extend_from_slice(&self.input_embd_table[start..end]);
            }
        }

        // Seq len is total tokens processed so far + this batch
        let seq_len = (self.current_pos + batch_size) as u32;

        // println!("[GPU] Processing batch of {} tokens. Pos: {} -> {}", batch_size, self.current_pos, seq_len);

        let (_pre_norm, _l21, logits) = self.pipeline.run_full_model_with_cache_state(
            &self.device,
            &self.queue,
            &self.model,
            &batched_embd,
            self.weights_f32.as_ref(),
            self.current_pos as u32,
            seq_len,
            Some((&self.kv_cache_k_layers, &self.kv_cache_v_layers)),
            &self.spec,
        ).expect("GPU forward pass failed");

        self.current_pos += batch_size;

        Ok(logits)
    }

    fn reset(&mut self) {
        self.current_sequence.clear();
        self.current_pos = 0;

        let kv_size_per_buffer = self.spec.kv_cache_size_per_layer;
        let zeros = vec![0u8; kv_size_per_buffer];
        for buffer in &self.kv_cache_k_layers {
            self.queue.write_buffer(buffer, 0, &zeros);
        }
        for buffer in &self.kv_cache_v_layers {
            self.queue.write_buffer(buffer, 0, &zeros);
        }
    }

    fn runtime_manifest(&self) -> serde_json::Value {
        json!({
            "engine": self.name(),
            "wgpu_adapter_name": self.adapter_name,
            "wgpu_backend": self.adapter_backend,
            "wgpu_driver": self.adapter_driver,
            "wgpu_driver_info": self.adapter_driver_info,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    println!("Shimmy Eval v2.3");

    let monitor = Monitor::new(args.monitor);

    println!("Loading Tokenizer...");
    let tokenizer = Tokenizer::from_gguf_file(&args.model).map_err(|e| {
        airframe::core::error::LibshimmyError::FixtureError {
            msg: format!("Tokenizer load failed: {:?}", e),
        }
    })?;

    if args.task == "l0probe" {
        run_l0probe(&args).await?;
        monitor.stop();
        return Ok(());
    }

    let mut engine: Box<dyn EvalEngine> = if args.backend == "gpu" {
        println!("Initializing GPU Backend...");
        let gpu_engine = GpuEvalEngine::new(&args.model).await?;
        Box::new(gpu_engine)
    } else {
        println!("Initializing CPU Backend...");
        let cpu_engine = CpuEvalEngine::new(&args.model)?;
        Box::new(cpu_engine)
    };

    println!("Engine: {}", engine.name());

    match args.task.as_str() {
        "wikitext" => run_wikitext(&args, &tokenizer, &mut *engine).await?,
        "lambada" => run_lambada(&args, &tokenizer, &mut *engine).await?,
        "hellaswag" => run_hellaswag(&args, &tokenizer, &mut *engine).await?,
        "arc" | "arc-easy" | "arc-challenge" => run_arc(&args, &tokenizer, &mut *engine).await?,
        "determinism" => run_determinism(&args, &tokenizer, &mut *engine).await?,
        // Unknown task string is a CLI misuse, not a runtime error — panic is appropriate here.
        _ => panic!(
            "Unknown task: {}. Available: wikitext, lambada, hellaswag, arc, determinism",
            args.task
        ),
    }

    monitor.stop();
    Ok(())
}

#[derive(Serialize)]
struct DeterminismConfig {
    trial_uuid: String,
    total_runs: usize,
    backend: String,
    model: String,
    model_sha256: String,
    model_size_bytes: u64,
    prompt: String,
    prompt_sha256: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    seed: u64,
    context_window: usize,
    attention_sink_token_count: usize,
    sliding_window_size: usize,
    airframe_commit: String,
    warmup_runs: usize,
}

#[derive(Serialize)]
struct RunMeta {
    run_number: usize,
    tokens_sha256: String,
    text_sha256: String,
    tokens_generated: usize,
    wall_clock_ms: u128,
    completed_at_utc: String,
    matches_canonical: bool,
    canonical_hash: String,
    first_divergence_token_pos: Option<usize>,
    expected_token_at_divergence: Option<u32>,
    actual_token_at_divergence: Option<u32>,
}

#[derive(Serialize)]
struct HashManifest {
    trial_uuid: String,
    backend: String,
    runs: usize,
    canonical_hash: String,
    match_count: usize,
    mismatches: Vec<serde_json::Value>,
    run_hashes: Vec<String>,
    started_at_utc: String,
    completed_at_utc: String,
}

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let val = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (val >> 40) as f32 / 16777216.0
    }
}

fn to_hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    to_hex_lower(&digest)
}

fn sha256_file(path: &Path) -> Result<String> {
    let data = fs::read(path)?;
    Ok(sha256_bytes(&data))
}

fn git_commit_hash() -> String {
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn rustc_version_string() -> String {
    let output = Command::new("rustc").arg("--version").output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn sample_token_from_logits(
    logits: &mut [f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    repetition_penalty: f32,
    recent_tokens: &[u32],
    rng: &mut XorShift64,
) -> usize {
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

    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    for logit in logits.iter_mut() {
        *logit /= temperature;
    }

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if top_k > 0 && top_k < indexed.len() {
        indexed.truncate(top_k);
    }

    let mut cum = 0.0_f32;
    let mut cutoff = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    let nucleus = &indexed[..cutoff];
    let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
    let r = rng.next_f32() * nucleus_sum;

    let mut acc = 0.0_f32;
    for &(idx, p) in nucleus {
        acc += p;
        if acc >= r {
            return idx;
        }
    }

    nucleus.last().map(|(i, _)| *i).unwrap_or(0)
}

fn find_first_divergence(
    expected: &[u32],
    actual: &[u32],
) -> Option<(usize, Option<u32>, Option<u32>)> {
    let min_len = expected.len().min(actual.len());
    for i in 0..min_len {
        if expected[i] != actual[i] {
            return Some((i, Some(expected[i]), Some(actual[i])));
        }
    }
    if expected.len() != actual.len() {
        return Some((
            min_len,
            expected.get(min_len).copied(),
            actual.get(min_len).copied(),
        ));
    }
    None
}

fn write_tokens_bin(path: &Path, tokens: &[u32]) -> Result<()> {
    let mut buf = Vec::with_capacity(tokens.len() * 4);
    for &t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    fs::write(path, &buf)?;
    Ok(())
}

fn generate_tokens_once(
    args: &Args,
    tokenizer: &Tokenizer,
    engine: &mut dyn EvalEngine,
    prompt_tokens: &[usize],
) -> Result<(Vec<u32>, String)> {
    let eos = tokenizer.eos_token() as usize;
    let mut rng = XorShift64::new(args.det_seed);
    let mut recent_tokens: Vec<u32> = Vec::new();

    engine.reset();
    let mut logits = engine.process_prompt(prompt_tokens)?;

    let mut out_tokens = Vec::with_capacity(args.det_max_tokens);
    let mut out_text = String::new();

    for _ in 0..args.det_max_tokens {
        let next = sample_token_from_logits(
            &mut logits,
            args.det_temperature,
            args.det_top_p,
            args.det_top_k,
            args.det_repetition_penalty,
            &recent_tokens,
            &mut rng,
        );

        if next == eos {
            break;
        }

        let next_u32 = next as u32;
        out_tokens.push(next_u32);
        let piece = tokenizer.decode_single(next_u32, true).map_err(|e| {
            airframe::core::error::LibshimmyError::FixtureError {
                msg: format!("Tokenizer decode failed: {:?}", e),
            }
        })?;
        out_text.push_str(&piece);

        recent_tokens.push(next_u32);
        if recent_tokens.len() > 64 {
            recent_tokens.remove(0);
        }

        logits = engine.process_prompt(&[next])?;
    }

    Ok((out_tokens, out_text))
}

async fn run_determinism(
    args: &Args,
    tokenizer: &Tokenizer,
    engine: &mut dyn EvalEngine,
) -> Result<()> {
    println!("=== DETERMINISM PROOF RUNNER START ===");

    let prompt = args.prompt.clone().unwrap_or_else(|| "Hello".to_string());

    let started_at = Utc::now();
    let trial_uuid = format!(
        "{}-{}",
        args.backend,
        started_at.format("%Y%m%dT%H%M%S%.3fZ")
    );

    let output_dir = args
        .det_output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("artifacts/determinism_proof/{}", trial_uuid)));
    fs::create_dir_all(&output_dir)?;

    let model_meta = fs::metadata(&args.model)?;
    let model_hash = sha256_file(&args.model)?;
    let prompt_hash = sha256_bytes(prompt.as_bytes());

    let cfg = DeterminismConfig {
        trial_uuid: trial_uuid.clone(),
        total_runs: args.det_runs,
        backend: args.backend.clone(),
        model: args.model.display().to_string(),
        model_sha256: model_hash.clone(),
        model_size_bytes: model_meta.len(),
        prompt,
        prompt_sha256: prompt_hash,
        max_tokens: args.det_max_tokens,
        temperature: args.det_temperature,
        top_p: args.det_top_p,
        top_k: args.det_top_k,
        repetition_penalty: args.det_repetition_penalty,
        seed: args.det_seed,
        context_window: args.det_context_window,
        attention_sink_token_count: args.det_attention_sink_token_count,
        sliding_window_size: args.det_sliding_window_size,
        airframe_commit: git_commit_hash(),
        warmup_runs: args.det_warmup_runs,
    };

    let cfg_json = serde_json::to_vec_pretty(&cfg)?;
    let cfg_path = output_dir.join("config.json");
    fs::write(&cfg_path, &cfg_json)?;
    fs::write(
        output_dir.join("config.json.sha256"),
        format!("{}  config.json\n", sha256_bytes(&cfg_json)),
    )?;
    fs::write(
        output_dir.join("model_file.sha256"),
        format!("{}  {}\n", model_hash, args.model.display()),
    )?;

    let hardware_manifest = json!({
        "trial_uuid": trial_uuid,
        "hostname": std::env::var("COMPUTERNAME").or_else(|_| std::env::var("HOSTNAME")).unwrap_or_else(|_| "unknown".to_string()),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "rustc_version": rustc_version_string(),
        "airframe_commit": cfg.airframe_commit,
        "engine": engine.runtime_manifest(),
    });
    let hw_json = serde_json::to_vec_pretty(&hardware_manifest)?;
    fs::write(output_dir.join("hardware_manifest.json"), &hw_json)?;
    fs::write(
        output_dir.join("hardware_manifest.json.sha256"),
        format!("{}  hardware_manifest.json\n", sha256_bytes(&hw_json)),
    )?;

    let prompt_tokens_u32 = tokenizer.encode(&cfg.prompt, true).map_err(|e| {
        airframe::core::error::LibshimmyError::FixtureError {
            msg: format!("Tokenizer encode failed: {:?}", e),
        }
    })?;
    let prompt_tokens: Vec<usize> = prompt_tokens_u32.iter().map(|&t| t as usize).collect();

    println!("Warmup: {} runs", args.det_warmup_runs);
    for i in 0..args.det_warmup_runs {
        let _ = generate_tokens_once(args, tokenizer, engine, &prompt_tokens)?;
        println!("Warmup {}/{} complete", i + 1, args.det_warmup_runs);
    }

    let trial_start = Utc::now();
    let mut canonical_hash = String::new();
    let mut canonical_tokens: Vec<u32> = Vec::new();
    let mut run_hashes = Vec::with_capacity(args.det_runs);
    let mut mismatches = Vec::new();
    let mut match_count = 0usize;

    let keep_mid = args.det_runs / 2;

    for run in 0..args.det_runs {
        let run_start = std::time::Instant::now();
        let (tokens, text) = generate_tokens_once(args, tokenizer, engine, &prompt_tokens)?;

        let token_bytes = {
            let mut b = Vec::with_capacity(tokens.len() * 4);
            for &t in &tokens {
                b.extend_from_slice(&t.to_le_bytes());
            }
            b
        };

        let tokens_hash = sha256_bytes(&token_bytes);
        let text_hash = sha256_bytes(text.as_bytes());
        let run_dir_prefix = format!("run_{:04}", run);
        let tokens_path = output_dir.join(format!("{}_tokens.bin", run_dir_prefix));
        let text_path = output_dir.join(format!("{}_text.txt", run_dir_prefix));
        let meta_path = output_dir.join(format!("{}_meta.json", run_dir_prefix));

        write_tokens_bin(&tokens_path, &tokens)?;
        fs::write(&text_path, text.as_bytes())?;

        let mut matches_canonical = true;
        let mut div_pos = None;
        let mut expected = None;
        let mut actual = None;

        if run == 0 {
            canonical_hash = tokens_hash.clone();
            canonical_tokens = tokens.clone();
        } else if tokens_hash != canonical_hash {
            matches_canonical = false;
            if let Some((p, e, a)) = find_first_divergence(&canonical_tokens, &tokens) {
                div_pos = Some(p);
                expected = e;
                actual = a;
            }
            mismatches.push(json!({
                "run": run,
                "expected_hash": canonical_hash,
                "actual_hash": tokens_hash,
                "first_divergence_token_pos": div_pos,
                "expected_token": expected,
                "actual_token": actual,
            }));
        }

        if matches_canonical {
            match_count += 1;
        }
        run_hashes.push(tokens_hash.clone());

        let meta = RunMeta {
            run_number: run,
            tokens_sha256: tokens_hash,
            text_sha256: text_hash,
            tokens_generated: tokens.len(),
            wall_clock_ms: run_start.elapsed().as_millis(),
            completed_at_utc: Utc::now().to_rfc3339(),
            matches_canonical,
            canonical_hash: canonical_hash.clone(),
            first_divergence_token_pos: div_pos,
            expected_token_at_divergence: expected,
            actual_token_at_divergence: actual,
        };
        fs::write(meta_path, serde_json::to_vec_pretty(&meta)?)?;

        let keep_full = run == 0 || run == keep_mid || run == args.det_runs.saturating_sub(1);
        if !keep_full {
            let _ = fs::remove_file(tokens_path);
            let _ = fs::remove_file(text_path);
        }

        if run % 50 == 0 || run + 1 == args.det_runs {
            println!(
                "DETERMINISM_PROGRESS run={}/{} matches={} mismatches={} canonical={}",
                run + 1,
                args.det_runs,
                match_count,
                mismatches.len(),
                canonical_hash
            );
        }
    }

    let manifest = HashManifest {
        trial_uuid: cfg.trial_uuid.clone(),
        backend: cfg.backend.clone(),
        runs: args.det_runs,
        canonical_hash: canonical_hash.clone(),
        match_count,
        mismatches,
        run_hashes,
        started_at_utc: trial_start.to_rfc3339(),
        completed_at_utc: Utc::now().to_rfc3339(),
    };
    fs::write(
        output_dir.join("hash_manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    println!(
        "DETERMINISM_COMPLETE runs={} matches={} mismatches={} canonical_hash={} output_dir={}",
        args.det_runs,
        match_count,
        args.det_runs.saturating_sub(match_count),
        canonical_hash,
        output_dir.display(),
    );

    Ok(())
}

async fn run_wikitext(
    args: &Args,
    tokenizer: &Tokenizer,
    engine: &mut dyn EvalEngine,
) -> Result<()> {
    println!("=== WIKITEXT-2 BENCHMARK START ===");
    let path = args
        .wikitext_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("fixtures/wikitext-2-raw/wiki.test.raw"));

    if !path.exists() {
        println!("ERROR: WikiText-2 file not found at {:?}. Skipping.", path);
        return Ok(());
    }

    println!("Loading WikiText-2 from: {:?}", path);

    let file = File::open(&path).expect("Failed to open WikiText-2");
    let reader = std::io::BufReader::new(file);

    let ctx_size = 2048;
    let mut nll_sum = 0.0;
    let mut count = 0;
    let mut total_tokens = 0;

    let benchmark_start = std::time::Instant::now();

    println!("=== PHASE 1: TOKENIZATION ===");
    let tokenization_start = std::time::Instant::now();

    // Collect all valid tokens first
    let mut all_tokens = Vec::new();
    let mut line_count = 0;
    let mut skipped_lines = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            skipped_lines += 1;
            continue;
        }

        // Try to encode this line
        let tokens = match tokenizer.encode(&line, true) {
            Ok(t) => t,
            Err(e) => {
                println!(
                    "WARNING: Failed to encode line {}: '{}...' - Error: {:?}, SKIPPING",
                    line_count + 1,
                    &line.chars().take(50).collect::<String>(),
                    e
                );
                skipped_lines += 1;
                continue;
            }
        };

        if tokens.len() < 2 {
            println!(
                "WARNING: Line {} too short ({} tokens), skipping",
                line_count + 1,
                tokens.len()
            );
            skipped_lines += 1;
            continue;
        }

        all_tokens.extend(tokens);
        line_count += 1;

        if line_count % 100 == 0 {
            let elapsed = tokenization_start.elapsed().as_secs_f64();
            println!(
                "Tokenized {} lines ({} skipped) - {} total tokens so far - {:.2} lines/sec",
                line_count,
                skipped_lines,
                all_tokens.len(),
                line_count as f64 / elapsed
            );
        }
    }

    let tokenization_time = tokenization_start.elapsed().as_secs_f64();
    println!("=== TOKENIZATION COMPLETE ===");
    println!(
        "Processed lines: {} (skipped: {})",
        line_count, skipped_lines
    );
    println!("Total tokens: {}", all_tokens.len());
    println!(
        "Tokenization time: {:.2}s ({:.2} tokens/sec)",
        tokenization_time,
        all_tokens.len() as f64 / tokenization_time
    );

    // Convert to usize and optionally cap run size for safety
    let mut tokens_usize: Vec<usize> = all_tokens.iter().map(|&t| t as usize).collect();
    if let Some(max_tokens) = args.max_eval_tokens {
        if tokens_usize.len() > max_tokens {
            tokens_usize.truncate(max_tokens);
            println!("Safety cap: truncated evaluation to {} tokens", max_tokens);
        }
    }

    // Continuous mode: if --keep-sink is set, process ALL tokens as one stream
    let chunks: Vec<&[usize]> = {
        let mut c: Vec<&[usize]> = tokens_usize.chunks(ctx_size).collect();
        if let Some(max_chunks) = args.max_chunks {
            if c.len() > max_chunks {
                c.truncate(max_chunks);
                println!("Safety cap: truncated evaluation to {} chunks", max_chunks);
            }
        }
        c
    };
    let total_chunks = chunks.len();

    println!("=== PHASE 2: GPU PROCESSING ===");
    println!(
        "Will process {} chunks of max {} tokens each",
        total_chunks, ctx_size
    );
    println!("Total tokens to process: {}", tokens_usize.len());

    let processing_start = std::time::Instant::now();

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_start = std::time::Instant::now();
        println!("");
        println!(
            ">>> STARTING CHUNK {}/{} ({} tokens)",
            i + 1,
            total_chunks,
            chunk.len()
        );

        if chunk.len() < 2 {
            println!(
                "WARNING: Chunk {} too small ({} tokens), skipping",
                i + 1,
                chunk.len()
            );
            continue;
        }

        println!("Resetting engine state...");
        engine.reset();
        println!("Engine reset complete");

        // --- BATCH OPTIMIZATION START (Auto-patched) ---
        // Instead of token-by-token, we use the engine's batch capability.
        // Input: chunk[0..N-1] -> Prediction: chunk[1..N]

        let mut chunk_predictions = 0;
        let mut chunk_nll_sum = 0.0_f32;

        if chunk.len() > 1 {
            println!("  [Batch] Processing {} tokens...", chunk.len() - 1);
            let batch_start = std::time::Instant::now();

            // Feed all but the last token as input context
            let input_tokens = &chunk[0..chunk.len() - 1];
            // Expect to predict from index 1 to end
            let targets = &chunk[1..chunk.len()];

            match engine.process_prompt(input_tokens) {
                Ok(all_logits) => {
                    let batch_ms = batch_start.elapsed().as_millis();
                    let vocab_size = all_logits.len() / input_tokens.len();
                    println!(
                        "  [Batch] Forward pass complete in {}ms ({:.2} tok/s)",
                        batch_ms,
                        input_tokens.len() as f64 / batch_start.elapsed().as_secs_f64()
                    );

                    for (i, &target) in targets.iter().enumerate() {
                        // logits for input[i] are at index i*vocab_size
                        let start_idx = i * vocab_size;
                        let end_idx = start_idx + vocab_size;

                        if end_idx <= all_logits.len() {
                            let logits_slice = &all_logits[start_idx..end_idx];
                            let nll = token_nll_from_logits(logits_slice, target);

                            if nll.is_finite() {
                                nll_sum += nll;
                                chunk_nll_sum += nll;
                            } else {
                                nll_sum += 100.0;
                                chunk_nll_sum += 100.0;
                            }
                            count += 1;
                            chunk_predictions += 1;
                        }
                    }
                }
                Err(e) => {
                    println!("ERROR: Batch computation failed: {:?}", e);
                }
            }
        }
        // --- BATCH OPTIMIZATION END ---

        let chunk_time = chunk_start.elapsed().as_secs_f64();
        let chunk_tokens_per_sec = chunk.len() as f64 / chunk_time;
        let chunk_nll_avg = if chunk_predictions > 0 {
            chunk_nll_sum / chunk_predictions as f32
        } else {
            f32::NAN
        };

        println!(
            "<<< CHUNK {}/{} COMPLETE - {} predictions, {:.2}s ({:.2} tokens/sec)",
            i + 1,
            total_chunks,
            chunk_predictions,
            chunk_time,
            chunk_tokens_per_sec
        );
        println!(
            "CHUNK_NLL chunk={} sum={:.8} avg={:.8} count={}",
            i + 1,
            chunk_nll_sum,
            chunk_nll_avg,
            chunk_predictions
        );

        total_tokens += chunk.len();

        // Progress update
        let total_elapsed = processing_start.elapsed().as_secs_f64();
        let overall_tokens_per_sec = total_tokens as f64 / total_elapsed;
        let percent_complete = (i + 1) as f64 / total_chunks as f64 * 100.0;

        println!("PROGRESS: {:.1}% complete - {} total tokens processed - {:.2} tokens/sec overall - elapsed: {:.1}s", 
                percent_complete, total_tokens, overall_tokens_per_sec, total_elapsed);
    }

    println!("");

    if count == 0 {
        println!("No predictions were generated; skipping perplexity calculation.");
        return Ok(());
    }

    let ppl = (nll_sum / count as f32).exp();
    println!("WikiText-2 Perplexity: {:.4}", ppl);
    println!("Processed {} tokens, {} predictions", total_tokens, count);
    Ok(())
}

async fn run_lambada(
    args: &Args,
    tokenizer: &Tokenizer,
    engine: &mut dyn EvalEngine,
) -> Result<()> {
    println!("=== LAMBADA BENCHMARK START ===");
    let run_start = std::time::Instant::now();

    let path = args
        .lambada_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("fixtures/lambada_test.jsonl"));

    if !path.exists() {
        println!("ERROR: LAMBADA file not found at {:?}.", path);
        println!("Hint: provide --lambada-path <path_to_lambada_test.jsonl>");
        return Ok(());
    }

    println!("Loading LAMBADA from: {:?}", path);

    let file = File::open(&path).expect("Failed to open LAMBADA dataset");
    let reader = std::io::BufReader::new(file);

    #[derive(serde::Deserialize)]
    struct LambadaRow {
        text: String,
    }

    let mut sample_index = 0usize;
    let mut evaluated = 0usize;
    let resume_from = args.lambada_resume_from;
    let mut matches = 0usize;
    let mut parse_errors = 0usize;
    let mut tokenization_errors = 0usize;
    let mut skipped_short = 0usize;
    let mut lines_seen = 0usize;
    let sample_cap = args.max_eval_tokens;

    if resume_from > 0 {
        println!("Resuming LAMBADA from evaluated sample {}", resume_from);
    }

    for line in reader.lines() {
        let line = line?;
        lines_seen += 1;

        if line.trim().is_empty() {
            continue;
        }

        let row: LambadaRow = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(_) => {
                parse_errors += 1;
                continue;
            }
        };

        let tokens = match tokenizer.encode(&row.text, true) {
            Ok(t) => t,
            Err(_) => {
                tokenization_errors += 1;
                continue;
            }
        };

        if tokens.len() < 2 {
            skipped_short += 1;
            continue;
        }

        let prompt: Vec<usize> = tokens[..tokens.len() - 1]
            .iter()
            .map(|&t| t as usize)
            .collect();
        let target = tokens[tokens.len() - 1] as usize;

        sample_index += 1;

        if sample_index <= resume_from {
            continue;
        }

        engine.reset();
        let logits = engine.process_prompt(&prompt)?;

        let best_token_id = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0);

        if best_token_id == target {
            matches += 1;
        }
        evaluated += 1;

        if evaluated % 10 == 0 {
            let elapsed = run_start.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 {
                evaluated as f64 / elapsed
            } else {
                0.0
            };
            let acc = (matches as f32 / evaluated as f32) * 100.0;
            println!(
                "LAMBADA_HEARTBEAT sample={} resumed_eval={} matches={} acc={:.2}% rate={:.2} samples/sec elapsed={:.1}s",
                sample_index,
                evaluated,
                matches,
                acc,
                rate,
                elapsed
            );
        }

        if sample_index % 50 == 0 {
            let acc = (matches as f32 / evaluated as f32) * 100.0;
            println!(
                "LAMBADA_PROGRESS samples={} accuracy={:.2}% (lines_seen={})",
                sample_index, acc, lines_seen
            );
        }

        if let Some(cap) = sample_cap {
            if sample_index >= cap {
                println!(
                    "Safety cap: truncated LAMBADA evaluation to {} samples",
                    cap
                );
                break;
            }
        }
    }

    if evaluated == 0 {
        println!("No valid LAMBADA samples were evaluated after resume point.");
        return Ok(());
    }

    let acc = (matches as f32 / evaluated as f32) * 100.0;
    println!("LAMBADA Accuracy: {:.4}%", acc);
    println!(
        "Processed {} resumed samples (matches={}, resume_from={}, final_sample_index={}, parse_errors={}, tokenization_errors={}, skipped_short={})",
        evaluated,
        matches,
        resume_from,
        sample_index,
        parse_errors,
        tokenization_errors,
        skipped_short
    );

    Ok(())
}

async fn run_l0probe(args: &Args) -> Result<()> {
    println!("=== L0.2 RMSNorm CPU↔GPU Probe ===");

    let container = CpuModelContainer::from_gguf(&args.model)?;
    let spec = container.spec;
    let weights = container.weights;

    let dim = spec.n_embd;
    let token_id = args.l0probe_token;

    let emb = weights
        .get(&WeightId::TokenEmbed)
        .expect("Missing token_embd.weight");

    let start = token_id * dim;
    let end = start + dim;
    if end > emb.data.len() {
        return Err(airframe::core::error::LibshimmyError::Unsupported(format!(
            "l0probe token {} out of range for embedding table",
            token_id
        )));
    }

    let input = &emb.data[start..end];
    let attn_norm = weights
        .get(&WeightId::AttnNorm { layer: 0 })
        .expect("Missing blk.0.attn_norm.weight");

    let cpu_l02 = rmsnorm_reference(input, &attn_norm.data, spec.rms_eps);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find wgpu adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("L0Probe"),
            required_features: wgpu::Features::empty() | wgpu::Features::TIMESTAMP_QUERY,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("Failed to create wgpu device");

    let model = BindlessModel::load_from_disk(&device, &args.model, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    let norm_offset = model
        .metadata
        .get_tensor_offset("blk.0.attn_norm.weight")
        .expect("Missing blk.0.attn_norm.weight offset");

    let params = RMSNormParams {
        count: dim as u32,
        weights_offset: norm_offset as u32,
        bias_offset: 0,
        eps: spec.rms_eps,
        norm_type: 0,
    };

    let gpu_l02 = pipeline.run_rmsnorm_test(&device, &queue, &model, input, params);

    let max_abs_err = cpu_l02
        .iter()
        .zip(gpu_l02.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let cpu_absmax = cpu_l02.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gpu_absmax = gpu_l02.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);

    println!("token_id={} dim={} eps={}", token_id, dim, spec.rms_eps);
    println!("CPU L0.2 absmax: {:.8}", cpu_absmax);
    println!("GPU L0.2 absmax: {:.8}", gpu_absmax);
    println!("L0.2 max_abs_err: {:.8}", max_abs_err);
    println!("CPU first20: {:?}", &cpu_l02[..20.min(cpu_l02.len())]);
    println!("GPU first20: {:?}", &gpu_l02[..20.min(gpu_l02.len())]);

    let artifact = format!(
        "token_id={token}\ncount={count}\neps={eps}\ncpu_absmax={cpu_absmax:.8}\ngpu_absmax={gpu_absmax:.8}\nmax_abs_err={max_abs_err:.8}\ncpu_first20={cpu_first20:?}\ngpu_first20={gpu_first20:?}\n",
        token = token_id,
        count = dim,
        eps = spec.rms_eps,
        cpu_absmax = cpu_absmax,
        gpu_absmax = gpu_absmax,
        max_abs_err = max_abs_err,
        cpu_first20 = &cpu_l02[..20.min(cpu_l02.len())],
        gpu_first20 = &gpu_l02[..20.min(gpu_l02.len())],
    );

    std::fs::write("artifacts/l0_2_probe.txt", artifact)
        .expect("Failed to write artifacts/l0_2_probe.txt");

    // L0.4 probe (post-attention residual)
    let ops = OpDispatcher::new();
    let mut kv_cache = KvCache::new(
        spec.n_ctx,
        spec.n_layer,
        spec.n_head_kv,
        spec.n_embd / spec.n_head,
    );

    let input_tensor = Tensor::new(input.to_vec(), vec![1, dim])?;
    let attn_input = ops.rmsnorm(&input_tensor, attn_norm, spec.rms_eps)?;

    let q_weight = weights
        .get(&WeightId::AttnQ { layer: 0 })
        .expect("Missing blk.0.attn_q.weight");
    let k_weight = weights
        .get(&WeightId::AttnK { layer: 0 })
        .expect("Missing blk.0.attn_k.weight");
    let v_weight = weights
        .get(&WeightId::AttnV { layer: 0 })
        .expect("Missing blk.0.attn_v.weight");
    let o_weight = weights
        .get(&WeightId::AttnO { layer: 0 })
        .expect("Missing blk.0.attn_output.weight");
    let ffn_norm_weight = weights
        .get(&WeightId::FfnNorm { layer: 0 })
        .expect("Missing blk.0.ffn_norm.weight");
    let gate_weight = weights
        .get(&WeightId::FfnGate { layer: 0 })
        .expect("Missing blk.0.ffn_gate.weight");
    let up_weight = weights
        .get(&WeightId::FfnUp { layer: 0 })
        .expect("Missing blk.0.ffn_up.weight");
    let down_weight = weights
        .get(&WeightId::FfnDown { layer: 0 })
        .expect("Missing blk.0.ffn_down.weight");

    let attn_output = ops.attention_with_cache(
        &attn_input,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        spec.n_head,
        spec.n_head_kv,
        spec.n_embd / spec.n_head,
        &[0],
        spec.rope_base,
        spec.rope_dim,
        spec.rope_scale,
        0,
        &mut kv_cache,
        None, // no QK norm for non-Qwen3
    )?;

    let post_attn = ops.add(&input_tensor, &attn_output)?;
    let cpu_l03 = attn_output.data.clone();
    let cpu_l04 = post_attn.data;

    let offsets = model
        .metadata
        .get_layer_offsets(0, "tinyllama")
        .expect("Missing layer 0 offsets");
    let layer_params = airframe::backend::bindless::pipeline::LayerParams {
        dim: spec.n_embd as u32,
        head_count: spec.n_head as u32,
        head_count_kv: spec.n_head_kv as u32,
        head_dim: (spec.n_embd / spec.n_head) as u32,
        rope_dim: spec.rope_dim as u32,
        rms_eps: spec.rms_eps,
        ffn_dim: spec.ff_dim as u32,
        temp_stride: spec.temp_buffer_size as u32,
        quant_type: 0,
        attn_logit_softcap: 0.0,
        post_norm_enabled: 0,
        qk_norm_enabled: 0,
        layer_norm_enabled: 0,
    };

    let (gpu_l04_mid, gpu_l0_final) = pipeline.run_layer_stepwise_test(
        &device,
        &queue,
        &model,
        input,
        offsets,
        layer_params,
        true,
    );

    let l04_max_abs_err = cpu_l04
        .iter()
        .zip(gpu_l04_mid.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let gpu_l05 = {
        let ffn_norm_offset = model
            .metadata
            .get_tensor_offset("blk.0.ffn_norm.weight")
            .expect("Missing blk.0.ffn_norm.weight offset");
        let ffn_norm_params = RMSNormParams {
            count: dim as u32,
            weights_offset: ffn_norm_offset as u32,
            bias_offset: 0,
            eps: spec.rms_eps,
            norm_type: 0,
        };
        pipeline.run_rmsnorm_test(&device, &queue, &model, &gpu_l04_mid, ffn_norm_params)
    };

    let post_attn_tensor = Tensor::new(cpu_l04.clone(), vec![1, dim])?;
    let cpu_l05_tensor = ops.rmsnorm(&post_attn_tensor, ffn_norm_weight, spec.rms_eps)?;
    let cpu_l05 = cpu_l05_tensor.data.clone();

    let l05_max_abs_err = cpu_l05
        .iter()
        .zip(gpu_l05.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let cpu_l06_tensor = ops.ffn_swiglu(&cpu_l05_tensor, gate_weight, up_weight, down_weight)?;
    let cpu_l06 = cpu_l06_tensor.data.clone();

    let gpu_l06: Vec<f32> = gpu_l0_final
        .iter()
        .zip(gpu_l04_mid.iter())
        .map(|(final_v, post_attn_v)| final_v - post_attn_v)
        .collect();

    let l06_max_abs_err = cpu_l06
        .iter()
        .zip(gpu_l06.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // L0.22 probe (final logits for single token)
    let llama_model = LlamaModel::from_spec(spec.clone());
    let mut cpu_engine = CpuCore::new(llama_model);
    let cpu_logits_tensor = cpu_engine.prefill(&[token_id], &weights)?;
    let cpu_l22 = cpu_logits_tensor.data;

    let (gpu_l20, gpu_l21, gpu_l22) = pipeline
        .run_full_model_with_cache_state(&device, &queue, &model, input, None, 0, 1, None, &spec)
        .expect("GPU forward pass failed");

    // F12 isolation: run with F32-dequantized output head override.
    let output_head_f32 = {
        let output_weight_type = model
            .metadata
            .get_tensor_type("output.weight")
            .expect("output.weight type not found");

        if output_weight_type != 14 {
            None
        } else {
            let file = std::fs::File::open(&args.model)?;
            let mmap = unsafe { Mmap::map(&file)? };

            let tensor_info = GgufTensorInfo {
                name: "output.weight".to_string(),
                dimensions: vec![spec.n_vocab, spec.n_embd],
                ggml_type: 14,
                offset: 0,
            };

            let tensor_f32 =
                dequantize_q6_k(&tensor_info, &mmap, model.metadata.data_start_offset)?;

            Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("L0Probe Output Head F32"),
                    contents: bytemuck::cast_slice(&tensor_f32.data),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            )
        }
    };

    let (_gpu_l20_f32_head, _gpu_l21_f32_head, gpu_l22_f32_head) =
        if let Some(ref f32_head) = output_head_f32 {
            pipeline.run_full_model_with_cache_state(
                &device,
                &queue,
                &model,
                input,
                Some(f32_head),
                0,
                1,
                None,
                &spec,
            ).expect("GPU forward pass failed (f32 head)")
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

    let cpu_l20 = {
        let token_embed = weights
            .get(&WeightId::TokenEmbed)
            .expect("Missing token_embd.weight");
        let row_start = token_id * dim;
        let row_end = row_start + dim;
        let input_embd = token_embed.data[row_start..row_end].to_vec();
        let mut hidden = Tensor::new(input_embd, vec![1, dim])?;

        let mut full_kv_cache = KvCache::new(
            spec.n_ctx,
            spec.n_layer,
            spec.n_head_kv,
            spec.n_embd / spec.n_head,
        );
        let position_ids = vec![0usize];
        let full_model = LlamaModel::from_spec(spec.clone());

        for layer in full_model.layers.iter() {
            hidden = layer.forward(&hidden, &weights, &mut full_kv_cache, &position_ids, &ops)?;
        }

        hidden.data
    };

    let output_norm_weight = weights
        .get(&WeightId::OutputNorm)
        .expect("Missing output_norm.weight");
    let cpu_l20_tensor = Tensor::new(cpu_l20.clone(), vec![1, dim])?;
    let cpu_l21_tensor = ops.rmsnorm(&cpu_l20_tensor, output_norm_weight, spec.rms_eps)?;
    let cpu_l21 = cpu_l21_tensor.data;

    let l20_max_abs_err = cpu_l20
        .iter()
        .zip(gpu_l20.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let l21_max_abs_err = cpu_l21
        .iter()
        .zip(gpu_l21.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let l22_max_abs_err = cpu_l22
        .iter()
        .zip(gpu_l22.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    let (l22_f32_head_max_abs_err, gpu_l22_f32_head_absmax, gpu_l22_f32_head_nan_count) =
        if gpu_l22_f32_head.is_empty() {
            (f32::NAN, f32::NAN, 0usize)
        } else {
            let max_err = cpu_l22
                .iter()
                .zip(gpu_l22_f32_head.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let absmax = gpu_l22_f32_head
                .iter()
                .map(|x| x.abs())
                .fold(0.0_f32, f32::max);
            let nan_count = gpu_l22_f32_head.iter().filter(|x| !x.is_finite()).count();
            (max_err, absmax, nan_count)
        };

    let cpu_l22_absmax = cpu_l22.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gpu_l22_absmax = gpu_l22.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gpu_l22_nan_count = gpu_l22.iter().filter(|x| !x.is_finite()).count();
    let cpu_l20_absmax = cpu_l20.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gpu_l20_absmax = gpu_l20.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gpu_l20_nan_count = gpu_l20.iter().filter(|x| !x.is_finite()).count();
    let cpu_l21_absmax = cpu_l21.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gpu_l21_absmax = gpu_l21.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    let gpu_l21_nan_count = gpu_l21.iter().filter(|x| !x.is_finite()).count();

    let gpu_l03: Vec<f32> = gpu_l04_mid
        .iter()
        .zip(input.iter())
        .map(|(post, inp)| post - inp)
        .collect();

    let l03_max_abs_err = cpu_l03
        .iter()
        .zip(gpu_l03.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    println!("L0.3 max_abs_err: {:.8}", l03_max_abs_err);
    println!("CPU L0.3 first20: {:?}", &cpu_l03[..20.min(cpu_l03.len())]);
    println!("GPU L0.3 first20: {:?}", &gpu_l03[..20.min(gpu_l03.len())]);

    println!("L0.4 max_abs_err: {:.8}", l04_max_abs_err);
    println!("CPU L0.4 first20: {:?}", &cpu_l04[..20.min(cpu_l04.len())]);
    println!(
        "GPU L0.4 first20: {:?}",
        &gpu_l04_mid[..20.min(gpu_l04_mid.len())]
    );

    println!("L0.5 max_abs_err: {:.8}", l05_max_abs_err);
    println!("CPU L0.5 first20: {:?}", &cpu_l05[..20.min(cpu_l05.len())]);
    println!("GPU L0.5 first20: {:?}", &gpu_l05[..20.min(gpu_l05.len())]);

    println!("L0.6 max_abs_err: {:.8}", l06_max_abs_err);
    println!("CPU L0.6 first20: {:?}", &cpu_l06[..20.min(cpu_l06.len())]);
    println!("GPU L0.6 first20: {:?}", &gpu_l06[..20.min(gpu_l06.len())]);

    println!("L0.20 max_abs_err: {:.8}", l20_max_abs_err);
    println!("CPU L0.20 absmax: {:.8}", cpu_l20_absmax);
    println!("GPU L0.20 absmax: {:.8}", gpu_l20_absmax);
    println!("GPU L0.20 non-finite count: {}", gpu_l20_nan_count);
    println!("CPU L0.20 first20: {:?}", &cpu_l20[..20.min(cpu_l20.len())]);
    println!("GPU L0.20 first20: {:?}", &gpu_l20[..20.min(gpu_l20.len())]);

    println!("L0.22 max_abs_err: {:.8}", l22_max_abs_err);
    println!("L0.21 max_abs_err: {:.8}", l21_max_abs_err);
    println!("CPU L0.21 absmax: {:.8}", cpu_l21_absmax);
    println!("GPU L0.21 absmax: {:.8}", gpu_l21_absmax);
    println!("GPU L0.21 non-finite count: {}", gpu_l21_nan_count);
    println!("CPU L0.21 first20: {:?}", &cpu_l21[..20.min(cpu_l21.len())]);
    println!("GPU L0.21 first20: {:?}", &gpu_l21[..20.min(gpu_l21.len())]);
    println!("CPU L0.22 absmax: {:.8}", cpu_l22_absmax);
    println!("GPU L0.22 absmax: {:.8}", gpu_l22_absmax);
    println!("GPU L0.22 non-finite count: {}", gpu_l22_nan_count);
    println!("CPU L0.22 first20: {:?}", &cpu_l22[..20.min(cpu_l22.len())]);
    println!("GPU L0.22 first20: {:?}", &gpu_l22[..20.min(gpu_l22.len())]);
    if !gpu_l22_f32_head.is_empty() {
        println!(
            "L0.22 (F32 head override) max_abs_err: {:.8}",
            l22_f32_head_max_abs_err
        );
        println!(
            "GPU L0.22 (F32 head) absmax: {:.8}",
            gpu_l22_f32_head_absmax
        );
        println!(
            "GPU L0.22 (F32 head) non-finite count: {}",
            gpu_l22_f32_head_nan_count
        );
        println!(
            "GPU L0.22 (F32 head) first20: {:?}",
            &gpu_l22_f32_head[..20.min(gpu_l22_f32_head.len())]
        );
    }

    let l04_artifact = format!(
        "token_id={token}\ncount={count}\nl03_max_abs_err={l03_max_abs_err:.8}\nl04_max_abs_err={l04_max_abs_err:.8}\nl05_max_abs_err={l05_max_abs_err:.8}\nl06_max_abs_err={l06_max_abs_err:.8}\nl20_max_abs_err={l20_max_abs_err:.8}\nl21_max_abs_err={l21_max_abs_err:.8}\nl22_max_abs_err={l22_max_abs_err:.8}\nl22_f32_head_max_abs_err={l22_f32_head_max_abs_err:.8}\ncpu_l20_absmax={cpu_l20_absmax:.8}\ngpu_l20_absmax={gpu_l20_absmax:.8}\ngpu_l20_non_finite_count={gpu_l20_non_finite_count}\ncpu_l21_absmax={cpu_l21_absmax:.8}\ngpu_l21_absmax={gpu_l21_absmax:.8}\ngpu_l21_non_finite_count={gpu_l21_non_finite_count}\ncpu_l22_absmax={cpu_l22_absmax:.8}\ngpu_l22_absmax={gpu_l22_absmax:.8}\ngpu_l22_non_finite_count={gpu_l22_non_finite_count}\ngpu_l22_f32_head_absmax={gpu_l22_f32_head_absmax:.8}\ngpu_l22_f32_head_non_finite_count={gpu_l22_f32_head_non_finite_count}\ncpu_l03_first20={cpu_l03_first20:?}\ngpu_l03_first20={gpu_l03_first20:?}\ncpu_l04_first20={cpu_l04_first20:?}\ngpu_l04_first20={gpu_l04_first20:?}\ncpu_l05_first20={cpu_l05_first20:?}\ngpu_l05_first20={gpu_l05_first20:?}\ncpu_l06_first20={cpu_l06_first20:?}\ngpu_l06_first20={gpu_l06_first20:?}\ncpu_l20_first20={cpu_l20_first20:?}\ngpu_l20_first20={gpu_l20_first20:?}\ncpu_l21_first20={cpu_l21_first20:?}\ngpu_l21_first20={gpu_l21_first20:?}\ncpu_l22_first20={cpu_l22_first20:?}\ngpu_l22_first20={gpu_l22_first20:?}\ngpu_l22_f32_head_first20={gpu_l22_f32_head_first20:?}\n",
        token = token_id,
        count = dim,
        l03_max_abs_err = l03_max_abs_err,
        l04_max_abs_err = l04_max_abs_err,
        l05_max_abs_err = l05_max_abs_err,
        l06_max_abs_err = l06_max_abs_err,
        l20_max_abs_err = l20_max_abs_err,
        l21_max_abs_err = l21_max_abs_err,
        l22_max_abs_err = l22_max_abs_err,
        l22_f32_head_max_abs_err = l22_f32_head_max_abs_err,
        cpu_l20_absmax = cpu_l20_absmax,
        gpu_l20_absmax = gpu_l20_absmax,
        gpu_l20_non_finite_count = gpu_l20_nan_count,
        cpu_l21_absmax = cpu_l21_absmax,
        gpu_l21_absmax = gpu_l21_absmax,
        gpu_l21_non_finite_count = gpu_l21_nan_count,
        cpu_l22_absmax = cpu_l22_absmax,
        gpu_l22_absmax = gpu_l22_absmax,
        gpu_l22_non_finite_count = gpu_l22_nan_count,
        gpu_l22_f32_head_absmax = gpu_l22_f32_head_absmax,
        gpu_l22_f32_head_non_finite_count = gpu_l22_f32_head_nan_count,
        cpu_l03_first20 = &cpu_l03[..20.min(cpu_l03.len())],
        gpu_l03_first20 = &gpu_l03[..20.min(gpu_l03.len())],
        cpu_l04_first20 = &cpu_l04[..20.min(cpu_l04.len())],
        gpu_l04_first20 = &gpu_l04_mid[..20.min(gpu_l04_mid.len())],
        cpu_l05_first20 = &cpu_l05[..20.min(cpu_l05.len())],
        gpu_l05_first20 = &gpu_l05[..20.min(gpu_l05.len())],
        cpu_l06_first20 = &cpu_l06[..20.min(cpu_l06.len())],
        gpu_l06_first20 = &gpu_l06[..20.min(gpu_l06.len())],
        cpu_l20_first20 = &cpu_l20[..20.min(cpu_l20.len())],
        gpu_l20_first20 = &gpu_l20[..20.min(gpu_l20.len())],
        cpu_l21_first20 = &cpu_l21[..20.min(cpu_l21.len())],
        gpu_l21_first20 = &gpu_l21[..20.min(gpu_l21.len())],
        cpu_l22_first20 = &cpu_l22[..20.min(cpu_l22.len())],
        gpu_l22_first20 = &gpu_l22[..20.min(gpu_l22.len())],
        gpu_l22_f32_head_first20 = &gpu_l22_f32_head[..20.min(gpu_l22_f32_head.len())],
    );

    std::fs::write("artifacts/l0_4_probe.txt", l04_artifact)
        .expect("Failed to write artifacts/l0_4_probe.txt");

    Ok(())
}

fn token_nll_from_logits(logits: &[f32], target: usize) -> f32 {
    if target >= logits.len() || logits.is_empty() {
        return f32::INFINITY;
    }

    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if !max.is_finite() {
        return f32::INFINITY;
    }

    let mut exp_sum = 0.0_f32;
    for &x in logits {
        exp_sum += (x - max).exp();
    }
    if exp_sum <= 0.0 || !exp_sum.is_finite() {
        return f32::INFINITY;
    }

    let log_sum_exp = max + exp_sum.ln();
    let target_logit = logits[target];
    if !target_logit.is_finite() {
        return f32::INFINITY;
    }

    -(target_logit - log_sum_exp)
}

fn rmsnorm_reference(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * inv_rms * w)
        .collect()
}

#[derive(serde::Deserialize)]
struct ArcChoices {
    text: Vec<String>,
    label: Vec<String>,
}

#[derive(serde::Deserialize)]
struct ArcSample {
    #[allow(dead_code)]
    id: String,
    question: String,
    choices: ArcChoices,
    #[serde(rename = "answerKey")]
    answer_key: String,
}

#[derive(serde::Deserialize)]
struct HellaSwagSample {
    #[allow(dead_code)]
    ind: usize,
    #[allow(dead_code)]
    activity_label: String,
    ctx_a: String,
    ctx_b: String,
    #[allow(dead_code)]
    ctx: String,
    endings: Vec<String>,
    label: usize,
}

fn log_softmax(logits: &[f32], token: usize) -> f32 {
    let max_l = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp: f32 = logits.iter().map(|l| (l - max_l).exp()).sum();
    let log_sum_exp = max_l + sum_exp.ln();
    logits[token] - log_sum_exp
}

async fn run_hellaswag(
    args: &Args,
    tokenizer: &Tokenizer,
    engine: &mut dyn EvalEngine,
) -> Result<()> {
    println!("=== HELLASWAG BENCHMARK START ===");
    let path = args
        .hellaswag_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("fixtures/hellaswag_val.jsonl"));

    if !path.exists() {
        println!("ERROR: HellaSwag file not found at {:?}. Skipping.", path);
        return Ok(());
    }

    println!("Loading HellaSwag from: {:?}", path);
    let file = File::open(&path).expect("Failed to open HellaSwag file");
    let reader = std::io::BufReader::new(file);

    let mut correct = 0;
    let mut total = 0;
    let resume_from = args.hellaswag_resume_from;
    let start_time = std::time::Instant::now();

    if resume_from > 0 {
        println!(
            "Resuming HellaSwag from sample {} (skipping first {})",
            resume_from, resume_from
        );
    }

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let sample: HellaSwagSample = serde_json::from_str(&line).map_err(|e| {
            airframe::core::error::LibshimmyError::FixtureError {
                msg: format!("Line {}: {}", line_idx, e),
            }
        })?;

        // Skip samples before resume point (use line_idx so total/correct stay clean)
        if line_idx < resume_from {
            continue;
        }

        let context_text = format!("{} {}", sample.ctx_a, sample.ctx_b);
        let context_tokens_u32 = tokenizer.encode(&context_text, true).unwrap_or(vec![]);
        let context_tokens: Vec<usize> = context_tokens_u32.iter().map(|&t| t as usize).collect();

        let mut best_ending_idx = 0;
        let mut best_ending_score = f32::NEG_INFINITY;

        for (i, ending_text) in sample.endings.iter().enumerate() {
            // Append ending with a space because ctx_b usually has no trailing space
            let ending_with_space = format!(" {}", ending_text);
            let ending_tokens_u32 = tokenizer
                .encode(&ending_with_space, false)
                .unwrap_or(vec![]);
            let ending_tokens: Vec<usize> = ending_tokens_u32.iter().map(|&t| t as usize).collect();

            if ending_tokens.is_empty() {
                continue;
            }

            // Reset engine for clean state
            engine.reset();

            // Run context
            let mut logits = engine.process_prompt(&context_tokens)?;

            // Calculate log-probability of ending
            let mut log_prob_sum = 0.0;
            for &token in &ending_tokens {
                let prob = log_softmax(&logits, token);
                log_prob_sum += prob;

                // Advance
                logits = engine.process_prompt(&[token])?;
            }

            // Normalize by length
            let score = log_prob_sum / (ending_tokens.len() as f32);

            if score > best_ending_score {
                best_ending_score = score;
                best_ending_idx = i;
            }
        }

        if best_ending_idx == sample.label {
            correct += 1;
        }
        total += 1;

        if total % 10 == 0 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let speed = total as f32 / elapsed;
            println!(
                "Processed {} samples ({:.2} samples/sec). Accuracy: {:.2}%",
                total,
                speed,
                (correct as f32 / total as f32) * 100.0
            );
        }

        if args.max_eval_tokens.is_some() && total >= args.max_eval_tokens.unwrap() {
            println!("Stopping early due to limit.");
            break;
        }
    }

    println!(
        "Final HellaSwag Accuracy: {:.2}% ({}/{})",
        (correct as f32 / total as f32) * 100.0,
        correct,
        total
    );
    Ok(())
}

/// ARC (AI2 Reasoning Challenge) evaluation.
/// Uses log-probability scoring: format question + choices as prompt, score each answer choice.
/// Two scoring methods available:
///   1. "completion" (default): Score the full answer text continuation (like HellaSwag)
///   2. Could also do single-token letter scoring, but completion is more robust
async fn run_arc(args: &Args, tokenizer: &Tokenizer, engine: &mut dyn EvalEngine) -> Result<()> {
    let task_name = args.task.as_str();
    let default_path = match task_name {
        "arc-challenge" => "fixtures/arc_challenge_test.jsonl",
        _ => "fixtures/arc_easy_test.jsonl",
    };
    let path = args
        .arc_path
        .clone()
        .unwrap_or_else(|| PathBuf::from(default_path));

    let display_name = match task_name {
        "arc-challenge" => "ARC-Challenge",
        _ => "ARC-Easy",
    };

    println!("=== {} BENCHMARK START ===", display_name);

    if !path.exists() {
        println!("ERROR: ARC file not found at {:?}. Skipping.", path);
        return Ok(());
    }

    println!("Loading {} from: {:?}", display_name, path);

    // Optionally load few-shot examples from train split
    let n_shot = args.arc_n_shot;
    let few_shot_prefix = if n_shot > 0 {
        // Try to load train split for few-shot examples
        let train_path = path.with_file_name(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .replace("test", "train"),
        );
        if train_path.exists() {
            let train_file = File::open(&train_path).expect("Failed to open ARC train file");
            let train_reader = std::io::BufReader::new(train_file);
            let mut examples = Vec::new();
            for line in train_reader.lines().take(n_shot) {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(sample) = serde_json::from_str::<ArcSample>(&line) {
                    // Find the correct answer text
                    let answer_idx = sample
                        .choices
                        .label
                        .iter()
                        .position(|l| *l == sample.answer_key)
                        .unwrap_or(0);
                    let answer_text = &sample.choices.text[answer_idx];
                    let formatted =
                        format!("Question: {}\nAnswer: {}\n\n", sample.question, answer_text);
                    examples.push(formatted);
                }
            }
            println!(
                "Loaded {} few-shot examples from {:?}",
                examples.len(),
                train_path
            );
            examples.join("")
        } else {
            println!(
                "WARNING: Train file not found at {:?}, using 0-shot",
                train_path
            );
            String::new()
        }
    } else {
        String::new()
    };

    let file = File::open(&path).expect("Failed to open ARC file");
    let reader = std::io::BufReader::new(file);

    let mut correct = 0;
    let mut total = 0;
    let resume_from = args.arc_resume_from;
    let start_time = std::time::Instant::now();

    if resume_from > 0 {
        println!(
            "Resuming {} from sample {} (skipping first {})",
            display_name, resume_from, resume_from
        );
    }

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let sample: ArcSample = serde_json::from_str(&line).map_err(|e| {
            airframe::core::error::LibshimmyError::FixtureError {
                msg: format!("Line {}: {}", line_idx, e),
            }
        })?;

        // Skip samples before resume point
        if line_idx < resume_from {
            continue;
        }

        // lm-eval-harness format: "Question: {question}\nAnswer:" (no choices listed)
        let question_prompt = format!("{}Question: {}\nAnswer:", few_shot_prefix, sample.question);
        let prompt_tokens_u32 = tokenizer.encode(&question_prompt, true).unwrap_or(vec![]);
        let prompt_tokens: Vec<usize> = prompt_tokens_u32.iter().map(|&t| t as usize).collect();

        // Score each choice by log-probability of its answer text
        let mut best_choice_idx = 0;
        let mut best_choice_score = f32::NEG_INFINITY;

        for (i, choice_text) in sample.choices.text.iter().enumerate() {
            // Format answer as " A. <full text>" to match natural continuation
            let answer_text = format!(" {}", choice_text);
            let answer_tokens_u32 = tokenizer.encode(&answer_text, false).unwrap_or(vec![]);
            let answer_tokens: Vec<usize> = answer_tokens_u32.iter().map(|&t| t as usize).collect();

            if answer_tokens.is_empty() {
                continue;
            }

            // Reset engine for clean state
            engine.reset();

            // Run prompt through engine
            let mut logits = engine.process_prompt(&prompt_tokens)?;

            // Calculate log-probability of this answer continuation
            let mut log_prob_sum = 0.0;
            for &token in &answer_tokens {
                let prob = log_softmax(&logits, token);
                log_prob_sum += prob;
                logits = engine.process_prompt(&[token])?;
            }

            // Normalize by length (prevents bias toward short answers)
            let score = log_prob_sum / (answer_tokens.len() as f32);

            if score > best_choice_score {
                best_choice_score = score;
                best_choice_idx = i;
            }
        }

        // Check if our best choice matches the answer key
        let predicted_label = &sample.choices.label[best_choice_idx];
        if *predicted_label == sample.answer_key {
            correct += 1;
        }
        total += 1;

        if total % 10 == 0 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let speed = total as f32 / elapsed;
            println!(
                "[{}] Processed {} samples ({:.2} samples/sec). Accuracy: {:.2}%",
                display_name,
                total,
                speed,
                (correct as f32 / total as f32) * 100.0
            );
        }

        if args.max_eval_tokens.is_some() && total >= args.max_eval_tokens.unwrap() {
            println!("Stopping early due to limit.");
            break;
        }
    }

    println!(
        "Final {} Accuracy: {:.2}% ({}/{})",
        display_name,
        (correct as f32 / total as f32) * 100.0,
        correct,
        total
    );
    Ok(())
}

// format_arc_question removed — lm-eval-harness uses simple "Question: ...\nAnswer:" format
