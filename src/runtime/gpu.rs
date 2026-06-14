//! GPU inference runtime — library-grade facade over the bindless pipeline.
//!
//! `GpuRuntime` owns the wgpu device, model weights, compute pipelines,
//! KV cache, and tokenizer. It exposes `load()` → `GpuSession` → `generate()`.

use crate::backend::bindless::kv_cache::KVCache;
use crate::backend::bindless::loader::BindlessModel;
use crate::backend::bindless::metadata::BindlessMetadata;
use crate::backend::bindless::pipeline::{BindlessPipeline, LayerParams, RMSNormParams};
use crate::backend::bindless::pipeline_shift::RopeShiftPipeline;
use crate::core::dequant::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};
use crate::core::model::GgufTensorInfo;
use crate::core::spec::ModelSpec;
use memmap2::Mmap;
use shimmytok::Tokenizer;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Sampling parameters for generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    /// Additional stop strings (e.g. `<|eot_id|>`, `<|im_end|>`).
    /// Encoded to token IDs at the start of generation and checked in the decode loop.
    pub extra_stop_tokens: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.8,
            top_p: 0.95,
            repetition_penalty: 1.1,
            seed: 42,
            extra_stop_tokens: Vec::new(),
        }
    }
}

/// Deterministic Xorshift64* PRNG.
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
        (val >> 40) as f32 / 16777216.0
    }
}

/// The GPU inference runtime. One per process — owns the device and all GPU resources.
pub struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    model: BindlessModel,
    pipeline: BindlessPipeline,
    shift_pipeline: RopeShiftPipeline,
    tokenizer: Tokenizer,
    spec: ModelSpec,
    output_head_f32: wgpu::Buffer,
    kv_cache: Arc<Mutex<KVCache>>,
    // Precomputed constants
    layer_params: LayerParams,
    norm_params: RMSNormParams,
    embd_weight_offset: u64,
    row_bytes: u64,
    eos_token: u32,
    im_end_token: Option<u32>,
}

impl GpuRuntime {
    /// Initialize the GPU runtime from a GGUF model path.
    ///
    /// This performs all expensive one-time setup:
    /// - wgpu device/queue creation
    /// - Model weight upload to VRAM
    /// - Pipeline compilation (shader creation)
    /// - KV cache allocation
    /// - Output head dequantization (Q6_K → F32)
    /// - Tokenizer initialization
    pub async fn load(model_path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let model_path_str = model_path.to_string_lossy().to_string();

        // GPU init
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("No GPU adapter found: {}", e))?;

        let adapter_limits = adapter.limits();

        // Pre-flight: check that the model file fits within the GPU's storage buffer binding
        // limit before uploading. Older GPUs (e.g. GTX 1050 Ti on some drivers) may report
        // a lower limit than the model requires, causing a deferred wgpu validation panic.
        let model_file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
        let max_binding = adapter_limits.max_storage_buffer_binding_size as u64;
        if model_file_size > max_binding {
            return Err(format!(
                "Model file ({:.0} MB) exceeds this GPU's storage buffer binding limit \
                 ({:.0} MB). Try a more quantized model or update your GPU drivers.",
                model_file_size as f64 / 1_048_576.0,
                max_binding as f64 / 1_048_576.0,
            )
            .into());
        }

        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        // Use the adapter's true max_buffer_size, not max_storage_buffer_binding_size.
        // On older GPUs these can differ, and capping max_buffer_size to the binding size
        // causes validation errors when creating large model buffers.
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage.max(14); // INT4 KV layout requires ≥14 storage buffers
        limits.max_compute_invocations_per_workgroup = 256;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await?;

        // Register an error handler so wgpu validation errors surface as descriptive
        // messages rather than the wgpu 27 default fatal panic.
        device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            eprintln!("[Airframe] GPU error: {}", error);
        }));

        // Load model
        let tokenizer = Tokenizer::from_gguf_file(&model_path_str)?;
        // Auto-derive model spec from GGUF metadata — works with any GGUF model
        let mut header_file = std::fs::File::open(model_path)?;
        let header_meta = BindlessMetadata::new(&mut header_file);
        drop(header_file);
        let mut spec = header_meta.to_model_spec();

        // Safe context window cap.
        // Consumer GPUs (4-8 GB VRAM) cannot sustain the full native context of
        // modern models (e.g. Llama-3.2 = 131072). The KV cache scales linearly:
        //   n_layers × n_kv_heads × head_dim × ctx × 2 × 4 bytes
        // Without a cap, a 131K-context model allocates ~28 GB of KV cache alone.
        //
        // If SHIMMY_MAX_CTX is explicitly set, honour it (user opted in).
        // Otherwise, cap at 4096 tokens — enough for practical use on consumer hardware.
        const DEFAULT_SAFE_CTX: usize = 8192;

        if let Some(max_ctx) = std::env::var("SHIMMY_MAX_CTX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            // Explicit override — apply YaRN RoPE scale if extending beyond native
            let rope_scale = std::env::var("SHIMMY_ROPE_SCALE")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or_else(|| {
                    if max_ctx > spec.n_ctx {
                        spec.n_ctx as f32 / max_ctx as f32
                    } else {
                        1.0
                    }
                });
            spec.n_ctx = max_ctx;
            spec.rope_scale = rope_scale;
        } else if spec.n_ctx > DEFAULT_SAFE_CTX {
            // Model native context exceeds safe default — cap it silently.
            // Set SHIMMY_MAX_CTX=<n> to override.
            eprintln!(
                "[GpuRuntime] Model native context {} tokens exceeds safe default {}. \
                 Capping to {} to protect GPU memory. Set SHIMMY_MAX_CTX=<n> to override.",
                spec.n_ctx, DEFAULT_SAFE_CTX, DEFAULT_SAFE_CTX
            );
            spec.n_ctx = DEFAULT_SAFE_CTX;
        }
        let gpu_model = BindlessModel::load_from_disk(&device, model_path, Some(&spec));
        let pipeline = BindlessPipeline::new(&device);
        let shift_pipeline = RopeShiftPipeline::new(&device);

        // Dequantize output head (Q6_K → F32)
        let output_head_f32 =
            Self::load_output_head_f32(&model_path_str, &gpu_model, &device, &spec)?;

        // === END DIAGNOSTIC ===

        // KV cache
        let max_ctx = spec.n_ctx as u32;
        let kv_cache = Arc::new(Mutex::new(KVCache::new(
            &device,
            spec.n_layer,
            spec.n_head_kv as u32,
            spec.head_dim as u32,
            max_ctx,
        )));

        let dim = spec.n_embd as u32;
        let embd_weight_offset = gpu_model
            .metadata
            .get_tensor_offset("token_embd.weight")
            .expect("token_embd.weight not found");
        let row_bytes = (dim as u64 / 32) * 18; // Q4_0 quantization

        let weight_quant_type = gpu_model
            .metadata
            .get_tensor_type("blk.0.attn_q.weight")
            .unwrap_or(2);
        let qt_v = gpu_model
            .metadata
            .get_tensor_type("blk.0.attn_v.weight")
            .unwrap_or(weight_quant_type);
        let qt_ffn_down = gpu_model
            .metadata
            .get_tensor_type("blk.0.ffn_down.weight")
            .unwrap_or(weight_quant_type);
        let packed_quant_type = weight_quant_type | (qt_v << 8) | (qt_ffn_down << 16);

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
            post_norm_enabled: if spec.arch_string().contains("gemma") {
                1
            } else {
                0
            },
            qk_norm_enabled: if spec.has_qk_norm { 1 } else { 0 },
            layer_norm_enabled: 0,
            ffn_kind_policy: 0,
            qkv_layout_policy: 0,
            batch_offset: 0,
            batch_count: 0, // placeholder — overridden per-dispatch in inference.rs
            q_weight_k: 0,
            k_weight_k: 0,
        };

        let norm_weight_offset = gpu_model
            .metadata
            .get_tensor_offset("output_norm.weight")
            .expect("output_norm.weight not found") as u32;
        let norm_params = RMSNormParams {
            count: dim,
            weights_offset: norm_weight_offset,
            bias_offset: 0,
            eps: spec.rms_eps,
            norm_type: 0,
        };

        let eos_token = tokenizer.eos_token();
        let im_end_token: Option<u32> = tokenizer.encode("<|im_end|>", false).ok().and_then(|v| {
            if v.len() == 1 {
                Some(v[0])
            } else {
                None
            }
        });

        Ok(Self {
            device,
            queue,
            model: gpu_model,
            pipeline,
            shift_pipeline,
            tokenizer,
            spec,
            output_head_f32,
            kv_cache,
            layer_params,
            norm_params,
            embd_weight_offset,
            row_bytes,
            eos_token,
            im_end_token,
        })
    }

    /// Generate text from a raw prompt string.
    ///
    /// `on_token` is called for each generated token (for streaming).
    /// Returns the full generated text.
    #[allow(clippy::type_complexity)]
    pub fn generate(
        &self,
        prompt: &str,
        params: &SamplingParams,
        mut on_token: Option<Box<dyn FnMut(&str) + Send>>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let prompt_tokens = self.tokenizer.encode(prompt, true)?;
        let dim = self.spec.n_embd as u32;
        let mut rng = Rng::new(params.seed);

        // Reset KV cache
        {
            let mut cache = self.kv_cache.lock().unwrap();
            cache.reset();
        }

        // Prefill: batch-dequant all prompt embeddings
        let mut batched_embd = Vec::with_capacity(prompt_tokens.len() * dim as usize);
        for &token_id in &prompt_tokens {
            let row_offset = self.embd_weight_offset + (token_id as u64 * self.row_bytes);
            let embd = self.pipeline.run_dequant_request(
                &self.device,
                &self.queue,
                &self.model,
                row_offset as u32,
                dim,
            );
            batched_embd.extend_from_slice(&embd);
        }

        // Run prefill through all layers
        let (_final_act, _l21, prefill_logits) = {
            let cache_guard = self.kv_cache.lock().unwrap();
            self.pipeline
                .run_full_model_prefill_chunked_with_cache_state(
                    &self.device,
                    &self.queue,
                    &self.model,
                    &batched_embd,
                    Some(&self.output_head_f32),
                    0,
                    Some((cache_guard.get_k_buffers(), cache_guard.get_v_buffers())),
                    &self.spec,
                    8,  // Forced small chunk for TDR safety on Q4_K models (1 of 8 test)
                )?
        };

        // Debug for garbage output diagnosis (1 of 8)
        let hidden_rms: f32 = _final_act.iter().map(|x| x * x).sum::<f32>().sqrt() / _final_act.len() as f32;
        eprintln!("[DEBUG 1/8] Final hidden rms: {:.6}, first5: {:?}", hidden_rms, &_final_act[..5.min(_final_act.len())]);
        let mut top: Vec<(usize, f32)> = prefill_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("[DEBUG 1/8] Prefill logits top5 tokens: {:?}", &top[..5]);

        // Advance KV cache position
        {
            let mut cache = self.kv_cache.lock().unwrap();
            for _ in 0..prompt_tokens.len() {
                cache.increment()?;
            }
        }

        let mut logits_vec = prefill_logits;
        let mut generated_text = String::new();
        let mut recent_tokens: Vec<u32> = Vec::new();

        // Encode extra stop tokens to IDs once before the decode loop.
        // Single-token strings only; multi-token stop strings are skipped.
        let extra_stop_ids: Vec<u32> = params
            .extra_stop_tokens
            .iter()
            .filter_map(|s| {
                self.tokenizer.encode(s, false).ok().and_then(|v| {
                    if v.len() == 1 {
                        Some(v[0])
                    } else {
                        None
                    }
                })
            })
            .collect();

        // Decode loop
        let log_logits = std::env::var("AIRFRAME_LOG_LOGITS").map(|v| v == "1").unwrap_or(false);
        for _step in 0..params.max_tokens {
            if log_logits {
                let mut top: Vec<(usize, f32)> = logits_vec.iter().enumerate()
                    .map(|(i, &v)| (i, v)).collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let argmax = top[0].0;
                let top5: Vec<(usize, f32)> = top.into_iter().take(5).collect();
                eprintln!("[LOGITS] step={} argmax={} top5={:?}", _step, argmax, top5);
            }
            let next_token = sample_token(
                &mut logits_vec,
                params.temperature,
                params.top_p,
                params.repetition_penalty,
                &recent_tokens,
                &mut rng,
            );

            recent_tokens.push(next_token);
            if recent_tokens.len() > 64 {
                recent_tokens.remove(0);
            }

            // EOS check
            if next_token == self.eos_token {
                break;
            }
            if let Some(im_end) = self.im_end_token {
                if next_token == im_end {
                    break;
                }
            }
            if extra_stop_ids.contains(&next_token) {
                break;
            }

            // Decode token to text
            let piece = self.tokenizer.decode_single(next_token, true)?;
            generated_text.push_str(&piece);

            if let Some(cb) = on_token.as_mut() {
                cb(&piece);
            }

            // Helical shift if approaching cache limit
            {
                let mut cache = self.kv_cache.lock().unwrap();
                let current_len = cache.get_seq_len();

                if current_len >= cache.max_len() - 4 {
                    let keep_sink = 4;
                    let shift_amt = cache.max_len() / 4;
                    for layer_idx in 0..self.spec.n_layer {
                        if cache.is_int4() {
                            self.shift_pipeline.execute_int4(
                                &self.device,
                                &self.queue,
                                cache.get_k_buffer(layer_idx),
                                cache.get_v_buffer(layer_idx),
                                cache.get_k_packed_buffer(layer_idx),
                                cache.get_v_packed_buffer(layer_idx),
                                cache.get_k_scale_buffer(layer_idx),
                                cache.get_v_scale_buffer(layer_idx),
                                keep_sink,
                                shift_amt,
                                current_len,
                                self.spec.n_head_kv as u32,
                                self.spec.head_dim as u32,
                                cache.max_len(),
                            );
                        } else {
                            self.shift_pipeline.execute(
                                &self.device,
                                &self.queue,
                                cache.get_k_buffer(layer_idx),
                                cache.get_v_buffer(layer_idx),
                                keep_sink,
                                shift_amt,
                                current_len,
                                self.spec.n_head_kv as u32,
                                self.spec.head_dim as u32,
                                self.spec.rope_dim as u32,
                                self.spec.rope_base,
                                cache.max_len(),
                            );
                        }
                    }
                    cache.set_seq_len(current_len - shift_amt);
                    cache.advance_window_base(shift_amt);
                }
            }

            // Compute next logits — use the full-model chunked path (same as prefill)
            // to avoid 22 individual layer readbacks per decode token.
            // run_dequant_request for embedding + run_full_model_with_cache_state
            // batches all 22 layers into ~3 submits instead of 22.
            let row_offset = self.embd_weight_offset + (next_token as u64 * self.row_bytes);
            let token_embd = self.pipeline.run_dequant_request(
                &self.device,
                &self.queue,
                &self.model,
                row_offset as u32,
                dim,
            );

            let current_pos = {
                let cache = self.kv_cache.lock().unwrap();
                cache.get_seq_len()
            };
            let (new_hidden, _l21, new_logits) = {
                let cache_guard = self.kv_cache.lock().unwrap();
                self.pipeline.run_full_model_prefill_chunked_with_cache_state(
                    &self.device,
                    &self.queue,
                    &self.model,
                    &token_embd,
                    Some(&self.output_head_f32),
                    current_pos,
                    Some((cache_guard.get_k_buffers(), cache_guard.get_v_buffers())),
                    &self.spec,
                    1, // single token decode
                )?
            };
            let layer_output = new_hidden;
            logits_vec = new_logits;

            // Increment KV cache
            {
                let mut cache = self.kv_cache.lock().unwrap();
                cache.increment()?;
            }
            // logits_vec already set above from run_full_model_prefill_chunked_with_cache_state
            let _ = layer_output; // suppress unused warning
        }

        Ok(generated_text)
    }

    /// Reset the KV cache (for a new conversation).
    pub fn reset(&self) {
        let mut cache = self.kv_cache.lock().unwrap();
        cache.reset();
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get a reference to the model spec.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn load_output_head_f32(
        model_path: &str,
        gpu_model: &BindlessModel,
        device: &wgpu::Device,
        spec: &ModelSpec,
    ) -> Result<wgpu::Buffer, Box<dyn std::error::Error + Send + Sync>> {
        use wgpu::util::DeviceExt;

        println!(
            "[OutputHead] load_output_head_f32 ENTERED for model: {}",
            model_path
        );
        println!(
            "[OutputHead] data_start_offset={} tensor_offset={} weight_type={}",
            gpu_model.metadata.data_start_offset,
            gpu_model
                .metadata
                .get_tensor_offset("output.weight")
                .unwrap_or(0),
            gpu_model
                .metadata
                .get_tensor_type("output.weight")
                .unwrap_or(0)
        );

        // Determine which tensor to use for the output head.
        // Models with tied embeddings (e.g. Llama-3.2) omit `output.weight`
        // and reuse `token_embd.weight` for the final projection.
        let (tensor_name, weight_type, tensor_offset) = {
            let has_output = gpu_model
                .metadata
                .get_tensor_type("output.weight")
                .is_some();
            if has_output {
                let wt = gpu_model
                    .metadata
                    .get_tensor_type("output.weight")
                    .expect("output.weight type not found");
                let off = gpu_model
                    .metadata
                    .get_tensor_offset("output.weight")
                    .unwrap_or(0);
                ("output.weight", wt, off)
            } else {
                // Tied embeddings: fall back to token_embd.weight
                let wt = gpu_model
                    .metadata
                    .get_tensor_type("token_embd.weight")
                    .ok_or("Neither output.weight nor token_embd.weight found in model")?;
                let off = gpu_model
                    .metadata
                    .get_tensor_offset("token_embd.weight")
                    .ok_or_else(|| format!("token_embd.weight not found in tensor_offsets map (tensor_count={})", gpu_model.metadata.tensor_count))?;
                ("token_embd.weight", wt, off)
            }
        };

        let file = std::fs::File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data_start = gpu_model.metadata.data_start_offset;

        // tensor_offset from get_tensor_offset() may be either:
        // - GGUF v2: absolute offset in file (offset >= data_start)
        // - GGUF v3: relative offset from data_start (offset < data_start)
        // Detect by checking if offset >= data_start.
        let tensor_offset_relative = if tensor_offset >= data_start {
            tensor_offset - data_start  // v2 absolute → convert to relative
        } else {
            tensor_offset               // v3 relative → use directly
        };

        let tensor_info = GgufTensorInfo {
            name: tensor_name.to_string(),
            dimensions: vec![spec.n_vocab, spec.n_embd],
            ggml_type: weight_type,
            offset: tensor_offset_relative,
        };

        // Dequantize to F32 — support all quant types used in output/embedding layers
        let tensor_f32 = match weight_type {
            0 => {
                // F32 — already float, just read directly
                use crate::core::tensor::Tensor;
                // tensor_offset_relative is already relative to data_start
                let byte_offset = data_start + tensor_offset_relative;
                let n_elements = spec.n_vocab * spec.n_embd;
                let bytes = &mmap[byte_offset as usize..(byte_offset as usize + n_elements * 4)];
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Tensor {
                    data: floats,
                    shape: vec![spec.n_vocab, spec.n_embd],
                }
            }
            2 => dequantize_q4_0(&tensor_info, &mmap, data_start)?,
            8 => dequantize_q8_0(&tensor_info, &mmap, data_start)?,
            12 => dequantize_q4_k(&tensor_info, &mmap, data_start)?,
            13 => dequantize_q5_k(&tensor_info, &mmap, data_start)?,
            14 => dequantize_q6_k(&tensor_info, &mmap, data_start)?,
            other => {
                return Err(format!(
                    "Unsupported quant type {} for output head tensor '{}'",
                    other, tensor_name
                )
                .into())
            }
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Output Head F32"),
            contents: bytemuck::cast_slice(&tensor_f32.data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // CPU-side verification — runs before GPU upload, guaranteed to print.
        let nan_in_head = tensor_f32.data.iter().filter(|v| v.is_nan()).count();
        let inf_in_head = tensor_f32.data.iter().filter(|v| v.is_infinite()).count();
        let max_abs = tensor_f32
            .data
            .iter()
            .cloned()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        println!(
            "[OutputHead-CPU] tensor={} quant={} elements={} NaN={} Inf={} max_abs={:.4e}",
            tensor_name,
            weight_type,
            tensor_f32.data.len(),
            nan_in_head,
            inf_in_head,
            max_abs
        );

        Ok(buffer)
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
    // Repetition penalty
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

    // Temperature = 0 → greedy
    if temperature < 1e-7 {
        return logits
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Apply temperature — map NaN/inf to -inf so they're excluded from sampling
    let inv_t = 1.0 / temperature;
    for v in logits.iter_mut() {
        if v.is_finite() {
            *v *= inv_t;
        } else {
            *v = f32::NEG_INFINITY;
        }
    }

    // Softmax
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p nucleus
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    // Filter non-finite before sorting
    indexed.retain(|(_, p)| p.is_finite());
    if indexed.is_empty() {
        return 0; // all logits were non-finite — safe fallback
    }
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = i + 1;
            break;
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_default_has_empty_stop_tokens() {
        let p = SamplingParams::default();
        assert!(p.extra_stop_tokens.is_empty());
    }

    #[test]
    fn test_sampling_params_stop_tokens_stored() {
        let p = SamplingParams {
            extra_stop_tokens: vec!["<|eot_id|>".to_string(), "<|im_end|>".to_string()],
            ..SamplingParams::default()
        };
        assert_eq!(p.extra_stop_tokens.len(), 2);
        assert!(p.extra_stop_tokens.contains(&"<|eot_id|>".to_string()));
    }

    #[test]
    fn test_sampling_params_clone_preserves_stop_tokens() {
        let p = SamplingParams {
            extra_stop_tokens: vec!["</s>".to_string()],
            ..SamplingParams::default()
        };
        let p2 = p.clone();
        assert_eq!(p2.extra_stop_tokens, p.extra_stop_tokens);
    }
}
