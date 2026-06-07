//! GPU inference runtime — library-grade facade over the bindless pipeline.
//!
//! `GpuRuntime` owns the wgpu device, model weights, compute pipelines,
//! KV cache, and tokenizer. It exposes `load()` → `GpuSession` → `generate()`.

use crate::backend::bindless::kv_cache::KVCache;
use crate::backend::bindless::loader::BindlessModel;
use crate::backend::bindless::metadata::BindlessMetadata;
use crate::backend::bindless::pipeline::{BindlessPipeline, LayerParams, RMSNormParams};
use crate::backend::bindless::pipeline_shift::RopeShiftPipeline;
use crate::core::dequant::dequantize_q6_k;
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
        // Respect SHIMMY_MAX_CTX for extended context (YaRN RoPE)
        if let Some(max_ctx) = std::env::var("SHIMMY_MAX_CTX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
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
        }
        let gpu_model = BindlessModel::load_from_disk(&device, model_path, Some(&spec));
        let pipeline = BindlessPipeline::new(&device);
        let shift_pipeline = RopeShiftPipeline::new(&device);

        // Dequantize output head (Q6_K → F32)
        let output_head_f32 =
            Self::load_output_head_f32(&model_path_str, &gpu_model, &device, &spec)?;

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
                    128,
                )?
        };

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
        for _step in 0..params.max_tokens {
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

            // Compute next logits
            let row_offset = self.embd_weight_offset + (next_token as u64 * self.row_bytes);
            let mut layer_output = self.pipeline.run_dequant_request(
                &self.device,
                &self.queue,
                &self.model,
                row_offset as u32,
                dim,
            );

            for layer_idx in 0..self.spec.n_layer {
                let layer_offsets = self
                    .model
                    .metadata
                    .get_layer_offsets(layer_idx, self.spec.arch_string())
                    .ok_or_else(|| format!("Missing offsets for layer {}", layer_idx))?;

                let mut cache = self.kv_cache.lock().unwrap();
                layer_output = self.pipeline.run_layer_with_cache(
                    &self.device,
                    &self.queue,
                    &self.model,
                    &mut cache,
                    layer_idx,
                    &layer_output,
                    layer_offsets,
                    self.layer_params,
                );
            }

            // Increment KV cache
            {
                let mut cache = self.kv_cache.lock().unwrap();
                cache.increment()?;
            }

            // Final RMSNorm + output head projection
            let normed = self.pipeline.run_rmsnorm_test(
                &self.device,
                &self.queue,
                &self.model,
                &layer_output,
                self.norm_params,
            );
            logits_vec = self.pipeline.run_matmul_f32(
                &self.device,
                &self.queue,
                &self.output_head_f32,
                &normed,
                self.spec.n_vocab as u32,
                dim,
            );
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
        let output_weight_type = gpu_model
            .metadata
            .get_tensor_type("output.weight")
            .expect("output.weight type not found");

        if output_weight_type != 14 {
            return Err(format!(
                "Expected Q6_K (type 14) for output.weight, got type {}",
                output_weight_type
            )
            .into());
        }

        let file = std::fs::File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let tensor_info = GgufTensorInfo {
            name: "output.weight".to_string(),
            dimensions: vec![spec.n_vocab, spec.n_embd],
            ggml_type: 14,
            offset: 0,
        };

        let data_start = gpu_model.metadata.data_start_offset;
        let tensor_f32 = dequantize_q6_k(&tensor_info, &mmap, data_start)?;

        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Output Head F32"),
            contents: bytemuck::cast_slice(&tensor_f32.data),
            usage: wgpu::BufferUsages::STORAGE,
        });

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
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
    }

    // Apply temperature
    let inv_t = 1.0 / temperature;
    for v in logits.iter_mut() {
        *v *= inv_t;
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
