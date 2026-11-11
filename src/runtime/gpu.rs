//! GPU inference runtime — library-grade facade over the bindless pipeline.
//!
//! `GpuRuntime` owns the wgpu device, model weights, compute pipelines,
//! KV cache, and tokenizer. It exposes `load()` → `GpuSession` → `generate()`.
#![allow(clippy::type_complexity)]

use crate::backend::bindless::kv_cache::KVCache;
use crate::backend::bindless::loader::BindlessModel;
use crate::backend::bindless::pipeline::BindlessPipeline;
use crate::backend::bindless::pipeline_shift::RopeShiftPipeline;
use crate::control::InferenceControl;
#[cfg(feature = "isf")]
use crate::control::{ControlDecision, InferenceEvent};
use crate::core::dequant::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};
use crate::core::model::GgufTensorInfo;
use crate::core::spec::ModelSpec;
#[cfg(feature = "isf")]
use crate::runtime::kvcache::KvSnapshot;
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
    request_gate: Mutex<()>,
    #[cfg(feature = "isf")]
    profile_config: Option<airframe_observe::profiling::ProfileConfig>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    model: BindlessModel,
    pipeline: BindlessPipeline,
    #[allow(dead_code)]
    shift_pipeline: RopeShiftPipeline,
    tokenizer: Arc<Tokenizer>,
    spec: ModelSpec,
    output_head_f32: Option<wgpu::Buffer>,
    kv_cache: Arc<Mutex<KVCache>>,
    // Precomputed constants
    embd_weight_offset: u64,
    row_bytes: u64,
    embd_quant_type: u32,
    eos_token: u32,
    im_end_token: Option<u32>,
}

// macOS Metal can time out a large dense prefill command before completion.
// Bound the outer batch only; preserve QKV microbatching and every operator.
fn prefill_chunk_for_platform(requested: u32, macos: bool) -> u32 {
    if macos {
        requested.min(64)
    } else {
        requested
    }
}

impl GpuRuntime {
    /// Construct a runtime from pre-loaded parts.
    ///
    /// For callers that load device/queue/model/pipeline/etc. independently
    /// (e.g. the dev server) — avoids duplicating the full [`load`] pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        device: wgpu::Device,
        queue: wgpu::Queue,
        model: BindlessModel,
        pipeline: BindlessPipeline,
        shift_pipeline: RopeShiftPipeline,
        tokenizer: Arc<Tokenizer>,
        spec: ModelSpec,
        output_head_f32: Option<wgpu::Buffer>,
        kv_cache: Arc<Mutex<KVCache>>,
        embd_weight_offset: u64,
        row_bytes: u64,
        embd_quant_type: u32,
        eos_token: u32,
        im_end_token: Option<u32>,
    ) -> Self {
        Self::try_from_parts(
            device,
            queue,
            model,
            pipeline,
            shift_pipeline,
            tokenizer,
            spec,
            output_head_f32,
            kv_cache,
            embd_weight_offset,
            row_bytes,
            embd_quant_type,
            eos_token,
            im_end_token,
        )
        .expect("runtime parts failed validated model admission")
    }

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
        Self::load_with_context(
            model_path,
            crate::execution::admission::configured_context()?,
        )
        .await
    }
    pub fn model_identity(&self) -> &crate::runtime::state::StateIdentity {
        self.model.validated().identity()
    }
    pub fn encode_prompt(
        &self,
        text: &str,
    ) -> Result<Vec<u32>, crate::execution::admission::ExecutionError> {
        let ids = self
            .tokenizer
            .encode_with_options(
                text,
                &shimmytok::EncodeOptions {
                    add_special_tokens: true,
                    parse_special: true,
                },
            )
            .map_err(|e| crate::execution::admission::ExecutionError::Request {
                code: "tokenization_failed",
                detail: e.to_string(),
            })?;
        if ids.is_empty()
            || ids.len() > self.spec.n_ctx
            || ids.iter().any(|id| *id as usize >= self.spec.n_vocab)
        {
            return Err(crate::execution::admission::ExecutionError::Request {
                code: "invalid_prompt_tokens",
                detail: "prompt must fit model vocabulary and context".into(),
            });
        }
        Ok(ids)
    }
    pub fn try_reset(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let _request = self.request_gate.try_lock().map_err(|_| {
            crate::execution::admission::ExecutionError::State(
                crate::runtime::state::StateError::Busy,
            )
        })?;
        self.kv_cache
            .lock()
            .map_err(|_| {
                crate::execution::admission::ExecutionError::Gpu("KV state lock poisoned".into())
            })?
            .reset();
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_parts(
        device: wgpu::Device,
        queue: wgpu::Queue,
        model: BindlessModel,
        pipeline: BindlessPipeline,
        shift_pipeline: RopeShiftPipeline,
        tokenizer: Arc<Tokenizer>,
        spec: ModelSpec,
        output_head_f32: Option<wgpu::Buffer>,
        kv_cache: Arc<Mutex<KVCache>>,
        embd_weight_offset: u64,
        row_bytes: u64,
        embd_quant_type: u32,
        eos_token: u32,
        im_end_token: Option<u32>,
    ) -> Result<Self, crate::execution::admission::ModelAdmissionError> {
        model.validated().check_spec(&spec)?;
        model.validated().check_device(&device.limits())?;
        let embedding = model
            .validated()
            .manifest()
            .tensor("token_embd.weight")
            .expect("validated embedding");
        if embedding.absolute_offset() != embd_weight_offset
            || embedding.row_bytes() != row_bytes
            || embedding.ggml_type() != embd_quant_type
            || tokenizer.vocab_size() != spec.n_vocab
            || tokenizer.eos_token() != eos_token
            || tokenizer.metadata() != &model.validated().manifest().header().values
        {
            return Err(
                crate::execution::admission::ModelAdmissionError::Configuration {
                    code: "model_parts_mismatch",
                    detail: "tokenizer/embedding parts differ from validated model".into(),
                },
            );
        }
        if let Some(head) = &output_head_f32 {
            let expected = (spec.n_embd as u64)
                .checked_mul(spec.n_vocab as u64)
                .and_then(|x| x.checked_mul(4));
            if Some(head.size()) != expected
                || head.size() > u64::from(device.limits().max_storage_buffer_binding_size)
            {
                return Err(
                    crate::execution::admission::ModelAdmissionError::Configuration {
                        code: "invalid_output_head",
                        detail: "external head must have exact F32 shape and fit its binding"
                            .into(),
                    },
                );
            }
        }
        Ok(Self {
            request_gate: Mutex::new(()),
            #[cfg(feature = "isf")]
            profile_config: airframe_observe::profiling::ProfileConfig::from_env(),
            device,
            queue,
            model,
            pipeline,
            shift_pipeline,
            tokenizer,
            spec,
            output_head_f32,
            kv_cache,
            embd_weight_offset,
            row_bytes,
            embd_quant_type,
            eos_token,
            im_end_token,
        })
    }
    pub async fn load_with_context(
        model_path: &Path,
        context: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        #[cfg(feature = "isf")]
        let profile_config = airframe_observe::profiling::ProfileConfig::from_env();
        #[cfg(feature = "isf")]
        let mut load_profile = profile_config.as_ref().map(|p| p.begin_load());
        #[cfg(not(feature = "isf"))]
        if std::env::var("AIRFRAME_TRACE_TIMING").ok().as_deref() == Some("1") {
            eprintln!("[Airframe] profiling requires the isf feature");
        }
        let model_path_str = model_path.to_string_lossy().to_string();

        let prepared = crate::execution::admission::PreparedDenseModel::load(model_path, context)?;
        let validated = prepared.validated().clone();
        let spec = validated.spec().clone();

        // GPU init
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: wgpu::InstanceFlags::default().with_env(),
            ..Default::default()
        });

        // Enumerate adapters and prefer DiscreteGpu over IntegratedGpu.
        // Fixes multi-GPU systems (laptops with NVIDIA/AMD + Intel iGPU)
        // where wgpu's default adapter selection often picks the integrated GPU.
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        let adapter = adapters
            .iter()
            .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .or_else(|| {
                adapters
                    .iter()
                    .find(|a| a.get_info().device_type == wgpu::DeviceType::IntegratedGpu)
            })
            .or_else(|| adapters.first())
            .ok_or_else(
                || crate::execution::admission::ModelAdmissionError::Device {
                    code: "gpu_unavailable",
                    detail: "No GPU adapter found".into(),
                },
            )
            .inspect(|a| {
                eprintln!(
                    "[GpuRuntime] Selected adapter: {} ({:?}, backend={:?})",
                    a.get_info().name,
                    a.get_info().device_type,
                    a.get_info().backend
                );
            })?;

        let adapter_limits = adapter.limits();
        validated.check_device(&adapter_limits)?;

        // Pre-flight: check that the model fits within GPU memory constraints.
        // The bindless architecture splits the model into up to 3 sub-range bindings
        // of BLOB_CHUNK_BYTES each, so the per-binding limit only needs to hold one chunk.
        // The overall buffer just needs to fit within max_buffer_size.
        let model_file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
        let max_buffer_size = adapter_limits.max_buffer_size;
        let max_binding = adapter_limits.max_storage_buffer_binding_size as u64;

        // Adapter-aware chunk plan. The bindless loader clamps BLOB_CHUNK_BYTES to the
        // adapter's storage-buffer binding limit itself (compute_chunk_plan), so a small
        // binding limit (e.g. llvmpipe 128 MB) is not fatal — it only raises the number
        // of resident chunks. Do not hard-fail on the raw 2 GB constant here.
        let chunk_plan =
            crate::backend::bindless::loader::compute_chunk_plan(model_file_size, max_binding);
        if chunk_plan.effective_chunk > max_buffer_size {
            return Err(crate::execution::admission::ModelAdmissionError::Device {
                code: "device_limit",
                detail: "blob chunk exceeds device buffer limit".into(),
            }
            .into());
        }
        if chunk_plan.num_chunks > 1 {
            eprintln!(
                "[GpuRuntime] Model ({:.0} MB) exceeds chunk size: using {}-chunk bindless split \
                 ({} MB each, binding limit {:.0} MB)",
                model_file_size as f64 / 1_048_576.0,
                chunk_plan.num_chunks,
                chunk_plan.effective_chunk as f64 / 1_048_576.0,
                max_binding as f64 / 1_048_576.0,
            );
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

        #[cfg(feature = "isf")]
        let profiling_features = if profile_config.as_ref().is_some_and(|p| p.gpu_requested())
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        #[cfg(not(feature = "isf"))]
        let profiling_features = wgpu::Features::empty();
        let request = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: profiling_features,
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await;
        let (device, queue) = match request {
            Ok(value) => value,
            Err(_) if !profiling_features.is_empty() => {
                #[cfg(feature = "isf")]
                if let Some(profile) = &profile_config {
                    profile.set_gpu_unavailable(
                        airframe_observe::profiling::MissingGpu::DeviceFeatureRequestFailed,
                    );
                }
                adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        required_limits: limits,
                        memory_hints: wgpu::MemoryHints::Performance,
                        ..Default::default()
                    })
                    .await?
            }
            Err(error) => {
                return Err(crate::execution::admission::ModelAdmissionError::Device {
                    code: "gpu_device_unavailable",
                    detail: error.to_string(),
                }
                .into())
            }
        };
        #[cfg(feature = "isf")]
        if let Some(config) = &profile_config {
            let info = adapter.get_info();
            config.set_adapter(info.name, format!("{:?}", info.backend));
        }

        // Register an error handler so wgpu validation errors surface as descriptive
        // messages rather than the wgpu 27 default fatal panic.
        device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            eprintln!("[Airframe] GPU error: {}", error);
        }));

        // Load model
        let tokenizer = prepared.tokenizer.clone();
        if tokenizer.metadata() != &validated.manifest().header().values {
            return Err(
                crate::execution::admission::ModelAdmissionError::Configuration {
                    code: "model_tokenizer_mismatch",
                    detail: "tokenizer metadata changed since model acquisition".into(),
                }
                .into(),
            );
        }
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let gpu_model = BindlessModel::try_load_prepared(&device, prepared)?;
        // Metal mapped initializers enqueue transfers until submission. Complete the
        // model phase before allocating the independently initialized output head:
        // Apple Paravirtual Metal returned zero storage when both large uploads
        // shared one pending batch. This is an ordering workaround, not a kernel fix.
        if adapter.get_info().backend == wgpu::Backend::Metal {
            let submission = queue.submit([]);
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(std::time::Duration::from_secs(120)),
                })
                .map_err(
                    |error| crate::execution::admission::ModelAdmissionError::Device {
                        code: "gpu_model_upload_failed",
                        detail: format!("Metal model initialization did not complete: {error}"),
                    },
                )?;
        }
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

        let validation_error = device.pop_error_scope().await;
        let allocation_error = device.pop_error_scope().await;
        if let Some(e) = allocation_error {
            return Err(crate::execution::admission::ModelAdmissionError::Device {
                code: "gpu_out_of_memory",
                detail: e.to_string(),
            }
            .into());
        }
        if let Some(e) = validation_error {
            return Err(crate::execution::admission::ModelAdmissionError::Device {
                code: "gpu_validation",
                detail: e.to_string(),
            }
            .into());
        }
        let embd_weight_offset = gpu_model
            .metadata
            .get_tensor_offset("token_embd.weight")
            .expect("token_embd.weight not found");

        // Quant type for token_embd.weight — may differ from layer weights.
        // Q4_K_M models often use Q4_K (type 12) for embeddings.
        let embd_quant_type = gpu_model
            .metadata
            .get_tensor_type("token_embd.weight")
            .unwrap_or(2); // default Q4_0

        // Row stride in bytes for the embedding table.
        // Q4_0: (dim/32)*18 = 2304 for dim=4096
        // Q4_K: (dim/256)*144 = 2304 for dim=4096  (coincidentally identical)
        // Q8_0: (dim/32)*34
        // F16:  dim*2
        // F32:  dim*4
        let row_bytes = gpu_model
            .validated()
            .manifest()
            .tensor("token_embd.weight")
            .expect("validated embedding")
            .row_bytes();
        eprintln!(
            "[GpuRuntime] token_embd.weight: quant_type={} row_bytes={}",
            embd_quant_type, row_bytes
        );

        let eos_token = tokenizer.eos_token();
        let im_end_token: Option<u32> = tokenizer.encode("<|im_end|>", false).ok().and_then(|v| {
            if v.len() == 1 {
                Some(v[0])
            } else {
                None
            }
        });

        #[cfg(feature = "isf")]
        if let Some(p) = &mut load_profile {
            p.complete();
        }
        Ok(Self {
            #[cfg(feature = "isf")]
            profile_config,
            request_gate: Mutex::new(()),
            device,
            queue,
            model: gpu_model,
            pipeline,
            shift_pipeline,
            tokenizer,
            spec,
            output_head_f32,
            kv_cache,
            embd_weight_offset,
            row_bytes,
            embd_quant_type,
            eos_token,
            im_end_token,
        })
    }
    /// Generate text using the Inference Saturation Fabric (ISF).
    ///
    /// Replaces the imperative for-loop in generate() with a D0 reactive graph.
    /// Rules registered at call time drive: embedding extraction, prefill dispatch,
    /// decode loop, EOS detection, and streaming — all as reactive facts.
    ///
    /// Patent Notice: Implements FSE + D0 Saturation Fabric.
    /// Pending patent by Michael A. Kuykendall. All rights reserved.
    #[cfg(feature = "isf")]
    #[allow(clippy::type_complexity)]
    pub fn generate(
        &self,
        prompt: &str,
        params: &SamplingParams,
        on_token: Option<Box<dyn FnMut(&str) + Send>>,
        control: Option<Box<dyn InferenceControl + Send + Sync>>,
        modify_logits: Option<Box<dyn Fn(&mut [f32]) + Send + Sync>>,
        trace_cb: Option<Box<dyn FnMut(usize, &[f32], f64) + Send>>,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.generate_with_metadata(prompt, params, on_token, control, modify_logits, trace_cb)
            .map(|output| output.text)
    }

    #[cfg(feature = "isf")]
    pub fn generate_with_metadata(
        &self,
        prompt: &str,
        params: &SamplingParams,
        on_token: Option<Box<dyn FnMut(&str) + Send>>,
        control: Option<Box<dyn InferenceControl + Send + Sync>>,
        modify_logits: Option<Box<dyn Fn(&mut [f32]) + Send + Sync>>,
        trace_cb: Option<Box<dyn FnMut(usize, &[f32], f64) + Send>>,
    ) -> Result<super::generation::GenerationOutput, Box<dyn std::error::Error + Send + Sync>> {
        self.generate_input_with_metadata(
            super::generation_input::GenerationInput::Text(prompt),
            params,
            on_token,
            control,
            modify_logits,
            trace_cb,
        )
    }
    #[cfg(feature = "isf")]
    fn generate_input_with_metadata(
        &self,
        input: super::generation_input::GenerationInput<'_>,
        params: &SamplingParams,
        on_token: Option<Box<dyn FnMut(&str) + Send>>,
        control: Option<Box<dyn InferenceControl + Send + Sync>>,
        modify_logits: Option<Box<dyn Fn(&mut [f32]) + Send + Sync>>,
        trace_cb: Option<Box<dyn FnMut(usize, &[f32], f64) + Send>>,
    ) -> Result<super::generation::GenerationOutput, Box<dyn std::error::Error + Send + Sync>> {
        let _generation_gate = self.request_gate.lock().map_err(|_| {
            crate::execution::admission::ExecutionError::Gpu(
                "runtime generation lock poisoned".into(),
            )
        })?;
        crate::execution::admission::validate_kernel_configuration()?;
        let mut profiling = self.profile_config.as_ref().map(|p| p.begin_generation());
        let result=(|| -> Result<super::generation::GenerationOutput,Box<dyn std::error::Error + Send + Sync>> {
        let profile_handle=profiling.as_ref().map(|p|p.handle());
        let contexts=profile_handle.as_ref().map(|p|crate::backend::bindless::profiling::generation_contexts(p.clone()));
        let prefill_profile=contexts.as_ref().map(|c|c.0.clone());
        let forward_profile=contexts.map(|c|c.1);
        use airframe_observe::facts::InferenceFact;
        use airframe_observe::isf::{ISFState, InferenceSaturationFabric};

        // Timestamped logging to /tmp/shimmy_isf_run.log — readable by Kiro
        let t0 = std::time::Instant::now();
        let log_path = "/tmp/shimmy_isf_run.log";
        let append_log = |msg: &str| {
            use std::io::Write;
            let line = format!("[T+{:.2}s] {}\n", t0.elapsed().as_secs_f64(), msg);
            if std::env::var("AIRFRAME_LOG_TDR_POLLS").is_ok() {
                eprintln!("[ISF] {}", line.trim());
            }
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
            {
                let _ = f.write_all(line.as_bytes());
            }
        };

        append_log("generate called");
        // Chat templates contain tokenizer-owned control markers. Parse them as
        // special tokens rather than feeding their literal spelling to the model.
        let prompt_tokens = match input {
            super::generation_input::GenerationInput::Text(prompt) => self.tokenizer.encode_with_options(prompt, &shimmytok::EncodeOptions { add_special_tokens: true, parse_special: true })?,
            super::generation_input::GenerationInput::TokenIds(ids) => {
                super::generation_input::validate_prompt_ids(ids, self.spec.n_vocab, self.spec.n_ctx)?;
                ids.to_vec()
            },
        };
        if prompt_tokens.is_empty() || prompt_tokens.len()>self.spec.n_ctx || prompt_tokens.iter().any(|id|*id as usize>=self.spec.n_vocab) {return Err(crate::execution::admission::ExecutionError::Request{code:"invalid_prompt_tokens",detail:"prompt must be nonempty, within vocabulary and context".into()}.into());}
        if !params.temperature.is_finite() || params.temperature<0.0 || !params.top_p.is_finite() || params.top_p<=0.0 || params.top_p>1.0 || !params.repetition_penalty.is_finite() || params.repetition_penalty<=0.0 || params.max_tokens>u32::MAX as usize {return Err(crate::execution::admission::ExecutionError::Request{code:"invalid_sampling_parameters",detail:"invalid sampling range or nonfinite value".into()}.into());}
        if prompt_tokens.len().checked_add(params.max_tokens.saturating_sub(1)).is_none_or(|end|end>self.spec.n_ctx){return Err(crate::execution::admission::ExecutionError::Request{code:"context_length_exceeded",detail:"prompt plus requested generation exceeds state capacity".into()}.into());}
        if let Some(p)=&profile_handle {p.set_prompt_tokens(prompt_tokens.len());}
        append_log(&format!("tokenized: {} tokens", prompt_tokens.len()));
        if params.max_tokens == 0 {
            return Ok(super::generation::GenerationOutput {
                text: String::new(),
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: 0,
                finish_reason: super::generation::FinishReason::Length,
            });
        }
        let dim = self.spec.n_embd as u32;
        let prompt_len = prompt_tokens.len() as u32;

        let mut request={let mut cache=self.kv_cache.lock().map_err(|_|crate::execution::admission::ExecutionError::Gpu("KV lock poisoned".into()))?; self.model.begin_request(&mut cache,prompt_tokens.len())?};
        let request_lease=request.lease();
        let execution_error=std::sync::Arc::new(std::sync::Mutex::new(None::<crate::execution::admission::ExecutionError>));

        // Encode extra stop tokens
        let extra_stop_ids: Vec<u32> = params
            .extra_stop_tokens
            .iter()
            .filter_map(|s| {
                self.tokenizer
                    .encode_with_options(
                        s,
                        &shimmytok::EncodeOptions {
                            add_special_tokens: false,
                            parse_special: true,
                        },
                    )
                    .ok()
                    .and_then(|v| if v.len() == 1 { Some(v[0]) } else { None })
            })
            .collect();

        // Build shared ISF state
        let state = std::sync::Arc::new(std::sync::Mutex::new(ISFState::new(
            prompt_len,
            params.max_tokens as u32,
            self.eos_token,
            extra_stop_ids,
            on_token,
        )));

        // Wire control / mask / trace hooks into the fabric. This is the fix for
        // the break where generate() dropped these args: the grammar pre-sample
        // mask and the post-sample InferenceControl gate now actually run on the
        // production ISF path. `airframe_observe` stays decoupled — it receives
        // plain `Arc<dyn Fn>` hooks, and we translate InferenceControl here.
        {
            let mut s=state.lock().unwrap();
            s.all_token_ids=prompt_tokens.clone();
            let mask_error=execution_error.clone();
            s.mask=Some(std::sync::Arc::new(move |logits:&mut[f32]|{
                if let Some(mask)=&modify_logits {mask(logits);}
                if logits.iter().any(|v|v.is_nan()||*v==f32::INFINITY)||!logits.iter().any(|v|v.is_finite()){
                    mask_error.lock().unwrap().get_or_insert(crate::execution::admission::ExecutionError::Request{code:"invalid_logit_mask",detail:"mask produced NaN/+infinity or no permitted token".into()});
                }
            }));
            let hook_error=execution_error.clone();let vocabulary=self.spec.n_vocab;
            let hook:std::sync::Arc<airframe_observe::isf::IsfControlHook>=std::sync::Arc::new(move |candidate,text,step,tokens,kv_len|{
                if hook_error.lock().unwrap().is_some(){return airframe_observe::isf::IsfControlDecision::Block("execution failed".into());}
                let token_ids:Vec<usize>=tokens.iter().map(|id|*id as usize).collect();
                let event=InferenceEvent{tokens:&token_ids,candidate_token:candidate,step,kv:KvSnapshot{len:kv_len,version:kv_len},text};
                let decision=control.as_ref().map(|c|c.intervene(&event)).unwrap_or(ControlDecision::Allow);
                match decision {
                    ControlDecision::Allow=>airframe_observe::isf::IsfControlDecision::Allow,
                    ControlDecision::ForceToken(id) if id<vocabulary=>airframe_observe::isf::IsfControlDecision::ForceToken(id),
                    ControlDecision::ForceToken(id)=>{
                        hook_error.lock().unwrap().get_or_insert(crate::execution::admission::ExecutionError::Request{code:"invalid_control_token",detail:format!("forced token {id} exceeds vocabulary")});
                        airframe_observe::isf::IsfControlDecision::Block("invalid forced token".into())
                    },
                    ControlDecision::EarlyExit=>airframe_observe::isf::IsfControlDecision::EarlyExit,
                    ControlDecision::BlockAndTerminate(reason)=>{
                        *hook_error.lock().unwrap()=Some(crate::execution::admission::ExecutionError::Request{code:"policy_blocked",detail:reason.clone()});
                        airframe_observe::isf::IsfControlDecision::Block(reason)
                    },
                }
            });
            s.control=Some(hook);
            s.trace=trace_cb.map(|b|std::sync::Arc::new(std::sync::Mutex::new(b)));
        }

        let embd_offset = self.embd_weight_offset;
        let row_bytes_val = self.row_bytes;
        let embd_quant_type_val = self.embd_quant_type;
        let prefill_chunk: u32 = std::env::var("SHIMMY_PREFILL_CHUNK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let prefill_chunk = prefill_chunk_for_platform(prefill_chunk, cfg!(target_os = "macos"));
        if let Some(profile) = &profile_handle { profile.set_prefill_chunk(prefill_chunk); }
        let temp = params.temperature;
        let top_p_val = params.top_p;
        let rep_penalty = params.repetition_penalty;
        let seed = params.seed;

        // Safety: all closures are called synchronously within this function's
        // lifetime. The references to self.device, self.queue, self.model,
        // self.pipeline, self.tokenizer are valid for the duration of generate().
        // We extend lifetimes here only because Arc<dyn Fn> requires 'static,
        // but these closures never escape this stack frame.
        let device_ref: &'static wgpu::Device = unsafe { &*(&self.device as *const _) };
        let queue_ref: &'static wgpu::Queue = unsafe { &*(&self.queue as *const _) };
        let model_ref: &'static crate::backend::bindless::loader::BindlessModel =
            unsafe { &*(&self.model as *const _) };
        let pipeline_ref: &'static crate::backend::bindless::pipeline::BindlessPipeline =
            unsafe { &*(&self.pipeline as *const _) };
        let tokenizer_ref: &'static shimmytok::Tokenizer =
            unsafe { &*(&*self.tokenizer as *const Tokenizer) };
        let output_head_ref: &'static Option<wgpu::Buffer> =
            unsafe { &*(&self.output_head_f32 as *const _) };
        let kv_cache_isf = self.kv_cache.clone();
        let spec_isf = self.spec.clone();

        // ── Closure: GPU embedding dequant ───────────────────────────────
        // Uses run_dequant_any_hot with the correct quant type for token_embd.weight.
        // This fixes NaN on Q4_K_M models where token_embd.weight is type=12 (Q4_K)
        // but the legacy run_dequant_request hardcodes the Q4_0 shader.
        let dequant_fn: std::sync::Arc<dyn Fn(u32, u32) -> Vec<f32> + Send + Sync> =
            std::sync::Arc::new(move |token_id: u32, dim: u32| {
                let row_offset = embd_offset + (token_id as u64 * row_bytes_val);
                pipeline_ref.run_dequant_any_hot(
                    device_ref,
                    queue_ref,
                    model_ref,
                    row_offset as u32,
                    dim,
                    embd_quant_type_val,
                )
            });

        // The dequantized F32 output head is only usable when it fits the device's
        // storage-buffer binding limit (it is bound as one storage binding). When it
        // does not (e.g. Gemma2 256K vocab on a 2GB-limit GPU, or ANY head on
        // llvmpipe's 128 MB limit), pass None — the pipeline reads the quantized
        // output.weight directly from the already-loaded blobs (inference.rs:
        // "blob-based quantized head").
        let max_binding = device_ref.limits().max_storage_buffer_binding_size as u64;
        let head_override: Option<&wgpu::Buffer> = match output_head_ref.as_ref() {
            Some(buf) if buf.size() <= max_binding => Some(buf),
            _ => {
                eprintln!(
                    "[GpuRuntime] output head F32 unavailable (binding limit {:.1} MB) — using blob-based quantized head",
                    max_binding as f64 / 1_048_576.0
                );
                None
            }
        };

        // ── Closure: GPU prefill dispatch ─────────────────────────────────
        let kv_for_prefill = kv_cache_isf.clone();
        let prefill_lease=request_lease.clone();
        let prefill_error=execution_error.clone();
        let spec_for_prefill = spec_isf.clone();
        let prefill_fn: std::sync::Arc<
            dyn Fn(Vec<f32>, u32) -> (Vec<f32>, Vec<f32>) + Send + Sync,
        > = std::sync::Arc::new(move |batched: Vec<f32>, _token_count: u32| {
            let mut cache_guard = kv_for_prefill.lock().unwrap();
            let result = match pipeline_ref.run_full_model_prefill_chunked_with_cache_state_profiled(
                device_ref,
                queue_ref,
                model_ref,
                &batched,
                head_override,
                0,
                Some((cache_guard.get_k_buffers(), cache_guard.get_v_buffers())),
                &spec_for_prefill,
                prefill_chunk,
                prefill_profile.clone(),
            ) {
                Ok((hidden,_l21,logits)) if !logits.is_empty() && logits.iter().all(|v|v.is_finite())=>(hidden,logits),
                Ok(_)=>{cache_guard.reset();prefill_error.lock().unwrap().get_or_insert(crate::execution::admission::ExecutionError::Gpu("nonfinite or empty GPU logits".into()));(vec![],vec![])},
                Err(error) => {cache_guard.reset();prefill_error.lock().unwrap().get_or_insert(crate::execution::admission::ExecutionError::Gpu(error));(vec![],vec![])},
            };
            cache_guard.set_seq_len(prefill_lease.lock().unwrap().snapshot().processed_positions as u32);
            // ── PPT Invariant Capture: the prefill pass is the golden [BOS,Hello]
            // forward; its last-token activation IS the position=1 layer output
            // the vault oracle rows reference. Emit FinalLogits at position=1. ──
            if std::env::var("AIRFRAME_CAPTURE_INVARIANT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            {
                if let Some(sink) =
                    crate::backend::bindless::pipeline::inference::invariant_capture_sink_mut()
                {
                    let rms = airframe_observe::facts::rms(&result.1);
                    let checksum = airframe_observe::facts::checksum(&result.1);
                    sink.push(airframe_observe::facts::CapturedLayer {
                        layer_idx: u32::MAX,
                        position: 1,
                        rms,
                        checksum,
                        is_final_logits: true,
                    });
                }
            }
            result
        });

        // ── Closure: GPU decode forward pass ─────────────────────────────
        let kv_for_forward = kv_cache_isf.clone();
        let forward_lease=request_lease.clone();
        let forward_error=execution_error.clone();
        let spec_for_forward = spec_isf.clone();
        let forward_fn: std::sync::Arc<
            dyn Fn(Vec<f32>, u32) -> (Vec<f32>, Vec<f32>) + Send + Sync,
        > = std::sync::Arc::new(move |token_data: Vec<f32>, current_pos: u32| {
            // Gate DIAG prints behind AIRFRAME_LOG_TDR_POLLS=1 (6ex)
            if std::env::var("AIRFRAME_LOG_TDR_POLLS").is_ok() {
                eprintln!(
                    "[DIAG] decode forward current_pos={} prompt_len={}",
                    current_pos, prompt_len
                );
            }
            let token_id = token_data[0] as u32;
            let row_offset = embd_offset + (token_id as u64 * row_bytes_val);
            let token_embd = pipeline_ref.run_dequant_any_hot(
                device_ref,
                queue_ref,
                model_ref,
                row_offset as u32,
                dim,
                embd_quant_type_val,
            );
            let mut cache_guard = kv_for_forward.lock().unwrap();
            let result = match pipeline_ref.run_full_model_prefill_chunked_with_cache_state_profiled(
                device_ref,
                queue_ref,
                model_ref,
                &token_embd,
                head_override,
                current_pos,
                Some((cache_guard.get_k_buffers(), cache_guard.get_v_buffers())),
                &spec_for_forward,
                1,
                forward_profile.clone(),
            ) {
                Ok((hidden,_l21,logits)) if !logits.is_empty() && logits.iter().all(|v|v.is_finite())=>(hidden,logits),
                Ok(_)=>{cache_guard.reset();forward_error.lock().unwrap().get_or_insert(crate::execution::admission::ExecutionError::Gpu("nonfinite or empty GPU logits".into()));(vec![],vec![])},
                Err(error) => {cache_guard.reset();forward_error.lock().unwrap().get_or_insert(crate::execution::admission::ExecutionError::Gpu(error));(vec![],vec![])},
            };
            cache_guard.set_seq_len(forward_lease.lock().unwrap().snapshot().processed_positions as u32);
            // ── PPT Invariant Capture: append FinalLogits to the probe sink ──
            if std::env::var("AIRFRAME_CAPTURE_INVARIANT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            {
                if let Some(sink) =
                    crate::backend::bindless::pipeline::inference::invariant_capture_sink_mut()
                {
                    let rms = airframe_observe::facts::rms(&result.1);
                    let checksum = airframe_observe::facts::checksum(&result.1);
                    sink.push(airframe_observe::facts::CapturedLayer {
                        layer_idx: u32::MAX,
                        position: current_pos,
                        rms,
                        checksum,
                        is_final_logits: true,
                    });
                }
            }
            result
        });

        // ── Closure: sampling ─────────────────────────────────────────────
        // NOTE: recent_tokens for repetition penalty are tracked in ISFState.recent_tokens
        // and passed to sample_token on each call via a snapshot. This fixes the
        // repetition loop issue where &[] was being passed (no penalty applied).
        let rng_cell = std::sync::Arc::new(std::sync::Mutex::new(Rng::new(seed)));
        let sample_fn: std::sync::Arc<dyn Fn(&mut Vec<f32>, &[u32]) -> u32 + Send + Sync> = {
            let rc = rng_cell.clone();
            std::sync::Arc::new(move |logits: &mut Vec<f32>, recent: &[u32]| {
                let mut rng = rc.lock().unwrap();
                sample_token(logits, temp, top_p_val, rep_penalty, recent, &mut rng)
            })
        };

        // ── Closure: token decode ─────────────────────────────────────────
        let decode_fn: std::sync::Arc<dyn Fn(u32) -> String + Send + Sync> =
            std::sync::Arc::new(move |token_id: u32| {
                tokenizer_ref
                    .decode_single(token_id, true)
                    .unwrap_or_default()
            });

        // GPU completion commits KV extent; sampling never claims a cache write.
        let kv_increment_fn:std::sync::Arc<dyn Fn()+Send+Sync>=std::sync::Arc::new(||{});

        // ── Build and run the ISF (Phase 3: reactive embedding via PromptToken facts) ─
        // Set prompt_token_ids in state so Rule 2 can check all_embeddings_cached.
        // Then assert PromptToken facts — Rule 1a derives EmbeddingRequest (unique per token_id),
        // Rule 1b dequants (exactly once per unique token_id via FactStore dedup),
        // Rule 2 detects all-ready and asserts PrefillBatchReady.
        // FSE invariant: N_unique dequants regardless of N_total tokens.
        append_log(&format!("pre_batch start, {} tokens", prompt_tokens.len()));
        let t_embed_start = std::time::Instant::now();
        let unique_count = {
            let mut s = state.lock().unwrap();
            s.prompt_token_ids = prompt_tokens.clone();
            let unique: std::collections::HashSet<u32> = prompt_tokens.iter().cloned().collect();
            unique.len()
        };

        let mut isf = InferenceSaturationFabric::new(
            state.clone(),
            dequant_fn,
            prefill_fn,
            forward_fn,
            sample_fn,
            decode_fn,
            kv_increment_fn,
            dim,
        );

        // Assert PromptToken facts — fabric drives embedding dequant reactively.
        // FactStore dedup ensures EmbeddingRequest fires exactly once per unique token_id.
        for (pos, &token_id) in prompt_tokens.iter().enumerate() {
            isf.fabric.assert(InferenceFact::PromptToken {
                position: pos as u32,
                token_id,
            });
        }
        append_log(&format!(
            "pre_batch: {} tokens, {} unique, asserted PromptToken facts",
            prompt_tokens.len(),
            unique_count
        ));
        let _ = t_embed_start; // timing logged after fixpoint

        let t_fixpoint = std::time::Instant::now();
        append_log("fixpoint start");
        let budget = airframe_observe::isf::request_run_budget(
            prompt_tokens.len(), params.max_tokens, 0,
        ).map_err(|error| crate::execution::admission::ExecutionError::Request {
            code: "isf_budget_overflow", detail: error.to_string(),
        })?;
        let run = isf.fabric.run_to_fixpoint(budget);
        append_log(&format!("ISF processed={} rounds={} saturated={} bound={}",
            run.total_facts_processed, run.total_rounds, run.saturated, budget.max_rounds));
        append_log(&format!(
            "fixpoint done, {:.2}s",
            t_fixpoint.elapsed().as_secs_f32()
        ));

        let s = state.lock().unwrap();
        append_log(&format!(
            "complete, {} chars, {} steps",
            s.generated_text.len(),
            s.all_token_ids.len().saturating_sub(s.prompt_len as usize)
        ));
        // Flush log
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
            {
                let _ = f.write_all(b"---\n");
            }
        }
        if let Some(error)=execution_error.lock().unwrap().take(){return Err(error.into());}
        // A control hook returning Block aborts generation — surface as an error.
        if let Some(reason) = &s.block_reason {
            return Err(format!("Blocked by control hook: {reason}").into());
        }
        if !run.saturated {
            return Err(crate::execution::admission::ExecutionError::Scheduler {
                code: "isf_budget_exhausted", detail: format!("processed {} facts under bound {}", run.total_facts_processed, budget.max_facts),
            }.into());
        }
        let finish_reason = match s.halt {
            Some(airframe_observe::facts::HaltReason::MaxTokensReached) => {
                super::generation::FinishReason::Length
            }
            Some(_) => super::generation::FinishReason::Stop,
            None => return Err(crate::execution::admission::ExecutionError::Scheduler {
                code: "isf_missing_terminal_reason", detail: "fact graph saturated without a terminal reason".into(),
            }.into()),
        };
        request.finish()?;
        self.kv_cache.lock().unwrap().reset();
        Ok(super::generation::GenerationOutput {
            text: s.generated_text.clone(),
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: s.all_token_ids.len().saturating_sub(prompt_tokens.len()),
            finish_reason,
        })
        })();
        if let Some(profile) = &mut profiling {
            profile.finish(result.is_ok());
        }
        result
    }
    #[cfg(feature = "isf")]
    pub fn generate_token_ids_with_metadata(
        &self,
        token_ids: &[u32],
        params: &SamplingParams,
        on_token: Option<Box<dyn FnMut(&str) + Send>>,
        control: Option<Box<dyn InferenceControl + Send + Sync>>,
        modify_logits: Option<Box<dyn Fn(&mut [f32]) + Send + Sync>>,
        trace_cb: Option<Box<dyn FnMut(usize, &[f32], f64) + Send>>,
    ) -> Result<super::generation::GenerationOutput, Box<dyn std::error::Error + Send + Sync>> {
        self.generate_input_with_metadata(
            super::generation_input::GenerationInput::TokenIds(token_ids),
            params,
            on_token,
            control,
            modify_logits,
            trace_cb,
        )
    }
    /// Generate text via the Inference Saturation Fabric (ISF) dispatch path.
    ///
    /// This is the single production entry point. `on_token` is called for each
    /// generated token (for streaming). `control`, `modify_logits`, and `trace_cb`
    /// are forwarded into the fabric and applied on every decode step: the mask
    /// runs pre-sample (grammar) and the control runs post-sample (grammar/FSE/
    /// math-bypass). A control returning `Block` aborts generation with an error.
    /// Reset the KV cache (for a new conversation).
    /// Compatibility wrapper; production callers should propagate try_reset().
    /// Panics when reset fails, including while a generation request is active.
    #[deprecated(note = "use try_reset() and propagate its error")]
    pub fn reset(&self) {
        self.try_reset()
            .expect("GPU reset failed; use try_reset() to handle reset errors");
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn tokenizer_arc(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
    pub fn eos_token(&self) -> u32 {
        self.eos_token
    }
    pub fn im_end_token(&self) -> Option<u32> {
        self.im_end_token
    }

    /// Get a reference to the model spec.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Returns the Jinja2 chat template from the model's GGUF metadata, if present.
    /// Use shimmyjinja::render_chat_template() to apply it.
    pub fn chat_template(&self) -> Option<&str> {
        self.spec.chat_template.as_deref()
    }

    // ── PPT Invariant Cage diagnostic accessors ────────────────────────────
    pub fn embd_quant_type(&self) -> u32 {
        self.embd_quant_type
    }
    pub fn embd_weight_offset(&self) -> u64 {
        self.embd_weight_offset
    }
    pub fn row_bytes(&self) -> u64 {
        self.row_bytes
    }
    /// Get the quant type for any tensor by name (diagnostic).
    pub fn tensor_type(&self, name: &str) -> Option<u32> {
        self.model.metadata.get_tensor_type(name)
    }
    /// Get the byte offset for any tensor by name (diagnostic).
    pub fn tensor_offset(&self, name: &str) -> Option<u64> {
        self.model.metadata.get_tensor_offset(name)
    }
    /// Dequantize any tensor from the GGUF blob (diagnostic).
    pub fn dequant_tensor_f32(&self, name: &str, count: u32) -> Vec<f32> {
        if let Some(off) = self.tensor_offset(name) {
            let qt = self.tensor_type(name).unwrap_or(0);
            self.pipeline.run_dequant_any_hot(
                &self.device,
                &self.queue,
                &self.model,
                off as u32,
                count,
                qt,
            )
        } else {
            vec![]
        }
    }
    /// Dequantize a single embedding row for diagnostics.
    pub fn dequant_token_embd(&self, token_id: u32) -> Vec<f32> {
        let dim = self.spec.n_embd as u32;
        let row_offset = self.embd_weight_offset + (token_id as u64 * self.row_bytes);
        self.pipeline.run_dequant_any_hot(
            &self.device,
            &self.queue,
            &self.model,
            row_offset as u32,
            dim,
            self.embd_quant_type,
        )
    }

    /// Dequantize the output head to F32 and upload it as a storage buffer.
    ///
    /// Returns `Ok(None)` when the F32 head cannot fit the device's buffer or
    /// storage-binding limits — callers must then fall back to the blob-based
    /// quantized head (pass `None` as the head override to the pipeline).
    pub fn load_output_head_f32(
        model_path: &str,
        gpu_model: &BindlessModel,
        device: &wgpu::Device,
        spec: &ModelSpec,
    ) -> Result<Option<wgpu::Buffer>, Box<dyn std::error::Error + Send + Sync>> {
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
                    .ok_or_else(|| {
                        format!(
                            "token_embd.weight not found in tensor_offsets map (tensor_count={})",
                            gpu_model.metadata.tensor_count
                        )
                    })?;
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
            tensor_offset - data_start // v2 absolute → convert to relative
        } else {
            tensor_offset // v3 relative → use directly
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
                    .as_chunks::<4>()
                    .0
                    .iter()
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

        let buf_size = (tensor_f32.data.len() as u64) * 4;
        let max_buf = device.limits().max_buffer_size;
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        if buf_size > max_buf || buf_size > max_binding {
            eprintln!(
                "[OutputHead] F32 head ({:.1} MB) exceeds device limits (max_buffer_size={:.1} MB, \
                 max_storage_buffer_binding_size={:.1} MB) — falling back to blob-based quantized head",
                buf_size as f64 / 1_048_576.0,
                max_buf as f64 / 1_048_576.0,
                max_binding as f64 / 1_048_576.0,
            );
            return Ok(None);
        }
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

        Ok(Some(buffer))
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
#[allow(clippy::items_after_test_module)]
mod tests {
    #[test]
    fn metal_prefill_bounds_the_observed_oversized_workload() {
        assert_eq!(super::prefill_chunk_for_platform(308, true), 64);
        assert_eq!(super::prefill_chunk_for_platform(512, true), 64);
        assert_eq!(super::prefill_chunk_for_platform(u32::MAX, true), 64);
    }

    #[test]
    fn metal_prefill_preserves_smaller_chunks_and_invalid_zero() {
        for requested in [0, 1, 16, 32, 64] {
            assert_eq!(
                super::prefill_chunk_for_platform(requested, true),
                requested
            );
        }
    }

    #[test]
    fn other_platform_prefill_configuration_is_unchanged() {
        for requested in [0, 1, 32, 64, 128, 308, 512, u32::MAX] {
            assert_eq!(
                super::prefill_chunk_for_platform(requested, false),
                requested
            );
        }
    }

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

/// Create an FSE (Fused Semantic Execution) control from reject patterns.
pub fn fse_control_from_patterns(
    patterns: &str,
) -> Option<Box<dyn InferenceControl + Send + Sync>> {
    if patterns.is_empty() {
        return None;
    }
    use crate::fse_control::FseControl;
    use libfse::{FseMap, FseOpcode, Rule};
    let rules: Vec<Rule> = if patterns.is_empty() {
        Vec::new()
    } else {
        patterns
            .split(',')
            .enumerate()
            .flat_map(|(i, p)| {
                let trimmed = p.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(Rule::new(trimmed.as_bytes(), FseOpcode::Reject(i as u32)))
                }
            })
            .collect()
    };
    FseMap::compile(rules).ok().and_then(|map| {
        FseControl::new(map)
            .ok()
            .map(|c| Box::new(c) as Box<dyn InferenceControl + Send + Sync>)
    })
}

/// Create a trace callback (stub — trace not yet wired).
pub fn trace_callback(_trace_path: &str) -> Option<Box<dyn FnMut(usize, &[f32], f64) + Send>> {
    None
}
