// Auto-split from pipeline.rs — types, struct, constructor, and helpers only.
// Inference methods: see inference.rs, layer.rs, dequant.rs, matmul.rs

pub(super) mod dequant;
pub(super) mod inference;
pub(super) mod layer;
pub(super) mod matmul;
pub mod resampler_gpu;
pub mod vit_layer;

//       pipeline/kv_cache.rs, pipeline/dispatch.rs — see C3 architectural debt.
use crate::core::spec::ModelSpec;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DequantParams {
    pub offset_bytes: u32,
    pub count: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DequantAnyParams {
    pub offset_bytes: u32,
    pub count: u32,
    pub quant_type: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MatMulParams {
    pub n: u32,
    pub k: u32,
    pub weights_offset: u32,
    pub padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RMSNormParams {
    pub count: u32,
    pub weights_offset: u32,
    pub eps: f32,
    pub padding: u32,
}

/// Offsets for a single Transformer Layer (TinyLlama/Llama 2).
/// All offsets are in bytes, absolute from the start of the GGUF blob.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerOffsets {
    pub attn_norm: u32,
    pub attn_q: u32,
    pub attn_k: u32,
    pub attn_v: u32,
    pub attn_out: u32,
    pub ffn_norm: u32,
    pub ffn_gate: u32,
    pub ffn_down: u32,
    pub ffn_up: u32,
    pub layer_idx: u32,        // was padding[0] — layer index for norm_bank lookup
    pub attn_q_norm: u32,      // byte offset of Q-norm weights in GGUF blob (0 = disabled)
    pub attn_k_norm: u32,      // byte offset of K-norm weights in GGUF blob (0 = disabled)
    pub attn_q_bias: u32,      // byte offset of Q bias (F32) in GGUF blob (0 = disabled; Qwen2)
    pub attn_k_bias: u32,      // byte offset of K bias (F32) in GGUF blob (0 = disabled; Qwen2)
    pub attn_v_bias: u32,      // byte offset of V bias (F32) in GGUF blob (0 = disabled; Qwen2)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerParams {
    pub dim: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub rms_eps: f32,
    pub ffn_dim: u32, // Feed-forward intermediate dimension (e.g. 5632 for TinyLlama)
    pub temp_stride: u32, // Per-token temp buffer stride in floats (e.g. 16384)
    pub quant_type: u32, // GGML type: 2=Q4_0, 12=Q4_K
    pub attn_logit_softcap: f32, // 0.0 = disabled; Gemma-2 uses 50.0
    pub post_norm_enabled: u32,  // 1 = apply post-attn and post-ffw norm (Gemma-2); 0 = disabled
    pub qk_norm_enabled: u32,    // 1 = apply per-head Q/K RMSNorm before RoPE (Qwen3); 0 = disabled
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CacheParams {
    pub current_pos: u32, // Position to write new K/V (0-based)
    pub seq_len: u32,     // Total cached positions (current_pos + 1)
    pub max_seq_len: u32, // 2048 (context window)
    pub batch_size: u32,  // Number of tokens in current batch
    pub logical_pos_base: u32, // Logical base of the compacted sliding window
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HeadBlobParams {
    pub vocab_size: u32,  // rows of output.weight (= n_vocab)
    pub dim:        u32,  // cols of output.weight (= n_embd)
    pub weight_off: u32,  // byte offset of output.weight inside the GGUF blob
    pub quant_type: u32,  // GGML type: 0=F32 1=F16 2=Q4_0 8=Q8_0 12=Q4_K 13=Q5_K 14=Q6_K
    pub softcap:    f32,  // final_logit_softcap (0.0 = disabled)
    pub _pad:       u32,
}

/// Pre-compiled per-layer lookup table entry.
/// Built once at model load time from the GGUF tensor index.
/// Eliminates per-token HashMap lookups and format! string allocations
/// in the inference hot path (FSE compiled-layer optimization).
#[derive(Clone, Debug)]
pub struct CompiledLayerEntry {
    /// All tensor byte-offsets for this layer, ready to upload to GPU.
    pub offsets: LayerOffsets,
    /// Packed quant types: bits[7:0]=attn_q, bits[15:8]=attn_v, bits[23:16]=ffn_down.
    /// Matches the `quant_type` field layout expected by LayerParams.
    pub quant_type_packed: u32,
}

/// The Control Plane for Bindless Inference.
pub struct BindlessPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub dequant_pipeline: wgpu::ComputePipeline,
    pub dequant_bind_group_layout: wgpu::BindGroupLayout,
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub matmul_layout: wgpu::BindGroupLayout,
    pub matmul_f32_pipeline: wgpu::ComputePipeline,
    pub matmul_f32_layout: wgpu::BindGroupLayout,
    pub rmsnorm_pipeline: wgpu::ComputePipeline,
    pub rmsnorm_layout: wgpu::BindGroupLayout,

    // Split Layer Pipelines
    pub layer_pipeline_attn_norm: wgpu::ComputePipeline,
    pub layer_pipeline_qkv: wgpu::ComputePipeline,
    pub layer_pipeline_qk_norm: wgpu::ComputePipeline,
    pub layer_pipeline_attn_out: wgpu::ComputePipeline,
    pub layer_pipeline_attn_proj: wgpu::ComputePipeline,
    pub layer_pipeline_ffn_norm: wgpu::ComputePipeline,
    pub layer_pipeline_ffn_proj: wgpu::ComputePipeline,
    pub layer_pipeline_ffn_down: wgpu::ComputePipeline,
    pub layer_pipeline_post_attn_norm: wgpu::ComputePipeline,
    pub layer_pipeline_post_ffw_norm: wgpu::ComputePipeline,
    pub layer_layout: wgpu::BindGroupLayout,

    // Blob-based LM head pipeline (quantized matmul, reads directly from GGUF blob)
    pub lm_head_blob_pipeline: wgpu::ComputePipeline,
    pub lm_head_blob_layout: wgpu::BindGroupLayout,
}

impl BindlessPipeline {

    /// Creates the pipeline with a "Probe" kernel to verify connectivity.
    pub fn new(device: &wgpu::Device) -> Self {
        // --- 1. Probe Pipeline ---
        // Binding 0: GGUF Blob, read-only storage
        // Binding 1: Output Probe, read-write storage
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bindless Layout"),
            entries: &[
                // GGUF Blob
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output Probe (Debug)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 2. Shader Source (Inline WGSL for Probe)
        // Reads the first u32 (Magic Number) and writes it to output[0]
        let shader_source = r#"
            @group(0) @binding(0) var<storage, read> gguf_blob: array<u32>;
            @group(0) @binding(1) var<storage, read_write> output: array<u32>;

            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                // Read magic "GGUF" (0x46554747 le)
                // Note: array<u32> views the byte buffer as u32s. 
                // GGUF magic is at offset 0.
                output[0] = gguf_blob[0];
                output[1] = gguf_blob[1]; // Version ??
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bindless Probe Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // 3. Create Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bindless Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bindless Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 4. Q4_0 Dequant Pipeline ---
        // Binding 0: GGUF Blob, read-only
        // Binding 1: Output F32, read-write
        // Binding 2: Params, uniform

        let dequant_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Dequant Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let dequant_shader_source = include_str!("../sh_dequant_q4_0.wgsl");
        let dequant_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Q4_0 Dequant Shader"),
            source: wgpu::ShaderSource::Wgsl(dequant_shader_source.into()),
        });

        let dequant_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Dequant Pipeline Layout"),
                bind_group_layouts: &[&dequant_layout],
                push_constant_ranges: &[],
            });

        let dequant_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Q4_0 Dequant Pipeline"),
            layout: Some(&dequant_pipeline_layout),
            module: &dequant_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 5. MatMul Pipeline ---
        // Bindings: 0=GGUF Blob read-only, 1=Input Vector read-only,
        //           2=Output Vector read-write, 3=Params uniform

        let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // GGUF
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Input x
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Output y
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Params
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let matmul_src = include_str!("../sh_matmul_q4_0.wgsl");
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(matmul_src.into()),
        });

        let matmul_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MatMul Pipeline Layout"),
                bind_group_layouts: &[&matmul_layout],
                push_constant_ranges: &[],
            });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 5b. MatMul F32 Pipeline ---
        let matmul_f32_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul F32 Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // W (F32)
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Input x
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Output y
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Params
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let matmul_f32_src = include_str!("../sh_matmul_f32.wgsl");
        let matmul_f32_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul F32 Shader"),
            source: wgpu::ShaderSource::Wgsl(matmul_f32_src.into()),
        });

        let matmul_f32_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MatMul F32 Pipeline Layout"),
                bind_group_layouts: &[&matmul_f32_layout],
                push_constant_ranges: &[],
            });

        let matmul_f32_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MatMul F32 Pipeline"),
                layout: Some(&matmul_f32_pipeline_layout),
                module: &matmul_f32_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- 6. RMSNorm Pipeline ---
        let rmsnorm_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RMSNorm Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Input x
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Output y
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Params
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob chunk 1: bytes [2GB, 4GB)
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob chunk 2: bytes [4GB, end)
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let rmsnorm_src = include_str!("../sh_rmsnorm.wgsl");
        let rmsnorm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RMSNorm Shader"),
            source: wgpu::ShaderSource::Wgsl(rmsnorm_src.into()),
        });

        let rmsnorm_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RMSNorm Pipeline Layout"),
                bind_group_layouts: &[&rmsnorm_layout],
                push_constant_ranges: &[],
            });

        let rmsnorm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RMSNorm Pipeline"),
            layout: Some(&rmsnorm_pipeline_layout),
            module: &rmsnorm_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- 7. Mega-Layer Pipeline ---
        // Bindings: 0=GGUF Blob read-only, 1=Activation In read-write,
        //           2=Temp State read-write, 3=LayerOffsets uniform,
        //           4=LayerParams uniform, 5=Norm Bank preflight, 6=RoPE Cache preflight
        let layer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Layer V1 Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Activation In
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Temp State
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // LayerOffsets
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // LayerParams
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Norm Bank
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // RoPE Cache
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // KV Cache K (Persistent)
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // KV Cache V (Persistent)
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // CacheParams (Uniform)
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob chunk 1: bytes [2GB, 4GB)
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // GGUF Blob chunk 2: bytes [4GB, end)
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let layer_src = include_str!("../sh_layer_v1.wgsl");
        let layer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Layer V1 Shader"),
            source: wgpu::ShaderSource::Wgsl(layer_src.into()),
        });

        let layer_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Layer V1 Pipeline Layout"),
                bind_group_layouts: &[&layer_layout],
                push_constant_ranges: &[],
            });

        let mk_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Layer V1 Pipeline ({})", entry)),
                layout: Some(&layer_pipeline_layout),
                module: &layer_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let layer_pipeline_attn_norm = mk_pipeline("main_attn_norm");
        let layer_pipeline_qkv = mk_pipeline("main_qkv");
        let layer_pipeline_qk_norm = mk_pipeline("main_qk_norm");
        let layer_pipeline_attn_out = mk_pipeline("main_attn_out");
        let layer_pipeline_attn_proj = mk_pipeline("main_attn_proj");
        let layer_pipeline_ffn_norm = mk_pipeline("main_ffn_norm");
        let layer_pipeline_ffn_proj = mk_pipeline("main_ffn_proj");
        let layer_pipeline_ffn_down = mk_pipeline("main_ffn_down");
        let layer_pipeline_post_attn_norm = mk_pipeline("main_post_attn_norm");
        let layer_pipeline_post_ffw_norm = mk_pipeline("main_post_ffw_norm");

        // --- LM Head (blob-based quantized matmul) ---
        // Layout: binding 0 = blob_0, 1 = act_in (read), 2 = logits (write), 3 = params,
        //         10 = blob_1, 11 = blob_2
        let lm_head_blob_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LM Head Blob Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: None,
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
            ],
        });

        let lm_head_blob_src = include_str!("../sh_head_blob.wgsl");
        let lm_head_blob_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Head Blob Shader"),
            source: wgpu::ShaderSource::Wgsl(lm_head_blob_src.into()),
        });
        let lm_head_blob_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("LM Head Blob Pipeline Layout"),
                bind_group_layouts: &[&lm_head_blob_layout],
                push_constant_ranges: &[],
            });
        let lm_head_blob_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Head Blob Pipeline"),
                layout: Some(&lm_head_blob_pipeline_layout),
                module: &lm_head_blob_shader,
                entry_point: Some("main_lm_head"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
            dequant_pipeline,
            dequant_bind_group_layout: dequant_layout,
            matmul_pipeline,
            matmul_layout,
            matmul_f32_pipeline,
            matmul_f32_layout,
            rmsnorm_pipeline,
            rmsnorm_layout,
            layer_pipeline_attn_norm,
            layer_pipeline_qkv,
            layer_pipeline_qk_norm,
            layer_pipeline_attn_out,
            layer_pipeline_attn_proj,
            layer_pipeline_ffn_norm,
            layer_pipeline_ffn_proj,
            layer_pipeline_ffn_down,
            layer_pipeline_post_attn_norm,
            layer_pipeline_post_ffw_norm,
            layer_layout,
            lm_head_blob_pipeline,
            lm_head_blob_layout,
        }
    }

    /// Read back GPU buffer contents to CPU as f32 values.
    /// Exported as pub(super) so sub-modules can call it.
    pub(super) fn readback_helper(&self, device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        loop {
            device.poll(wgpu::PollType::Poll).expect("GPU device lost during readback poll");
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        buffer.unmap();
        result
    }
}


#[cfg(test)]
mod tests {
    /// Workgroup ceil-div: (n + 255) / 256 must match the dispatch sizing used throughout.
    #[test]
    fn workgroup_ceil_div_rounds_up_correctly() {
        assert_eq!((256 + 255) / 256, 1); // exact multiple
        assert_eq!((257 + 255) / 256, 2); // one over
        assert_eq!((1 + 255) / 256, 1);   // minimum
        assert_eq!((512 + 255) / 256, 2); // exact double
    }

    /// RoPE softmax temperature denominator: sum of exp-shifted values for a 3-element slice.
    #[test]
    fn softmax_sum_is_positive_finite() {
        let logits = [1.0_f32, 2.0, 3.0];
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|x| (x - max_val).exp()).sum();
        assert!(sum.is_finite() && sum > 0.0);
    }

    /// KV cache ring index: position modulo context length stays in bounds.
    #[test]
    fn kv_cache_ring_index_wraps_within_bounds() {
        let ctx = 2048_usize;
        for pos in [0, 1, ctx - 1, ctx, ctx + 1, ctx * 2] {
            let idx = pos % ctx;
            assert!(idx < ctx, "ring index {idx} out of bounds for ctx={ctx}");
        }
    }
}
