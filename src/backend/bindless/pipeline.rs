use super::loader::BindlessModel;
use crate::core::spec::ModelSpec;
use wgpu::util::DeviceExt;

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
    pub padding: [u32; 3], // Pad to 48 bytes (12 * 4) for alignment
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
    pub padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CacheParams {
    pub current_pos: u32, // Position to write new K/V (0-based)
    pub seq_len: u32,     // Total cached positions (current_pos + 1)
    pub max_seq_len: u32, // 2048 (context window)
    pub batch_size: u32,  // Number of tokens in current batch
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
    pub layer_pipeline_qkv: wgpu::ComputePipeline,
    pub layer_pipeline_attn_out: wgpu::ComputePipeline,
    pub layer_pipeline_attn_proj: wgpu::ComputePipeline,
    pub layer_pipeline_ffn_proj: wgpu::ComputePipeline,
    pub layer_pipeline_ffn_down: wgpu::ComputePipeline,
    pub layer_layout: wgpu::BindGroupLayout,
}

impl BindlessPipeline {
    /// Creates the pipeline with a "Probe" kernel to verify connectivity.
    pub fn new(device: &wgpu::Device) -> Self {
        // --- 1. Probe Pipeline (Keep Existing) ---
        // Binding 0: The GGUF Blob (ReadOnly Storage)
        // Binding 1: Output Probe (ReadWrite Storage)
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
        // Binding 0: GGUF Blob (ReadOnly)
        // Binding 1: Output F32 (ReadWrite)
        // Binding 2: Params (Uniform)

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

        let dequant_shader_source = include_str!("sh_dequant_q4_0.wgsl");
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
        // Bindings:
        // 0: GGUF Blob (Storage Read)
        // 1: Input Vector (Storage Read)
        // 2: Output Vector (Storage ReadWrite)
        // 3: Params (Uniform)

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

        let matmul_src = include_str!("sh_matmul_q4_0.wgsl");
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

        let matmul_f32_src = include_str!("sh_matmul_f32.wgsl");
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
            ],
        });

        let rmsnorm_src = include_str!("sh_rmsnorm.wgsl");
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
        // Bindings:
        // 0: GGUF Blob (RO Storage)
        // 1: Activation In (RW Storage)
        // 2: Temp State (RW Storage)
        // 3: LayerOffsets (Uniform)
        // 4: LayerParams (Uniform)
        // 5: Norm Bank (Preflight)
        // 6: RoPE Cache (Preflight)
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
            ],
        });

        let layer_src = include_str!("sh_layer_v1.wgsl");
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

        let layer_pipeline_qkv = mk_pipeline("main_qkv");
        let layer_pipeline_attn_out = mk_pipeline("main_attn_out");
        let layer_pipeline_attn_proj = mk_pipeline("main_attn_proj");
        let layer_pipeline_ffn_proj = mk_pipeline("main_ffn_proj");
        let layer_pipeline_ffn_down = mk_pipeline("main_ffn_down");

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
            layer_pipeline_qkv,
            layer_pipeline_attn_out,
            layer_pipeline_attn_proj,
            layer_pipeline_ffn_proj,
            layer_pipeline_ffn_down,
            layer_layout,
        }
    }

    /// Dispatch the probe to verify we can read the GGUF magic number.
    pub fn run_probe(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
    ) -> Vec<u32> {
        println!("[BindlessPipeline] Running Probe...");

        // 1. Create Output Buffer (Size 8 bytes = 2 u32s)
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Probe Output"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 2. Create BindGroup
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Probe BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // 3. Encode Command
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Probe Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // 4. Copier for Readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, 8);

        // 5. Submit
        queue.submit(Some(encoder.finish()));

        // 6. Readback
        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        // Loop poll until mapped
        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Verify Q4_0 Block Dequantization
    pub fn run_dequant_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
    ) -> Vec<f32> {
        println!("[BindlessPipeline] Running Dequant Test...");

        // Output: 32 elements * 4 bytes = 128 bytes
        let output_size = 32 * 4;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dequant Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create DequantParams uniform buffer (offset=0, count=32)
        use wgpu::util::DeviceExt;
        let params_data: [u32; 4] = [0, 32, 0, 0]; // offset_bytes, count, pad1, pad2
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dequant Params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dequant BindGroup"),
            layout: &self.dequant_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dequant Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.dequant_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1); // 1 group of 32 threads
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Run Dequant Test (or get embedding)
    pub fn run_dequant_request(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        offset_bytes: u32,
        count: u32,
    ) -> Vec<f32> {
        // Upload Params
        let params = DequantParams {
            offset_bytes,
            count,
            pad1: 0,
            pad2: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dequant Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create Output Buffer
        let output_size = (count as usize * 4) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Dequant"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dequant BindGroup"),
            layout: &self.dequant_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.dequant_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (count + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(res) = rx.try_recv() {
                res.unwrap();
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        result
    }

    /// Run MatMul Test (GEMV)
    pub fn run_matmul_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input: &[f32],
        params: MatMulParams,
    ) -> Vec<f32> {
        println!(
            "[BindlessPipeline] Running MatMul Test (N={}, K={})",
            params.n, params.k
        );

        // Upload Input Vector
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input X"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create Output Buffer
        let output_size = (params.n as usize * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Y"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload Params
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MatMul Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul BindGroup"),
            layout: &self.matmul_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.matmul_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (params.n + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Run MatMul with pre-dequantized F32 weights (for Q6_K workaround)
    pub fn run_matmul_f32(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weights_f32: &wgpu::Buffer, // Pre-uploaded F32 weights
        input: &[f32],
        n: u32, // Output dimension (vocab_size)
        k: u32, // Input dimension (hidden_size)
    ) -> Vec<f32> {
        println!("[BindlessPipeline] Running MatMul F32 (N={}, K={})", n, k);

        // Upload Input Vector
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input X"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create Output Buffer
        let output_size = (n as usize * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Y"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload Params (offset is ignored for F32, but struct requires it)
        let params = MatMulParams {
            n,
            k,
            weights_offset: 0, // Unused for F32 path
            padding: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MatMul Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul F32 BindGroup"),
            layout: &self.matmul_f32_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights_f32.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul F32 Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.matmul_f32_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (n + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Run RMSNorm Test
    pub fn run_rmsnorm_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input: &[f32],
        params: RMSNormParams,
    ) -> Vec<f32> {
        println!(
            "[BindlessPipeline] Running RMSNorm Test (Size={}, Eps={:e})",
            params.count, params.eps
        );

        // Upload Input Vector
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input X"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Create Output Buffer
        let output_size = (params.count as usize * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Y"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload Params
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMSNorm Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RMSNorm BindGroup"),
            layout: &self.rmsnorm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RMSNorm Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.rmsnorm_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Runs the full model loop (Layers 0..N + Final Norm + Head)
    pub fn run_full_model(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        spec: &ModelSpec,
    ) -> Vec<f32> {
        self.run_full_model_with_cache(
            device,
            queue,
            model,
            input_embd,
            head_weights_override,
            0,
            1,
            spec,
        )
    }

    pub fn run_full_model_with_cache(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        current_pos: u32,
        seq_len: u32,
        spec: &ModelSpec,
    ) -> Vec<f32> {
        self.run_full_model_with_cache_state(
            device,
            queue,
            model,
            input_embd,
            head_weights_override,
            current_pos,
            seq_len,
            None,
            spec,
        )
        .2
    }

    pub fn run_full_model_with_cache_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input_embd: &[f32],
        head_weights_override: Option<&wgpu::Buffer>,
        current_pos: u32,
        seq_len: u32,
        kv_state: Option<(&[wgpu::Buffer], &[wgpu::Buffer])>,
        spec: &ModelSpec,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Derive all constants from ModelSpec
        let dim = spec.n_embd as u32;
        let layer_count = spec.n_layer;
        let vocab_size = spec.n_vocab as u32;
        let ffn_dim = spec.ff_dim as u32;
        let temp_stride = spec.temp_buffer_size as u32;

        let params_base = LayerParams {
            dim,
            head_count: spec.n_head as u32,
            head_count_kv: spec.n_head_kv as u32,
            head_dim: spec.head_dim as u32,
            rms_eps: spec.rms_eps,
            ffn_dim,
            temp_stride,
            padding: 0,
        };

        // 1. Buffers
        let batch_size = (input_embd.len() as u32) / dim;
        // A. Activation (Residual Stream) - Init with Embeddings
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation"),
            contents: bytemuck::cast_slice(input_embd),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // B. Temp Buffer
        // Needs to hold FFN Gate + Up + scratch space per token
        let temp_buffer_size = batch_size as u64 * temp_stride as u64 * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp State"),
            size: temp_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // C. Layer Params (Constant)
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params_base),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // D. Output Logits
        // Only computed for the LAST token in the sequence (usually).
        // If we want all logits, we'd need batch_size * vocab_size.
        // For now, let's stick to last token logic for compatibility.
        let logits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Logits"),
            size: (vocab_size as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let l21_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("L21 Final Norm Output"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pre_norm_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pre-Final-Norm Output"),
            size: (dim as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // E. KV Cache
        // Shader cache indexing is layer-local: [pos, kv_head, head_dim] with no layer axis.
        // Therefore full-model loop must bind a distinct K/V buffer per layer.
        let kv_size_per_buffer = spec.kv_cache_size_per_layer as u64;
        let local_kv_storage_per_layer = if kv_state.is_none() {
            let mut bufs = Vec::with_capacity(layer_count as usize);
            for i in 0..layer_count {
                let kv_buffer_k = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("KV Cache K L{}", i)),
                    size: kv_size_per_buffer,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let kv_buffer_v = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("KV Cache V L{}", i)),
                    size: kv_size_per_buffer,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                bufs.push((kv_buffer_k, kv_buffer_v));
            }
            Some(bufs)
        } else {
            None
        };

        // F. Cache Params
        let cache_params = CacheParams {
            current_pos,
            seq_len, // Total cached positions (including this batch)
            max_seq_len: spec.n_ctx as u32,
            batch_size,
        };

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 2. Prepare Layers (Offsets & BindGroups)
        let mut layer_bind_groups = Vec::new();
        let mut _offset_buffers = Vec::new(); // Keep alive

        for i in 0..layer_count {
            let offsets = model
                .metadata
                .get_layer_offsets(i, spec.arch_string())
                .expect(&format!("Missing offsets for layer {}", i));

            let (kv_buffer_k_ref, kv_buffer_v_ref): (&wgpu::Buffer, &wgpu::Buffer) =
                if let Some((kv_k_layers, kv_v_layers)) = kv_state {
                    (&kv_k_layers[i as usize], &kv_v_layers[i as usize])
                } else {
                    let (local_k, local_v) = &local_kv_storage_per_layer
                        .as_ref()
                        .expect("local KV storage missing")[i as usize];
                    (local_k, local_v)
                };

            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} Offsets", i)),
                contents: bytemuck::bytes_of(&offsets),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Layer {} BindGroup", i)),
                layout: &self.layer_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: model.gpu_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: activation_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: temp_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: model
                            .preflight
                            .as_ref()
                            .unwrap()
                            .norm_bank_buffer
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: model
                            .preflight
                            .as_ref()
                            .unwrap()
                            .rope_cache_buffer
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: kv_buffer_k_ref.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: kv_buffer_v_ref.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: cache_params_buffer.as_entire_binding(),
                    },
                ],
            });

            _offset_buffers.push(buf);
            layer_bind_groups.push(bg);
        }

        // 3. Final Norm
        let norm_weight = model
            .metadata
            .get_tensor_offset("output_norm.weight")
            .expect("output_norm missing");
        let norm_params = RMSNormParams {
            count: dim,
            weights_offset: norm_weight as u32,
            eps: spec.rms_eps,
            padding: 0,
        };
        let norm_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Final Norm Params"),
            contents: bytemuck::bytes_of(&norm_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Offset for the LAST token in the batch
        let last_token_offset = (batch_size as u64 - 1u64) * (dim as u64) * 4u64;
        let token_size = std::num::NonZeroU64::new((dim as u64) * 4u64).unwrap();

        let norm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Final Norm BG"),
            layout: &self.rmsnorm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &activation_buffer,
                        offset: last_token_offset,
                        size: Some(token_size),
                    }),
                },
                // Use temp_buffer for output to avoid read-write aliasing on activation_buffer
                // Output to the BEGINNING of temp_buffer (reusing space)
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &temp_buffer,
                        offset: 0,
                        size: Some(token_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: norm_param_buf.as_entire_binding(),
                },
            ],
        });

        // 4. Output Head (MatMul)
        let head_weight = model
            .metadata
            .get_tensor_offset("output.weight")
            .expect("output.weight missing");

        let head_params = MatMulParams {
            n: vocab_size,
            k: dim,
            weights_offset: head_weight as u32,
            padding: 0,
        };
        let head_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Head Params"),
            contents: bytemuck::bytes_of(&head_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let head_bg = if let Some(override_buf) = head_weights_override {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Head BG F32"),
                layout: &self.matmul_f32_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: override_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &temp_buffer,
                            offset: 0,
                            size: Some(token_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: logits_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: head_param_buf.as_entire_binding(),
                    },
                ],
            })
        } else {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Head BG"),
                layout: &self.matmul_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: model.gpu_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &temp_buffer,
                            offset: 0,
                            size: Some(token_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: logits_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: head_param_buf.as_entire_binding(),
                    },
                ],
            })
        };

        // 5. Command Encoding
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Full Model"),
        });

        let wg_dim = (dim + 255) / 256;
        let ffn_total = ffn_dim * 2; // Gate + Up need this many threads
        let wg_ffn = (ffn_total + 255) / 256; // Ceil div by workgroup size (256)
        let wg_norm = (dim + 255) / 256;
        let wg_head = (vocab_size + 255) / 256;

        // QKV Dispatch Calculation
        let q_len = params_base.head_count * params_base.head_dim;
        let kv_len = params_base.head_count_kv * params_base.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = (total_qkv + 255) / 256;

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Loop"),
                timestamp_writes: None,
            });

            // Loop Layers
            for (_i, bg) in layer_bind_groups.iter().enumerate() {
                cpass.set_bind_group(0, bg, &[]);

                // QKV (Calculates Q/K/V and applies Softmax + Attention)
                cpass.set_pipeline(&self.layer_pipeline_qkv);
                cpass.dispatch_workgroups(wg_qkv, batch_size, 1);

                // Attn Out (Project weighted V back to Residual)
                cpass.set_pipeline(&self.layer_pipeline_attn_out);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);

                // Attn Proj (Apply output projection + residual add)
                cpass.set_pipeline(&self.layer_pipeline_attn_proj);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);

                // FFN Proj
                cpass.set_pipeline(&self.layer_pipeline_ffn_proj);
                cpass.dispatch_workgroups(wg_ffn, batch_size, 1);

                // FFN Down
                cpass.set_pipeline(&self.layer_pipeline_ffn_down);
                cpass.dispatch_workgroups(wg_dim, batch_size, 1);
            }
        }

        // Snapshot h20 (post-layer-loop, pre-final-norm)
        // Copy only the LAST token's state for validation
        encoder.copy_buffer_to_buffer(
            &activation_buffer,
            last_token_offset,
            &pre_norm_buffer,
            0,
            (dim as u64) * 4,
        );

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Norm+Head"),
                timestamp_writes: None,
            });

            // Final Norm
            cpass.set_bind_group(0, &norm_bg, &[]);
            cpass.set_pipeline(&self.rmsnorm_pipeline);
            cpass.dispatch_workgroups(wg_norm, 1, 1);

            // Head
            cpass.set_bind_group(0, &head_bg, &[]);
            if head_weights_override.is_some() {
                cpass.set_pipeline(&self.matmul_f32_pipeline);
            } else {
                cpass.set_pipeline(&self.matmul_pipeline);
            }
            cpass.dispatch_workgroups(wg_head, 1, 1);
        }

        // 6. Readback
        let output_size = (vocab_size * 4) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&temp_buffer, 0, &l21_buffer, 0, (dim as u64) * 4);
        encoder.copy_buffer_to_buffer(&logits_buffer, 0, &staging_buffer, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let pre_norm_slice = pre_norm_buffer.slice(..);
        let (tx_pre, rx_pre) = std::sync::mpsc::channel();
        pre_norm_slice.map_async(wgpu::MapMode::Read, move |res| tx_pre.send(res).unwrap());

        let l21_slice = l21_buffer.slice(..);
        let (tx_l21, rx_l21) = std::sync::mpsc::channel();
        l21_slice.map_async(wgpu::MapMode::Read, move |res| tx_l21.send(res).unwrap());

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(res) = rx_pre.try_recv() {
                res.expect("Pre-norm buffer map failed");
            }
            if let Ok(res) = rx_l21.try_recv() {
                res.expect("L21 buffer map failed");
            }
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let pre_norm_data = pre_norm_slice.get_mapped_range();
        let pre_norm_result: Vec<f32> = bytemuck::cast_slice(&pre_norm_data).to_vec();
        drop(pre_norm_data);
        pre_norm_buffer.unmap();

        let l21_data = l21_slice.get_mapped_range();
        let l21_result: Vec<f32> = bytemuck::cast_slice(&l21_data).to_vec();
        drop(l21_data);
        l21_buffer.unmap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        (pre_norm_result, l21_result, result)
    }

    /// Run ONLY QKV kernel and capture Q, K, V values for debugging
    pub fn run_qkv_only_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        println!(
            "[BindlessPipeline] Running QKV ONLY Test (Dim={})",
            params.dim
        );

        // 1. Activation In (ReadWrite)
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation In"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // 2. Temp State for Q (ReadWrite)
        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp State"),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 3. Uniforms
        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // KV Cache - K and V go here
        let kv_size = 2048u64 * (params.head_count_kv as u64) * (params.head_dim as u64) * 4 * 2; // max_seq * n_head_kv * head_dim * sizeof(f32) * 2 (K+V)
        let kv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KV Cache Temp"),
            size: kv_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 4. Bind Group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("QKV Test BindGroup"),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: activation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: model
                        .preflight
                        .as_ref()
                        .unwrap()
                        .norm_bank_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: model
                        .preflight
                        .as_ref()
                        .unwrap()
                        .rope_cache_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: kv_buffer.as_entire_binding(),
                },
            ],
        });

        // 5. Run ONLY QKV kernel
        let q_len = params.head_count * params.head_dim;
        let kv_len = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = (total_qkv + 255) / 256;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("QKV Only"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qkv);
            cpass.dispatch_workgroups(wg_qkv, 1, 1);
        }

        // Stage readback buffers
        let q_size = (q_len as u64) * 4; // Q: 2048 floats
        let kv_unit = (kv_len as u64) * 4; // K or V: 256 floats

        let staging_q = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Q"),
            size: q_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging K"),
            size: kv_unit,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging V"),
            size: kv_unit,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Q from temp_state[0..q_len]
        encoder.copy_buffer_to_buffer(&temp_buffer, 0, &staging_q, 0, q_size);
        // K from kv_cache[0..kv_len]
        encoder.copy_buffer_to_buffer(&kv_buffer, 0, &staging_k, 0, kv_unit);
        // V from kv_cache[kv_len..kv_len*2]
        encoder.copy_buffer_to_buffer(&kv_buffer, kv_unit, &staging_v, 0, kv_unit);

        queue.submit(Some(encoder.finish()));

        // Readback
        let q_vec = self.readback_helper(device, &staging_q);
        let k_vec = self.readback_helper(device, &staging_k);
        let v_vec = self.readback_helper(device, &staging_v);

        (q_vec, k_vec, v_vec)
    }

    /// Run Full Layer Test
    pub fn run_layer_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> Vec<f32> {
        self.run_layer_stepwise_test(device, queue, model, input, offsets, params, false)
            .1
    }

    pub fn run_layer_stepwise_test(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
        read_halfway: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        println!(
            "[BindlessPipeline] Running Layer V1 Test Stepwise (Dim={})",
            params.dim
        );

        // 1. Activation In (ReadWrite)
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Activation In"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // 2. Temp State (ReadWrite, Scratchpad)
        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Temp State"),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 3. Uniforms
        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // KV Caches (Temp for Test - K and V separate)
        let kv_size_per_buffer = (params.dim as u64 / params.head_count as u64)
            * (params.head_count_kv as u64)
            * 2048u64
            * 4; // head_dim * n_head_kv * max_seq * sizeof(f32)
        let kv_buffer_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KV Cache K Temp"),
            size: kv_size_per_buffer,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let kv_buffer_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KV Cache V Temp"),
            size: kv_size_per_buffer,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Cache params (temporary - position 0)
        let cache_params = CacheParams {
            current_pos: 0,
            seq_len: 1,
            max_seq_len: 2048, // test default
            batch_size: 1,
        };

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params Temp"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 4. Bind Group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Layer Test BindGroup"),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: activation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: model
                        .preflight
                        .as_ref()
                        .unwrap()
                        .norm_bank_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: model
                        .preflight
                        .as_ref()
                        .unwrap()
                        .rope_cache_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: kv_buffer_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: kv_buffer_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: cache_params_buffer.as_entire_binding(),
                },
            ],
        });

        // 5. Dispatch Sequence
        // Part 1: QKV + Attn
        let dim = params.dim as u64;
        let wg_dim = (params.dim + 255) / 256;

        // QKV Needs more threads: Q + K + V
        let q_len = params.head_count * params.head_dim;
        let kv_len = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = (total_qkv + 255) / 256;

        // FFN needs threads for both Gate and Up: ffn_dim * 2
        let ffn_total = params.ffn_dim * 2; // Gate + Up
        let wg_ffn = (ffn_total + 255) / 256; // Ceil div by workgroup size

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Kernel 1: QKV
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("QKV"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qkv);
            cpass.dispatch_workgroups(wg_qkv, 1, 1);
        }

        // Kernel 2: Attention
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attn"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_out);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // Kernel 3: Attention Projection
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AttnProj"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_proj);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        let staging_mid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Mid"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if read_halfway {
            encoder.copy_buffer_to_buffer(&activation_buffer, 0, &staging_mid, 0, dim * 4);
        }

        // Kernel 4: FFN Projection
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFN"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_proj);
            cpass.dispatch_workgroups(wg_ffn, 1, 1);
        }

        // Kernel 5: FFN Down
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFNDown"),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_down);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        let staging_final = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Final"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&activation_buffer, 0, &staging_final, 0, dim * 4);
        queue.submit(Some(encoder.finish()));

        // Readbacks
        let mut mid_vec = Vec::new();
        if read_halfway {
            mid_vec = self.readback_helper(device, &staging_mid);
        }
        let final_vec = self.readback_helper(device, &staging_final);

        (mid_vec, final_vec)
    }

    /// Run a single transformer layer with stateful KV cache
    ///
    /// This method:
    /// 1. Uses the provided KVCache for attention context
    /// 2. Writes new K/V at cache.get_seq_len() position
    /// 3. Reads all cached K/V for attention (with causal masking)
    /// 4. Increments cache position after success
    ///
    /// # Arguments
    /// * `device` - WGPU device
    /// * `queue` - WGPU queue
    /// * `model` - Loaded GGUF model with preflight data
    /// * `kv_cache` - Mutable KVCache for this layer
    /// * `layer_idx` - Layer index (0..21)
    /// * `input` - Input activation vector (dim elements)
    /// * `offsets` - Weight offsets for this layer
    /// * `params` - Layer hyperparameters
    ///
    /// # Returns
    /// Output activation vector (dim elements)
    pub fn run_layer_with_cache(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        kv_cache: &mut super::kv_cache::KVCache,
        layer_idx: usize,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> Vec<f32> {
        let dim = params.dim as u64;

        // 1. Create activation buffer (input/output)
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Activation Layer {}", layer_idx)),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // 2. Temp state buffer (for Q, attention scores, FFN intermediate)
        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Temp State Layer {}", layer_idx)),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // 3. Uniform buffers
        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // DIAGNOSTIC: Verify offsets are correct for first 2 layers
        if layer_idx <= 1 {
            eprintln!("[💾 GPU UPLOAD] Layer {} offsets:", layer_idx);
            eprintln!(
                "    attn_q: {} | attn_norm: {} | layer_idx (padding[0]): {}",
                offsets.attn_q, offsets.attn_norm, offsets.padding[0]
            );
        }

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 4. Cache params uniform
        let cache_params = CacheParams {
            current_pos: kv_cache.get_seq_len(), // Position to write new K/V
            seq_len: kv_cache.get_seq_len() + 1, // After write, this many positions cached
            max_seq_len: kv_cache.max_len(),
            batch_size: 1,
        };

        // DIAGNOSTIC: Print cache params for first 2 layers
        if layer_idx <= 1 {
            eprintln!(
                "    [CACHE] Layer {}: current_pos={}, seq_len={}, max={}",
                layer_idx, cache_params.current_pos, cache_params.seq_len, cache_params.max_seq_len
            );
        }

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 5. Create bind group with all 10 bindings
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Layer {} BindGroup", layer_idx)),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: activation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: model
                        .preflight
                        .as_ref()
                        .unwrap()
                        .norm_bank_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: model
                        .preflight
                        .as_ref()
                        .unwrap()
                        .rope_cache_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: kv_cache.get_k_buffer(layer_idx).as_entire_binding(),
                }, // K cache for this layer
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: kv_cache.get_v_buffer(layer_idx).as_entire_binding(),
                }, // V cache for this layer
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: cache_params_buffer.as_entire_binding(),
                },
            ],
        });

        // 6. Calculate workgroup counts
        let wg_dim = (params.dim + 255) / 256;
        let q_len = params.head_count * params.head_dim;
        let kv_len = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = (total_qkv + 255) / 256;
        let ffn_total = params.ffn_dim * 2;
        let wg_ffn = (ffn_total + 255) / 256;

        // 7. Dispatch compute passes(5 kernels, each in separate pass for synchronization)
        // CRITICAL: Each compute pass ensures GPU work completes before next starts
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Layer {} Encoder", layer_idx)),
        });

        // Kernel 1: QKV generation + RoPE + cache write
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - QKV", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qkv);
            cpass.dispatch_workgroups(wg_qkv, 1, 1);
        } // ← GPU waits for QKV to finish before proceeding

        // Kernel 2: Attention (Q @ cached_K, softmax, weighted sum of cached_V)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - Attn", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_out);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // ← GPU waits for Attn to finish

        // Kernel 3: Attention output projection
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - AttnProj", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_proj);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // ← GPU waits for AttnProj to finish

        // Kernel 4: FFN gate/up + SiLU
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFN", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_proj);
            cpass.dispatch_workgroups(wg_ffn, 1, 1);
        } // ← GPU waits for FFN to finish

        // Kernel 5: FFN down + residual
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFNDown", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_down);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        } // ← GPU waits for FFNDown to finish

        // 8. Readback result
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&activation_buffer, 0, &staging_buffer, 0, dim * 4);
        queue.submit(Some(encoder.finish()));

        let output = self.readback_helper(device, &staging_buffer);

        // NOTE: Cache increment happens ONCE per token (after all layers), not per layer
        // Caller must call kv_cache.increment() after processing all 22 layers

        output
    }

    /// Debug version that extracts Q/K/V tensors for verification
    pub fn run_layer_with_cache_debug(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        kv_cache: &mut super::kv_cache::KVCache,
        layer_idx: usize,
        input: &[f32],
        offsets: LayerOffsets,
        params: LayerParams,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let dim = params.dim as u64;

        // 1. Create buffers (same as run_layer_with_cache)
        let activation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Activation Layer {}", layer_idx)),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let temp_size = (params.temp_stride as u64) * 4;
        let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Temp State Layer {}", layer_idx)),
            size: temp_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, // Added COPY_SRC for debug
            mapped_at_creation: false,
        });

        let offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Offsets"),
            contents: bytemuck::bytes_of(&offsets),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let cache_params = CacheParams {
            current_pos: kv_cache.get_seq_len(),
            seq_len: kv_cache.get_seq_len() + 1,
            max_seq_len: kv_cache.max_len(),
            batch_size: 1,
        };

        let cache_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cache Params"),
            contents: bytemuck::bytes_of(&cache_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Layer {} BindGroup", layer_idx)),
            layout: &self.layer_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: activation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: temp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offsets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: model
                        .preflight
                        .as_ref()
                        .unwrap()
                        .norm_bank_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: model
                        .preflight
                        .as_ref()
                        .unwrap()
                        .rope_cache_buffer
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: kv_cache.get_k_buffer(layer_idx).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: kv_cache.get_v_buffer(layer_idx).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: cache_params_buffer.as_entire_binding(),
                },
            ],
        });

        let wg_dim = (params.dim + 255) / 256;
        let q_len = params.head_count * params.head_dim;
        let kv_len = params.head_count_kv * params.head_dim;
        let total_qkv = q_len + kv_len * 2;
        let wg_qkv = (total_qkv + 255) / 256;
        let ffn_total = params.ffn_dim * 2;
        let wg_ffn = (ffn_total + 255) / 256;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Layer {} Encoder (Debug)", layer_idx)),
        });

        // Kernel 1: QKV generation + RoPE + cache write
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - QKV", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_qkv);
            cpass.dispatch_workgroups(wg_qkv, 1, 1);
        }

        // CAPTURE Q from temp_state (first dim elements = head_count * head_dim)
        let q_size = (params.head_count as u64) * (params.head_dim as u64) * 4;
        let q_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Q Staging"),
            size: q_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&temp_buffer, 0, &q_staging, 0, q_size);

        // CAPTURE K from kv_cache_k for CURRENT position only
        let kv_slice_size = (params.head_count_kv as u64) * (params.head_dim as u64) * 4;
        let k_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("K Staging"),
            size: kv_slice_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k_offset = (cache_params.current_pos as u64)
            * (params.head_count_kv as u64)
            * (params.head_dim as u64)
            * 4;
        encoder.copy_buffer_to_buffer(
            kv_cache.get_k_buffer(layer_idx),
            k_offset,
            &k_staging,
            0,
            kv_slice_size,
        );

        // CAPTURE V from kv_cache_v for CURRENT position only
        let v_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("V Staging"),
            size: kv_slice_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v_offset = (cache_params.current_pos as u64)
            * (params.head_count_kv as u64)
            * (params.head_dim as u64)
            * 4;
        encoder.copy_buffer_to_buffer(
            kv_cache.get_v_buffer(layer_idx),
            v_offset,
            &v_staging,
            0,
            kv_slice_size,
        );

        // Continue with remaining kernels (2-5) - same as run_layer_with_cache
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - Attn", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_out);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - AttnProj", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_attn_proj);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // CAPTURE post-attention state after AttnProj (before FFN)
        let post_attn_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Post-Attn Staging"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&activation_buffer, 0, &post_attn_staging, 0, dim * 4);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFN", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_proj);
            cpass.dispatch_workgroups(wg_ffn, 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Layer {} - FFNDown", layer_idx)),
                timestamp_writes: None,
            });
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&self.layer_pipeline_ffn_down);
            cpass.dispatch_workgroups(wg_dim, 1, 1);
        }

        // CAPTURE FFN down output (stored at temp_state[ffn_dim*2..ffn_dim*2+dim])
        let ffn_debug_offset = (params.ffn_dim as u64) * 2 * 4;
        let ffn_down_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFN Down Staging"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &temp_buffer,
            ffn_debug_offset,
            &ffn_down_staging,
            0,
            dim * 4,
        );

        // Readback final output
        let output_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging"),
            size: dim * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&activation_buffer, 0, &output_staging, 0, dim * 4);

        queue.submit(Some(encoder.finish()));

        // Read all buffers
        let q_vals = self.readback_helper(device, &q_staging);
        let k_vals = self.readback_helper(device, &k_staging);
        let v_vals = self.readback_helper(device, &v_staging);
        let post_attn_vals = self.readback_helper(device, &post_attn_staging);
        let ffn_down_vals = self.readback_helper(device, &ffn_down_staging);
        let output = self.readback_helper(device, &output_staging);

        (
            output,
            post_attn_vals,
            ffn_down_vals,
            q_vals,
            k_vals,
            v_vals,
        )
    }

    fn readback_helper(&self, device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<f32> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
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
