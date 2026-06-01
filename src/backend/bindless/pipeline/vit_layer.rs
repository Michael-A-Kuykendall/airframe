//! GPU dispatch for SigLIP ViT transformer blocks.
//! Shader: `sh_vit_layer.wgsl`  (8 entry points)
//!
//! The `VitPipeline` struct holds all 8 compiled `wgpu::ComputePipeline`s and
//! the shared `wgpu::BindGroupLayout`.  It is constructed once at model-load
//! time and reused for every ViT block.
//!
//! Usage pattern (called by `GpuVisionModel::encode_image`):
//! ```
//! for layer in 0..N_VIT_LAYERS {
//!     vit_pipeline.run_vit_block(device, queue,
//!         vit_blob, activations, temp_kqv, temp_ffn,
//!         layer_offsets[layer], vit_params);
//! }
//! vit_pipeline.run_post_ln(device, queue,
//!     vit_blob, activations, temp_kqv, temp_ffn,
//!     post_ln_offsets, vit_params);
//! ```

use wgpu::util::DeviceExt;

// ─── Uniform structs (must match WGSL byte-for-byte) ─────────────────────────

/// Byte offsets into the mmproj GGUF blob for one ViT transformer block.
/// 16 × u32 = 64 bytes.  Passed to the shader via uniform binding(4).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VitBlockOffsets {
    pub ln1_w:    u32,    // LayerNorm 1 weight  [hidden_dim] F32
    pub ln1_b:    u32,    // LayerNorm 1 bias    [hidden_dim] F32
    pub attn_q_w: u32,    // Attn Q weight       [hidden_dim × hidden_dim] F16
    pub attn_q_b: u32,    // Attn Q bias         [hidden_dim] F32
    pub attn_k_w: u32,    // Attn K weight       [hidden_dim × hidden_dim] F16
    pub attn_k_b: u32,    // Attn K bias         [hidden_dim] F32
    pub attn_v_w: u32,    // Attn V weight       [hidden_dim × hidden_dim] F16
    pub attn_v_b: u32,    // Attn V bias         [hidden_dim] F32
    pub attn_o_w: u32,    // Attn out weight     [hidden_dim × hidden_dim] F16
    pub attn_o_b: u32,    // Attn out bias       [hidden_dim] F32
    pub ln2_w:    u32,    // LayerNorm 2 weight  [hidden_dim] F32
    pub ln2_b:    u32,    // LayerNorm 2 bias    [hidden_dim] F32
    pub ffn_up_w: u32,    // FFN up weight       [mlp_dim × hidden_dim] F16
    pub ffn_up_b: u32,    // FFN up bias         [mlp_dim] F32
    pub ffn_dn_w: u32,    // FFN down weight     [hidden_dim × mlp_dim] F16
    pub ffn_dn_b: u32,    // FFN down bias       [hidden_dim] F32
}

/// ViT model dimensions and runtime parameters.
/// 8 × 4 = 32 bytes.  Passed to the shader via uniform binding(5).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VitParams {
    pub hidden_dim: u32,   // 1152
    pub n_heads:    u32,   // 16
    pub head_dim:   u32,   // 72
    pub mlp_dim:    u32,   // 4304
    pub n_tokens:   u32,   // 1024 (patches per tile, no CLS token)
    pub ln_eps:     f32,   // 1e-6
    pub pad0:       u32,
    pub pad1:       u32,
}

// ─── VitPipeline ─────────────────────────────────────────────────────────────

/// Compiled GPU pipelines for one SigLIP ViT transformer block.
/// One instance covers ALL 27 layers (offsets vary per call, not per pipeline).
pub struct VitPipeline {
    pub layout:     wgpu::BindGroupLayout,
    pub ln1:        wgpu::ComputePipeline,
    pub qkv:        wgpu::ComputePipeline,
    pub attn:       wgpu::ComputePipeline,
    pub attn_proj:  wgpu::ComputePipeline,
    pub ln2:        wgpu::ComputePipeline,
    pub ffn_up:     wgpu::ComputePipeline,
    pub ffn_down:   wgpu::ComputePipeline,
    pub post_ln:    wgpu::ComputePipeline,
}

impl VitPipeline {
    /// Compile all 8 ViT kernels from `sh_vit_layer.wgsl`.
    /// Call once at model-load time.
    pub fn new(device: &wgpu::Device) -> Self {
        // ── Bind Group Layout (6 bindings matching WGSL) ──────────────────────
        let make_storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let make_storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let make_uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VitLayer Layout"),
            entries: &[
                make_storage_ro(0),  // vit_blob
                make_storage_rw(1),  // activations
                make_storage_rw(2),  // temp_kqv
                make_storage_rw(3),  // temp_ffn
                make_uniform(4),     // offsets
                make_uniform(5),     // params
            ],
        });

        // ── Shader module ─────────────────────────────────────────────────────
        let src = include_str!("../sh_vit_layer.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VitLayer Shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VitLayer Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("VitLayer::{}", entry)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            layout,
            ln1:       mk("main_vit_ln1"),
            qkv:       mk("main_vit_qkv"),
            attn:      mk("main_vit_attn"),
            attn_proj: mk("main_vit_attn_proj"),
            ln2:       mk("main_vit_ln2"),
            ffn_up:    mk("main_vit_ffn_up"),
            ffn_down:  mk("main_vit_ffn_down"),
            post_ln:   mk("main_vit_post_ln"),
        }
    }

    /// Run one ViT transformer block (kernels 1–7).
    /// Submits a single `CommandEncoder` with 7 sequential compute passes.
    ///
    /// # Arguments
    /// * `vit_blob`    – entire mmproj GGUF file on GPU (read-only)
    /// * `activations` – `[n_tokens × hidden_dim]` f32, read/write residual stream
    /// * `temp_kqv`    – `[n_tokens × hidden_dim × 4]` f32 scratch
    /// * `temp_ffn`    – `[n_tokens × mlp_dim]` f32 scratch
    pub fn run_vit_block(
        &self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        vit_blob:    &wgpu::Buffer,
        activations: &wgpu::Buffer,
        temp_kqv:    &wgpu::Buffer,
        temp_ffn:    &wgpu::Buffer,
        offsets: VitBlockOffsets,
        params:  VitParams,
    ) {
        let (offsets_buf, params_buf) = self.upload_uniforms(device, &offsets, &params);
        let bg = self.make_bind_group(device, vit_blob, activations, temp_kqv, temp_ffn,
                                      &offsets_buf, &params_buf);

        let n  = params.n_tokens;
        let h  = params.hidden_dim;
        let nh = params.n_heads;
        let m  = params.mlp_dim;

        let wg = |x: u32| (x + 255) / 256;

        let mut enc = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("VitBlock") });

        dispatch(&mut enc, &bg, &self.ln1,       wg(n),       "vit_ln1");
        dispatch(&mut enc, &bg, &self.qkv,        wg(n*h*3),   "vit_qkv");
        dispatch(&mut enc, &bg, &self.attn,        wg(n*nh),    "vit_attn");
        dispatch(&mut enc, &bg, &self.attn_proj,   wg(n*h),     "vit_attn_proj");
        dispatch(&mut enc, &bg, &self.ln2,         wg(n),       "vit_ln2");
        dispatch(&mut enc, &bg, &self.ffn_up,      wg(n*m),     "vit_ffn_up");
        dispatch(&mut enc, &bg, &self.ffn_down,    wg(n*h),     "vit_ffn_down");

        queue.submit(Some(enc.finish()));
    }

    /// Run the post-block LayerNorm (after all 27 ViT blocks).
    /// Caller sets `offsets.ln1_w` / `offsets.ln1_b` to the post-LN tensor offsets.
    pub fn run_post_ln(
        &self,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        vit_blob:    &wgpu::Buffer,
        activations: &wgpu::Buffer,
        temp_kqv:    &wgpu::Buffer,
        temp_ffn:    &wgpu::Buffer,
        offsets: VitBlockOffsets,
        params:  VitParams,
    ) {
        let (offsets_buf, params_buf) = self.upload_uniforms(device, &offsets, &params);
        let bg = self.make_bind_group(device, vit_blob, activations, temp_kqv, temp_ffn,
                                      &offsets_buf, &params_buf);

        let wg = (params.n_tokens + 255) / 256;
        let mut enc = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("VitPostLN") });
        dispatch(&mut enc, &bg, &self.post_ln, wg, "vit_post_ln");
        queue.submit(Some(enc.finish()));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn upload_uniforms(
        &self,
        device: &wgpu::Device,
        offsets: &VitBlockOffsets,
        params:  &VitParams,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("VitBlockOffsets"),
            contents: bytemuck::bytes_of(offsets),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("VitParams"),
            contents: bytemuck::bytes_of(params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        (offsets_buf, params_buf)
    }

    fn make_bind_group(
        &self,
        device:      &wgpu::Device,
        vit_blob:    &wgpu::Buffer,
        activations: &wgpu::Buffer,
        temp_kqv:    &wgpu::Buffer,
        temp_ffn:    &wgpu::Buffer,
        offsets_buf: &wgpu::Buffer,
        params_buf:  &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("VitLayer BG"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: vit_blob.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: activations.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: temp_kqv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: temp_ffn.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: offsets_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
            ],
        })
    }
}

/// Helper: begin a compute pass, set pipeline + bind group, dispatch, end pass.
fn dispatch(
    enc:      &mut wgpu::CommandEncoder,
    bg:       &wgpu::BindGroup,
    pipeline: &wgpu::ComputePipeline,
    wg_x:     u32,
    label:    &str,
) {
    let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, bg, &[]);
    cpass.dispatch_workgroups(wg_x, 1, 1);
}
