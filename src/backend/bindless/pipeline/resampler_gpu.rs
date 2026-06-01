//! GPU dispatch for the Perceiver Resampler cross-attention module.
//! Shader: `sh_resampler_gpu.wgsl`  (13 entry points)
//!
//! The `ResamplerPipeline` struct holds all compiled pipelines and the shared
//! `wgpu::BindGroupLayout`.  It is constructed once at model-load time.
//!
//! kv_state buffer layout (3 slots, each `n_vit × d_model` f32):
//! - slot 0 `[0      .. n_vit×D)`:   kv_lin / kv_ln intermediate
//! - slot 1 `[n_vit×D.. 2×n_vit×D)`: K projected
//! - slot 2 `[2n×D   .. 3×n_vit×D)`: V projected
//! - slot 0 is also reused as Q projection temp and final_proj output.
//!   After `run_resampler` returns, `kv_state[0 .. n_queries * d_model]`
//!   contains the 64 visual tokens `[64 × 3584]`.

use wgpu::util::DeviceExt;

// ─── Uniform structs (must match WGSL byte-for-byte) ─────────────────────────

/// Byte offsets into the mmproj GGUF blob for the Resampler module.
/// 20 × u32 = 80 bytes.  Passed to the shader via uniform binding(4).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResamplerOffsets {
    pub query_embeds: u32,   // [n_queries × d_model] F32
    pub kv_weight:    u32,   // [kv_dim × d_model]    F16
    pub ln_q_w:       u32,   // [d_model] F32
    pub ln_q_b:       u32,   // [d_model] F32
    pub ln_kv_w:      u32,   // [d_model] F32
    pub ln_kv_b:      u32,   // [d_model] F32
    pub attn_q_w:     u32,   // [d_model × d_model] F16
    pub attn_q_b:     u32,   // [d_model] F32
    pub attn_k_w:     u32,   // [d_model × d_model] F16
    pub attn_k_b:     u32,   // [d_model] F32
    pub attn_v_w:     u32,   // [d_model × d_model] F16
    pub attn_v_b:     u32,   // [d_model] F32
    pub attn_out_w:   u32,   // [d_model × d_model] F16
    pub attn_out_b:   u32,   // [d_model] F32
    pub pos_embed_k:  u32,   // [4900 × d_model] F32
    pub ln_post_w:    u32,   // [d_model] F32
    pub ln_post_b:    u32,   // [d_model] F32
    pub proj_w:       u32,   // [d_model × d_model] F16
    pub pad0:         u32,
    pub pad1:         u32,
}

/// Resampler dimensions.
/// 8 × 4 = 32 bytes.  Passed to the shader via uniform binding(5).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResamplerParams {
    pub n_queries:  u32,   // 64
    pub n_vit:      u32,   // 1024
    pub d_model:    u32,   // 3584
    pub kv_dim:     u32,   // 1152
    pub n_heads:    u32,   // 16
    pub head_dim:   u32,   // 224
    pub ln_eps:     f32,   // 1e-6
    pub pad0:       u32,
}

// ─── ResamplerPipeline ────────────────────────────────────────────────────────

/// Compiled GPU pipelines for the Perceiver Resampler.
pub struct ResamplerPipeline {
    pub layout:      wgpu::BindGroupLayout,
    pub init_q:      wgpu::ComputePipeline,
    pub kv_lin:      wgpu::ComputePipeline,
    pub ln_kv:       wgpu::ComputePipeline,
    pub proj_k:      wgpu::ComputePipeline,
    pub proj_v:      wgpu::ComputePipeline,
    pub ln_q:        wgpu::ComputePipeline,
    pub proj_q:      wgpu::ComputePipeline,
    pub copy_q:      wgpu::ComputePipeline,
    pub attn:        wgpu::ComputePipeline,
    pub out_proj:    wgpu::ComputePipeline,
    pub post_ln:     wgpu::ComputePipeline,
    pub final_proj:  wgpu::ComputePipeline,
}

impl ResamplerPipeline {
    /// Compile all 13 Resampler kernels from `sh_resampler_gpu.wgsl`.
    /// Call once at model-load time.
    pub fn new(device: &wgpu::Device) -> Self {
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
            label: Some("Resampler Layout"),
            entries: &[
                make_storage_ro(0),  // vit_blob
                make_storage_ro(1),  // vit_features  (ViT encoder output, read-only)
                make_storage_rw(2),  // query_state
                make_storage_rw(3),  // kv_state (3-slot scratch)
                make_uniform(4),     // offsets
                make_uniform(5),     // params
            ],
        });

        let src = include_str!("../sh_resampler_gpu.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Resampler Shader"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Resampler Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Resampler::{}", entry)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            layout,
            init_q:     mk("main_rsp_init_q"),
            kv_lin:     mk("main_rsp_kv_lin"),
            ln_kv:      mk("main_rsp_ln_kv"),
            proj_k:     mk("main_rsp_proj_k"),
            proj_v:     mk("main_rsp_proj_v"),
            ln_q:       mk("main_rsp_ln_q"),
            proj_q:     mk("main_rsp_proj_q"),
            copy_q:     mk("main_rsp_copy_q"),
            attn:       mk("main_rsp_attn"),
            out_proj:   mk("main_rsp_out_proj"),
            post_ln:    mk("main_rsp_post_ln"),
            final_proj: mk("main_rsp_final_proj"),
        }
    }

    /// Run the full Resampler forward pass.
    ///
    /// Submits 13 compute passes in sequence (each is a separate `CommandEncoder`
    /// submit to guarantee ordering — no barrier primitives needed).
    ///
    /// On return, `kv_state[0 .. n_queries * d_model]` (i.e. the first
    /// `64 × 3584 × 4` bytes of the `kv_state` buffer) contains the 64 visual
    /// token embeddings ready to be read back to the CPU.
    ///
    /// # Arguments
    /// * `vit_blob`     – entire mmproj GGUF file on GPU
    /// * `vit_features` – `[n_vit × kv_dim]` f32, ViT encoder output (read-only)
    /// * `query_state`  – `[n_queries × d_model]` f32 working buffer
    /// * `kv_state`     – `[n_vit × d_model × 3]` f32 scratch (3 slots)
    pub fn run_resampler(
        &self,
        device:       &wgpu::Device,
        queue:        &wgpu::Queue,
        vit_blob:     &wgpu::Buffer,
        vit_features: &wgpu::Buffer,
        query_state:  &wgpu::Buffer,
        kv_state:     &wgpu::Buffer,
        offsets:      ResamplerOffsets,
        params:       ResamplerParams,
    ) {
        let (off_buf, par_buf) = self.upload_uniforms(device, &offsets, &params);

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("Resampler BG"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: vit_blob.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vit_features.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: query_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: kv_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: off_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: par_buf.as_entire_binding() },
            ],
        });

        let nq  = params.n_queries;
        let nv  = params.n_vit;
        let d   = params.d_model;
        let nh  = params.n_heads;

        let wg = |x: u32| (x + 255) / 256;

        // Each kernel is a separate encoder submit so WebGPU ordering guarantees apply.
        // (Single-encoder multi-pass would also work but this is clearest.)

        submit_one(device, queue, &bg, &self.init_q,     wg(nq * d),  "rsp_init_q");
        submit_one(device, queue, &bg, &self.kv_lin,     wg(nv * d),  "rsp_kv_lin");
        submit_one(device, queue, &bg, &self.ln_kv,      wg(nv),      "rsp_ln_kv");
        submit_one(device, queue, &bg, &self.proj_k,     wg(nv * d),  "rsp_proj_k");
        submit_one(device, queue, &bg, &self.proj_v,     wg(nv * d),  "rsp_proj_v");
        submit_one(device, queue, &bg, &self.ln_q,       wg(nq),      "rsp_ln_q");
        submit_one(device, queue, &bg, &self.proj_q,     wg(nq * d),  "rsp_proj_q");
        submit_one(device, queue, &bg, &self.copy_q,     wg(nq * d),  "rsp_copy_q");
        submit_one(device, queue, &bg, &self.attn,       wg(nq * nh), "rsp_attn");
        submit_one(device, queue, &bg, &self.out_proj,   wg(nq * d),  "rsp_out_proj");
        submit_one(device, queue, &bg, &self.post_ln,    wg(nq),      "rsp_post_ln");
        submit_one(device, queue, &bg, &self.final_proj, wg(nq * d),  "rsp_final_proj");
    }

    fn upload_uniforms(
        &self,
        device:  &wgpu::Device,
        offsets: &ResamplerOffsets,
        params:  &ResamplerParams,
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let off_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("ResamplerOffsets"),
            contents: bytemuck::bytes_of(offsets),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let par_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("ResamplerParams"),
            contents: bytemuck::bytes_of(params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        (off_buf, par_buf)
    }
}

/// Submit a single compute pass (one encoder, one pass, immediate submit).
fn submit_one(
    device:   &wgpu::Device,
    queue:    &wgpu::Queue,
    bg:       &wgpu::BindGroup,
    pipeline: &wgpu::ComputePipeline,
    wg_x:     u32,
    label:    &str,
) {
    let mut enc = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, bg, &[]);
        cpass.dispatch_workgroups(wg_x, 1, 1);
    }
    queue.submit(Some(enc.finish()));
}
