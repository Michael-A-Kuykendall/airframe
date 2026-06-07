use wgpu::util::DeviceExt;

pub struct RopeShiftPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    // INT4 packed+scale shift extension (TurboQuant — feat/turboquant-wgsl)
    int4_pipeline: wgpu::ComputePipeline,
    int4_bind_group_layout: wgpu::BindGroupLayout,
}

#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct CompactionParams {
    keep_sink: u32,
    shift_amt: u32,
    old_seq_len: u32,
    n_head_kv: u32,
    head_dim: u32,
    rope_dim: u32,
    rope_base: f32,
    max_seq_len: u32,
}

impl RopeShiftPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader_src = include_str!("sh_rope_shift.wgsl");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Helical Context Shift"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RopeShift Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // K source (read-only snapshot)
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
                    // V source (read-only snapshot)
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // K destination (live cache)
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // V destination (live cache)
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // Params
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RopeShift Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Helical Context Shift Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        // --- INT4 packed+scale shift pipeline ---
        let int4_shader_src = include_str!("sh_rope_shift_int4.wgsl");
        let int4_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Helical Context Shift INT4"),
            source: wgpu::ShaderSource::Wgsl(int4_shader_src.into()),
        });

        let int4_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RopeShift INT4 Bind Group Layout"),
            entries: &[
                // 0: params
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 1: packed_k_src (read)
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 2: scale_k_src (read)
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 3: packed_v_src (read)
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 4: scale_v_src (read)
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 5: packed_k_dst (read_write)
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 6: scale_k_dst (read_write)
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 7: packed_v_dst (read_write)
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                // 8: scale_v_dst (read_write)
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let int4_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RopeShift INT4 Pipeline Layout"),
            bind_group_layouts: &[&int4_bind_group_layout],
            push_constant_ranges: &[],
        });

        let int4_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Helical Context Shift INT4 Pipeline"),
            layout: Some(&int4_pipeline_layout),
            module: &int4_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        Self {
            pipeline,
            bind_group_layout,
            int4_pipeline,
            int4_bind_group_layout,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        k_buffer: &wgpu::Buffer,
        v_buffer: &wgpu::Buffer,
        keep_sink: u32,
        shift_amt: u32,
        old_seq_len: u32,
        n_head_kv: u32,
        head_dim: u32,
        rope_dim: u32,
        rope_base: f32,
        max_seq_len: u32,
    ) {
        let params = CompactionParams {
            keep_sink,
            shift_amt,
            old_seq_len,
            n_head_kv,
            head_dim,
            rope_dim,
            rope_base,
            max_seq_len,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RopeShift Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Allocate scratch buffers and snapshot the live cache before shift.
        // This eliminates the overlap hazard: shader reads scratch (frozen), writes live cache.
        let buf_size = k_buffer.size();
        let scratch_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RopeShift Scratch K"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RopeShift Scratch V"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RopeShift Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch_k.as_entire_binding(), // read-only snapshot
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch_v.as_entire_binding(), // read-only snapshot
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: k_buffer.as_entire_binding(), // write destination
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_buffer.as_entire_binding(), // write destination
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RopeShift Encoder"),
        });

        // Step 1: Snapshot live cache → scratch (frozen read-only source)
        encoder.copy_buffer_to_buffer(k_buffer, 0, &scratch_k, 0, buf_size);
        encoder.copy_buffer_to_buffer(v_buffer, 0, &scratch_v, 0, buf_size);

        // Step 2: Dispatch shift kernel (reads scratch, writes live cache)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RopeShift Compute Pass"),
                timestamp_writes: None,
            });

            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            // Pure copy kernel — one thread per element, indexed by dim x kv_head x seq offset.
            // Shader workgroup is 64x4x1; dispatch covers head_dim x n_head_kv x elements_to_shift.
            let elements_to_shift = old_seq_len.saturating_sub(keep_sink + shift_amt);

            if elements_to_shift > 0 {
                let dim_x = head_dim.div_ceil(64); // head_dim threads, workgroup 64
                let dim_y = n_head_kv.div_ceil(4); // n_head_kv threads, workgroup 4
                let dim_z = elements_to_shift; // one workgroup per seq position

                cpass.dispatch_workgroups(dim_x, dim_y, dim_z);
            }
        }

        queue.submit(Some(encoder.finish()));
    }

    /// INT4 variant — shifts F32 staging buffers AND packed+scale buffers in one submission.
    ///
    /// Call this instead of `execute()` when the KV cache was created with `KVCache::new_int4()`.
    /// Internally freezes all 6 buffers into scratch copies before dispatching both kernels so
    /// there is no overlap hazard between source and destination ranges.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_int4(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        // F32 staging buffers
        k_buffer: &wgpu::Buffer,
        v_buffer: &wgpu::Buffer,
        // INT4 packed nibble buffers
        packed_k_buffer: &wgpu::Buffer,
        packed_v_buffer: &wgpu::Buffer,
        // INT4 scale buffers
        scale_k_buffer: &wgpu::Buffer,
        scale_v_buffer: &wgpu::Buffer,
        // Compaction parameters
        keep_sink: u32,
        shift_amt: u32,
        old_seq_len: u32,
        n_head_kv: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) {
        let elements_to_shift = old_seq_len.saturating_sub(keep_sink + shift_amt);
        if elements_to_shift == 0 { return; }

        let params = CompactionParams {
            keep_sink,
            shift_amt,
            old_seq_len,
            n_head_kv,
            head_dim,
            rope_dim: head_dim,
            rope_base: 10000.0,
            max_seq_len,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RopeShiftINT4 Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Scratch buffers for F32 K/V (same as execute())
        let f32_buf_size = k_buffer.size();
        let scratch_k = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RopeShiftINT4 Scratch K F32"),
            size: f32_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RopeShiftINT4 Scratch V F32"),
            size: f32_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Scratch buffers for INT4 packed K/V
        let packed_size = packed_k_buffer.size();
        let scratch_pk = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RopeShiftINT4 Scratch PK"),
            size: packed_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_pv = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RopeShiftINT4 Scratch PV"),
            size: packed_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Scratch buffers for INT4 scale K/V
        let scale_size = scale_k_buffer.size();
        let scratch_sk = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RopeShiftINT4 Scratch SK"),
            size: scale_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scratch_sv = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RopeShiftINT4 Scratch SV"),
            size: scale_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let f32_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RopeShiftINT4 F32 BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scratch_k.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch_v.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: k_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buffer.as_entire_binding() },
            ],
        });

        let int4_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RopeShiftINT4 Packed BindGroup"),
            layout: &self.int4_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scratch_pk.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scratch_sk.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scratch_pv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scratch_sv.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: packed_k_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: scale_k_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: packed_v_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: scale_v_buffer.as_entire_binding() },
            ],
        });

        let dim_x_f32   = head_dim.div_ceil(64);
        let dim_x_int4  = (head_dim / 8).div_ceil(64);
        let dim_y       = n_head_kv.div_ceil(4);
        let dim_z       = elements_to_shift;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RopeShiftINT4 Encoder"),
        });

        // Freeze all live buffers into scratch (eliminates overlap hazard)
        encoder.copy_buffer_to_buffer(k_buffer,        0, &scratch_k,  0, f32_buf_size);
        encoder.copy_buffer_to_buffer(v_buffer,        0, &scratch_v,  0, f32_buf_size);
        encoder.copy_buffer_to_buffer(packed_k_buffer, 0, &scratch_pk, 0, packed_size);
        encoder.copy_buffer_to_buffer(packed_v_buffer, 0, &scratch_pv, 0, packed_size);
        encoder.copy_buffer_to_buffer(scale_k_buffer,  0, &scratch_sk, 0, scale_size);
        encoder.copy_buffer_to_buffer(scale_v_buffer,  0, &scratch_sv, 0, scale_size);

        // Dispatch F32 shift
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RopeShiftINT4 F32 Pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.pipeline);
            cp.set_bind_group(0, &f32_bg, &[]);
            cp.dispatch_workgroups(dim_x_f32, dim_y, dim_z);
        }

        // Dispatch INT4 packed+scale shift
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RopeShiftINT4 Packed Pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.int4_pipeline);
            cp.set_bind_group(0, &int4_bg, &[]);
            cp.dispatch_workgroups(dim_x_int4, dim_y, dim_z);
        }

        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;

    #[test]
    fn test_helical_gpu_shift_pure_copy() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("No GPU adapter");
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("Failed to create device");

        let pipeline = RopeShiftPipeline::new(&device);

        let max_seq_len: u32 = 16;
        let n_head_kv: u32 = 2;
        let head_dim: u32 = 64;
        let keep_sink: u32 = 4;
        let shift_amt: u32 = 5;
        let old_seq_len: u32 = 12;
        // Elements at positions [keep_sink+shift_amt .. old_seq_len) = [9,10,11]
        // shift to [keep_sink .. keep_sink+3) = [4,5,6]

        let total_floats = (max_seq_len * n_head_kv * head_dim) as usize;
        let buffer_size = (total_floats * 4) as u64;

        // Fill K and V with recognizable per-position patterns
        let mut k_data = vec![0.0f32; total_floats];
        let mut v_data = vec![0.0f32; total_floats];
        for pos in 0..max_seq_len {
            for h in 0..n_head_kv {
                for d in 0..head_dim {
                    let idx = (pos * n_head_kv * head_dim + h * head_dim + d) as usize;
                    k_data[idx] = (pos as f32) * 0.1 + (h as f32) * 0.01 + (d as f32) * 0.001;
                    v_data[idx] = (pos as f32) + (h as f32) * 0.1 + (d as f32) * 0.01;
                }
            }
        }

        let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test K"),
            contents: bytemuck::cast_slice(&k_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test V"),
            contents: bytemuck::cast_slice(&v_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        pipeline.execute(
            &device,
            &queue,
            &k_buf,
            &v_buf,
            keep_sink,
            shift_amt,
            old_seq_len,
            n_head_kv,
            head_dim,
            head_dim,
            10000.0,
            max_seq_len,
        );

        // Readback K
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&k_buf, 0, &staging, 0, buffer_size);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(r) = rx.try_recv() {
                r.unwrap();
                break;
            }
        }
        let data = slice.get_mapped_range();
        let gpu_k: &[f32] = bytemuck::cast_slice(&data);

        // Verify: data from old positions [9,10,11] must now be at [4,5,6]
        // Pure copy means K values are BIT-IDENTICAL to the original source positions.
        let elements_to_shift = old_seq_len - (keep_sink + shift_amt); // 3
        let mut max_err: f32 = 0.0;
        for offset in 0..elements_to_shift {
            let old_pos = keep_sink + shift_amt + offset;
            let new_pos = keep_sink + offset;
            for h in 0..n_head_kv {
                for d in 0..head_dim {
                    let old_idx = (old_pos * n_head_kv * head_dim + h * head_dim + d) as usize;
                    let new_idx = (new_pos * n_head_kv * head_dim + h * head_dim + d) as usize;
                    let err = (gpu_k[new_idx] - k_data[old_idx]).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
        }
        drop(data);

        println!("PURE COPY TEST: max K error = {:.2e}", max_err);
        assert!(
            max_err == 0.0,
            "K shift must be bit-exact copy, got max_err={:.2e}",
            max_err
        );
        println!("PASS: K values are bit-identical after shift (pure copy, no RoPE)");
    }

    /// Production-like test where source/destination ranges OVERLAP.
    /// keep=2, shift=3, old_len=16 → elements_to_shift=11 (pos 5..16 → 2..13)
    /// Source [5,15] and destination [2,12] overlap at [5,12] — 8 collisions.
    /// Without scratch buffers this would corrupt data.
    /// Pure copy: both K and V must be bit-identical to the original source data.
    #[test]
    fn test_helical_gpu_shift_overlapping() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("No GPU adapter");
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("Failed to create device");

        let pipeline = RopeShiftPipeline::new(&device);

        let max_seq_len: u32 = 16;
        let n_head_kv: u32 = 2;
        let head_dim: u32 = 64;
        let keep_sink: u32 = 2;
        let shift_amt: u32 = 3;
        let old_seq_len: u32 = 16;

        let total_floats = (max_seq_len * n_head_kv * head_dim) as usize;
        let buffer_size = (total_floats * 4) as u64;

        // Fill every position with a unique, recognizable pattern.
        let mut k_data = vec![0.0f32; total_floats];
        let mut v_data = vec![0.0f32; total_floats];
        for pos in 0..max_seq_len {
            for h in 0..n_head_kv {
                for d in 0..head_dim {
                    let idx = (pos * n_head_kv * head_dim + h * head_dim + d) as usize;
                    k_data[idx] = (pos as f32) * 0.1 + (h as f32) * 0.01 + (d as f32) * 0.001;
                    v_data[idx] = (pos as f32) + (h as f32) * 0.1 + (d as f32) * 0.01;
                }
            }
        }

        let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test K Overlap"),
            contents: bytemuck::cast_slice(&k_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test V Overlap"),
            contents: bytemuck::cast_slice(&v_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        pipeline.execute(
            &device,
            &queue,
            &k_buf,
            &v_buf,
            keep_sink,
            shift_amt,
            old_seq_len,
            n_head_kv,
            head_dim,
            head_dim,
            10000.0,
            max_seq_len,
        );

        // Readback K
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&k_buf, 0, &staging, 0, buffer_size);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(r) = rx.try_recv() {
                r.unwrap();
                break;
            }
        }
        let data = slice.get_mapped_range();
        let gpu_k: &[f32] = bytemuck::cast_slice(&data);

        // Readback V
        let staging_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc2.copy_buffer_to_buffer(&v_buf, 0, &staging_v, 0, buffer_size);
        queue.submit(Some(enc2.finish()));
        let slice_v = staging_v.slice(..);
        let (tx2, rx2) = std::sync::mpsc::channel();
        slice_v.map_async(wgpu::MapMode::Read, move |v| tx2.send(v).unwrap());
        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(r) = rx2.try_recv() {
                r.unwrap();
                break;
            }
        }
        let data_v = slice_v.get_mapped_range();
        let gpu_v: &[f32] = bytemuck::cast_slice(&data_v);

        // Verify: every shifted element must be bit-identical to its source
        let elements_to_shift = old_seq_len - (keep_sink + shift_amt); // 11
        let mut max_k_err: f32 = 0.0;
        let mut max_v_err: f32 = 0.0;
        for offset in 0..elements_to_shift {
            let old_pos = keep_sink + shift_amt + offset;
            let new_pos = keep_sink + offset;
            for h in 0..n_head_kv {
                for d in 0..head_dim {
                    let old_idx = (old_pos * n_head_kv * head_dim + h * head_dim + d) as usize;
                    let new_idx = (new_pos * n_head_kv * head_dim + h * head_dim + d) as usize;
                    let k_err = (gpu_k[new_idx] - k_data[old_idx]).abs();
                    let v_err = (gpu_v[new_idx] - v_data[old_idx]).abs();
                    if k_err > max_k_err {
                        max_k_err = k_err;
                    }
                    if v_err > max_v_err {
                        max_v_err = v_err;
                    }
                }
            }
        }
        drop(data);
        drop(data_v);

        println!(
            "OVERLAP TEST: max K error = {:.2e}, max V error = {:.2e}",
            max_k_err, max_v_err
        );
        println!(
            "  params: keep={}, shift={}, old_len={}, elements_to_shift={}",
            keep_sink, shift_amt, old_seq_len, elements_to_shift
        );
        println!(
            "  source range: [{}..{}], dest range: [{}..{}]",
            keep_sink + shift_amt,
            old_seq_len - 1,
            keep_sink,
            keep_sink + elements_to_shift - 1
        );

        assert!(
            max_k_err == 0.0,
            "K must be bit-exact copy, got max_err={:.2e}",
            max_k_err
        );
        assert!(
            max_v_err == 0.0,
            "V must be bit-exact copy, got max_err={:.2e}",
            max_v_err
        );
        println!("PASS: Overlapping shift is bit-identical for both K and V (pure copy)");
    }
}
