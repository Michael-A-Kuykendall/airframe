//! Dequantisation and probe dispatch methods for `BindlessPipeline`.
use super::super::loader::BindlessModel;
use super::*;
use wgpu::util::DeviceExt;

impl BindlessPipeline {
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
                    resource: model.blob_binding_0(),
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
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU device lost during readback poll");
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
                    resource: model.blob_binding_0(),
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
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU device lost during readback poll");
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
                    resource: model.blob_binding_0(),
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
            let workgroups = count.div_ceil(64);
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
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU device lost during readback poll");
            if let Ok(res) = rx.try_recv() {
                res.unwrap();
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        result
    }

    /// Dequantize any supported GGML quant type to f32 via GPU — HOT PATH.
    ///
    /// Uses the pre-compiled `dequant_any_pipeline` (compiled once at startup).
    /// Safe to call in the embedding pre-batch loop.
    /// Supports: 0=F32, 1=F16, 2=Q4_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
    pub fn run_dequant_any_hot(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        offset_bytes: u32,
        count: u32,
        quant_type: u32,
    ) -> Vec<f32> {
        // The model is uploaded to VRAM as word-granular blob buffers; the
        // shader reads absolute word indices, so the bytes backing blob_0 are
        // the raw GGUF bytes for the model's first chunk.
        let blob = {
            let buf = &model.gpu_buffers[0];
            let size = buf.size();
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DequantAny Hot Staging"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size);
            queue.submit(Some(encoder.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
            loop {
                device.poll(wgpu::PollType::Poll).ok();
                if let Ok(res) = rx.try_recv() {
                    res.expect("DequantAny hot staging map failed");
                    break;
                }
            }
            let data = slice.get_mapped_range();
            let bytes: Vec<u8> = data.to_vec();
            drop(data);
            staging.unmap();
            bytes
        };
        self.run_dequant_any_blob(device, queue, &blob, offset_bytes, count, quant_type)
    }

    /// Dequantize a tensor directly from raw GGUF blob bytes on the GPU.
    ///
    /// This is the core of [`Self::run_dequant_any_hot`] but takes a `&[u8]` blob
    /// instead of a [`BindlessModel`], so it can be driven by synthetic blocks in
    /// tests (e.g. the P2 algebraic audit) without loading a full model or
    /// satisfying the multi-buffer split geometry. The blob must be 4-byte aligned
    /// at `offset_bytes` (offsets are word-granular in `sh_dequant_any.wgsl`).
    ///
    /// Supports: 0=F32, 1=F16, 2=Q4_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K.
    pub fn run_dequant_any_blob(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        blob: &[u8],
        offset_bytes: u32,
        count: u32,
        quant_type: u32,
    ) -> Vec<f32> {
        let params = DequantAnyParams {
            offset_bytes,
            count,
            formula_index: formula_index_for_ggml(quant_type),
            pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DequantAny Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Place the blob bytes at byte 0 so word index 0 → blob[0..4].
        // The shader reads at word `offset_bytes/4`, which is byte `offset_bytes`
        // in this buffer → blob[offset_bytes] → correct absolute position.
        let mut blob_full = blob.to_vec();
        // Pad to a multiple of 4 bytes so the last word is fully readable.
        while !blob_full.len().is_multiple_of(4) {
            blob_full.push(0u8);
        }
        let blob_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DequantAny Blob"),
            contents: &blob_full,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DequantAny DummyBlob"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_size = (count as usize * 4) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DequantAny Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DequantAny BindGroup Blob"),
            layout: &self.dequant_any_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: blob_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: dummy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: dummy.as_entire_binding(),
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
            cpass.set_pipeline(&self.dequant_any_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DequantAny Staging"),
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
            device.poll(wgpu::PollType::Poll).ok();
            if let Ok(res) = rx.try_recv() {
                res.expect("DequantAny staging map failed");
                break;
            }
        }
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        result
    }

    /// Dequantize any supported GGML quant type to f32 via GPU.
    ///
    /// Uses `sh_dequant_any.wgsl` which supports:
    /// 0=F32, 1=F16, 2=Q4_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K
    ///
    /// The pipeline is created on each call — intended for validation/testing,
    /// not for hot-path inference.
    pub fn run_dequant_any_request(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &BindlessModel,
        offset_bytes: u32,
        count: u32,
        quant_type: u32,
    ) -> Vec<f32> {
        let params = DequantAnyParams {
            offset_bytes,
            count,
            formula_index: formula_index_for_ggml(quant_type),
            pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DequantAny Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let output_size = (count as usize * 4) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DequantAny Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Build a one-shot pipeline for the multi-type dequant shader.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DequantAny Layout"),
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
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader_src = include_str!("../sh_dequant_any.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DequantAny Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DequantAny Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DequantAny Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DequantAny BindGroup"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.blob_binding_0(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: model.blob_binding_1(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: model.blob_binding_2(),
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
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = count.div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DequantAny Staging"),
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
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("GPU device lost during readback poll");
            if let Ok(res) = rx.try_recv() {
                res.unwrap();
                break;
            }
        }

        let data = slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }
}
