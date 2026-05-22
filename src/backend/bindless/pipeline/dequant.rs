//! Dequantisation and probe dispatch methods for `BindlessPipeline`.
use super::*;
use super::super::loader::BindlessModel;
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
            device.poll(wgpu::PollType::Poll).expect("GPU device lost during readback poll");
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
            device.poll(wgpu::PollType::Poll).expect("GPU device lost during readback poll");
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
            device.poll(wgpu::PollType::Poll).expect("GPU device lost during readback poll");
            if let Ok(res) = rx.try_recv() {
                res.unwrap();
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
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
            quant_type,
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
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (count + 63) / 64;
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
            device.poll(wgpu::PollType::Poll).expect("GPU device lost during readback poll");
            if let Ok(res) = rx.try_recv() {
                res.unwrap();
                break;
            }
        }

        let data = slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    }
}
