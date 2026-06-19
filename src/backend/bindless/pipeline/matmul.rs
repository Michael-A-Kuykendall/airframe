//! MatMul and RMSNorm dispatch methods for `BindlessPipeline`.
use super::super::loader::BindlessModel;
use super::*;
use wgpu::util::DeviceExt;

impl BindlessPipeline {
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
            let workgroups = params.n.div_ceil(256);
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
            device
                .poll(wgpu::PollType::Poll)
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
            let workgroups = n.div_ceil(64);
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
            device
                .poll(wgpu::PollType::Poll)
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

    /// Run the blob-based LM head matmul on CPU-side normed activation.
    /// Uploads `input` to a transient GPU buffer, dispatches `main_lm_head`,
    /// reads back the logits.  No F32 dequant buffer required — weights are
    /// read directly from the GGUF blob.
    #[allow(clippy::too_many_arguments)]
    pub fn run_lm_head_blob(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &super::super::loader::BindlessModel,
        input: &[f32], // normed activation [dim]
        vocab_size: u32,
        dim: u32,
        weight_off: u32, // word offset (byte_offset / 4) of output.weight in GGUF blob
        quant_type: u32, // GGML type
        softcap: f32,
    ) -> Vec<f32> {
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LM Head Input"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_size = (vocab_size as u64) * 4;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Head Logits"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let head_params = HeadBlobParams {
            vocab_size,
            dim,
            weight_off,
            quant_type,
            softcap,
            base_row: 0,
            _pad: 0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LM Head Blob Params"),
            contents: bytemuck::bytes_of(&head_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Head Blob BG"),
            layout: &self.lm_head_blob_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.blob_binding_0(),
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
                label: Some("LM Head Blob Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.lm_head_blob_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = vocab_size.div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Head Staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
        loop {
            device
                .poll(wgpu::PollType::Poll)
                .expect("GPU device lost during head readback");
            if let Ok(res) = rx.try_recv() {
                res.expect("LM head buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Run the blob-based LM head matmul with dispatch splitting (TDR-safe).
    ///
    /// Dispatches the head in tiles of `max_safe_wgs` workgroups, each with
    /// an incremented `base_row` so the shader writes to the correct output
    /// region.  All tiles write into the same output buffer.
    ///
    /// This is the TDR-safe replacement for `run_lm_head_blob()`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_lm_head_blob_tiled(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &super::super::loader::BindlessModel,
        input: &[f32],
        vocab_size: u32,
        dim: u32,
        weight_off: u32,
        quant_type: u32,
        softcap: f32,
        max_safe_wgs: u32,
    ) -> Vec<f32> {
        let tile_size = 64u32; // @workgroup_size in sh_head_blob.wgsl
        let total_wgs = vocab_size.div_ceil(tile_size);
        let output_size = (vocab_size as u64) * 4;

        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("LM Head Input (tiled)"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Head Logits (tiled)"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Head Tiled"),
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("LM Head Tiled Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.lm_head_blob_pipeline);

        // Keep bind groups alive until after submit
        let mut tile_bind_groups: Vec<wgpu::BindGroup> = Vec::new();
        let mut wgs_dispatched = 0u32;

        while wgs_dispatched < total_wgs {
            let this_tile = (total_wgs - wgs_dispatched).min(max_safe_wgs);
            let base_row = wgs_dispatched * tile_size;

            let head_params = HeadBlobParams {
                vocab_size,
                dim,
                weight_off,
                quant_type,
                softcap,
                base_row,
                _pad: 0,
            };
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!(
                    "LM Head Params tile-{}",
                    wgs_dispatched / max_safe_wgs
                )),
                contents: bytemuck::bytes_of(&head_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!(
                    "LM Head BG tile-{}",
                    wgs_dispatched / max_safe_wgs
                )),
                layout: &self.lm_head_blob_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: model.blob_binding_0(),
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

            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(this_tile, 1, 1);

            tile_bind_groups.push(bg);
            wgs_dispatched += this_tile;
        }

        drop(cpass);

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Head Staging (tiled)"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, output_size);
        queue.submit(Some(encoder.finish()));

        // Bind groups must outlive the submit — they do (tile_bind_groups lives until drop)
        drop(tile_bind_groups);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
        loop {
            device
                .poll(wgpu::PollType::Poll)
                .expect("GPU device lost during head tile readback");
            if let Ok(res) = rx.try_recv() {
                res.expect("LM head tile buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
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
                    resource: model.blob_binding_0(),
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
            device
                .poll(wgpu::PollType::Poll)
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
}
