#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use wgpu::util::DeviceExt;

    async fn get_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("No adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        (device, queue)
    }

    #[tokio::test]
    async fn test_gpu_q4_0_micro_audit() {
        let (device, queue) = get_device().await;

        // 1. Construct Manual Q4_0 Block
        // Scale = 2.0. f16(2.0) = 0x4000.

        // Quants: All 0xBA (Low=10, High=11).
        // Expected:
        //   Low: (10 - 8) * 2.0 = 4.0
        //   High: (11 - 8) * 2.0 = 6.0

        let mut block_bytes: Vec<u8> = Vec::with_capacity(32);
        // Scale (Little Endian)
        block_bytes.push(0x00);
        block_bytes.push(0x40);

        // Quants (16 bytes)
        for _ in 0..16 {
            block_bytes.push(0xBA); // 186
        }

        // Pad to u32 alignment (2 bytes needed to reach 20)
        block_bytes.push(0x00);
        block_bytes.push(0x00);

        // Ensure u32 interpretation matches what we expect
        let input_u32s: Vec<u32> = block_bytes
            .chunks(4)
            .map(|c| {
                let mut val = 0u32;
                for i in 0..c.len() {
                    val |= (c[i] as u32) << (i * 8);
                }
                val
            })
            .collect();

        // 2. Shader
        let shader_src = r#"
        @group(0) @binding(0) var<storage, read> input_blob: array<u32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;

        fn decode_q4_0_val(block_byte_offset: u32, local_idx: u32) -> f32 {
            // Read Scale
            let scale_u32_idx = block_byte_offset / 4u;
            let scale_u32_val = input_blob[scale_u32_idx];
            let scale_packed = extractBits(scale_u32_val, (block_byte_offset % 4u) * 8u, 16u);
            let scale = unpack2x16float(scale_packed).x;

            // Read Quant
            // i=0 -> byte 0 low
            // i=1 -> byte 0 high -> Wait, this matches my NEW fix? 
            // My NEW fix was: i / 2u. 
            // Let's implement the EXACT logic from sh_layer_v1.wgsl
            
            // NOTE: Copy-pasting logic from sh_layer_v1.wgsl to verify IT specifically
            
            // Copied from sh_layer_v1 (Fixed Splitted Layout):
            // i=0..15  -> Low Nibbles of bytes 0..15
            // i=16..31 -> High Nibbles of bytes 0..15
            
            var byte_idx = 0u;
            var shift = 0u;
            if (local_idx < 16u) {
                byte_idx = local_idx;
                shift = 0u;
            } else {
                byte_idx = local_idx - 16u;
                shift = 4u;
            }
            
            let q_byte_off = block_byte_offset + 2u + byte_idx;
            let q_u32_idx = q_byte_off / 4u;
            let q_u32_val = input_blob[q_u32_idx];
            let q_byte_val = extractBits(q_u32_val, (q_byte_off % 4u) * 8u, 8u);
            
            let nib = (q_byte_val >> shift) & 0x0Fu;
            
            return (f32(nib) - 8.0) * scale;
        }

        @compute @workgroup_size(32)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let i = gid.x;
            if (i < 32u) {
                output[i] = decode_q4_0_val(0u, i);
            }
        }
        "#;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Test Kernel"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_src)),
        });

        // 3. Buffers
        let buffer_in = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("In"),
            contents: bytemuck::cast_slice(&input_u32s),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Out"),
            size: 32 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 4. Pipeline
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_out.as_entire_binding(),
                },
            ],
        });

        // 5. Run
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // 6. Readback
        let buffer_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read"),
            size: 32 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer_out, 0, &buffer_read, 0, 32 * 4);
        queue.submit(Some(encoder.finish()));

        let slice = buffer_read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel(); // oneshot not avail? using mpsc is fine
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

        loop {
            device.poll(wgpu::PollType::Poll).unwrap();
            if let Ok(res) = rx.try_recv() {
                res.expect("Buffer map failed");
                break;
            }
        }

        let data = slice.get_mapped_range();
        let result: &[f32] = bytemuck::cast_slice(&data);

        println!("GPU Results: {:?}", result);

        // 7. Verify
        // Input: All bytes 0xBA (Low=A=10 -> 4.0, High=B=11 -> 6.0)
        // Splitted Layout:
        // Indices 0..15 -> Low Nibbles -> All 4.0
        // Indices 16..31 -> High Nibbles -> All 6.0

        for i in 0..16 {
            assert!(
                (result[i] - 4.0).abs() < 1e-5,
                "Idx {} (Low Group) Expected 4.0, got {}",
                i,
                result[i]
            );
        }
        for i in 16..32 {
            assert!(
                (result[i] - 6.0).abs() < 1e-5,
                "Idx {} (High Group) Expected 6.0, got {}",
                i,
                result[i]
            );
        }
    }
}
