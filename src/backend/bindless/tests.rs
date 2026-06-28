#[cfg(test)]
mod tests {
    use super::super::loader::BindlessModel;
    use super::super::pipeline::BindlessPipeline;
    use crate::core::spec::ModelSpec;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use wgpu;

    #[allow(dead_code)]
    fn get_tinyllama_spec() -> ModelSpec {
        ModelSpec::tinylama_1_1b_chat_v1_0()
    }

    fn dummy_metadata() -> super::super::metadata::BindlessMetadata {
        super::super::metadata::BindlessMetadata {
            version: 3,
            tensor_count: 0,
            tensor_offsets: std::collections::HashMap::new(),
            tensor_types: std::collections::HashMap::new(),
            data_start_offset: 0,
            gguf_metadata: std::collections::HashMap::new(),
            tensor_dims: std::collections::HashMap::new(),
            compiled_layers: Vec::new(),
        }
    }

    async fn get_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No adapter");

        let adapter_limits = adapter.limits();
        println!(
            "[BindlessTest] Adapter Max Storage Buffer: {} MB",
            adapter_limits.max_storage_buffer_binding_size / 1024 / 1024
        );

        // We need generous limits for storage buffers
        // Default WebGPU limit is 128MB. We need more for GGUF.
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size; // Use what the card has
        limits.max_buffer_size = adapter_limits.max_storage_buffer_binding_size as u64;
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage; // INT4 bind group uses 14 bindings; use adapter max

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("No device");

        println!(
            "[BindlessTest] Allocated Device Limits: Max Storage Buffer = {} MB",
            device.limits().max_storage_buffer_binding_size / 1024 / 1024
        );

        (device, queue)
    }

    #[tokio::test]
    async fn test_bindless_probe_magic() {
        // 1. Setup WGPU
        let (device, queue) = get_device().await;

        // 2. Create Dummy GGUF file
        // Magic: 'G', 'G', 'U', 'F' = 0x46554747 (LE)
        // Ver: 3 = 0x00000003
        let mut tmp = NamedTempFile::new().unwrap();
        let magic: u32 = 0x46554747;
        let ver: u32 = 3;

        let mut content = Vec::new();
        content.extend_from_slice(&magic.to_le_bytes());
        content.extend_from_slice(&ver.to_le_bytes());
        // Pad with junk
        for _ in 0..100 {
            content.push(0);
        }

        tmp.write_all(&content).unwrap();
        let path = tmp.path();

        // 3. Load Model (Bindless)
        // This uploads it to GPU
        let model = BindlessModel::load_from_disk(&device, path, None);
        assert_eq!(model.size, content.len() as u64);

        // 4. Create Pipeline
        let pipeline = BindlessPipeline::new(&device);

        // 5. Run Probe
        let result = pipeline.run_probe(&device, &queue, &model);

        println!("Probe Result: {:x?}", result);

        // 6. Verify
        assert_eq!(result[0], 0x46554747, "Magic number mismatch");
        assert_eq!(result[1], 3, "Version mismatch");
    }

    #[tokio::test]
    async fn test_bindless_dequant_q4_0() {
        // 1. Setup
        let (device, queue) = get_device().await;

        // 2. Create Dummy Q4_0 Block in "GGUF Buffer"
        // Block 0 offset = 0.
        // d (f16) = 1.0.  f16::to_bits(1.0) = 0x3C00.
        // qs[0] = 0x88 (nibbles 8, 8). (8-8)*1.0 = 0.0.
        // qs[1] = 0x97 (nibbles 9, 7). (7-8)=-1.0, (9-8)=1.0.

        // We need 18 bytes for Q4_0 block, but wgpu STORAGE buffers require 4-byte alignment.
        // Pad to next multiple of 4: 20 bytes total (18 data + 2 padding).
        let mut block_bytes: Vec<u8> = Vec::with_capacity(20);

        // Offset 0: scale = 1.0 (0x3C00, LE = 00 3C)
        block_bytes.push(0x00);
        block_bytes.push(0x3C);

        // Offset 2..18: 16 bytes of u8 qs (Q4_0: simplified test with 16 elements)
        // Byte 0 (qs[0]): 0x88 -> (8, 8) -> val 0.0, 0.0
        block_bytes.push(0x88);

        // Byte 1 (qs[1]): 0x0F -> (15, 0) -> val (15-8)=7.0, (0-8)=-8.0
        block_bytes.push(0x0F);

        // Fill rest with 0x88 (zeros)
        for _ in 2..16 {
            block_bytes.push(0x88);
        }

        // Padding to 4-byte alignment (wgpu STORAGE buffer requirement)
        block_bytes.push(0x00); // padding byte 0
        block_bytes.push(0x00); // padding byte 1

        // Upload
        use wgpu::util::DeviceExt;
        let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dequant Test Buffer"),
            contents: &block_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let dummy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Blob"),
            contents: &[0u8; 8], // Pad to 4-byte alignment for STORAGE buffer
            usage: wgpu::BufferUsages::STORAGE,
        });
        let model = BindlessModel {
            gpu_buffer,
            size: block_bytes.len() as u64,
            dummy_buf,
            metadata: dummy_metadata(),
            preflight: None,
        };

        // 3. Run Dequant
        let pipeline = BindlessPipeline::new(&device);
        let result = pipeline.run_dequant_test(&device, &queue, &model);

        println!("Dequant Result: {:?}", &result[0..8]);

        // 4. Verify
        // Lane 0: qs[0] low = 8. (8-8)*1.0 = 0.0
        assert!((result[0] - 0.0).abs() < 1e-5);

        // Lane 1: qs[1] low = 15. (15-8)*1.0 = 7.0
        assert!((result[1] - 7.0).abs() < 1e-5);

        // Lane 16: qs[0] high = 8. (8-8)*1.0 = 0.0
        assert!((result[16] - 0.0).abs() < 1e-5);

        // Lane 17: qs[1] high = 0. (0-8)*1.0 = -8.0
        // Wait, did I parse high nibble logic correctly?
        // sh_dequant_q4_0.wgsl:
        // } else {
        //     let qs_byte = get_u8(block_offset + 2u + (lane_idx - 16u));
        //     nibble = (qs_byte >> 4u) & 0x0Fu;
        // }
        // lane 17 -> 17-16=1. qs_byte is qs[1] (0x0F).
        // 0x0F >> 4 = 0. Nibble = 0. 0-8 = -8. Correct.
        assert!((result[17] - (-8.0)).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_bindless_matmul_q4_0() {
        use super::super::pipeline::MatMulParams;
        use wgpu::util::DeviceExt;

        // 1. Setup
        let (device, queue) = get_device().await;

        // Matrix size: N=2, K=64.
        // We need 2 rows of 64 elements (2 blocks per row).
        // Total blocks = 4.
        // Row bytes = 2 * 18 = 36.
        // Total bytes = 72.

        // Values:
        // Row 0: All 1.0 (via Scale=1.0, qs=0x88 -> 0.0 + bias? No wait)
        // Let's make it simple.
        // Scale = 1.0.
        // qs = 0x88 (8,8) -> (0,0).
        // qs = 0x97 (9,7) -> (1, -1).

        let mut block_bytes: Vec<u8> = Vec::with_capacity(72);

        // Row 0, Block 0 (Cols 0..31):
        // Scale 1.0
        // qs[0] = 0x97 -> (1, -1). Val 0 = -1.0, Val 16 = 1.0.
        // All others 0x88 -> 0.0.
        block_bytes.push(0x00);
        block_bytes.push(0x3C); // 1.0
        block_bytes.push(0x97);
        for _ in 1..16 {
            block_bytes.push(0x88);
        } // Zeros

        // Row 0, Block 1 (Cols 32..63): All Zeros
        block_bytes.push(0x00);
        block_bytes.push(0x3C); // 1.0
        for _ in 0..16 {
            block_bytes.push(0x88);
        }

        // Row 1, Block 0 (Cols 0..31):
        // Scale 2.0 (0x4000)
        // qs[0] = 0x99 -> (1, 1). Val 0 = (9-8)*2 = 2.0. Val 16 = 2.0.
        block_bytes.push(0x00);
        block_bytes.push(0x40); // 2.0
        block_bytes.push(0x99);
        for _ in 1..16 {
            block_bytes.push(0x88);
        }

        // Row 1, Block 1: All Zeros
        block_bytes.push(0x00);
        block_bytes.push(0x40); // 2.0
        for _ in 0..16 {
            block_bytes.push(0x88);
        }

        assert_eq!(block_bytes.len(), 72);

        // Upload Model
        let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MatMul Test Buffer"),
            contents: &block_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let dummy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Blob"),
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::STORAGE,
        });
        let model = BindlessModel {
            gpu_buffer,
            size: 72,
            dummy_buf,
            metadata: dummy_metadata(),
            preflight: None,
        };

        // Input Vector x [64]
        // x[0] = 1.0
        // x[16] = 1.0
        // rest 0.0
        let mut input_x = vec![0.0f32; 64];
        input_x[0] = 1.0;
        input_x[16] = 1.0;

        let pipeline = BindlessPipeline::new(&device);
        let params = MatMulParams {
            n: 2,
            k: 64,
            weights_offset: 0,
            padding: 0,
        };

        let result = pipeline.run_matmul_test(&device, &queue, &model, &input_x, params);

        println!("MatMul Result: {:?}", result);

        // Expected:
        // Row 0:
        // Col 0 val = -1.0. Col 16 val = 1.0.
        // Dot = (-1.0 * 1.0) + (1.0 * 1.0) = 0.0.
        assert!((result[0] - 0.0).abs() < 1e-5);

        // Let's change input to make it non-zero.
        // x[0] = 1.0, x[16] = 0.0.
        // Row 0 dot = -1.0 * 1.0 = -1.0.

        // Row 1:
        // Col 0 val = 2.0. Col 16 val = 2.0.
        // Dot = (2.0 * 1.0) + (2.0 * 1.0) = 4.0.
        assert!((result[1] - 4.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_bindless_rmsnorm() {
        use super::super::pipeline::RMSNormParams;
        use wgpu::util::DeviceExt;

        // 1. Setup
        let (device, queue) = get_device().await;

        // 2. Create Dummy GGUF with F32 Weights
        // Let's make weights = [0.5, 0.5, 0.5, 0.5]
        let mut block_bytes: Vec<u8> = Vec::new(); // Starts at offset 0

        let weights = vec![0.5f32; 4];
        for w in weights {
            block_bytes.extend_from_slice(&w.to_le_bytes());
        }

        let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMSNorm Weight Buffer"),
            contents: &block_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let dummy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Blob"),
            contents: &[0u8; 8], // Pad to 4-byte alignment for STORAGE buffer
            usage: wgpu::BufferUsages::STORAGE,
        });
        let model = BindlessModel {
            gpu_buffer,
            size: block_bytes.len() as u64,
            dummy_buf,
            metadata: dummy_metadata(),
            preflight: None,
        };

        // 3. Input [3.0, 0.0, 4.0, 0.0]
        let input = vec![3.0f32, 0.0, 4.0, 0.0];

        // 4. Run Pipeline
        let pipeline = BindlessPipeline::new(&device);
        let params = RMSNormParams {
            count: 4,
            weights_offset: 0, // Weights st start of buffer
            bias_offset: 0,
            eps: 0.0, // Simplify math
            norm_type: 0,
        };

        let result = pipeline.run_rmsnorm_test(&device, &queue, &model, &input, params);

        // 5. Verify
        // MeanSq = (9+16)/4 = 6.25. Sqrt = 2.5. Inv = 0.4.
        // x_norm = [1.2, 0.0, 1.6, 0.0]
        // w = 0.5
        // y = [0.6, 0.0, 0.8, 0.0]

        println!("RMSNorm Result: {:?}", result);
        assert!((result[0] - 0.6).abs() < 1e-5);
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - 0.8).abs() < 1e-5);
        assert!((result[3] - 0.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_bindless_chain() {
        // Test RMSNorm -> MatMul chaining without CPU readback
        // Input A -> [RMSNorm] -> Buffer B -> [MatMul] -> Output C

        use super::super::pipeline::{MatMulParams, RMSNormParams};
        use wgpu::util::DeviceExt;

        let (device, queue) = get_device().await;

        // 1. Construct GGUF
        // Offset 0: RMSNorm Weights (32 x F32) = 128 bytes. All 1.0.
        // Offset 128: MatMul Weights (1 x Q4_0 Block) = 18 bytes. All 1.0.

        let mut block_bytes: Vec<u8> = Vec::new();

        // RMSNorm Weights: 32 * 1.0
        for _ in 0..32 {
            block_bytes.extend_from_slice(&1.0f32.to_le_bytes());
        }

        // MatMul Weights: 1 Block. Ref value 1.0.
        // Scale = 1.0 (0x3C00)
        block_bytes.push(0x00);
        block_bytes.push(0x3C);
        // Quants: 0x99 (since (9-8)*1.0 = 1.0)
        for _ in 0..16 {
            block_bytes.push(0x99);
        }

        // Upload Model
        let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Chain Model Buffer"),
            contents: &block_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dummy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Blob"),
            contents: &[0u8; 8], // Pad to 4-byte alignment for STORAGE buffer
            usage: wgpu::BufferUsages::STORAGE,
        });
        let model = BindlessModel {
            gpu_buffer,
            size: block_bytes.len() as u64,
            dummy_buf,
            metadata: dummy_metadata(),
            preflight: None,
        };

        // 2. Buffers
        // Input A (User provided)
        let input_a = vec![1.0f32; 32];
        let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Buffer A"),
            contents: bytemuck::cast_slice(&input_a),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Buffer B (Intermediate)
        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer B"),
            size: 32 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Buffer C (Output)
        // MatMul output: N=1 (1 row). Size = 1 * 4 = 4 bytes.
        let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer C"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 3. Params
        let rms_params = RMSNormParams {
            count: 32,
            weights_offset: 0,
            bias_offset: 0,
            eps: 0.0,
            norm_type: 0,
        };
        let rms_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMS Params"),
            contents: bytemuck::bytes_of(&rms_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mm_params = MatMulParams {
            n: 1,                // 1 Row
            k: 32,               // 32 Cols
            weights_offset: 128, // After RMSNorm weights
            padding: 0,
        };
        let mm_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MatMul Params"),
            contents: bytemuck::bytes_of(&mm_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 4. Bind Groups
        let pipeline = BindlessPipeline::new(&device);

        // RMSNorm: In=A, Out=B
        let bg_rms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG RMS"),
            layout: &pipeline.rmsnorm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rms_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: model.dummy_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: model.dummy_buf.as_entire_binding(),
                },
            ],
        });

        // MatMul: In=B, Out=C
        let bg_mm = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG MatMul"),
            layout: &pipeline.matmul_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: model.gpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mm_params_buf.as_entire_binding(),
                },
            ],
        });

        // 5. Record & Submit
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

            // Dispatch RMS
            cpass.set_pipeline(&pipeline.rmsnorm_pipeline);
            cpass.set_bind_group(0, &bg_rms, &[]);
            cpass.dispatch_workgroups(1, 1, 1);

            // Dispatch MatMul
            cpass.set_pipeline(&pipeline.matmul_pipeline);
            cpass.set_bind_group(0, &bg_mm, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            label: None,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer_c, 0, &staging, 0, 4);
        queue.submit(Some(encoder.finish()));

        // 6. Readback
        let slice = staging.slice(..);
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
        println!("Chain Result: {:?}", result);

        // Expectation:
        // RMSNorm(Input A) -> Input is all 1.0. MeanSq=1.0. Normed=1.0. Mul W(1.0)=1.0.
        // B = [1.0; 32].
        // MatMul(B) -> Dot(B, W_mm). W_mm=1.0.
        // Dot = 32 * 1.0 * 1.0 = 32.0.

        assert!((result[0] - 32.0).abs() < 1e-4); // Slightly looser tol due to accumulation
    }
}
