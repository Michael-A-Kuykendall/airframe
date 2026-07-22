//! Verify Layer 1 weights are correctly uploaded to GPU
//! Tests that gguf_blob on GPU contains same bytes as file at Layer 1 offsets

use airframe::backend::bindless::loader::BindlessModel;
use airframe::core::spec::ModelSpec;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

#[tokio::test]
#[ignore]
async fn test_layer1_weights_on_gpu() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Verify Layer 1 Weights on GPU ===\n");

    // Initialize GPU
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No GPU adapter found");

    let adapter_limits = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter_limits.max_storage_buffer_binding_size as u64;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await?;

    // Load model
    let model_path =
        PathBuf::from("D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
    let gpu_model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));

    // Get Layer 1 attn_q offset
    let layer1_offsets = gpu_model
        .metadata
        .get_layer_offsets(1, "tinyllama")
        .expect("Layer 1 not found");
    let attn_q_offset = layer1_offsets.attn_q as u64;

    println!("Layer 1 attn_q offset: {} bytes", attn_q_offset);

    // Read first 32 bytes from FILE at this offset
    let mut file = File::open(&model_path)?;
    file.seek(SeekFrom::Start(attn_q_offset))?;
    let mut file_bytes = vec![0u8; 32];
    file.read_exact(&mut file_bytes)?;

    println!("File bytes [{}..{}]:", attn_q_offset, attn_q_offset + 32);
    println!("{:02x?}", file_bytes);

    // Read same bytes from GPU buffer
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: 32,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });
    // Map the absolute file offset into the correct blob buffer (multi-buffer).
    let chunk0 = gpu_model.effective_chunk;
    let buf_idx = (attn_q_offset / chunk0) as usize;
    let buf_off = attn_q_offset % chunk0;
    encoder.copy_buffer_to_buffer(
        &gpu_model.gpu_buffers[buf_idx],
        buf_off,
        &staging_buffer,
        0,
        32,
    );
    let idx = queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        })
        .unwrap();

    let gpu_data = buffer_slice.get_mapped_range();
    let gpu_bytes = &gpu_data[0..32];

    println!(
        "\nGPU buffer bytes [{}..{}]:",
        attn_q_offset,
        attn_q_offset + 32
    );
    println!("{:02x?}", gpu_bytes);

    // Compare
    let matches = file_bytes.iter().zip(gpu_bytes.iter()).all(|(a, b)| a == b);

    if matches {
        println!("\n✅ GPU buffer MATCHES file at Layer 1 attn_q offset");
    } else {
        println!("\n❌ GPU buffer DIFFERS from file!");
        for (i, (f, g)) in file_bytes.iter().zip(gpu_bytes.iter()).enumerate() {
            if f != g {
                println!("  Byte {}: file=0x{:02x}, gpu=0x{:02x}", i, f, g);
            }
        }
    }

    assert!(
        matches,
        "GPU buffer does not match file for Layer 1 attn_q!"
    );

    Ok(())
}
