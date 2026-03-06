// Algebraic verification: Upload a focused GGUF byte window around Layer 1 attn_q
// and verify byte-for-byte GPU roundtrip + Q4_0 scale extraction math.

use airframe::backend::bindless::metadata::BindlessMetadata;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

#[tokio::test]
#[ignore]
async fn verify_gpu_buffer_matches_file() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize GPU
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No GPU adapter found");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await?;

    // 2. Parse metadata directly to get authoritative offset
    let model_path =
        PathBuf::from("C:/Users/micha/repos/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf");
    let mut file = File::open(&model_path)?;
    let metadata = BindlessMetadata::new(&mut file);

    // 3. Get Layer 1 attn_q offset (authoritative from metadata)
    let l1_offsets = metadata.get_layer_offsets(1, "tinyllama").unwrap();
    let layer1_attn_q_offset = l1_offsets.attn_q;

    println!("\n=== ALGEBRAIC VERIFICATION ===");
    println!("Layer 1 attn_q offset: {} bytes", layer1_attn_q_offset);
    println!(
        "U32 index: {} (= {} / 4)",
        layer1_attn_q_offset / 4,
        layer1_attn_q_offset
    );

    // 4. Read a focused byte window from file around that offset
    file.seek(SeekFrom::Start(layer1_attn_q_offset as u64))?;
    let mut file_bytes = [0u8; 32];
    file.read_exact(&mut file_bytes)?;

    println!("\n=== FILE DATA (Ground Truth) ===");
    println!(
        "Bytes [{}..{}]:",
        layer1_attn_q_offset,
        layer1_attn_q_offset + 32
    );
    println!("{:02x?}", &file_bytes);

    // 5. Upload only this 32-byte window to GPU (device-limit safe)
    let gpu_window = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Algebraic Window Buffer"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&gpu_window, 0, &file_bytes);

    // 6. Read back from GPU buffer and compare byte-for-byte
    let read_size = 32;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: read_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy from GPU buffer to staging
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&gpu_window, 0, &staging_buffer, 0, read_size);
    let idx = queue.submit(Some(encoder.finish()));

    // Read back
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

    println!("\n=== GPU BUFFER DATA (What Shader Sees) ===");
    println!(
        "Bytes [{}..{}]:",
        layer1_attn_q_offset,
        layer1_attn_q_offset + 32
    );
    println!("{:02x?}", gpu_bytes);

    // 7. Compare
    let matches = file_bytes.iter().zip(gpu_bytes.iter()).all(|(a, b)| a == b);

    println!("\n=== ALGEBRAIC RESULT ===");
    if matches {
        println!("✅ GPU byte window MATCHES file byte-for-byte");
        println!("   → Focused upload/readback path is CORRECT");
    } else {
        println!("❌ GPU byte window DIFFERS from file");
        println!("   → Upload/readback path is corrupted");

        println!("\n    Differences:");
        for (i, (file_byte, gpu_byte)) in file_bytes.iter().zip(gpu_bytes.iter()).enumerate() {
            if file_byte != gpu_byte {
                println!(
                    "      Byte {}: file=0x{:02x} gpu=0x{:02x}",
                    layer1_attn_q_offset + i as u32,
                    file_byte,
                    gpu_byte
                );
            }
        }
    }

    // 8. Decode first Q4_0 block scale from both
    let file_scale_u16 = u16::from_le_bytes([file_bytes[0], file_bytes[1]]);
    let gpu_scale_u16 = u16::from_le_bytes([gpu_bytes[0], gpu_bytes[1]]);

    println!("\n=== Q4_0 SCALE COMPARISON ===");
    println!("File scale (F16 bits): 0x{:04x}", file_scale_u16);
    println!("GPU scale (F16 bits):  0x{:04x}", gpu_scale_u16);

    if file_scale_u16 == gpu_scale_u16 {
        println!("✅ Scales match - byte transport for scale is correct");
    } else {
        println!("❌ Scales differ - scale transport is broken");
    }

    // 9. Algebraic extractBits-equivalent validation for low 16 bits at aligned block start
    let packed = u32::from_le_bytes([gpu_bytes[0], gpu_bytes[1], gpu_bytes[2], gpu_bytes[3]]);
    let extracted_low16 = (packed & 0xFFFF) as u16;
    println!("Extracted low16 from first u32: 0x{:04x}", extracted_low16);
    assert_eq!(
        file_scale_u16, extracted_low16,
        "Low16 extraction mismatch at aligned Q4_0 block start"
    );
    assert!(
        matches,
        "GPU bytes do not match file bytes for focused algebraic window"
    );

    Ok(())
}
