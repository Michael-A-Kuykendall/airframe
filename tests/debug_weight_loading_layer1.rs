// Minimal test to isolate Layer 1 weight loading bug
// Tests ONLY weight reading from GGUF blob for Layer 0 vs Layer 1

use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::core::spec::ModelSpec;
use std::fs::File;
use std::path::PathBuf;

#[tokio::test]
#[ignore]
async fn debug_weight_read_layer1() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{Read, Seek, SeekFrom};

    let model_path =
        PathBuf::from("C:/Users/micha/repos/llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();

    // Load metadata
    let mut file = File::open(&model_path)?;
    let metadata = BindlessMetadata::new(&mut file);
    // Get offsets for both layers
    let l0_offsets = metadata.get_layer_offsets(0, "tinyllama").unwrap();
    let l1_offsets = metadata.get_layer_offsets(1, "tinyllama").unwrap();

    println!("\n=== WEIGHT OFFSET VERIFICATION ===");
    println!("Layer 0 attn_q offset: {} bytes", l0_offsets.attn_q);
    println!("Layer 1 attn_q offset: {} bytes", l1_offsets.attn_q);
    println!(
        "Difference: {} bytes",
        l1_offsets.attn_q - l0_offsets.attn_q
    );

    // Read first Q4_0 block from each layer's attn_q weight
    // Q4_0 format: [F16 scale (2 bytes)][16 bytes of nibbles] = 18 bytes/block

    file.seek(SeekFrom::Start(l0_offsets.attn_q as u64))?;
    let mut l0_block = [0u8; 18];
    file.read_exact(&mut l0_block)?;

    file.seek(SeekFrom::Start(l1_offsets.attn_q as u64))?;
    let mut l1_block = [0u8; 18];
    file.read_exact(&mut l1_block)?;

    println!("\n=== FIRST Q4_0 BLOCK DATA ===");
    println!("Layer 0 first 18 bytes: {:?}", &l0_block);
    println!("Layer 1 first 18 bytes: {:?}", &l1_block);

    // Decode scale (F16 -> F32)
    let l0_scale_bytes = u16::from_le_bytes([l0_block[0], l0_block[1]]);
    let l01_scale_bytes = u16::from_le_bytes([l1_block[0], l1_block[1]]);

    println!("\n=== DECODED SCALES ===");
    println!("Layer 0 scale (F16 bits): 0x{:04x}", l0_scale_bytes);
    println!("Layer 1 scale (F16 bits): 0x{:04x}", l01_scale_bytes);

    // Check if they're different (they MUST be for different layers)
    if l0_block == l1_block {
        println!("\n❌ CRITICAL BUG: Layer 0 and Layer 1 weights are IDENTICAL!");
        println!("   This means GPU is reading same offset for both layers!");
        return Err("Weight loading bug detected".into());
    } else {
        println!("\n✅ PASS: Layer 0 and Layer 1 weights are DIFFERENT (as expected)");
        println!("   File offset calculation is CORRECT");
    }

    Ok(())
}
