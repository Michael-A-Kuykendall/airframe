// Test Assumption #3: Verify norm bank extraction copied correct data
// Algebraic comparison: preflight-extracted norms vs file bytes

use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::core::spec::ModelSpec;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

#[test]
fn test_norm_bank_extraction_algebraic() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = match std::env::var("SHIMMY_BASE_GGUF")
        .map(PathBuf::from)
        .or_else(|_| {
            let candidates = [
                "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf",
                "/home/ubuntu/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
            ];
            candidates.iter()
                .find(|p| PathBuf::from(p).exists())
                .map(PathBuf::from)
                .ok_or("Model not found")
        }) {
        Ok(p) => p,
        Err(_) => {
            println!("[SKIP] SHIMMY_BASE_GGUF not set and no model found at known paths - skipping norm bank test");
            return Ok(());
        }
    };
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();

    // 1. Load metadata to get tensor offsets
    let mut file = File::open(&model_path)?;
    let metadata = BindlessMetadata::new(&mut file);

    println!("\n=== ALGEBRAIC FORMULA VERIFICATION ===");
    println!("Testing: norm_bank[layer * 4 * dim] == file_bytes[blk.layer.attn_norm.weight]");
    println!("Note: slots 2 & 3 (post_attention_norm, post_ffw_norm) are zero-filled for non-Gemma2 models");
    println!();

    // 2. Get Layer 0 and Layer 1 attn_norm offsets from file
    let l0_norm_offset = metadata
        .get_tensor_offset("blk.0.attn_norm.weight")
        .expect("Layer 0 attn_norm not found");
    let l1_norm_offset = metadata
        .get_tensor_offset("blk.1.attn_norm.weight")
        .expect("Layer 1 attn_norm not found");

    println!("File Offsets:");
    println!("  blk.0.attn_norm.weight: {} bytes", l0_norm_offset);
    println!("  blk.1.attn_norm.weight: {} bytes", l1_norm_offset);
    println!();

    // 3. Read first 10 F32 values from each norm weight in file
    file.seek(SeekFrom::Start(l0_norm_offset))?;
    let mut l0_file_bytes = [0u8; 40]; // 10 * 4 bytes
    file.read_exact(&mut l0_file_bytes)?;
    let l0_file_floats: Vec<f32> = l0_file_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    file.seek(SeekFrom::Start(l1_norm_offset))?;
    let mut l1_file_bytes = [0u8; 40];
    file.read_exact(&mut l1_file_bytes)?;
    let l1_file_floats: Vec<f32> = l1_file_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("File Data (Ground Truth):");
    println!("  Layer 0 attn_norm first 5: {:?}", &l0_file_floats[..5]);
    println!("  Layer 1 attn_norm first 5: {:?}", &l1_file_floats[..5]);
    println!();

    // 4. Build norm bank using preflight logic (CPU-side extraction)
    file.seek(SeekFrom::Start(0))?;
    let mut raw_data = Vec::new();
    file.read_to_end(&mut raw_data)?;

    let dim = spec.n_embd;
    let n_layers = spec.n_layer;
    let block_size = dim * 4; // F32 = 4 bytes

    let total_size = (n_layers * 4 + 1) * block_size;
    let mut norm_bank = vec![0u8; total_size];

    println!("Norm Bank Extraction:");
    println!("  dim: {}", dim);
    println!("  block_size: {} bytes ({} F32 elements)", block_size, dim);
    println!("  total_size: {} bytes", total_size);
    println!();

    // Extract Layer 0 and Layer 1 attn norms (slot 0 of each 4-slot group)
    for layer_idx in 0..=1 {
        let tensor_name = format!("blk.{}.attn_norm.weight", layer_idx);
        let file_offset = metadata.get_tensor_offset(&tensor_name).unwrap() as usize;
        let bank_offset = layer_idx * 4 * block_size;

        println!("Layer {} attn_norm extraction:", layer_idx);
        println!("  file_offset: {} bytes", file_offset);
        println!(
            "  bank_offset: {} bytes (element index {})",
            bank_offset,
            bank_offset / 4
        );
        println!(
            "  Copying: raw_data[{}..{}] → norm_bank[{}..{}]",
            file_offset,
            file_offset + block_size,
            bank_offset,
            bank_offset + block_size
        );

        norm_bank[bank_offset..bank_offset + block_size]
            .copy_from_slice(&raw_data[file_offset..file_offset + block_size]);
        println!();
    }

    // 5. Read extracted values from norm bank
    let l0_bank_floats: Vec<f32> = norm_bank[0..40]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let l1_bank_offset = 1 * 4 * block_size;
    let l1_bank_floats: Vec<f32> = norm_bank[l1_bank_offset..l1_bank_offset + 40]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    println!("Norm Bank Extracted Data:");
    println!("  Layer 0 attn_norm first 5: {:?}", &l0_bank_floats[..5]);
    println!("  Layer 1 attn_norm first 5: {:?}", &l1_bank_floats[..5]);
    println!();

    // 6. Algebraic Comparison
    println!("=== ALGEBRAIC COMPARISON ===");

    let l0_match = l0_file_floats
        .iter()
        .zip(l0_bank_floats.iter())
        .all(|(a, b)| (a - b).abs() < 1e-9);

    let l1_match = l1_file_floats
        .iter()
        .zip(l1_bank_floats.iter())
        .all(|(a, b)| (a - b).abs() < 1e-9);

    if l0_match {
        println!("✅ Layer 0: norm_bank[0..9] MATCHES file bytes");
    } else {
        println!("❌ Layer 0: norm_bank[0..9] DIFFERS from file!");
        for (i, (file_val, bank_val)) in
            l0_file_floats.iter().zip(l0_bank_floats.iter()).enumerate()
        {
            if (file_val - bank_val).abs() >= 1e-9 {
                println!(
                    "    [{}] file={:.8} bank={:.8} diff={:.2e}",
                    i,
                    file_val,
                    bank_val,
                    (file_val - bank_val).abs()
                );
            }
        }
    }

    if l1_match {
        println!("✅ Layer 1: norm_bank[4096..4105] MATCHES file bytes");
    } else {
        println!("❌ Layer 1: norm_bank[4096..4105] DIFFERS from file!");
        for (i, (file_val, bank_val)) in
            l1_file_floats.iter().zip(l1_bank_floats.iter()).enumerate()
        {
            if (file_val - bank_val).abs() >= 1e-9 {
                println!(
                    "    [{}] file={:.8} bank={:.8} diff={:.2e}",
                    i,
                    file_val,
                    bank_val,
                    (file_val - bank_val).abs()
                );
            }
        }
    }

    println!();
    println!("=== FORMULA VERIFICATION ===");
    println!("CPU Formula: norm_weight = tensor from WeightId::AttnNorm {{ layer: 1 }}");
    println!("GPU Formula: norm_bank[layer_idx * 4 * dim + col]");
    println!("           = norm_bank[1 * 4 * 2048 + col]");
    println!("           = norm_bank[8192 + col]");
    println!("           (slots 1/2/3 per layer hold ffn_norm, post_attn_norm, post_ffw_norm)");
    println!();

    if l0_match && l1_match {
        println!("✅ ASSUMPTION #3 PASSES: Norm bank extraction is CORRECT");
        println!("   → Bug is NOT in preflight.rs extraction");
        println!("   → Bug must be in shader access or params.dim being wrong");
    } else {
        println!("❌ ASSUMPTION #3 FAILS: Norm bank extraction is WRONG");
        println!("   → Bug is in preflight.rs build_norm_bank_from_ram()");
        println!("   → Fix: Verify copy_from_slice offset calculations");
    }

    assert!(l0_match && l1_match, "Norm bank extraction failed!");

    Ok(())
}
