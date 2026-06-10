//! Verify that dequantize_q6_k produces zero NaN/Inf for TinyLlama output.weight.
//! Run: cargo test --test output_head_nan_check --release -- --nocapture --ignored

use airframe::core::dequant::dequantize_q6_k;
use airframe::core::model::GgufTensorInfo;
use memmap2::Mmap;
use std::fs::File;

#[test]
#[ignore]
fn test_output_head_nan_count() {
    let model_path = "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf";
    let file = File::open(model_path).expect("model file not found");
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    // data_start_offset confirmed from metadata log: 1,709,440
    let data_start: u64 = 1_709_440;

    let tensor_info = GgufTensorInfo {
        name: "output.weight".to_string(),
        dimensions: vec![32000, 2048],
        ggml_type: 14, // Q6_K
        offset: 0,     // output.weight is first tensor in data section
    };

    let result = dequantize_q6_k(&tensor_info, &mmap, data_start)
        .expect("dequantize_q6_k failed");

    let nan_count = result.data.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = result.data.iter().filter(|&&x| x.is_infinite()).count();
    let max_abs = result.data.iter().cloned().map(f32::abs).fold(0.0f32, f32::max);
    let first5: Vec<f32> = result.data.iter().take(5).copied().collect();

    println!("[output_head_nan_check]");
    println!("  elements:  {}", result.data.len());
    println!("  NaN count: {}", nan_count);
    println!("  Inf count: {}", inf_count);
    println!("  max_abs:   {:.4e}", max_abs);
    println!("  first5:    {:?}", first5);

    assert_eq!(nan_count, 0, "dequantize_q6_k produced {} NaN values", nan_count);
    assert_eq!(inf_count, 0, "dequantize_q6_k produced {} Inf values", inf_count);
    assert!(max_abs < 10.0, "max_abs={} is suspiciously large", max_abs);
}
