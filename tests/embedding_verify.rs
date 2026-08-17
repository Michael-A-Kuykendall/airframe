//! Embedding Row Equality Verification
//!
//! Verifies that GPU embedding lookup matches CPU-side dequant of token_embd rows.
//! Gate 2 of airframe-f41.1: row equality max_err == 0.

use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::BindlessPipeline;
use airframe::core::dequant::q4_0::dequantize_q4_0;
use airframe::core::dequant::q6_k::dequantize_q6_k;
use airframe::core::model::GgufTensorInfo;
use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;

fn get_model_path(name: &str) -> PathBuf {
    let base = std::env::var("AIRFRAME_TEST_MODELS")
        .unwrap_or_else(|_| "/home/michael/models".to_string());
    PathBuf::from(base).join(name)
}

async fn test_embedding_row_equality(
    model_name: &str,
    model_path: PathBuf,
    quant_type: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping", model_path);
        return Ok(());
    }

    println!("\n=== Embedding Row Equality Test ({}) ===\n", model_name);

    // Load metadata to get spec
    let mut header_file = File::open(&model_path)?;
    let header_meta = BindlessMetadata::new(&mut header_file);
    let spec = header_meta.to_model_spec();
    drop(header_file);

    // Create GPU device
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("No adapter found");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    // Load model
    let gpu_model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));
    let pipeline = BindlessPipeline::new(&device);

    let dim = spec.n_embd;
    let n_vocab = spec.n_vocab;

    // Get token_embd tensor info
    let embd_quant_type = gpu_model
        .metadata
        .get_tensor_type("token_embd.weight")
        .unwrap_or(quant_type);
    let embd_weight_offset = gpu_model
        .metadata
        .get_tensor_offset("token_embd.weight")
        .expect("token_embd.weight not found");

    println!(
        "dim={}, n_vocab={}, quant_type={}, offset=0x{:x}",
        dim, n_vocab, embd_quant_type, embd_weight_offset
    );

    // Test token 0 (BOS)
    let token = 0u32;
    let row_bytes = match embd_quant_type {
        0 => dim * 4,
        1 => dim * 2,
        2 => (dim / 32) * 18,    // Q4_0
        6 => (dim / 32) * 22,    // Q5_0
        8 => (dim / 32) * 34,    // Q8_0
        12 => (dim / 256) * 144, // Q4_K
        13 => (dim / 256) * 176, // Q5_K
        14 => (dim / 256) * 210, // Q6_K
        _ => panic!("unsupported embedding quant type: {}", embd_quant_type),
    };

    // CPU dequant - dequantize the whole tensor and pick the row
    // tensor offsets are absolute; dequant functions expect offset relative to data_start_offset
    let file = File::open(&model_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data_start = gpu_model.metadata.data_start_offset;
    let tensor_offset_relative = if embd_weight_offset >= data_start {
        embd_weight_offset - data_start
    } else {
        embd_weight_offset
    };

    let tensor_info = GgufTensorInfo {
        name: "token_embd.weight".to_string(),
        ggml_type: embd_quant_type,
        offset: tensor_offset_relative,
        dimensions: vec![n_vocab, dim],
    };

    let cpu_tensor = match embd_quant_type {
        2 => dequantize_q4_0(&tensor_info, &mmap, data_start)?,
        14 => dequantize_q6_k(&tensor_info, &mmap, data_start)?,
        _ => return Err(format!("quant type {} not implemented in test", embd_quant_type).into()),
    };

    let cpu_embd: Vec<f32> = cpu_tensor
        .data
        .chunks(dim)
        .nth(token as usize)
        .expect("token row not found")
        .to_vec();

    println!(
        "CPU embedding row {}: first 5 = {:?}",
        token,
        &cpu_embd[..5.min(dim)]
    );

    // GPU dequant via run_dequant_any_hot
    let row_offset = embd_weight_offset + (token as u64 * row_bytes as u64);
    let gpu_embd = pipeline.run_dequant_any_hot(
        &device,
        &queue,
        &gpu_model,
        row_offset as u32,
        dim as u32,
        embd_quant_type,
    );

    println!(
        "GPU embedding row {}: first 5 = {:?}",
        token,
        &gpu_embd[..5.min(dim)]
    );

    // Compare
    let mut max_err: f32 = 0.0;
    for i in 0..dim {
        let err = (cpu_embd[i] - gpu_embd[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }

    println!("Max error: {}", max_err);
    assert_eq!(
        max_err, 0.0,
        "Embedding row equality failed: max_err = {}",
        max_err
    );

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_embedding_row_equality_tinyllama() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        get_model_path("Llama/TinyLlama-1.1B-Chat-Q4_0/tinyllama-1.1b-chat-v1.0.Q4_0.gguf");
    test_embedding_row_equality("TinyLlama", model_path, 2).await
}

#[tokio::test]
#[ignore]
async fn test_embedding_row_equality_gemma4() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = get_model_path("Gemma-4/gemma-4-E4B-it-Q4_K_M/gemma-4-E4B-it-Q4_K_M.gguf");
    test_embedding_row_equality("Gemma-4-E4B", model_path, 14).await
}
