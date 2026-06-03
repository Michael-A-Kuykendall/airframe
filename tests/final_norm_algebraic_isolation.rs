// Focused algebraic isolation for final RMSNorm math.
// Verifies GPU RMSNorm kernel equals CPU reference on the same 2048-vector input and weights.

use std::collections::HashMap;

use airframe::backend::bindless::loader::BindlessModel;
use airframe::backend::bindless::metadata::BindlessMetadata;
use airframe::backend::bindless::pipeline::{BindlessPipeline, RMSNormParams};
use airframe::core::tensor::Tensor;
use airframe::ops::reference::rmsnorm::rmsnorm_f32;
use wgpu::util::DeviceExt;

fn make_input(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = i as f32;
            (x * 0.013).sin() * 0.75 + (x * 0.007).cos() * 0.25
        })
        .collect()
}

fn make_weights(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = i as f32;
            0.95 + 0.1 * (x * 0.011).sin()
        })
        .collect()
}

#[tokio::test]
#[ignore]
async fn verify_final_norm_algebraic_isolation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Final Norm Algebraic Isolation ===\n");

    let dim = 2048usize;
    let eps = 1e-5f32;

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

    let pipeline = BindlessPipeline::new(&device);

    let input = make_input(dim);
    let weights = make_weights(dim);

    // Build a minimal fake bindless model whose blob starts with the final norm weights at offset 0.
    let weight_blob = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Algebraic FinalNorm Weight Blob"),
        contents: bytemuck::cast_slice(&weights),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });

    let dummy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dummy Blob"),
        contents: &[0u8; 4],
        usage: wgpu::BufferUsages::STORAGE,
    });
    let model = BindlessModel {
        gpu_buffer: weight_blob,
        size: (dim * std::mem::size_of::<f32>()) as u64,
        dummy_buf,
        metadata: BindlessMetadata {
            version: 3,
            tensor_count: 0,
            tensor_offsets: HashMap::new(),
            tensor_types: HashMap::new(),
            data_start_offset: 0,
            gguf_metadata: HashMap::new(),
            tensor_dims: HashMap::new(),
            compiled_layers: Vec::new(),
        },
        preflight: None,
    };

    let gpu = pipeline.run_rmsnorm_test(
        &device,
        &queue,
        &model,
        &input,
        RMSNormParams {
            count: dim as u32,
            weights_offset: 0,
            eps,
            padding: 0,
        },
    );

    let cpu = rmsnorm_f32(
        &Tensor::new(input.clone(), vec![dim])?,
        &Tensor::new(weights.clone(), vec![dim])?,
        eps,
    )?
    .data;

    let mut max_abs_err = 0.0f32;
    let mut non_finite_gpu = 0usize;

    for (c, g) in cpu.iter().zip(gpu.iter()) {
        if !g.is_finite() {
            non_finite_gpu += 1;
        }
        let err = (c - g).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
    }

    println!("dim={} eps={}", dim, eps);
    println!("max_abs_err={:.10}", max_abs_err);
    println!("gpu_non_finite_count={}", non_finite_gpu);
    println!("cpu_first10={:?}", &cpu[0..10]);
    println!("gpu_first10={:?}", &gpu[0..10]);

    assert_eq!(non_finite_gpu, 0, "GPU RMSNorm produced non-finite values");
    assert!(
        max_abs_err < 1e-5,
        "Final norm algebraic mismatch: max_abs_err={} (expected < 1e-5)",
        max_abs_err
    );

    Ok(())
}
