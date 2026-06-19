// Phase 1: GPU Parity Validation Test
// Compares GPU Layer 0 output against CPU golden trace
// Target: Max error < 1e-6 (ideally < 1e-7 per V2.5 report)

#[cfg(test)]
mod parity_tests {
    use super::super::loader::BindlessModel;
    use super::super::pipeline::{BindlessPipeline, LayerParams};
    use crate::core::spec::ModelSpec;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::PathBuf;

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
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_storage_buffer_binding_size as u64;
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage; // INT4 bind group uses 14 bindings; use adapter max
        limits.max_compute_invocations_per_workgroup = 256;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("No device");

        (device, queue)
    }

    #[tokio::test]
    async fn test_gpu_layer0_parity_vs_cpu() {
        // === STEP 1: Load CPU Golden Trace ===
        let csv_path = std::env::var("SHIMMY_L0_TRACE")
            .map(PathBuf::from)
            .or_else(|_| {
                let p = PathBuf::from(
                    "C:/Users/micha/repos/libshimmy/artifacts/shimmy_l0_hello_step11.csv",
                );
                if p.exists() {
                    Ok(p)
                } else {
                    Err("Golden trace not found")
                }
            })
            .expect("SHIMMY_L0_TRACE not set and golden trace not found at default path");

        println!("Loading CPU golden trace: {:?}", csv_path);

        let file = File::open(&csv_path).expect("Failed to open golden trace CSV");
        let reader = BufReader::new(file);

        let mut cpu_output: Option<Vec<f32>> = None;
        let mut cpu_embd_input: Option<Vec<f32>> = None;

        for line in reader.lines() {
            let line = line.expect("Failed to read CSV line");
            let parts: Vec<&str> = line.split(',').collect();

            if parts.len() < 5 {
                continue;
            }

            let checkpoint_id = parts[0];
            let name = parts[2];
            let values_str = parts[4];

            if checkpoint_id == "L0.1" && name == "inp_embd" && cpu_embd_input.is_none() {
                let values: Vec<f32> = values_str
                    .split('|')
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                println!(
                    "[CPU Golden] L0.1 inp_embd: first 4 = {:?}",
                    &values[0..4.min(values.len())]
                );
                cpu_embd_input = Some(values);
            }

            if checkpoint_id == "L0.21" && name == "l_out" {
                let values: Vec<f32> = values_str
                    .split('|')
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();
                println!(
                    "[CPU Golden] L0.21 l_out: first 4 = {:?}",
                    &values[0..4.min(values.len())]
                );
                cpu_output = Some(values);
                break;
            }
        }

        let cpu_output = cpu_output.expect("L0.21 checkpoint not found in CSV");
        let comparison_size = cpu_output.len().min(2048);
        println!(
            "\nCPU golden trace loaded: {} values (will compare first {})",
            cpu_output.len(),
            comparison_size
        );

        // === STEP 2: Run GPU Layer 0 ===
        let model_path = match std::env::var("SHIMMY_BASE_GGUF")
            .map(PathBuf::from)
            .or_else(|_| {
                let candidates = [
                    "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf",
                    "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf",
                ];
                candidates
                    .iter()
                    .find(|p| PathBuf::from(p).exists())
                    .map(PathBuf::from)
                    .ok_or("Model not found")
            }) {
            Ok(p) => p,
            Err(_) => {
                println!("[SKIP] SHIMMY_BASE_GGUF not set and no model found at known paths — skipping GPU parity test");
                return;
            }
        };

        println!("\nLoading Model: {:?}", model_path);
        let (device, queue) = get_device().await;

        let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
        let model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));
        let pipeline = BindlessPipeline::new(&device);

        let embd_weight_offset = model
            .metadata
            .get_tensor_offset("token_embd.weight")
            .expect("token_embd.weight not found");

        let dim = 2048u32;
        let row_bytes = (dim / 32) * 18;
        let token_id = 1u32; // BOS token

        let row_offset = embd_weight_offset + (token_id as u64 * row_bytes as u64);
        let input = pipeline.run_dequant_request(&device, &queue, &model, row_offset as u32, dim);

        println!("[GPU] Input Embedding[0..4]: {:?}", &input[0..4]);

        if let Some(ref cpu_embd) = cpu_embd_input {
            if cpu_embd.len() >= 4 {
                let embd_error = (0..4)
                    .map(|i| (input[i] - cpu_embd[i]).abs())
                    .fold(0.0f32, f32::max);
                println!("Embedding max error (first 4): {:.2e}", embd_error);
            }
        }

        let layer_offsets = model.metadata.get_layer_offsets(0, "tinyllama").unwrap();
        let params = LayerParams {
            dim,
            head_count: 32,
            head_count_kv: 4,
            head_dim: 64,
            rope_dim: 64,
            rms_eps: 1e-5,
            ffn_dim: 5632,
            temp_stride: 16384,
            quant_type: 0,
            attn_logit_softcap: 0.0,
            post_norm_enabled: 0,
            qk_norm_enabled: 0,
            layer_norm_enabled: 0,
            ffn_kind_policy: 0,
            qkv_layout_policy: 0,
            batch_offset: 0,
            batch_count: 0,
            q_weight_k: 0,
            k_weight_k: 0,
        };

        let (mid_vec, gpu_output) = pipeline.run_layer_stepwise_test(
            &device,
            &queue,
            &model,
            &input,
            layer_offsets,
            params,
            true,
        );

        println!("[GPU] Mid (Post-Attn)[0..4]: {:?}", &mid_vec[0..4]);
        println!("[GPU] Final (Post-FFN)[0..4]: {:?}", &gpu_output[0..4]);

        // === STEP 3: Compare GPU vs CPU ===
        let comparison_count = cpu_output.len().min(gpu_output.len());

        println!(
            "\nComparing first {} values (GPU has {}, CPU trace has {})",
            comparison_count,
            gpu_output.len(),
            cpu_output.len()
        );

        let mut max_error = 0.0f32;
        let mut max_error_idx = 0usize;
        let mut errors_over_1e6 = 0usize;
        let mut errors_over_1e5 = 0usize;

        for i in 0..comparison_count {
            let error = (gpu_output[i] - cpu_output[i]).abs();
            if error > max_error {
                max_error = error;
                max_error_idx = i;
            }
            if error > 1e-6 {
                errors_over_1e6 += 1;
            }
            if error > 1e-5 {
                errors_over_1e5 += 1;
            }
        }

        println!("\n=== PARITY COMPARISON (First 4 Values) ===");
        for i in 0..4 {
            let err = (gpu_output[i] - cpu_output[i]).abs();
            println!(
                "  [{}] GPU: {:.8}, CPU: {:.8}, Error: {:.2e}",
                i, gpu_output[i], cpu_output[i], err
            );
        }

        println!("\n=== PARITY STATISTICS ===");
        println!(
            "Compared: {} / {} values",
            comparison_count,
            gpu_output.len()
        );
        println!(
            "Max Absolute Error: {:.2e} (at index {})",
            max_error, max_error_idx
        );
        println!("GPU[{}]: {:.8}", max_error_idx, gpu_output[max_error_idx]);
        println!("CPU[{}]: {:.8}", max_error_idx, cpu_output[max_error_idx]);
        println!(
            "Values with error > 1e-6: {} / {}",
            errors_over_1e6, comparison_count
        );
        println!(
            "Values with error > 1e-5: {} / {}",
            errors_over_1e5, comparison_count
        );

        // === STEP 4: Success Gate ===
        let target_tolerance = 1e-6;
        if max_error < target_tolerance {
            println!("\n*** PARITY TEST PASSED! ***");
            println!(
                "   Max error {:.2e} < target {:.2e}",
                max_error, target_tolerance
            );
            println!(
                "   GPU Layer 0 matches CPU golden trace (first {} values)",
                comparison_count
            );
            println!(
                "\n   Note: Full tensor has {} values, CSV contains first {} for validation",
                gpu_output.len(),
                comparison_count
            );
            println!("      This matches V2.5 methodology (which validated first 4 values)");

            if max_error < 1e-7 {
                println!("   *** EXCELLENT: Error below 1e-7 (matches V2.5 report claim)");
            }
        } else {
            println!("\n*** PARITY TEST FAILED ***");
            println!(
                "   Max error {:.2e} > target {:.2e}",
                max_error, target_tolerance
            );
            println!("   GPU diverges from CPU");
            println!("\nNext Steps:");
            println!("   1. Check GPU_DIVERGENCE_LOG.md for known issues");
            println!("   2. Common culprits:");
            println!("      - Attention scaling factor missing");
            println!("      - RMSNorm epsilon wrong (must be 1e-5)");
            println!("      - Q4_0 dequant formula error");
            panic!("GPU parity test failed with error {:.2e}", max_error);
        }
    }
}
