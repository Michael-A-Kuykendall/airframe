// Layer-by-layer GPU diagnostic for notebook analysis
// Outputs JSON matching CPU golden trace format for direct comparison

#[cfg(test)]
mod layer_dump_tests {
    use super::super::kv_cache::KVCache;
    use super::super::loader::BindlessModel;
    use super::super::pipeline::{BindlessPipeline, LayerParams};
    use crate::core::spec::ModelSpec;
    use std::fs::File;
    use std::io::Write;
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
    async fn test_dump_gpu_layers_for_notebook() {
        println!("\n=== GPU Layer Dump for Notebook Analysis ===\n");

        // Load model
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
                println!("[SKIP] SHIMMY_BASE_GGUF not set and no model found at known paths — skipping layer dump test");
                return;
            }
        };
        println!("Loading model: {:?}", model_path);

        let (device, queue) = get_device().await;
        let spec = ModelSpec::tinylama_1_1b_chat_v1_0();
        let model = BindlessModel::load_from_disk(&device, &model_path, Some(&spec));
        let pipeline = BindlessPipeline::new(&device);

        // Initialize KV cache (n_layers, n_head_kv, head_dim, max_seq_len)
        let mut kv_cache = KVCache::new(&device, 22, 4, 64, 2048);

        let embd_weight_offset = model
            .metadata
            .get_tensor_offset("token_embd.weight")
            .expect("token_embd.weight not found");

        let dim = 2048u32;
        let row_bytes = (dim / 32) * 18;

        let layer_params = LayerParams {
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
        };

        // Process sequence: BOS (1), "Hello" (15043), then 1 decode step
        let tokens = vec![1u32, 15043u32];

        let mut all_results = serde_json::json!({
            "test": "gpu_layer_dump",
            "tokens": tokens,
            "positions": [],
        });

        // === PREFILL: Process tokens 0 and 1 ===
        for (pos, &token_id) in tokens.iter().enumerate() {
            println!("\n--- Position {} (Token {}) ---", pos, token_id);

            // Get embedding
            let row_offset = embd_weight_offset + (token_id as u64 * row_bytes as u64);
            let mut layer_input =
                pipeline.run_dequant_request(&device, &queue, &model, row_offset as u32, dim);

            let mut position_data = serde_json::json!({
                "position": pos,
                "token_id": token_id,
                "cache_len_before": kv_cache.get_seq_len(),
                "embedding": &layer_input[0..4],
                "layers": [],
            });

            // Run all 22 layers
            for layer_idx in 0..22 {
                let layer_offsets = model
                    .metadata
                    .get_layer_offsets(layer_idx, "tinyllama")
                    .unwrap_or_else(|| panic!("Layer {} offsets not found", layer_idx));

                layer_input = pipeline.run_layer_with_cache(
                    &device,
                    &queue,
                    &model,
                    &mut kv_cache,
                    layer_idx,
                    &layer_input,
                    layer_offsets,
                    layer_params,
                );

                // Record first 8 values for analysis
                position_data["layers"]
                    .as_array_mut()
                    .unwrap()
                    .push(serde_json::json!({
                        "layer": layer_idx,
                        "output": &layer_input[0..8],
                    }));

                if layer_idx == 0 || layer_idx == 21 {
                    println!(
                        "  L{}: [{:.8}, {:.8}, {:.8}, {:.8}]",
                        layer_idx, layer_input[0], layer_input[1], layer_input[2], layer_input[3]
                    );
                }
            }

            // Increment cache
            kv_cache.increment();
            position_data["cache_len_after"] = serde_json::json!(kv_cache.get_seq_len());

            all_results["positions"]
                .as_array_mut()
                .unwrap()
                .push(position_data);
        }

        // === DECODE: 1 generation step ===
        println!("\n--- Decode Step 0 (Position 2) ---");

        // Use last layer output as input (no embedding lookup for generated token yet)
        // For real decode, we'd argmax logits → get token → embed. Simplified here.
        let token_id = 29892u32; // Comma (from server logs)
        let row_offset = embd_weight_offset + (token_id as u64 * row_bytes as u64);
        let mut layer_input =
            pipeline.run_dequant_request(&device, &queue, &model, row_offset as u32, dim);

        let mut position_data = serde_json::json!({
            "position": 2,
            "token_id": token_id,
            "cache_len_before": kv_cache.get_seq_len(),
            "embedding": &layer_input[0..4],
            "layers": [],
        });

        for layer_idx in 0..22 {
            let layer_offsets = model
                .metadata
                .get_layer_offsets(layer_idx, "tinyllama")
                .unwrap_or_else(|| panic!("Layer {} offsets not found", layer_idx));

            layer_input = pipeline.run_layer_with_cache(
                &device,
                &queue,
                &model,
                &mut kv_cache,
                layer_idx,
                &layer_input,
                layer_offsets,
                layer_params,
            );

            position_data["layers"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({
                    "layer": layer_idx,
                    "output": &layer_input[0..8],
                }));

            if layer_idx == 0 || layer_idx == 21 {
                println!(
                    "  L{}: [{:.8}, {:.8}, {:.8}, {:.8}]",
                    layer_idx, layer_input[0], layer_input[1], layer_input[2], layer_input[3]
                );
            }
        }

        all_results["positions"]
            .as_array_mut()
            .unwrap()
            .push(position_data);

        // === Write to artifacts for notebook ===
        let output_path =
            PathBuf::from("C:/Users/micha/repos/libshimmy/artifacts/gpu_layer_dump.json");
        let mut file = File::create(&output_path).expect("Failed to create output file");
        let json_str = serde_json::to_string_pretty(&all_results).unwrap();
        file.write_all(json_str.as_bytes())
            .expect("Failed to write JSON");

        println!("\n=== Output written to: {:?} ===", output_path);
        println!("Run notebook cell 2 to analyze GPU vs CPU divergence");
    }
}
