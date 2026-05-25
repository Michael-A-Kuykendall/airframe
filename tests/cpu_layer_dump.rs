//! CPU Layer-by-layer diagnostic for notebook analysis
//! Outputs JSON matching GPU trace format for direct comparison
//!
//! NOTE: Currently disabled - needs MultiTokenEngine API updates.
//! Use layer_dump_gpu binary instead.

// Placeholder test - real implementation requires Engine API refactor
#[test]
fn test_cpu_layer_dump_placeholder() {
    // CPU layer dump needs Engine::capture_layers() - deferred to Phase 2.2+
    println!("CPU layer dump stubbed - use layer_dump_gpu binary");
}

/*
// DISABLED - Requires private field access
#[test]
#[ignore]
fn OLD_test_cpu_layer_dump_for_notebook() {
    println!("\n=== CPU Layer Dump for Notebook Analysis ===\n");

    // Load model
    let model_path = PathBuf::from("D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    println!("Loading model: {:?}", model_path);

    let model_data = Model::from_tinylama_q4_0_gguf(&model_path).expect("Failed to load model");
    let llama_model = LlamaModel::from_spec(model_data.spec.clone());
    let mut engine = MultiTokenEngine::new(llama_model);

    // Process sequence: BOS (1), "Hello" (15043)
    let tokens = vec![1usize, 15043usize];

    let mut all_results = serde_json::json!({
        "test": "cpu_layer_dump",
        "tokens": tokens,
        "positions": [],
    });

    println!("Processing {} token positions...", tokens.len());

    // === Process each token position individually ===
    for (pos_idx, &token_id) in tokens.iter().enumerate() {
        println!("\n--- Position {} (Token {}) ---", pos_idx, token_id);

        let cache_len_before = engine.engine.kv_cache.len();

        // Get embedding directly from weights
        // We need to manually compute embedding since we can't access embed_tokens
        // Token embedding is (n_vocab x dim) matrix, row-major
        let token_embed_weight = model_data.weights.get(&airframe::core::weight_id::WeightId::TokenEmbed)
            .expect("token_embed weight not found");

        let dim = model_data.spec.n_embd;
        let embedding: Vec<f32> = token_embed_weight.data[token_id * dim..(token_id + 1) * dim].to_vec();

        // Run forward pass for just this one token
        let logits = engine.engine.prefill(&[token_id], &model_data.weights)
            .expect(&format!("Failed to process token {}", token_id));

        let cache_len_after = engine.engine.kv_cache.len();

        // For CPU, we don't have layer-by-layer outputs easily accessible
        // So we'll just record embedding and final logits
        // To get layer outputs, we'd need to modify LlamaModel to expose them
        let position_data = serde_json::json!({
            "position": pos_idx,
            "token_id": token_id,
            "cache_len_before": cache_len_before,
            "embedding": &embedding[0..4],
            "cache_len_after": cache_len_after,
            "final_logits": &logits.data[0..8],
        });

        println!("  Embedding: [{:.8}, {:.8}, {:.8}, {:.8}]",
            embedding[0], embedding[1], embedding[2], embedding[3]
        );
        println!("  Final logits: [{:.8}, {:.8}, {:.8}, {:.8}]",
            logits.data[0], logits.data[1], logits.data[2], logits.data[3]
        );

        all_results["positions"].as_array_mut().unwrap().push(position_data);
    }

    // === Write output ===
    let output_path = PathBuf::from("artifacts/cpu_layer_dump.json");
    println!("\n✅ Writing results to: {:?}", output_path);

    let mut file = File::create(&output_path).expect("Failed to create output file");
    let json_str = serde_json::to_string_pretty(&all_results).expect("Failed to serialize JSON");
    file.write_all(json_str.as_bytes()).expect("Failed to write JSON");

    println!("✅ CPU layer dump complete!");
    println!("\nTo compare GPU vs CPU:");
    println!("  - GPU: artifacts/gpu_layer_dump.json");
    println!("  - CPU: artifacts/cpu_layer_dump.json");
    println!("\nRun the notebook cell to see the comparison.");
    println!("\nNOTE: CPU trace only includes embedding and final logits.");
    println!("For full layer-by-layer comparison, use the existing CSV traces.");
}
*/
