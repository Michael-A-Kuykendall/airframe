// Quick diagnostic test to print Layer 0 vs Layer 1 offsets

#[tokio::test]
#[ignore]
async fn debug_layer_offsets() -> Result<(), Box<dyn std::error::Error>> {
    use airframe::backend::bindless::metadata::BindlessMetadata;
    use airframe::core::spec::ModelSpec;
    use std::fs::File;
    use std::path::PathBuf;

    let model_path =
        PathBuf::from("C:/Users/micha/repos/llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf");
    let spec = ModelSpec::tinylama_1_1b_chat_v1_0();

    // Load metadata only
    let mut file = File::open(&model_path)?;
    let metadata = BindlessMetadata::new(&mut file);

    println!("\n=== ALGEBRAIC OFFSET VERIFICATION ===\n");

    // Layer 0
    let l0 = metadata.get_layer_offsets(0, "tinyllama").unwrap();
    println!("Layer 0 Offsets:");
    println!("  attn_norm:   {}", l0.attn_norm);
    println!("  attn_q:      {}", l0.attn_q);
    println!("  attn_k:      {}", l0.attn_k);
    println!("  attn_v:      {}", l0.attn_v);
    println!("  attn_out:    {}", l0.attn_out);
    println!("  ffn_norm:    {}", l0.ffn_norm);
    println!("  ffn_gate:    {}", l0.ffn_gate);
    println!("  ffn_up:      {}", l0.ffn_up);
    println!("  ffn_down:    {}", l0.ffn_down);
    println!("  layer_idx:   {}", l0.padding[0]);

    println!("\nLayer 1 Offsets:");
    let l1 = metadata.get_layer_offsets(1, "tinyllama").unwrap();
    println!("  attn_norm:   {}", l1.attn_norm);
    println!("  attn_q:      {}", l1.attn_q);
    println!("  attn_k:      {}", l1.attn_k);
    println!("  attn_v:      {}", l1.attn_v);
    println!("  attn_out:    {}", l1.attn_out);
    println!("  ffn_norm:    {}", l1.ffn_norm);
    println!("  ffn_gate:    {}", l1.ffn_gate);
    println!("  ffn_up:      {}", l1.ffn_up);
    println!("  ffn_down:    {}", l1.ffn_down);
    println!("  layer_idx:   {}", l1.padding[0]);

    println!("\n=== ALGEBRAIC VERIFICATION ===");
    println!("Formula: offset(L1.tensor) SHOULD BE > offset(L0.tensor)");
    println!("\nChecking:");
    println!(
        "  L1.attn_q ({}) > L0.attn_q ({}): {}",
        l1.attn_q,
        l0.attn_q,
        l1.attn_q > l0.attn_q
    );
    println!(
        "  L1.attn_norm ({}) > L0.attn_norm ({}): {}",
        l1.attn_norm,
        l0.attn_norm,
        l1.attn_norm > l0.attn_norm
    );
    println!(
        "  L1.ffn_gate ({}) > L0.ffn_gate ({}): {}",
        l1.ffn_gate,
        l0.ffn_gate,
        l1.ffn_gate > l0.ffn_gate
    );
    println!(
        "  L1.layer_idx ({}) == 1: {}",
        l1.padding[0],
        l1.padding[0] == 1
    );

    // Now check the norm bank directly
    println!("\n=== NORM BANK VALIDATION ===");
    let expected_l0_attn_offset = metadata
        .get_tensor_offset("blk.0.attn_norm.weight")
        .unwrap();
    let expected_l1_attn_offset = metadata
        .get_tensor_offset("blk.1.attn_norm.weight")
        .unwrap();

    println!("Expected file offsets:");
    println!("  blk.0.attn_norm.weight: {}", expected_l0_attn_offset);
    println!("  blk.1.attn_norm.weight: {}", expected_l1_attn_offset);
    println!("\nNorm bank should have extracted FROM these offsets");
    println!("and placed them at norm_bank indices:");
    println!("  Layer 0 attn: index 0 (bytes 0-8191)");
    println!(
        "  Layer 1 attn: index {} (bytes {})",
        1 * 2 * spec.n_embd,
        1 * 2 * spec.n_embd * 4
    );

    println!("\n=== ALGEBRAIC FORMULA CHECK ===");
    println!("Shader formula for Layer 1 attn norm:");
    println!("  norm_offset_base = layer_idx * 2 * dim");
    println!("  = {} * 2 * {}", l1.padding[0], spec.n_embd);
    println!("  = {}", (l1.padding[0] * 2) as usize * spec.n_embd);
    println!(
        "\n✅ If formula produces {}, norm index is CORRECT",
        1 * 2 * spec.n_embd
    );
    println!("❌ If formula produces anything else, THAT'S THE BUG");

    Ok(())
}
