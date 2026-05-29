/// Quick tokenizer probe — run with:
///   cargo run --example tok_probe
/// to see exactly what token IDs the model produces for digit strings.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::var("LIBSHIMMY_MODEL_PATH").unwrap_or_else(|_| {
        "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf".to_string()
    });

    let tok = shimmytok::Tokenizer::from_gguf_file(&model_path)?;

    println!("EOS token id = {}", tok.eos_token());

    // Check im_end
    let im_end = tok.encode("<|im_end|>", false).ok();
    println!("encode('<|im_end|>', false) = {:?}", im_end);

    // Probe digit strings
    for s in &["6", "9", "12", "198", "1000", " 6", " 9", " 198"] {
        let ids = tok.encode(s, false)?;
        let decoded: Vec<String> = ids
            .iter()
            .map(|&id| {
                let raw = tok.decode_single(id, false).unwrap_or_default();
                let skip = tok.decode_single(id, true).unwrap_or_default();
                format!("{}({:?}/{:?})", id, raw, skip)
            })
            .collect();
        println!("encode({:?}, false) = [{}]", s, decoded.join(", "));
    }

    Ok(())
}
