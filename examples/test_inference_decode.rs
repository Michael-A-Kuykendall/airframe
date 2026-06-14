use shimmytok::Tokenizer;

fn main() {
    let model_path = "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf";
    
    // Load tokenizer directly
    let tokenizer = Tokenizer::from_gguf_file(model_path).expect("Failed to load tokenizer");
    
    // Test decode_single on various tokens
    let test_tokens = [1, 2, 15043, 3186, 29989, 4, 5]; // BOS, EOS, plus normal tokens
    
    println!("Testing decode_single with skip_special=true:");
    for token in test_tokens {
        let result = tokenizer.decode_single(token, true);
        println!("  Token {} -> {:?}", token, result);
    }
    
    println!("\nTesting decode_single with skip_special=false:");
    for token in test_tokens {
        let result = tokenizer.decode_single(token, false);
        println!("  Token {} -> {:?}", token, result);
    }
}
