use shimmytok::Tokenizer;
fn main() {
    let models = [
        ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama32"),
        ("TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf", "tinyllama"),
        ("gemma-2-2b-it-Q4_K_M.gguf", "gemma2"),
        ("Phi-3.5-mini-instruct.Q4_K_M.gguf", "phi35"),
        ("phi3-mini-4k-instruct-q4.gguf", "phi3mini"),
        ("qwen2-7b-instruct-q4_k_m.gguf", "qwen2"),
        ("Qwen3-0.6B-Q4_K_M.gguf", "qwen3"),
        ("deepseek-llm-7b-chat.Q4_K_M.gguf", "deepseek"),
    ];
    for (fname, label) in models {
        let path = format!("D:/shimmy-test-models/gguf_collection/{}", fname);
        match Tokenizer::from_gguf_file(&path) {
            Ok(tok) => {
                let bos_id = tok.bos_token();
                let eos_id = tok.eos_token();
                let bos_str = tok.token_to_piece(bos_id).unwrap_or_else(|_| "?".into());
                let eos_str = tok.token_to_piece(eos_id).unwrap_or_else(|_| "?".into());
                println!("{}: bos_id={} bos={:?}  eos_id={} eos={:?}", label, bos_id, bos_str, eos_id, eos_str);
            }
            Err(e) => println!("{}: ERROR {}", label, e),
        }
    }
}
