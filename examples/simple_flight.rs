//! Simple Flight: A minimal CLI to verify Airframe + Shimmytok integration.
//!
//! Usage: cargo run -p airframe --example simple_flight -- "Hello now is the time"

use airframe::core::model::Model;
use airframe::family::llama::LlamaModel;
use airframe::runtime::{engine::Engine, sampling::Sampler};
use shimmytok::Tokenizer;
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let prompt = if args.len() > 1 {
        &args[1]
    } else {
        "Hello, world!"
    };

    let model_path = std::env::var("LIBSHIMMY_MODEL_PATH").unwrap_or_else(|_| {
        "../../llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf".to_string()
    });

    println!("✈️  SIMPLE FLIGHT CHECK");
    println!("---------------------");
    println!("Model: {}", model_path);
    println!("Prompt: \"{}\"", prompt);

    // 1. Tokenize (using shimmytok, which is re-exported or dep of airframe?)
    // Note: Airframe deps on shimmytok, so we use that.
    let tokenizer = Tokenizer::from_gguf_file(&model_path)?;
    let tokens = tokenizer.encode(prompt, true)?; // Add special tokens
    println!("Tokens: {:?}", tokens);

    // 2. Load Model (Airframe)
    println!("Loading Airframe...");
    let model = Model::from_tinylama_q4_0_gguf(Path::new(&model_path))?;
    println!("Weights loaded: {}", model.weights.len());

    // 3. Initialize Engine
    let llama_model = LlamaModel::from_spec(model.spec.clone());
    let mut engine = Engine::new(Box::new(llama_model));
    let sampler = Sampler::greedy();

    // 4. Prefill
    let prompt_ids: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
    let mut logits = engine.prefill(&prompt_ids, &model.weights)?;

    print!("\nOutput: {}", prompt);
    std::io::stdout().flush()?;

    // 5. Decode Loop
    for _ in 0..32 {
        let next_token = sampler.sample(&logits)?;

        // Break on EOS
        if next_token == tokenizer.eos_token() as usize {
            break;
        }

        let token_str = tokenizer.decode(&[next_token as u32], true).unwrap();
        print!("{}", token_str);
        std::io::stdout().flush()?;

        logits = engine.decode(next_token, &model.weights)?;
    }
    println!("\n\n✅ Flight Complete.");
    Ok(())
}
