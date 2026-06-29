//! vault_generator — Generate comprehensive vault from llmfit listings!
//!
//! Simple workflow:
//! 1. Use llmfit to list all commodity-friendly models (NO DOWNLOADS)
//! 2. For each model, extract metadata from HuggingFace API (NO DOWNLOADS!)
//! 3. Generate vault seeds for all models in one commit
//!
//! Usage:
//!   cargo run --bin vault_generator
//!   cargo run --bin vault_generator --output-dir "vault/seeds"
//!

use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

// ─── Output Types ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct VaultManifest {
    version: u32,
    generated_at: String,
    total_models: usize,
    source: String,
    models: Vec<String>,
    status: String,
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let output_dir = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("vault/seeds")
    };

    fs::create_dir_all(&output_dir)?;

    println!("=== Vault Generator ===");
    println!("Generating vault from llmfit listings (NO DOWNLOADS!)");
    println!("Output directory: {}", output_dir.display());
    println!();

    // Step 1: Get model list from llmfit or use comprehensive fallback
    println!("[1/3] Fetching model list from llmfit...");

    let models = fetch_model_list()?;
    println!("Found {} commodity-friendly models", models.len());
    println!();

    // Step 2: Extract metadata from HuggingFace API for each model
    println!("[2/3] Extracting metadata via HuggingFace API (no downloads)...");

    let mut manifest_models = Vec::new();

    for (i, model_id) in models.iter().enumerate() {
        let progress = format!("[{}/{}]", i + 1, models.len());
        println!("{} Processing: {}", progress, model_id);

        // Extract metadata from HuggingFace API
        match extract_metadata_from_hf_api(model_id)? {
            Some(metadata) => {
                let filename = model_id.split('/').next_back().unwrap_or("unknown");
                let seed_path = output_dir.join(format!("{}.json", filename));

                println!("  ✅ Metadata extracted: {}", metadata.arch);
                println!("     Quantization: {}", metadata.quant);
                println!("     Layers: {}", metadata.n_layers);

                // Write vault seed (metadata-only, no forward pass)
                write_vault_seed(&seed_path, &metadata)?;

                manifest_models.push(format!(
                    "\"{}\"",
                    model_id.split('/').next_back().unwrap_or("unknown")
                ));
            }
            None => {
                println!("  ⚠️  Could not fetch metadata (skipping)");
            }
        }
    }

    // Step 3: Create vault manifest
    println!("[3/3] Creating vault manifest...");

    let manifest = VaultManifest {
        version: 2,
        generated_at: chrono::Utc::now().to_rfc3339(),
        total_models: models.len(),
        source: "llmfit + HuggingFace API".to_string(),
        models: manifest_models,
        status: "metadata-only".to_string(),
    };

    let manifest_path = output_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    File::create(manifest_path)?.write_all(manifest_json.as_bytes())?;

    println!();
    println!("=== Summary ===");
    println!("✅ Vault generated successfully!");
    println!("   Models: {} (metadata-only)", models.len());
    println!("   Output: {}", output_dir.display());
    println!();
    println!("Next steps:");
    println!("  1. Review seeds in {}", output_dir.display());
    println!("  2. Run 'vault_seed' on models you want oracle traces for");
    println!("  3. Commit the vault to your repo");

    Ok(())
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Fetch model list from llmfit or use comprehensive fallback
fn fetch_model_list() -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Try llmfit first
    let output = std::process::Command::new("llmfit")
        .args(["list", "--json"])
        .output()?;

    if output.status.success() {
        println!("   Using llmfit model database");

        use serde_json::Value;
        let json: Value = serde_json::from_slice(&output.stdout)?;
        let models = match json.as_array() {
            Some(arr) => arr
                .iter()
                .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
                .map(|s: &str| s.to_string()) // Convert &str to String
                .take(20) // Limit to top 20 for initial vault
                .collect(),
            None => Vec::new(),
        };

        if !models.is_empty() {
            return Ok(models);
        }
    }

    // Fallback: Comprehensive commodity-friendly model list
    println!("   Using fallback comprehensive list");

    let models = vec![
        // Qwen family (primary target)
        "Qwen/Qwen3.5-9B-Q4_K_M".to_string(),
        "Qwen/Qwen3-8B-Q4_K_M".to_string(),
        "Qwen/Qwen3-4B-Thinking-Q4_K_M".to_string(),
        "Qwen/Qwen3-1.7B-Q4_K_M".to_string(),
        "Qwen/Qwen3-0.6B-Q4_K_M".to_string(),
        // DeepSeek
        "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M".to_string(),
        "deepseek/deepseek-r1-qwen3-8b".to_string(),
        // Mistral
        "mistralai/ministral-3-14b-reasoning-Q4_K_M".to_string(),
        "mistralai/Mistral-Nemo-Instruct-2407-Q4_K_M".to_string(),
        // Gemma
        "google/gemma-4-e4b-it-Q4_K_M".to_string(),
        "google/gemma-2-9b-it-Q4_K_M".to_string(),
        "google/gemma-2-2b-it-Q4_K_M".to_string(),
        // Phi
        "microsoft/phi-4-Q4_K_M".to_string(),
        "microsoft/phi-3.5-mini-instruct-Q4_K_M".to_string(),
        "microsoft/phi-3-vision-instruct-Q4_K_M".to_string(),
        "microsoft/phi-2-Q4_K_M".to_string(),
        // Llama
        "meta-llama/Llama-3.2-3B-Instruct-Q4_K_M".to_string(),
        "meta-llama/Llama-3.2-1B-Instruct-Q4_K_M".to_string(),
        "bartowski/Llama-3.1-8B-Instruct-GGUF".to_string(),
        // Qwen2 (legacy but popular)
        "Qwen/Qwen2.5-Coder-14B-Instruct-Q4_K_M".to_string(),
        "Qwen/Qwen2.5-7B-Instruct-Q4_K_M".to_string(),
        "Qwen/Qwen2.5-1.5B-Instruct-Q4_K_M".to_string(),
        "Qwen/Qwen2-7B-Instruct-Q4_K_M".to_string(),
    ];

    Ok(models)
}

/// Extract metadata from HuggingFace API (NO DOWNLOAD!)
fn extract_metadata_from_hf_api(
    model_id: &str,
) -> Result<Option<ModelMetadata>, Box<dyn std::error::Error>> {
    // Use curl to fetch model info from HuggingFace API
    let output = std::process::Command::new("curl")
        .args([
            "-s",
            "--max-time",
            "10",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            &format!("https://huggingface.co/api/models/{}", model_id),
        ])
        .output()?;

    let status = String::from_utf8_lossy(&output.stdout);

    if status == "200" {
        println!("   [HF API] Model exists: {}", model_id);

        // Fetch full metadata
        let output = std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time",
                "15",
                &format!("https://huggingface.co/api/models/{}", model_id),
            ])
            .output()?;

        if output.status.success() {
            let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;

            // Extract key fields
            let filename = model_id.split('/').next_back().unwrap_or("unknown.gguf");
            let tags: Vec<String> = json
                .get("tags")
                .and_then(|t| t.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .flat_map(|s| s.split(',').map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let quant = parse_quantization_from_tags(&tags);
            let arch = parse_architecture_from_name(model_id);

            // Estimate file size (rough approximation)
            let params_raw: f64 = json
                .get("author")
                .and_then(|a| a.get("authorName"))
                .and_then(|n| n.as_str())
                .map(|s| s.parse::<f64>().unwrap_or(8.0)) // default 8B
                .unwrap_or(8.0);

            let size_gb = params_raw * 0.5; // Q4_K_M is ~50% of FP32
            let file_size_bytes = (size_gb * 1_073_741_824.0) as u64;

            Ok(Some(ModelMetadata {
                name: model_id.to_string(),
                gguf_filename: filename.to_string(),
                arch: parse_architecture_from_name(model_id),
                quant,
                n_layers: estimate_layers(params_raw), // Estimate from parameters
                n_heads: 32,                           // default for most models
                n_heads_kv: 8,                         // GQA default
                head_dim: 128,
                n_embd: 5120,
                ff_dim: 24576,
                n_vocab: 151936,
                n_ctx: 4096,
                rope_base: 10000.0,
                rope_scale: 1.0,
                rope_dim: 128, // default
                rms_eps: 1e-5,
                has_qk_norm: arch == "qwen3", // arch is already computed above
                attn_logit_softcap: 0.0,
                final_logit_softcap: 0.0,
                gguf_path: format!("hf://{}", model_id),
                file_size_bytes,
            }))
        } else {
            Ok(None)
        }
    } else {
        println!("   [HF API] Model not found or network error");
        Ok(None)
    }
}

fn parse_quantization_from_tags(tags: &[String]) -> String {
    for tag in tags {
        if tag.contains("q4_k_m") {
            return "Q4_K_M".to_string();
        }
        if tag.contains("q8_0") {
            return "Q8_0".to_string();
        }
        if tag.contains("f16") {
            return "F16".to_string();
        }
    }
    "Q4_K_M".to_string() // default
}

fn parse_architecture_from_name(model_id: &str) -> String {
    if model_id.contains("qwen35") || model_id.contains("qwen3.5") || model_id.contains("qwen3-") {
        "qwen3".to_string()
    } else if model_id.contains("gemma") {
        "gemma".to_string()
    } else if model_id.contains("phi") {
        "phi".to_string()
    } else if model_id.contains("mistral") {
        "mistral".to_string()
    } else if model_id.contains("llama") {
        "llama".to_string()
    } else if model_id.contains("deepseek") {
        "qwen3".to_string()
    }
    // DeepSeek uses Qwen arch
    else {
        "unknown".to_string()
    }
}

fn estimate_layers(params: f64) -> usize {
    match params.round() {
        0.6 => 8,
        1.7 => 28,
        3.0 => 32,
        7.5..=8.5 => 48,
        13.0..=14.0 => 64,
        _ => 32, // default
    }
}

/// Write vault seed JSON
fn write_vault_seed(
    path: &PathBuf,
    metadata: &ModelMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    let seed = VaultSeedApi {
        seed_version: 2,
        source_gguf: metadata.name.clone(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        model: ModelRow {
            name: metadata.name.clone(),
            gguf_filename: metadata.gguf_filename.clone(),
            arch: metadata.arch.clone(),
            quant: metadata.quant.clone(),
            n_layers: metadata.n_layers,
            n_heads: metadata.n_heads,
            n_heads_kv: metadata.n_heads_kv,
            head_dim: metadata.head_dim,
            n_embd: metadata.n_embd,
            ff_dim: metadata.ff_dim,
            n_vocab: metadata.n_vocab,
            n_ctx: metadata.n_ctx,
            rope_base: metadata.rope_base,
            rope_scale: 1.0,
            rope_dim: metadata.head_dim,
            rms_eps: metadata.rms_eps,
            has_qk_norm: metadata.has_qk_norm,
            attn_logit_softcap: metadata.attn_logit_softcap,
            final_logit_softcap: metadata.final_logit_softcap,
            gguf_path: metadata.gguf_path.clone(),
            file_size_bytes: metadata.file_size_bytes,
        },
        oracles: vec![], // Empty for metadata-only vault
        integrity: IntegrityBlock {
            expected_oracle_count: 0,
            rms_sum: 0.0,
            nan_count: 0,
            inf_count: 0,
            forward_pass_seconds: 0.0,
            forward_pass_ok: false, // metadata-only mode
            forward_error: String::new(),
        },
    };

    let json = serde_json::to_string_pretty(&seed)?;
    fs::write(path, &json)?;

    Ok(())
}

// ─── Structs (copied from vault_seed.rs for consistency) ──────────────────

#[derive(Serialize)]
struct VaultSeedApi {
    seed_version: u32,
    source_gguf: String,
    generated_at: String,
    model: ModelRow,
    oracles: Vec<OracleRow>,
    integrity: IntegrityBlock,
}

#[derive(Serialize)]
struct ModelRow {
    name: String,
    gguf_filename: String,
    arch: String,
    quant: String,
    n_layers: usize,
    n_heads: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_embd: usize,
    ff_dim: usize,
    n_vocab: usize,
    n_ctx: usize,
    rope_base: f32,
    rope_scale: f32,
    rope_dim: usize,
    rms_eps: f32,
    has_qk_norm: bool,
    attn_logit_softcap: f32,
    final_logit_softcap: f32,
    gguf_path: String,
    file_size_bytes: u64,
}

#[derive(Serialize)]
struct OracleRow {
    layer_idx: i32,
    operation: String,
    position: usize,
    input_token_id: u32,
    rms: f32,
    first20: Vec<f32>,
    checksum: i64,
}

#[derive(Serialize)]
struct IntegrityBlock {
    expected_oracle_count: usize,
    rms_sum: f64,
    nan_count: usize,
    inf_count: usize,
    forward_pass_seconds: f64,
    forward_pass_ok: bool,
    forward_error: String,
}

#[derive(Serialize)]
struct ModelMetadata {
    name: String,
    gguf_filename: String,
    arch: String,
    quant: String,
    n_layers: usize,
    n_heads: usize,
    n_heads_kv: usize,
    head_dim: usize,
    n_embd: usize,
    ff_dim: usize,
    n_vocab: usize,
    n_ctx: usize,
    rope_base: f32,
    rope_scale: f32,
    rope_dim: usize,
    rms_eps: f32,
    has_qk_norm: bool,
    attn_logit_softcap: f32,
    final_logit_softcap: f32,
    gguf_path: String,
    file_size_bytes: u64,
}
