//! vault_seed — Golden Reference Vault population tool
//!
//! Loads a GGUF file, runs a fixed CPU forward pass (BOS → "Hello"),
//! captures per-layer hidden state RMS and first-20 values, then writes
//! a JSON seed file ready for DuckDB import.
//!
//! Usage:
//!   vault_seed <gguf_path> [output_json]
//!
//! Output JSON contains:
//!   - model metadata row (for models table)
//!   - layer oracle rows (for layer_oracles table)
//!   - integrity check block (row counts, checksums)
//!
//! Example:
//!   vault_seed "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"
//!   vault_seed "D:/shimmy-test-models/gguf_collection/Qwen3-8B-Q4_K_M.gguf" vault/seeds/qwen3-8b.json

use airframe::core::model::Model;
use airframe::core::spec::{GgufFileType, ModelArch};
use airframe::core::tensor::Tensor;
use airframe::core::weight_id::WeightId;
use airframe::family::llama::LlamaBlock;
use airframe::ops::dispatch::OpDispatcher;
use airframe::runtime::kvcache::KvCache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ─── Output types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct VaultSeed {
    /// Tool version — bump when output format changes
    seed_version: u32,
    /// Absolute path of the GGUF file this seed was generated from
    source_gguf: String,
    /// When this seed was generated (ISO 8601)
    generated_at: String,
    /// Model metadata row (maps directly to vault `models` table)
    model: ModelRow,
    /// Layer oracle rows (maps to vault `layer_oracles` table)
    oracles: Vec<OracleRow>,
    /// Integrity block — checked by import script before any INSERT
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
    layer_idx: i32,   // -1 = embedding, 0..n_layer-1 = transformer layer
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
    /// Sum of all RMS values — cheap cross-check after import
    rms_sum: f64,
    /// Number of NaN values found in any oracle — must be 0
    nan_count: usize,
    /// Number of Inf values found in any oracle — must be 0
    inf_count: usize,
    /// Wall-clock seconds for the forward pass
    forward_pass_seconds: f64,
    /// Whether the forward pass completed (false = metadata-only seed)
    forward_pass_ok: bool,
    /// Error message if forward pass failed
    forward_error: String,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn rms(v: &[f32]) -> f32 {
    let sq_sum: f32 = v.iter().map(|x| x * x).sum();
    (sq_sum / v.len() as f32).sqrt()
}

/// Row-wise checksum: reinterpret each f32 as its bit pattern (u32), sum as i64.
/// Deterministic across runs; catches silent value corruption.
fn checksum(v: &[f32]) -> i64 {
    v.iter()
        .map(|x| x.to_bits() as i64)
        .fold(0i64, |acc, b| acc.wrapping_add(b))
}

fn nan_inf_count(v: &[f32]) -> (usize, usize) {
    let nans = v.iter().filter(|x| x.is_nan()).count();
    let infs = v.iter().filter(|x| x.is_infinite()).count();
    (nans, infs)
}

fn embed_token(token_id: usize, embed_weight: &Tensor, n_embd: usize) -> Tensor {
    let start = token_id * n_embd;
    Tensor {
        data: embed_weight.data[start..start + n_embd].to_vec(),
        shape: vec![1, n_embd],
    }
}

fn arch_str(arch: &ModelArch) -> &'static str {
    match arch {
        ModelArch::Llama => "llama",
        ModelArch::Mistral => "mistral",
        ModelArch::Phi => "phi",
        ModelArch::Gemma => "gemma",
        ModelArch::Qwen2 => "qwen2",
        ModelArch::Qwen3 => "qwen3",
        ModelArch::Other(_) => "other",
    }
}

fn quant_str(ft: &GgufFileType) -> &'static str {
    match ft {
        GgufFileType::F32 => "f32",
        GgufFileType::F16 => "f16",
        GgufFileType::Q4_0 => "q4_0",
        GgufFileType::Q4_1 => "q4_1",
        GgufFileType::Q5_0 => "q5_0",
        GgufFileType::Q5_1 => "q5_1",
        GgufFileType::Q8_0 => "q8_0",
        GgufFileType::Q2_K => "q2_k",
        GgufFileType::Q3_K => "q3_k",
        GgufFileType::Q4_K => "q4_k_m",
        GgufFileType::Q5_K => "q5_k_m",
        GgufFileType::Q6_K => "q6_k",
        GgufFileType::Unknown => "unknown",
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: vault_seed <gguf_path> [output_json]");
        eprintln!("  gguf_path   : path to the GGUF file");
        eprintln!("  output_json : where to write the seed (default: vault/seeds/<model>.json)");
        std::process::exit(1);
    }

    let gguf_path = PathBuf::from(&args[1]);
    if !gguf_path.exists() {
        eprintln!("ERROR: file not found: {:?}", gguf_path);
        std::process::exit(1);
    }

    let filename = gguf_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown.gguf");

    let stem = gguf_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let output_path = if args.len() >= 3 {
        PathBuf::from(&args[2])
    } else {
        PathBuf::from("vault/seeds").join(format!("{}.json", stem))
    };

    eprintln!("=== vault_seed ===");
    eprintln!("  GGUF   : {}", gguf_path.display());
    eprintln!("  Output : {}", output_path.display());
    eprintln!();

    // ── 1. Load model (dequantizes weights to F32) ──────────────────────────
    eprintln!("[1/5] Loading model ...");
    let file_size = std::fs::metadata(&gguf_path)?.len();
    let load_start = Instant::now();
    let model_result = Model::from_gguf(&gguf_path);
    let (model, spec, load_error) = match model_result {
        Ok(m) => {
            let spec = m.spec.clone();
            eprintln!(
                "      arch={} quant={} layers={} embd={} ({:.1}s)",
                arch_str(&spec.arch),
                quant_str(&spec.file_type),
                spec.n_layer,
                spec.n_embd,
                load_start.elapsed().as_secs_f64()
            );
            (Some(m), Some(spec), String::new())
        }
        Err(e) => {
            let err_str = format!("{:?}", e);
            eprintln!("      load failed: {}", err_str);
            (None, None, err_str)
        }
    };

    // If load failed, write a metadata-only seed and exit cleanly
    if model.is_none() {
        eprintln!("[!] Model load failed — writing error record");
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let seed = VaultSeed {
            seed_version: 1,
            source_gguf: gguf_path.to_string_lossy().to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            model: ModelRow {
                name: filename.to_string(),
                gguf_filename: filename.to_string(),
                arch: "unknown".to_string(),
                quant: "unknown".to_string(),
                n_layers: 0, n_heads: 0, n_heads_kv: 0, head_dim: 0,
                n_embd: 0, ff_dim: 0, n_vocab: 0, n_ctx: 0,
                rope_base: 0.0, rope_scale: 0.0, rope_dim: 0, rms_eps: 0.0,
                has_qk_norm: false, attn_logit_softcap: 0.0, final_logit_softcap: 0.0,
                gguf_path: gguf_path.to_string_lossy().to_string(),
                file_size_bytes: file_size,
            },
            oracles: vec![],
            integrity: IntegrityBlock {
                expected_oracle_count: 0,
                rms_sum: 0.0,
                nan_count: 0,
                inf_count: 0,
                forward_pass_seconds: 0.0,
                forward_pass_ok: false,
                forward_error: load_error,
            },
        };
        let json = serde_json::to_string_pretty(&seed)?;
        std::fs::write(&output_path, &json)?;
        eprintln!("✅  Seed written (load-failed): {}", output_path.display());
        return Ok(());
    }

    let model = model.unwrap();
    let spec = spec.unwrap();

    // ── 2. Set up layers, ops, KV cache ─────────────────────────────────────
    eprintln!("[2/5] Setting up forward pass ...");
    let layers: Vec<LlamaBlock> = (0..spec.n_layer)
        .map(|i| LlamaBlock::new(i, spec.clone()))
        .collect();
    let ops = OpDispatcher;
    let mut kv_cache = KvCache::new(
        spec.n_ctx.min(4096), // cap to 4096 for oracle gen — we only need 2 positions
        spec.n_layer,
        spec.n_head_kv,
        spec.n_embd / spec.n_head,
    );

    let embed_weight = model
        .weights
        .get(&WeightId::TokenEmbed)
        .ok_or("token_embd.weight not found in model")?;

    // ── 3. BOS prefill at position 0 (warms KV cache) ───────────────────────
    eprintln!("[3/5] BOS prefill (position 0) ...");
    let bos_id = 1usize;
    let mut hidden = embed_token(bos_id, embed_weight, spec.n_embd);
    let pos0 = vec![0usize];
    let mut forward_pass_ok = true;
    let mut forward_error = String::new();

    for layer in &layers {
        match layer.forward(&hidden, &model.weights, &mut kv_cache, &pos0, &ops) {
            Ok(h) => hidden = h,
            Err(e) => {
                forward_pass_ok = false;
                forward_error = format!("{:?}", e);
                eprintln!("  ⚠️  BOS prefill failed at layer {}: {}", layer.layer_idx, forward_error);
                break;
            }
        }
    }
    if forward_pass_ok {
        let _ = kv_cache.complete_decode();
    }

    // ── 4. Capture oracles at position 1 ("Hello" token) ────────────────────
    eprintln!("[4/5] Oracle capture (position 1) ...");
    let hello_id = 15043u32;
    let mut oracles: Vec<OracleRow> = Vec::new();
    let mut total_nan = 0usize;
    let mut total_inf = 0usize;
    let mut rms_sum = 0f64;
    let fwd_start = Instant::now();

    if forward_pass_ok {
        let mut hidden = embed_token(hello_id as usize, embed_weight, spec.n_embd);
        let pos1 = vec![1usize];
        let mut kv_cache2 = KvCache::new(
            spec.n_ctx.min(4096),
            spec.n_layer,
            spec.n_head_kv,
            spec.n_embd / spec.n_head,
        );
        // Re-run BOS to warm the second cache
        let bos_hidden = embed_token(bos_id, embed_weight, spec.n_embd);
        let mut bos_h = bos_hidden;
        for layer in &layers {
            match layer.forward(&bos_h, &model.weights, &mut kv_cache2, &pos0, &ops) {
                Ok(h) => bos_h = h,
                Err(_) => { forward_pass_ok = false; break; }
            }
        }
        if forward_pass_ok {
            let _ = kv_cache2.complete_decode();
            for layer in &layers {
                match layer.forward(&hidden, &model.weights, &mut kv_cache2, &pos1, &ops) {
                    Ok(h) => {
                        hidden = h;
                        let r = rms(&hidden.data);
                        let cs = checksum(&hidden.data);
                        let first20: Vec<f32> = hidden.data.iter().take(20).copied().collect();
                        let (nans, infs) = nan_inf_count(&hidden.data);
                        total_nan += nans;
                        total_inf += infs;
                        rms_sum += r as f64;
                        oracles.push(OracleRow {
                            layer_idx: layer.layer_idx as i32,
                            operation: "layer_output".to_string(),
                            position: 1,
                            input_token_id: hello_id,
                            rms: r,
                            first20,
                            checksum: cs,
                        });
                    }
                    Err(e) => {
                        forward_pass_ok = false;
                        forward_error = format!("{:?}", e);
                        eprintln!("  ⚠️  forward failed at layer {}: {}", layer.layer_idx, forward_error);
                        break;
                    }
                }
            }
        }
    }

    let fwd_elapsed = fwd_start.elapsed().as_secs_f64();
    eprintln!(
        "      {} layers captured, NaN={}, Inf={} ({:.1}s){}",
        oracles.len(),
        total_nan,
        total_inf,
        fwd_elapsed,
        if !forward_pass_ok { " [PARTIAL - forward pass failed]" } else { "" }
    );

    // ── 5. Assemble and write seed ───────────────────────────────────────────
    eprintln!("[5/5] Writing seed ...");

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let model_name = spec
        .model_name
        .trim_start_matches("tinyllama_")
        .to_string();

    let seed = VaultSeed {
        seed_version: 1,
        source_gguf: gguf_path.to_string_lossy().to_string(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        model: ModelRow {
            name: model_name,
            gguf_filename: filename.to_string(),
            arch: arch_str(&spec.arch).to_string(),
            quant: quant_str(&spec.file_type).to_string(),
            n_layers: spec.n_layer,
            n_heads: spec.n_head,
            n_heads_kv: spec.n_head_kv,
            head_dim: spec.head_dim,
            n_embd: spec.n_embd,
            ff_dim: spec.ff_dim,
            n_vocab: spec.n_vocab,
            n_ctx: spec.n_ctx,
            rope_base: spec.rope_base,
            rope_scale: spec.rope_scale,
            rope_dim: spec.rope_dim,
            rms_eps: spec.rms_eps,
            has_qk_norm: spec.has_qk_norm,
            attn_logit_softcap: spec.attn_logit_softcap,
            final_logit_softcap: spec.final_logit_softcap,
            gguf_path: gguf_path.to_string_lossy().to_string(),
            file_size_bytes: file_size,
        },
        integrity: IntegrityBlock {
            expected_oracle_count: oracles.len(),
            rms_sum,
            nan_count: total_nan,
            inf_count: total_inf,
            forward_pass_seconds: fwd_elapsed,
            forward_pass_ok,
            forward_error,
        },
        oracles,
    };

    let json = serde_json::to_string_pretty(&seed)?;
    std::fs::write(&output_path, &json)?;

    eprintln!();
    eprintln!("✅  Seed written: {}", output_path.display());
    eprintln!("    Models row       : 1");
    eprintln!("    Oracle rows      : {}", seed.integrity.expected_oracle_count);
    eprintln!("    Forward pass OK  : {}", seed.integrity.forward_pass_ok);
    if !seed.integrity.forward_error.is_empty() {
        eprintln!("    Forward error    : {}", seed.integrity.forward_error);
    }
    eprintln!("    NaN count        : {}", seed.integrity.nan_count);
    eprintln!("    Inf count        : {}", seed.integrity.inf_count);
    if seed.integrity.expected_oracle_count > 0 {
        eprintln!("    RMS sum          : {:.6}", seed.integrity.rms_sum);
    }
    if seed.integrity.nan_count > 0 || seed.integrity.inf_count > 0 {
        eprintln!("⚠️   WARNING: NaN or Inf values detected — do not import this seed");
        std::process::exit(2);
    }

    Ok(())
}
