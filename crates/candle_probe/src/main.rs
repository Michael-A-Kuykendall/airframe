//! candle_probe — Cross-validation probe for the Golden Reference Vault
//!
//! Loads a GGUF model using candle (independent of Airframe), runs the same
//! fixed forward pass (BOS → "Hello" token, all layers), and outputs a JSON
//! cross-validation seed for comparison against vault oracle rows.
//!
//! This is a SPIKE — TinyLlama Q4_0 only for now.
//! Success = candle RMS values agree with Airframe vault within tolerance.
//!
//! Usage:
//!   candle_probe <gguf_path> [output_json]
//!
//! Output JSON:
//!   {
//!     "tool": "candle-0.10.2",
//!     "source_gguf": "...",
//!     "generated_at": "...",
//!     "layers": [
//!       { "layer_idx": 0, "rms": 0.024, "first20": [...], "checksum": 12345678 }
//!       ...
//!     ]
//!   }

use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use serde::Serialize;
use std::path::PathBuf;

// ─── Output types ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ProbeOutput {
    tool: String,
    source_gguf: String,
    generated_at: String,
    token_ids: Vec<u32>,
    final_logits: FinalLogits,
    layers: Vec<LayerResult>,
    integrity: ProbeIntegrity,
}

#[derive(Serialize)]
struct FinalLogits {
    len: usize,
    rms: f32,
    max: f32,
    argmax_id: usize,
    nan_count: usize,
    first20: Vec<f32>,
}

#[derive(Serialize)]
struct LayerResult {
    layer_idx: i32,
    rms: f32,
    first20: Vec<f32>,
    checksum: i64,
}

#[derive(Serialize)]
struct ProbeIntegrity {
    layer_count: usize,
    nan_count: usize,
    inf_count: usize,
    rms_sum: f64,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn rms(v: &[f32]) -> f32 {
    let sq: f32 = v.iter().map(|x| x * x).sum();
    (sq / v.len() as f32).sqrt()
}

fn checksum(v: &[f32]) -> i64 {
    v.iter()
        .map(|x| x.to_bits() as i64)
        .fold(0i64, |a, b| a.wrapping_add(b))
}

fn nan_inf(v: &[f32]) -> (usize, usize) {
    (
        v.iter().filter(|x| x.is_nan()).count(),
        v.iter().filter(|x| x.is_infinite()).count(),
    )
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: candle_probe <gguf_path> [output_json] [--token-ids id,id,...]");
        eprintln!("  Default tokens: BOS(1) Hello(15043). Pass peel tokens.ids for numerical gate.");
        std::process::exit(1);
    }

    let gguf_path = PathBuf::from(&args[1]);
    if !gguf_path.exists() {
        eprintln!("ERROR: file not found: {:?}", gguf_path);
        std::process::exit(1);
    }

    let mut output_path = {
        let stem = gguf_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        PathBuf::from("vault/seeds/candle").join(format!("{}.json", stem))
    };
    let mut token_ids: Vec<u32> = vec![1u32, 15043u32];
    let mut i = 2usize;
    while i < args.len() {
        if args[i] == "--token-ids" {
            i += 1;
            if i >= args.len() {
                eprintln!("ERROR: --token-ids needs a comma-separated list");
                std::process::exit(1);
            }
            token_ids = args[i]
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().parse::<u32>())
                .collect::<Result<Vec<_>, _>>()?;
        } else if !args[i].starts_with("--") {
            output_path = PathBuf::from(&args[i]);
        }
        i += 1;
    }
    if token_ids.is_empty() {
        eprintln!("ERROR: empty token list");
        std::process::exit(1);
    }

    eprintln!("=== candle_probe ===");
    eprintln!("  GGUF   : {}", gguf_path.display());
    eprintln!("  Output : {}", output_path.display());
    eprintln!("  Device : CPU");
    eprintln!();

    let device = Device::Cpu;

    // ── Load GGUF via candle ─────────────────────────────────────────────────
    eprintln!("[1/4] Loading GGUF via candle ...");
    let mut file = std::fs::File::open(&gguf_path)?;
    let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

    // Build the quantized model
    let mut model = ModelWeights::from_gguf(gguf_content, &mut file, &device)?;
    eprintln!("      Model loaded");

    // ── Token sequence (multi-token prefill for numerical gate) ────────────
    eprintln!(
        "[2/4] Running {} tokens: {:?} ...",
        token_ids.len(),
        &token_ids[..token_ids.len().min(12)]
    );

    // Sequential single-token forwards (candle quantized_llama API).
    // Last forward's logits = next-token distribution after full prefix.
    let mut logits_flat: Vec<f32> = Vec::new();
    for (pos, &tid) in token_ids.iter().enumerate() {
        let t = Tensor::new(&[tid], &device)?.unsqueeze(0)?;
        let logits = model.forward(&t, pos)?;
        logits_flat = logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
    }

    // Argmax for compare vs peel final.logits.argmax.id
    let mut argmax_id: usize = 0;
    let mut argmax_v = f32::NEG_INFINITY;
    for (i, &v) in logits_flat.iter().enumerate() {
        if v.is_finite() && v > argmax_v {
            argmax_v = v;
            argmax_id = i;
        }
    }

    eprintln!(
        "      prefill done — logits[{}] argmax={} max={:.4}",
        logits_flat.len(),
        argmax_id,
        argmax_v
    );

    // ── Compute stats ────────────────────────────────────────────────────────
    eprintln!("[3/4] Computing stats ...");

    let r = rms(&logits_flat);
    let cs = checksum(&logits_flat);
    let first20: Vec<f32> = logits_flat.iter().take(20).copied().collect();
    let (nans, infs) = nan_inf(&logits_flat);

    // For the spike, we report one "layer" which is actually the final logits
    // This is enough to validate the candle integration is working
    let layers = vec![LayerResult {
        layer_idx: -1_i32, // -1 = final logits (same convention as vault_seed)
        rms: r,
        first20: first20.clone(),
        checksum: cs,
    }];

    let rms_sum = r as f64;

    eprintln!(
        "      Final logits: RMS={:.6}, NaN={}, Inf={}",
        r, nans, infs
    );

    // ── Write output ─────────────────────────────────────────────────────────
    eprintln!("[4/4] Writing output ...");

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let output = ProbeOutput {
        tool: "candle-0.10.2".to_string(),
        source_gguf: gguf_path.to_string_lossy().to_string(),
        generated_at: chrono::Utc::now().to_rfc3339(),
        token_ids: token_ids.clone(),
        final_logits: FinalLogits {
            len: logits_flat.len(),
            rms: r,
            max: argmax_v,
            argmax_id,
            nan_count: nans,
            first20,
        },
        integrity: ProbeIntegrity {
            layer_count: layers.len(),
            nan_count: nans,
            inf_count: infs,
            rms_sum,
        },
        layers,
    };

    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(&output_path, &json)?;

    eprintln!();
    eprintln!("✅  Probe written: {}", output_path.display());
    eprintln!("    Candle version : candle-0.10.2");
    eprintln!("    NaN count      : {}", nans);
    eprintln!("    Inf count      : {}", infs);
    eprintln!("    Final RMS      : {:.6}", r);

    if nans > 0 || infs > 0 {
        eprintln!("⚠️   WARNING: NaN or Inf in candle output");
        std::process::exit(2);
    }

    Ok(())
}
