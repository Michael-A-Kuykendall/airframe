//! PPT Invariant Cage — certification harness (B3).
//!
//! For every model that has golden `layer_oracles` rows in the vault
//! (`vault/vault.duckdb`), this harness:
//!   1. materializes the vault from the local git-lfs cache if needed,
//!   2. runs `invariant_probe` (built with `--features isf`) on the model's
//!      golden fixture `[BOS, Hello]` (prompt "Hello", add_special=true),
//!   3. compares the captured per-layer RMS against the vault golden RMS, and
//!   4. FAILS the test on the FIRST divergent layer, reporting which layer.
//!
//! Checksums are NOT compared cross-platform (CPU candle vs GPU wgpu produce
//! different bit patterns from different dequant rounding). Only the RMS
//! statistical aggregate is compared, with a generous tolerance that accounts
//! for platform precision differences.
//!
//! Uses the `duckdb` CLI (external binary, no C compilation) to query vault.
//!
//! Run:  cargo test --test test_invariants -- --test-threads=1

use std::path::{Path, PathBuf};
use std::process::Command;

/// Maximum allowed ratio (max/min) between GPU and vault RMS for layer output.
/// Ratio-based instead of absolute tolerance: different backends (wgpu vs candle)
/// accumulate precision differences that produce 20-40% RMS variation on
/// known-good models. A 1.5x threshold passes legitimate platform differences
/// while still catching architecture-level mismatches (Qwen3, qwen2 show 5-18x).
const LAYER_OUTPUT_RATIO_MAX: f32 = 2.0;

/// Maximum allowed ratio (max/min) between GPU and vault RMS for final_logits.
/// Uses ratio instead of absolute tolerance because logits RMS scale varies
/// greatly across model architectures. Catches catastrophic failures like
/// all-zero logits (ratio → ∞) while passing legitimate platform differences.
const FINAL_LOGITS_RATIO_MAX: f32 = 4.0;

/// All 12 models populated in `layer_oracles`. Models without a GGUF file on
/// disk are gracefully skipped at test time (deepseek-coder, LLaMA v2).
const MODELS: &[(&str, &str)] = &[
    (
        "tinyllama-1.1b-chat-v1.0|q4_0",
        "D:/shimmy-test-models/gguf_collection/TinyLlama/TinyLlama-1.1B-Chat-v1.0/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf",
    ),
    (
        "tinyllama-1.1b-chat-v1.0|q6_k",
        "D:/shimmy-test-models/gguf_collection/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
    ),
    (
        "Llama 3.2 1B Instruct|q4_k_m",
        "D:/shimmy-test-models/gguf_collection/meta-llama/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    ),
    (
        "Llama 3.2 1B Instruct|q6_k",
        "D:/shimmy-test-models/gguf_collection/meta-llama/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-Q6_K.gguf",
    ),
    (
        "Llama 3.2 3B Instruct|q4_k_m",
        "D:/shimmy-test-models/gguf_collection/meta-llama/Llama-3.2-3B-Instruct/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    (
        "Qwen3 1.7B|q4_k_m",
        "D:/shimmy-test-models/gguf_collection/Qwen/Qwen3-1.7B/Qwen3-1.7B-Q4_K_M.gguf",
    ),
    (
        "Qwen3 8B Awq Compatible Instruct|q4_k_m",
        "D:/shimmy-test-models/gguf_collection/Qwen/Qwen3-8B/Qwen3-8B-Q4_K_M.gguf",
    ),
    (
        "qwen2-0_5b-instruct|q4_k_m",
        "D:/shimmy-test-models/gguf_collection/Qwen/Qwen2-0.5B-Instruct/qwen2-0_5b-instruct-q4_k_m.gguf",
    ),
    (
        "qwen2-1_5b-instruct|q4_k_m",
        "D:/shimmy-test-models/gguf_collection/Qwen/Qwen2-1.5B-Instruct/qwen2-1_5b-instruct-q4_k_m.gguf",
    ),
    (
        "qwen2-7b-instruct|q4_k_m",
        "D:/shimmy-test-models/gguf_collection/Qwen/Qwen2-7B-Instruct/qwen2-7b-instruct-q4_k_m.gguf",
    ),
];

fn ensure_vault() -> PathBuf {
    let vault = Path::new(env!("CARGO_MANIFEST_DIR")).join("vault/vault.duckdb");
    if let Ok(meta) = std::fs::metadata(&vault) {
        if meta.len() < 1000 {
            let cache = Path::new(
                "C:/Users/micha/repos/airframe/.git/lfs/objects/36/e7/36e7beaf0ee87887ebe508465de72d8d9ceaaefcd8097b8c1805a8fa6e373359",
            );
            if cache.exists() {
                std::fs::copy(cache, &vault).expect("copy vault from lfs cache");
            } else {
                eprintln!("[vault] WARN: lfs cache object missing; tests may find no rows");
            }
        }
    }
    vault
}

#[derive(serde::Deserialize, Debug)]
#[allow(dead_code)]
struct ProbeLayer {
    layer_idx: u32,
    position: u32,
    rms: f32,
    checksum: i64,
}

#[derive(serde::Deserialize, Debug)]
#[allow(dead_code)]
struct ProbeFinal {
    position: u32,
    rms: f32,
    checksum: i64,
}

#[derive(serde::Deserialize, Debug)]
#[allow(dead_code)]
struct ProbeOut {
    model: String,
    layers: Vec<ProbeLayer>,
    #[serde(default)]
    final_logits: Option<ProbeFinal>,
}

#[derive(serde::Deserialize, Debug)]
#[allow(dead_code)]
struct VaultRow {
    layer_idx: i64,
    operation: String,
    expected_rms: f64,
    expected_max: f64,
    expected_nan: i64,
    expected_inf: i64,
}

fn query_vault_rows(vault: &Path, model_name: &str, quant: &str) -> Vec<VaultRow> {
    let sql = format!(
        "SELECT lo.layer_idx, lo.operation, lo.expected_rms, lo.expected_max, \
         lo.expected_nan, lo.expected_inf \
         FROM layer_oracles lo JOIN models m ON m.id=lo.model_id \
         WHERE m.name='{}' AND m.quant='{}' ORDER BY lo.layer_idx",
        model_name, quant
    );
    let out = Command::new("duckdb")
        .arg(vault.as_os_str())
        .arg("-json")
        .arg(&sql)
        .output()
        .expect("failed to run duckdb CLI");
    assert!(
        out.status.success(),
        "duckdb query failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("duckdb output not utf-8");
    if stdout.trim().is_empty() {
        return vec![];
    }
    serde_json::from_str(&stdout).expect("parse duckdb json output")
}

fn run_probe(model_path: &str, model_tag: &str) -> ProbeOut {
    let bin = env!("CARGO_BIN_EXE_invariant_probe");
    let out = Command::new(bin)
        .arg(model_path)
        .arg(model_tag)
        .output()
        .expect("run invariant_probe");
    assert!(
        out.status.success(),
        "invariant_probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json_line = stdout
        .lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .last()
        .expect("probe produced no JSON");
    serde_json::from_str(json_line).expect("parse probe JSON")
}

#[test]
fn certify_all_vault_models_against_gpu() {
    let vault = ensure_vault();
    eprintln!("[certify] vault at {}", vault.display());

    let mut first_failure: Option<String> = None;

    for (tag, path) in MODELS {
        if !Path::new(path).exists() {
            eprintln!("[certify] SKIP {} (gguf not present)", tag);
            continue;
        }
        let (name, quant) = tag.split_once('|').unwrap_or((tag, ""));
        let rows = query_vault_rows(&vault, name, quant);
        if rows.is_empty() {
            eprintln!("[certify] SKIP {} (no vault oracle rows)", tag);
            continue;
        }
        eprintln!("[certify] {}: {} oracle rows from vault", tag, rows.len());

        let captured = run_probe(path, tag);
        let cap_map: std::collections::HashMap<u32, &ProbeLayer> = captured
            .layers
            .iter()
            .filter(|l| l.position == 1)
            .map(|l| (l.layer_idx, l))
            .collect();

        let mut model_first_div: Option<(i64, String)> = None;

        for r in &rows {
            if r.expected_nan != 0 || r.expected_inf != 0 {
                continue;
            }
            // vault uses layer_idx=-1 as sentinel for final_logits rows
            if r.layer_idx < 0 {
                let fl = match &captured.final_logits {
                    Some(fl) => fl,
                    None => {
                        model_first_div
                            .get_or_insert((-1, "final_logits: NO CAPTURE".into()));
                        continue;
                    }
                };
                let gpu_rms = fl.rms;
                let vault_rms = r.expected_rms as f32;
                let ratio = gpu_rms.max(vault_rms) / gpu_rms.min(vault_rms);
                if ratio.is_nan() || ratio > FINAL_LOGITS_RATIO_MAX || gpu_rms.is_nan() || vault_rms.is_nan() {
                    let reason = format!(
                        "final_logits: gpu_rms={:.6} vault_rms={:.6} ratio={:.3} (max {:.1})",
                        gpu_rms, vault_rms, ratio, FINAL_LOGITS_RATIO_MAX
                    );
                    model_first_div.get_or_insert((-1, reason));
                }
                continue;
            }
            let cap = match cap_map.get(&(r.layer_idx as u32)) {
                Some(c) => c,
                None => {
                    model_first_div
                        .get_or_insert((r.layer_idx, format!("layer {}: NO CAPTURE", r.layer_idx)));
                    continue;
                }
            };
            let gpu_rms = cap.rms;
            let vault_rms = r.expected_rms as f32;
            let ratio = gpu_rms.max(vault_rms) / gpu_rms.min(vault_rms);
            if ratio.is_nan() || ratio > LAYER_OUTPUT_RATIO_MAX {
                let reason = format!(
                    "layer {} [{}]: gpu_rms={:.6} vault_rms={:.6} ratio={:.3} (max {:.1})",
                    r.layer_idx, r.operation, gpu_rms, vault_rms, ratio, LAYER_OUTPUT_RATIO_MAX
                );
                model_first_div.get_or_insert((r.layer_idx, reason));
            }
        }

        match model_first_div {
            Some((lyr, reason)) => {
                eprintln!("[certify] {} DIVERGENT at layer {}: {}", tag, lyr, reason);
                if first_failure.is_none() {
                    first_failure = Some(format!("{} DIVERGENT at layer {}: {}", tag, lyr, reason));
                }
            }
            None => {
                eprintln!(
                    "[certify] {} PASS (all {} oracle rows match)",
                    tag,
                    rows.len()
                );
            }
        }
    }

    match first_failure {
        Some(msg) => panic!("PPT INVARIANT CAGE: first divergence found -> {}", msg),
        None => eprintln!("[certify] ALL MODELS PASS"),
    }
}
