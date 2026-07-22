//! Bead V3 — Divergence localizer (diagnostic, REPORT-ONLY).
//!
//! For every model that has golden `layer_oracles` rows in the local vault
//! (`vault/vault.duckdb`), this harness:
//!   1. materializes the vault from the local git-lfs cache if needed,
//!   2. runs `invariant_probe` (built with `--features isf`) on the model's
//!      golden fixture `[BOS, Hello]` (prompt "Hello", add_special=true),
//!   3. feeds the captured per-layer RMS + the vault oracle RMS through an
//!      `ObservationSession` registered with `register_certification_default`
//!      (the V2 spec-vs-vault certification rule), and
//!   4. reports, per model, PASS or the FIRST divergent layer
//!      (`model=X layer=Y obs_rms=.. exp_rms=.. delta=..`).
//!
//! This is a DIAGNOSTIC, not a gate: it prints results and NEVER panics. The
//! certification authority is the V2 fabric rule (vault RMS vs GPU RMS), which
//! is itself anchored to the P2 spec-math gate. Golden traces (vault oracles)
//! are used ONLY as a localization hint — never as the source of truth.
//!
//! The vault is read via the `duckdb` CLI (external binary, no C compilation,
//! dev/test only). No `duckdb` Rust crate, no `vault/seeds` JSON detour.
//!
//! Run:  cargo test --features isf --test test_invariants -- --test-threads=1

use airframe_observe::facts::InferenceFact;
use airframe_observe::observers::FINAL_LOGITS_LAYER;
use airframe_observe::session::ObservationSession;
use std::path::{Path, PathBuf};
use std::process::Command;

/// All models populated in `layer_oracles`. Models without a GGUF file on disk
/// are gracefully skipped at test time (deepseek-coder, LLaMA v2 have no local
/// GGUF; they are simply reported SKIP).
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
    position: u32,
    expected_rms: f64,
    expected_max: f64,
    expected_nan: i64,
    expected_inf: i64,
    checksum: i64,
}

fn query_vault_rows(vault: &Path, model_name: &str, quant: &str) -> Vec<VaultRow> {
    let sql = format!(
        "SELECT lo.layer_idx, lo.operation, lo.position, lo.expected_rms, lo.expected_max, \
         lo.expected_nan, lo.expected_inf, lo.checksum \
         FROM layer_oracles lo JOIN models m ON m.id=lo.model_id \
         WHERE m.name='{}' AND m.quant='{}' ORDER BY lo.layer_idx",
        model_name, quant
    );
    let out = Command::new("duckdb")
        .arg(vault.as_os_str())
        .arg("-json")
        .arg(&sql)
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[certify] WARN: duckdb CLI unavailable ({}), skipping", e);
            return vec![];
        }
    };
    if !out.status.success() {
        eprintln!(
            "[certify] WARN: duckdb query failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        return vec![];
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    if stdout.trim().is_empty() {
        return vec![];
    }
    serde_json::from_str(&stdout).unwrap_or_else(|e| {
        eprintln!("[certify] WARN: parse duckdb json failed: {}", e);
        vec![]
    })
}

/// Run `invariant_probe` for the model. Returns `None` if the probe fails
/// (model load / GPU error) so the harness can SKIP gracefully — report-only,
/// never panics.
fn run_probe(model_path: &str, model_tag: &str) -> Option<ProbeOut> {
    let bin = env!("CARGO_BIN_EXE_invariant_probe");
    let out = Command::new(bin).arg(model_path).arg(model_tag).output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "[certify] SKIP {} (cannot launch invariant_probe: {})",
                model_tag, e
            );
            return None;
        }
    };
    if !out.status.success() {
        eprintln!(
            "[certify] SKIP {} (invariant_probe failed: {})",
            model_tag,
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("")
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json_line = stdout
        .lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .last();
    match json_line {
        Some(line) => match serde_json::from_str::<ProbeOut>(line) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!(
                    "[certify] SKIP {} (probe JSON unparsable: {})",
                    model_tag, e
                );
                None
            }
        },
        None => {
            eprintln!("[certify] SKIP {} (probe produced no JSON)", model_tag);
            None
        }
    }
}

#[test]
fn localize_divergence_across_vault_models() {
    // Diagnostic / report-only: this harness must never panic. A divergent
    // model is reported, not failed.
    let vault = ensure_vault();
    eprintln!("[certify] vault at {}", vault.display());

    let mut n_pass = 0u32;
    let mut n_div = 0u32;
    let mut n_skip = 0u32;

    for (tag, path) in MODELS {
        if !Path::new(path).exists() {
            eprintln!("[certify] SKIP {} (gguf not present)", tag);
            n_skip += 1;
            continue;
        }
        let (name, quant) = tag.split_once('|').unwrap_or((tag, ""));
        let rows = query_vault_rows(&vault, name, quant);
        if rows.is_empty() {
            eprintln!("[certify] SKIP {} (no vault oracle rows)", tag);
            n_skip += 1;
            continue;
        }

        let captured = match run_probe(path, tag) {
            Some(c) => c,
            None => {
                n_skip += 1;
                continue;
            }
        };

        // Feed captures + oracles through the V2 certification fabric.
        let mut session = ObservationSession::new();
        session.register_certification_default();

        // Oracle reference facts (vault) — skip rows flagged nan/inf.
        for r in &rows {
            if r.expected_nan != 0 || r.expected_inf != 0 {
                continue;
            }
            session.emit(InferenceFact::VaultOracle {
                model_id: 0,
                layer_idx: r.layer_idx as i32,
                position: r.position,
                expected_rms_bits: (r.expected_rms as f32).to_bits(),
                checksum: r.checksum,
            });
        }

        // Live captures (GPU) at the golden position.
        for l in &captured.layers {
            if l.position != 1 {
                continue;
            }
            session.emit(InferenceFact::LayerOutput {
                layer_idx: l.layer_idx,
                position: l.position,
                rms_bits: l.rms.to_bits(),
                checksum: l.checksum,
            });
        }
        if let Some(fl) = &captured.final_logits {
            session.emit(InferenceFact::FinalLogits {
                position: fl.position,
                rms_bits: fl.rms.to_bits(),
                checksum: fl.checksum,
            });
        }

        session.saturate();
        let results = session.certification().unwrap().drain();
        let fails: Vec<_> = results.iter().filter(|r| !r.passed).collect();

        if fails.is_empty() {
            eprintln!(
                "[certify] {} PASS (all {} covered oracle rows match)",
                tag,
                results.len()
            );
            n_pass += 1;
        } else {
            // FIRST divergent layer = smallest layer_idx (final_logits sorts last).
            let first = fails
                .iter()
                .min_by_key(|r| r.layer_idx)
                .expect("has a failing result");
            let where_str = if first.layer_idx == FINAL_LOGITS_LAYER {
                "final_logits".to_string()
            } else {
                format!("layer {}", first.layer_idx)
            };
            eprintln!(
                "[certify] {} DIVERGENT at {}: obs_rms={:.6} exp_rms={:.6} delta={:.3}",
                tag, where_str, first.observed_rms, first.expected_rms, first.rel_delta
            );
            n_div += 1;
        }
    }

    eprintln!(
        "[certify] SUMMARY: pass={} divergent={} skip={}",
        n_pass, n_div, n_skip
    );
}
