//! Vault oracle loading.
//! Reads pre-extracted JSON seed files from `vault/seeds/<model_name>.json`.
//! These are produced by `vault_seed` and imported by `import_seeds.py`.

use crate::facts::{InferenceFact, KEY_VAULT_ORACLE};

/// A single oracle record from a seed file.
#[derive(Clone, Debug, serde::Deserialize)]
struct OracleRecord {
    layer_idx: i32,
    operation: String,
    position: u32,
    hidden_state_rms: Option<f32>,
    checksum: Option<i64>,
}

/// A vault seed file.
#[derive(Clone, Debug, serde::Deserialize)]
struct SeedFile {
    model: ModelMeta,
    oracles: Vec<OracleRecord>,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct ModelMeta {
    name: String,
}

/// Find the seed file path for a given model name.
/// Checks `vault/seeds/<name>.json` and `vault/seeds/<name>.json` with
/// common name variations (lowercase, underscore vs hyphen).
fn find_seed_path(base_dir: &str, model_name: &str) -> Option<String> {
    let candidates = vec![
        format!("{}/{}.json", base_dir, model_name),
        format!("{}/{}.json", base_dir, model_name.to_lowercase()),
        format!("{}/{}.json", base_dir, model_name.replace('-', "_")),
        format!("{}/{}.json", base_dir, model_name.to_lowercase().replace('-', "_")),
    ];
    for path in candidates {
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }
    None
}

/// Load `VaultOracle` facts for a given model from its JSON seed file.
/// `base_dir`: directory containing seed files (e.g., `vault/seeds`).
/// `model_name`: model name to match (used to find the JSON file).
///
/// Returns one `InferenceFact::VaultOracle` per oracle record.
pub fn load_vault_oracles(base_dir: &str, model_name: &str) -> Result<Vec<InferenceFact>, String> {
    let path = find_seed_path(base_dir, model_name)
        .ok_or_else(|| format!("No seed file found for model '{}' in {}", model_name, base_dir))?;

    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read seed file '{}': {}", path, e))?;

    let seed: SeedFile = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse seed file '{}': {}", path, e))?;

    let mut facts = Vec::new();
    for oracle in &seed.oracles {
        if oracle.operation != "layer_output" && oracle.operation != "final_logits" {
            continue;
        }
        let rms = oracle.hidden_state_rms.unwrap_or(0.0);
        let cs = oracle.checksum.unwrap_or(0);
        facts.push(InferenceFact::VaultOracle {
            model_id: 0,
            layer_idx: oracle.layer_idx,
            position: oracle.position,
            expected_rms_bits: rms.to_bits(),
            checksum: cs,
        });
    }

    Ok(facts)
}

/// Get the set of model names that have seed files in the given directory.
pub fn list_models(base_dir: &str) -> Result<Vec<String>, String> {
    let dir = std::path::Path::new(base_dir);
    if !dir.is_dir() {
        return Err(format!("Not a directory: {}", base_dir));
    }

    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("Failed to read dir: {}", e))? {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                names.push(stem.to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vault_oracle_key_is_22() {
        assert_eq!(KEY_VAULT_ORACLE, 22);
    }

    #[test]
    fn load_tinyllama_oracles() {
        let base_dir = "vault/seeds";
        if !std::path::Path::new(base_dir).is_dir() {
            eprintln!("Skipping: vault/seeds directory not found");
            return;
        }
        let facts = load_vault_oracles(base_dir, "tinyllama_q4_0").unwrap_or_else(|e| {
            // Try alternative names
            load_vault_oracles(base_dir, "TinyLlama-1.1B-Chat-v1.0.Q4_0")
                .expect(&format!("Fallback also failed: {}", e))
        });
        assert!(!facts.is_empty(), "Should have at least one oracle");
        for fact in &facts {
            match fact {
                InferenceFact::VaultOracle { expected_rms_bits, checksum, .. } => {
                    let rms = f32::from_bits(*expected_rms_bits);
                    assert!(rms >= 0.0, "RMS should be non-negative");
                    assert!(*checksum != 0 || rms == 0.0, "Checksum should be non-zero for non-zero RMS");
                }
                _ => panic!("Expected VaultOracle facts"),
            }
        }
    }
}
