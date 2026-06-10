//! Configuration for shimmy console
//!
//! Config file lives at: ~/.shimmy/config.toml
//!
//! Example:
//! ```toml
//! default_theme = "arcade"
//! default_model_path = "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"
//!
//! [model_dirs]
//! extra = [
//!     "D:/shimmy-test-models/gguf_collection",
//! ]
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Console configuration — loaded from ~/.shimmy/config.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Path to history database
    pub history_db: PathBuf,
    /// Path to session database
    pub session_db: PathBuf,
    /// Backend URL for inference (used when NOT in local mode)
    pub backend_url: String,
    /// Discovery port for finding backends
    pub discovery_port: u16,
    /// Default model name (display name, used to match discovered models)
    pub default_model: Option<String>,
    /// Direct path to a specific GGUF file — bypasses discovery, uses LocalInferenceAdapter
    pub default_model_path: Option<PathBuf>,
    /// Extra directories to scan for .gguf files (in addition to standard paths)
    pub model_dirs: Vec<PathBuf>,
    /// Default theme name (e.g. "arcade")
    pub default_theme: String,
    /// Maximum context window tokens
    pub max_context_tokens: usize,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("shimmy");

        Self {
            history_db: data_dir.join("history.db"),
            session_db: data_dir.join("sessions.db"),
            backend_url: "http://localhost:11435".to_string(),
            discovery_port: 11430,
            default_model: None,
            default_model_path: None,
            model_dirs: Vec::new(),
            default_theme: "arcade".to_string(),
            max_context_tokens: 8192,
            debug: false,
        }
    }
}

impl Config {
    /// Standard config file path: ~/.shimmy/config.toml
    pub fn config_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".shimmy")
            .join("config.toml")
    }

    /// Load from ~/.shimmy/config.toml, falling back to defaults if not found
    pub fn load() -> Self {
        let path = Self::config_path();
        if path.exists() {
            Self::from_file(&path).unwrap_or_else(|e| {
                eprintln!("⚠️  Config parse error ({}): {} — using defaults", path.display(), e);
                Self::default()
            })
        } else {
            Self::default()
        }
    }

    /// Load from environment variables, then override with file if present
    pub fn from_env() -> Self {
        let mut config = Self::load();

        if let Ok(url) = std::env::var("SHIMMY_BACKEND_URL") {
            config.backend_url = url;
        }
        if let Ok(port) = std::env::var("SHIMMY_DISCOVERY_PORT") {
            if let Ok(p) = port.parse() {
                config.discovery_port = p;
            }
        }
        if let Ok(model) = std::env::var("SHIMMY_DEFAULT_MODEL") {
            config.default_model = Some(model);
        }
        if let Ok(path) = std::env::var("SHIMMY_MODEL_PATH") {
            config.default_model_path = Some(PathBuf::from(path));
        }
        if let Ok(dirs) = std::env::var("SHIMMY_MODEL_DIRS") {
            for d in dirs.split(';') {
                let p = PathBuf::from(d.trim());
                if !config.model_dirs.contains(&p) {
                    config.model_dirs.push(p);
                }
            }
        }
        if let Ok(tokens) = std::env::var("SHIMMY_MAX_CONTEXT_TOKENS") {
            if let Ok(t) = tokens.parse() {
                config.max_context_tokens = t;
            }
        }
        config.debug = std::env::var("SHIMMY_DEBUG").is_ok();
        config
    }

    /// Load configuration from a TOML file
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to ~/.shimmy/config.toml
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(&path, contents)?;
        Ok(())
    }

    /// Returns true if a local model path is configured (skip server, use LocalInferenceAdapter)
    pub fn has_local_model(&self) -> bool {
        if let Some(ref p) = self.default_model_path {
            p.exists()
        } else {
            false
        }
    }

    /// All directories to scan for GGUF models (standard + configured extras)
    pub fn all_model_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = standard_model_dirs();
        for d in &self.model_dirs {
            if !dirs.contains(d) {
                dirs.push(d.clone());
            }
        }
        dirs
    }
}

/// Standard locations to scan for .gguf files across platforms
pub fn standard_model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // Current directory models/
    dirs.push(PathBuf::from("models"));

    // Home-based locations
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".shimmy").join("models"));
        dirs.push(home.join(".cache").join("huggingface").join("hub"));
        dirs.push(home.join(".cache").join("lm-studio").join("models"));
        dirs.push(home.join(".ollama").join("models"));
        dirs.push(home.join("models"));
    }

    // Data-local locations
    if let Some(data) = dirs::data_local_dir() {
        dirs.push(data.join("shimmy").join("models"));
        dirs.push(data.join("lm-studio").join("models"));
        dirs.push(data.join("ollama").join("models"));
    }

    dirs
}

/// Scan a list of directories for .gguf files, return (display_name, full_path) pairs
pub fn discover_gguf_files(dirs: &[PathBuf]) -> Vec<(String, PathBuf)> {
    let mut results = Vec::new();
    for dir in dirs {
        if !dir.exists() { continue; }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    let name = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    results.push((name, path));
                }
            }
        }
    }
    results
}
