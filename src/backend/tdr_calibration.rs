//! TDR Calibration Cache — per-(pipeline, n_embd) safe workgroup limits.
//!
//! Calibration measures actual GPU time for a range of workgroup counts using
//! the real shader + real model weight buffers, then caches the largest safe
//! dispatch size for each pipeline/dimension combination.
//!
//! Cache is stored at:
//!   Windows: %LOCALAPPDATA%/Airframe/tdr-calibration.json
//!   Linux:   ~/.cache/airframe/tdr-calibration.json
//!   Override: $AIRFRAME_CACHE_DIR/tdr-calibration.json
//!
//! ## Cache file structure
//! ```json
//! {
//!   "version": 1,
//!   "gpu": "NVIDIA GeForce RTX 3060",
//!   "last_updated": "2026-06-18T12:00:00Z",
//!   "pipelines": {
//!     "head_blob": {
//!       "2048": {
//!         "safe_workgroups": 512,
//!         "measured_ms": 1180,
//!         "budget_ms": 1400
//!       }
//!     }
//!   }
//! }
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level calibration cache.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalibrationCache {
    pub version: u32,
    pub gpu: String,
    pub last_updated: String,
    pub pipelines: HashMap<String, HashMap<String, PipelineCalibration>>,
}

/// Per-pipeline, per-dimension calibration result.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PipelineCalibration {
    /// Largest workgroup count that completed within budget.
    pub safe_workgroups: u32,
    /// Actual GPU time measured at safe_workgroups (ms).
    pub measured_ms: u64,
    /// TDR budget used for this calibration (ms).
    pub budget_ms: u64,
}

impl CalibrationCache {
    pub fn empty(gpu_name: &str) -> Self {
        Self {
            version: 1,
            gpu: gpu_name.to_string(),
            last_updated: String::new(),
            pipelines: HashMap::new(),
        }
    }

    /// Look up safe workgroups for a (pipeline_name, n_embd) combination.
    /// Returns None if no calibration exists for that key.
    pub fn get_safe_workgroups(&self, pipeline: &str, n_embd: u32) -> Option<u32> {
        self.pipelines
            .get(pipeline)
            .and_then(|dims| dims.get(&n_embd.to_string()))
            .map(|cal| cal.safe_workgroups)
    }

    /// Store a calibration result for (pipeline, n_embd).
    pub fn set_calibration(
        &mut self,
        pipeline: &str,
        n_embd: u32,
        safe_workgroups: u32,
        measured_ms: u64,
        budget_ms: u64,
    ) {
        let entry = PipelineCalibration {
            safe_workgroups,
            measured_ms,
            budget_ms,
        };
        self.pipelines
            .entry(pipeline.to_string())
            .or_default()
            .insert(n_embd.to_string(), entry);
    }
}

/// Determine the cache file path.
pub fn cache_path() -> PathBuf {
    if let Ok(dir) = std::env::var("AIRFRAME_CACHE_DIR") {
        let mut p = PathBuf::from(dir);
        p.push("tdr-calibration.json");
        return p;
    }
    #[cfg(windows)]
    {
        let dir = std::env::var("LOCALAPPDATA")
            .unwrap_or_else(|_| "C:\\Users\\Default\\AppData\\Local".to_string());
        let mut p = PathBuf::from(dir);
        p.push("Airframe");
        p.push("tdr-calibration.json");
        p
    }
    #[cfg(not(windows))]
    {
        let dir = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        let mut p = PathBuf::from(dir);
        p.push(".cache");
        p.push("airframe");
        p.push("tdr-calibration.json");
        p
    }
}

/// Load the calibration cache from disk. Returns None if no cache exists.
pub fn load_cache() -> Option<CalibrationCache> {
    let path = cache_path();
    if !path.exists() {
        return None;
    }
    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Save the calibration cache to disk. Creates parent directories if needed.
pub fn save_cache(cache: &CalibrationCache) -> Result<(), String> {
    let path = cache_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create cache dir: {}", e))?;
    }
    let data = serde_json::to_string_pretty(cache)
        .map_err(|e| format!("Failed to serialize cache: {}", e))?;
    std::fs::write(&path, &data).map_err(|e| format!("Failed to write cache: {}", e))?;
    Ok(())
}

/// Remove the calibration cache from disk.
pub fn clear_cache() -> Result<(), String> {
    let path = cache_path();
    if path.exists() {
        std::fs::remove_file(&path).map_err(|e| format!("Failed to remove cache: {}", e))?;
    }
    Ok(())
}

/// Ensure calibration exists for a given (pipeline, n_embd) combination.
///
/// Returns the safe number of workgroups for a single dispatch of this
/// pipeline/dimension pair. Uses disk cache if available; if not, returns a
/// conservative default and logs a message.
///
/// When GPU timestamp queries (airframe-mbt) land, this function will also
/// run the adaptive calibration sweep on first-use.
pub fn ensure_calibrated(gpu_name: &str, pipeline: &str, n_embd: u32) -> u32 {
    let mut cache = load_cache().unwrap_or_else(|| CalibrationCache::empty(gpu_name));

    if let Some(safe) = cache.get_safe_workgroups(pipeline, n_embd) {
        return safe;
    }

    // No cache entry — use a conservative default.
    // The adaptive calibration sweep will replace this when GPU timestamp
    // queries are available (bead airframe-mbt).
    let conservative_default: u32 = 512;
    eprintln!(
        "[TDR-CAL] no calibration for {} dim={}; using conservative default {} WGs. \
         Run --recalibrate-tdr after timestamp queries land for optimal splitting.",
        pipeline, n_embd, conservative_default
    );

    cache.set_calibration(pipeline, n_embd, conservative_default, 0, 0);
    let _ = save_cache(&cache);
    conservative_default
}
