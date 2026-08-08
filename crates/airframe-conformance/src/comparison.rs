use serde::{Deserialize, Serialize};

/// Comparison result between two engine captures.

/// Comparison status for a single point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PointStatus {
    /// Values match within tolerance.
    Match,
    /// Values exceed tolerance.
    Mismatch,
    /// One or both captures unavailable.
    Unavailable,
    /// Error during comparison.
    Error(String),
}

/// Per-point comparison result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PointComparison {
    /// Point identity.
    pub identity: crate::capture_protocol::PointIdentity,

    /// Comparison status.
    pub status: PointStatus,

    /// Reference (expected) RMS.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_rms: Option<f64>,

    /// Candidate (actual) RMS.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_rms: Option<f64>,

    /// RMS ratio (max/min).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_ratio: Option<f64>,

    /// Tolerance used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,

    /// Reference NaN count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_nan: Option<u64>,

    /// Candidate NaN count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_nan: Option<u64>,

    /// Detailed message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Layer-level comparison summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LayerComparison {
    /// Layer index.
    pub layer: u32,

    /// Overall layer status.
    pub status: LayerStatus,

    /// Per-point comparisons in this layer.
    pub points: Vec<PointComparison>,

    /// Whether RMS is monotonic up to this layer (embedding -> this layer).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monotonic_rms: Option<bool>,

    /// First point where monotonicity breaks (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monotonic_break: Option<String>,
}

/// Overall layer status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerStatus {
    AllMatch,
    HasMismatches,
    Unavailable,
    Error,
}

/// Final logits comparison.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LogitsComparison {
    /// Comparison status.
    pub status: PointStatus,

    /// Reference logits RMS.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_rms: Option<f64>,

    /// Candidate logits RMS.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_rms: Option<f64>,

    /// Logits RMS ratio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_ratio: Option<f64>,

    /// Tolerance used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<f64>,

    /// Reference argmax token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_argmax: Option<u32>,

    /// Candidate argmax token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_argmax: Option<u32>,

    /// Whether argmax matches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argmax_match: Option<bool>,

    /// Top-k overlap count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topk_overlap: Option<u32>,

    /// Reference NaN count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_nan: Option<u64>,

    /// Candidate NaN count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_nan: Option<u64>,
}

/// Complete comparison result for a conformance run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonResult {
    /// Schema identifier: "airframe.conformance.comparison.v1"
    #[serde(rename = "$schema")]
    pub schema: String,

    /// Run ID.
    pub run_id: String,

    /// Conformance version.
    pub conformance_version: String,

    /// Comparison timestamp.
    pub compared_at: String,

    /// Reference engine.
    pub reference_engine: String,

    /// Candidate engine.
    pub candidate_engine: String,

    /// Per-layer comparisons.
    pub layers: Vec<LayerComparison>,

    /// Final logits comparison.
    pub final_logits: LogitsComparison,

    /// Overall result.
    pub overall: OverallResult,

    /// Summary statistics.
    pub summary: ComparisonSummary,
}

/// Overall conformance result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OverallResult {
    Pass,
    Fail,
    Inconclusive,
}

/// Summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonSummary {
    /// Total points compared.
    pub total_points: u32,

    /// Points matching within tolerance.
    pub matched_points: u32,

    /// Points exceeding tolerance.
    pub mismatched_points: u32,

    /// Points unavailable.
    pub unavailable_points: u32,

    /// Layers fully matching.
    pub matching_layers: u32,

    /// Layers with mismatches.
    pub mismatched_layers: u32,

    /// Monotonic RMS layers.
    pub monotonic_layers: u32,

    /// Maximum RMS ratio observed.
    pub max_rms_ratio: f64,

    /// Maximum logits RMS ratio observed.
    pub max_logits_rms_ratio: f64,
}

impl ComparisonResult {
    pub fn new(run_id: String, reference_engine: String, candidate_engine: String) -> Self {
        Self {
            schema: "airframe.conformance.comparison.v1".to_string(),
            run_id,
            conformance_version: crate::CONFORMANCE_VERSION.to_string(),
            compared_at: chrono::Utc::now().to_rfc3339(),
            reference_engine,
            candidate_engine,
            layers: Vec::new(),
            final_logits: LogitsComparison {
                status: PointStatus::Unavailable,
                reference_rms: None,
                candidate_rms: None,
                rms_ratio: None,
                tolerance: None,
                reference_argmax: None,
                candidate_argmax: None,
                argmax_match: None,
                topk_overlap: None,
                reference_nan: None,
                candidate_nan: None,
            },
            overall: OverallResult::Inconclusive,
            summary: ComparisonSummary {
                total_points: 0,
                matched_points: 0,
                mismatched_points: 0,
                unavailable_points: 0,
                matching_layers: 0,
                mismatched_layers: 0,
                monotonic_layers: 0,
                max_rms_ratio: 0.0,
                max_logits_rms_ratio: 0.0,
            },
        }
    }

    pub fn validate(&self) -> Result<(), crate::schemas::ValidationError> {
        crate::schemas::validate_comparison(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_result_roundtrip() {
        let result = ComparisonResult::new(
            "run-123".to_string(),
            "candle".to_string(),
            "airframe_product".to_string(),
        );
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ComparisonResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, parsed);
    }

    #[test]
    fn comparison_schema() {
        let result = ComparisonResult::new(
            "run-123".to_string(),
            "candle".to_string(),
            "airframe_product".to_string(),
        );
        assert_eq!(result.schema, "airframe.conformance.comparison.v1");
    }
}
