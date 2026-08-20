// GLOSSARY: meaning (b) of `evidence` — EvidencePackage, the conformance
// evidence bundle (serialized, schema-valid). NOT validation::EvidenceChecklist
// (meaning a, in-memory slice-validation tracker in src/validation/evidence.rs).
// See that module's glossary.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evidence package metadata — complete conformance evidence bundle.

/// Evidence package containing all conformance artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvidencePackage {
    /// Schema identifier: "airframe.conformance.evidence.v1"
    #[serde(rename = "$schema")]
    pub schema: String,

    /// Package metadata.
    pub metadata: PackageMetadata,

    /// Conformance manifest used.
    pub manifest: crate::manifest::ConformanceManifest,

    /// Capture manifests (one per engine).
    pub captures: Vec<crate::capture_protocol::CaptureManifest>,

    /// Declared inputs.
    pub declared_inputs: Vec<crate::input_declaration::DeclaredInput>,

    /// Build provenance.
    pub build_provenance: crate::input_declaration::BuildProvenance,

    /// Comparison results.
    pub comparisons: Vec<crate::comparison::ComparisonResult>,

    /// Overall conformance verdict.
    pub verdict: Verdict,
}

/// Package metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PackageMetadata {
    /// Unique package ID (UUID v4).
    pub package_id: String,

    /// Package creation timestamp (RFC3339).
    pub created_at: String,

    /// Conformance version.
    pub conformance_version: String,

    /// Schema version of this package format.
    pub schema_version: String,

    /// Human-readable description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Tags for categorization.
    #[serde(default)]
    pub tags: Vec<String>,

    /// Additional metadata.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Overall conformance verdict.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// All comparisons pass within tolerance.
    Pass,
    /// One or more comparisons fail.
    Fail,
    /// Insufficient data to determine.
    Inconclusive,
    /// Package is malformed or incomplete.
    Invalid,
}

impl EvidencePackage {
    pub fn new(
        manifest: crate::manifest::ConformanceManifest,
        captures: Vec<crate::capture_protocol::CaptureManifest>,
        declared_inputs: Vec<crate::input_declaration::DeclaredInput>,
        build_provenance: crate::input_declaration::BuildProvenance,
        comparisons: Vec<crate::comparison::ComparisonResult>,
    ) -> Self {
        let verdict = Self::compute_verdict(&comparisons);
        Self {
            schema: "airframe.conformance.evidence.v1".to_string(),
            metadata: PackageMetadata {
                package_id: uuid::Uuid::new_v4().to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                conformance_version: crate::CONFORMANCE_VERSION.to_string(),
                schema_version: "1".to_string(),
                description: None,
                tags: Vec::new(),
                extra: HashMap::new(),
            },
            manifest,
            captures,
            declared_inputs,
            build_provenance,
            comparisons,
            verdict,
        }
    }

    fn compute_verdict(comparisons: &[crate::comparison::ComparisonResult]) -> Verdict {
        if comparisons.is_empty() {
            return Verdict::Inconclusive;
        }
        for comp in comparisons {
            if comp.overall == crate::comparison::OverallResult::Fail {
                return Verdict::Fail;
            }
            if comp.overall == crate::comparison::OverallResult::Inconclusive {
                return Verdict::Inconclusive;
            }
        }
        Verdict::Pass
    }

    pub fn validate(&self) -> Result<(), crate::schemas::ValidationError> {
        crate::schemas::validate_evidence(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_package_roundtrip() {
        let manifest = crate::manifest::ConformanceManifest::new(
            "/path/model.gguf".to_string(),
            "llama".to_string(),
            vec!["test".to_string()],
        );
        let pkg = EvidencePackage::new(
            manifest,
            vec![],
            vec![],
            crate::input_declaration::BuildProvenance::new(
                "2.5.0".to_string(),
                "abc".to_string(),
                "0.3.0".to_string(),
            ),
            vec![],
        );
        let json = serde_json::to_string(&pkg).unwrap();
        let parsed: EvidencePackage = serde_json::from_str(&json).unwrap();
        assert_eq!(pkg, parsed);
    }

    #[test]
    fn evidence_schema() {
        let manifest = crate::manifest::ConformanceManifest::new(
            "/path/model.gguf".to_string(),
            "llama".to_string(),
            vec!["test".to_string()],
        );
        let pkg = EvidencePackage::new(
            manifest,
            vec![],
            vec![],
            crate::input_declaration::BuildProvenance::new(
                "2.5.0".to_string(),
                "abc".to_string(),
                "0.3.0".to_string(),
            ),
            vec![],
        );
        assert_eq!(pkg.schema, "airframe.conformance.evidence.v1");
    }
}
