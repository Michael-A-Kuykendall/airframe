use crate::{
    capture_protocol, comparison, evidence_package, input_declaration, manifest, semantic_manifest,
};
use jsonschema::{Draft, Validator};
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

/// Validation error for schema validation.
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Schema compilation failed: {0}")]
    SchemaCompilation(String),

    #[error("Validation failed: {0}")]
    Validation(String),

    #[error("JSON serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Schema validator holding compiled JSON schemas.
pub struct SchemaValidator {
    schemas: HashMap<String, Validator>,
}

impl SchemaValidator {
    /// Create a new validator with all conformance schemas compiled.
    pub fn new() -> Result<Self, ValidationError> {
        let mut schemas = HashMap::new();

        // Manifest schema
        let manifest_schema =
            Self::compile_schema(include_str!("../schemas/manifest.schema.json"))?;
        schemas.insert(
            "airframe.conformance.manifest.v1".to_string(),
            manifest_schema,
        );

        // Capture schema
        let capture_schema = Self::compile_schema(include_str!("../schemas/capture.schema.json"))?;
        schemas.insert(
            "airframe.conformance.capture.v1".to_string(),
            capture_schema,
        );

        // Declared input schema
        let declared_input_schema =
            Self::compile_schema(include_str!("../schemas/declared_input.schema.json"))?;
        schemas.insert(
            "airframe.conformance.declared_input.v1".to_string(),
            declared_input_schema,
        );

        // Build provenance schema
        let build_provenance_schema =
            Self::compile_schema(include_str!("../schemas/build_provenance.schema.json"))?;
        schemas.insert(
            "airframe.conformance.build_provenance.v1".to_string(),
            build_provenance_schema,
        );

        // Comparison schema
        let comparison_schema =
            Self::compile_schema(include_str!("../schemas/comparison.schema.json"))?;
        schemas.insert(
            "airframe.conformance.comparison.v1".to_string(),
            comparison_schema,
        );

        // Evidence package schema
        let evidence_schema =
            Self::compile_schema(include_str!("../schemas/evidence.schema.json"))?;
        schemas.insert(
            "airframe.conformance.evidence.v1".to_string(),
            evidence_schema,
        );

        // Semantic manifest schema
        let semantic_manifest_schema =
            Self::compile_schema(include_str!("../schemas/semantic_manifest.schema.json"))?;
        schemas.insert(
            "airframe.conformance.semantic_manifest.v1".to_string(),
            semantic_manifest_schema,
        );

        Ok(Self { schemas })
    }

    fn compile_schema(schema_str: &str) -> Result<Validator, ValidationError> {
        let schema_value: Value = serde_json::from_str(schema_str)
            .map_err(|e| ValidationError::SchemaCompilation(format!("Invalid JSON: {}", e)))?;
        Validator::options()
            .with_draft(Draft::Draft202012)
            .build(&schema_value)
            .map_err(|e| ValidationError::SchemaCompilation(e.to_string()))
    }

    /// Validate a value against a schema by schema ID.
    pub fn validate(&self, schema_id: &str, value: &Value) -> Result<(), ValidationError> {
        let schema = self
            .schemas
            .get(schema_id)
            .ok_or_else(|| ValidationError::Validation(format!("Unknown schema: {}", schema_id)))?;
        let result = schema.validate(value);
        if let Err(errors) = result {
            let messages: Vec<String> = errors.map(|e| e.to_string()).collect();
            return Err(ValidationError::Validation(messages.join("; ")));
        }
        Ok(())
    }

    /// Validate a manifest.
    pub fn validate_manifest(
        &self,
        manifest: &manifest::ConformanceManifest,
    ) -> Result<(), ValidationError> {
        let value = serde_json::to_value(manifest)?;
        self.validate("airframe.conformance.manifest.v1", &value)
    }

    /// Validate a capture manifest.
    pub fn validate_capture(
        &self,
        capture: &capture_protocol::CaptureManifest,
    ) -> Result<(), ValidationError> {
        let value = serde_json::to_value(capture)?;
        self.validate("airframe.conformance.capture.v1", &value)
    }

    /// Validate a declared input.
    pub fn validate_declared_input(
        &self,
        input: &input_declaration::DeclaredInput,
    ) -> Result<(), ValidationError> {
        let value = serde_json::to_value(input)?;
        self.validate("airframe.conformance.declared_input.v1", &value)
    }

    /// Validate build provenance.
    pub fn validate_build_provenance(
        &self,
        prov: &input_declaration::BuildProvenance,
    ) -> Result<(), ValidationError> {
        let value = serde_json::to_value(prov)?;
        self.validate("airframe.conformance.build_provenance.v1", &value)
    }

    /// Validate a comparison result.
    pub fn validate_comparison(
        &self,
        comp: &comparison::ComparisonResult,
    ) -> Result<(), ValidationError> {
        let value = serde_json::to_value(comp)?;
        self.validate("airframe.conformance.comparison.v1", &value)
    }

    /// Validate an evidence package.
    pub fn validate_evidence(
        &self,
        pkg: &evidence_package::EvidencePackage,
    ) -> Result<(), ValidationError> {
        let value = serde_json::to_value(pkg)?;
        self.validate("airframe.conformance.evidence.v1", &value)
    }

    /// Validate a semantic manifest.
    pub fn validate_semantic_manifest(
        &self,
        manifest: &semantic_manifest::SemanticManifest,
    ) -> Result<(), ValidationError> {
        let value = serde_json::to_value(manifest)?;
        self.validate("airframe.conformance.semantic_manifest.v1", &value)
    }
}

impl Default for SchemaValidator {
    fn default() -> Self {
        Self::new().expect("Failed to create SchemaValidator")
    }
}

/// Validate a manifest (convenience function).
pub fn validate_manifest(manifest: &manifest::ConformanceManifest) -> Result<(), ValidationError> {
    SchemaValidator::new()?.validate_manifest(manifest)
}

/// Validate a capture manifest.
pub fn validate_capture(
    capture: &capture_protocol::CaptureManifest,
) -> Result<(), ValidationError> {
    SchemaValidator::new()?.validate_capture(capture)
}

/// Validate a declared input.
pub fn validate_declared_input(
    input: &input_declaration::DeclaredInput,
) -> Result<(), ValidationError> {
    SchemaValidator::new()?.validate_declared_input(input)
}

/// Validate build provenance.
pub fn validate_build_provenance(
    prov: &input_declaration::BuildProvenance,
) -> Result<(), ValidationError> {
    SchemaValidator::new()?.validate_build_provenance(prov)
}

/// Validate a comparison result.
pub fn validate_comparison(comp: &comparison::ComparisonResult) -> Result<(), ValidationError> {
    SchemaValidator::new()?.validate_comparison(comp)
}

/// Validate an evidence package.
pub fn validate_evidence(pkg: &evidence_package::EvidencePackage) -> Result<(), ValidationError> {
    SchemaValidator::new()?.validate_evidence(pkg)
}

/// Validate a semantic manifest.
pub fn validate_semantic_manifest(
    manifest: &semantic_manifest::SemanticManifest,
) -> Result<(), ValidationError> {
    SchemaValidator::new()?.validate_semantic_manifest(manifest)
}

/// Validate all schemas can be compiled (used by validate_schemas.py).
pub fn validate_all_schemas() -> Result<(), ValidationError> {
    SchemaValidator::new()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_creates() {
        let validator = SchemaValidator::new();
        assert!(validator.is_ok());
    }

    #[test]
    fn validate_all_schemas_ok() {
        let result = validate_all_schemas();
        assert!(result.is_ok());
    }
}
