use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Declared numerical token inputs — binds tokenization to provenance.
/// This is the "declared input" that Shimmy must produce and conformance verifies.

/// A single declared token with full provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeclaredToken {
    /// Token ID.
    pub id: u32,

    /// Token text piece (UTF-8).
    pub piece: String,

    /// Byte offset in original prompt.
    pub byte_offset: u32,

    /// Byte length in original prompt.
    pub byte_length: u32,

    /// Tokenizer that produced this token.
    pub tokenizer: TokenizerProvenance,

    /// Whether this token was added by the tokenizer (BOS, EOS, etc.).
    #[serde(default)]
    pub is_special: bool,
}

/// Tokenizer provenance information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TokenizerProvenance {
    /// Tokenizer identifier (e.g., "shimmytok", "tiktoken", "sentencepiece").
    pub name: String,

    /// Tokenizer version.
    pub version: String,

    /// Git commit of tokenizer (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,

    /// Tokenizer configuration hash.
    pub config_hash: String,

    /// Chat template applied (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
}

/// Complete declared input for a prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeclaredInput {
    /// Schema identifier: "airframe.conformance.declared_input.v1"
    #[serde(rename = "$schema")]
    pub schema: String,

    /// Original prompt text.
    pub prompt: String,

    /// Declared tokens in order.
    pub tokens: Vec<DeclaredToken>,

    /// Tokenizer provenance.
    pub tokenizer: TokenizerProvenance,

    /// Declaration timestamp (RFC3339).
    pub declared_at: String,

    /// Conformance version.
    pub conformance_version: String,

    /// Run ID this declaration belongs to.
    pub run_id: String,
}

/// Build provenance for Shimmy — binds binary to source.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BuildProvenance {
    /// Schema identifier: "airframe.conformance.build_provenance.v1"
    #[serde(rename = "$schema")]
    pub schema: String,

    /// Shimmy version.
    pub shimmy_version: String,

    /// Git commit.
    pub git_commit: String,

    /// Build timestamp (RFC3339).
    pub built_at: String,

    /// Build profile (debug/release).
    pub profile: String,

    /// Target triple.
    pub target: String,

    /// Cargo features enabled.
    pub features: Vec<String>,

    /// Dependency versions (from Cargo.lock).
    #[serde(default)]
    pub dependencies: HashMap<String, String>,

    /// Airframe version linked (from Cargo.lock).
    pub airframe_version: String,

    /// Airframe git commit (if available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub airframe_git_commit: Option<String>,

    /// Conformance version.
    pub conformance_version: String,
}

impl DeclaredInput {
    pub fn new(
        prompt: String,
        tokens: Vec<DeclaredToken>,
        tokenizer: TokenizerProvenance,
        run_id: String,
    ) -> Self {
        Self {
            schema: "airframe.conformance.declared_input.v1".to_string(),
            prompt,
            tokens,
            tokenizer,
            declared_at: chrono::Utc::now().to_rfc3339(),
            conformance_version: crate::CONFORMANCE_VERSION.to_string(),
            run_id,
        }
    }

    pub fn validate(&self) -> Result<(), crate::schemas::ValidationError> {
        crate::schemas::validate_declared_input(self)
    }
}

impl BuildProvenance {
    pub fn new(shimmy_version: String, git_commit: String, airframe_version: String) -> Self {
        Self {
            schema: "airframe.conformance.build_provenance.v1".to_string(),
            shimmy_version,
            git_commit,
            built_at: chrono::Utc::now().to_rfc3339(),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .to_string(),
            target: std::env::consts::ARCH.to_string() + "-" + std::env::consts::OS,
            features: Vec::new(),
            dependencies: HashMap::new(),
            airframe_version,
            airframe_git_commit: None,
            conformance_version: crate::CONFORMANCE_VERSION.to_string(),
        }
    }

    pub fn validate(&self) -> Result<(), crate::schemas::ValidationError> {
        crate::schemas::validate_build_provenance(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_input_roundtrip() {
        let tokenizer = TokenizerProvenance {
            name: "shimmytok".to_string(),
            version: "0.8.2".to_string(),
            git_commit: None,
            config_hash: "abc123".to_string(),
            chat_template: None,
        };
        let input = DeclaredInput::new(
            "test prompt".to_string(),
            vec![DeclaredToken {
                id: 1,
                piece: "test".to_string(),
                byte_offset: 0,
                byte_length: 4,
                tokenizer: tokenizer.clone(),
                is_special: false,
            }],
            tokenizer,
            "run-123".to_string(),
        );
        let json = serde_json::to_string(&input).unwrap();
        let parsed: DeclaredInput = serde_json::from_str(&json).unwrap();
        assert_eq!(input, parsed);
    }

    #[test]
    fn build_provenance_schema() {
        let prov = BuildProvenance::new(
            "2.5.0".to_string(),
            "abc123".to_string(),
            "0.3.0".to_string(),
        );
        assert_eq!(prov.schema, "airframe.conformance.build_provenance.v1");
    }
}
