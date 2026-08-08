//! airframe-conformance — Independent conformance verification for Airframe GPU inference engine.
//!
//! This crate is TEST-ONLY and must NOT be a production dependency.
//! It enforces a one-way dependency boundary: conformance code may only import
//! specification APIs, never production Airframe semantic/loader/dispatch/offset/cache/capture implementations.

pub mod capture_protocol;
pub mod comparison;
pub mod evidence_package;
pub mod input_declaration;
pub mod manifest;
pub mod schemas;

/// Re-export schema validation for external tools.
pub use schemas::{validate_all_schemas, SchemaValidator};

/// The conformance crate version — used in evidence packages.
pub const CONFORMANCE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Forbidden production module prefixes that conformance code must never import.
/// This list is used by the dependency_policy test to enforce the boundary.
pub const FORBIDDEN_PRODUCTION_PREFIXES: &[&str] = &[
    "airframe::semantic",
    "airframe::loader",
    "airframe::dispatch",
    "airframe::offset",
    "airframe::cache",
    "airframe::capture::production",
    "airframe::inference",
    "airframe::quant",
    "airframe::rope",
    "airframe::rms_norm",
    "airframe::attention",
    "airframe::ffn",
    "airframe::lm_head",
];

/// Allowed specification-only module prefixes that conformance code MAY import.
pub const ALLOWED_SPEC_PREFIXES: &[&str] = &[
    "airframe::capture::spec",
    "airframe::capture::telemetry",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformance_version_is_present() {
        assert!(!CONFORMANCE_VERSION.is_empty());
    }

    #[test]
    fn forbidden_prefixes_not_empty() {
        assert!(!FORBIDDEN_PRODUCTION_PREFIXES.is_empty());
    }

    #[test]
    fn allowed_prefixes_not_empty() {
        assert!(!ALLOWED_SPEC_PREFIXES.is_empty());
    }
}