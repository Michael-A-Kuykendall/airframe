//! airframe-conformance-protocol — Neutral conformance protocol definitions.
//!
//! This crate contains the shared protocol types and schemas used by both
//! production Airframe capture and the independent evaluator. It is a
//! specification-only crate with no implementation dependencies.

pub mod capture_protocol;
pub mod comparison;
pub mod evidence_package;
pub mod input_declaration;
pub mod manifest;
pub mod raw_gguf;
pub mod schemas;
pub mod semantic_manifest;

/// Re-export schema validation for external tools.
pub use schemas::{validate_all_schemas, SchemaValidator};

/// The protocol crate version.
pub const PROTOCOL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The conformance protocol version — used in evidence packages and protocol types.
pub const CONFORMANCE_VERSION: &str = env!("CARGO_PKG_VERSION");
