//! V2 slice-gated validation infrastructure.
//!
//! Provides evidence checklists, oracle conformance, and
//! determinism proofs for slice gate criteria.

pub mod artifacts;
pub mod errors;
pub mod evidence;
pub mod projection;
pub mod slice_validator;

pub use artifacts::*;
pub use errors::*;
pub use evidence::EvidenceChecklist;
pub use projection::canonical_projection;
pub use slice_validator::{DecodeResult, KVCacheResult, OracleResult, SliceValidator};
