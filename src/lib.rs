//! Airframe: Meticulous Rust port of llama.cpp inference.
//!
//! Provides CPU-only FP32 inference for TinyLlama Q4_0 GGUF models
//! with exact numerical parity to llama.cpp.

pub mod conformance;
pub mod control;
pub mod core;
pub mod debug_trace;
pub mod family;
pub mod fixtures;
pub mod grammar;
pub mod ops;
pub mod runtime;
pub mod validation;

pub mod adapter;

// Re-export diagnostic control
pub use family::llama::init_verbose_diagnostics;
pub mod backend;
pub mod fse_control;
pub mod math_bypass_control;
pub mod schoolmarm_control;

// PPT + Invariant testing framework: objective, AI-independent quality gates.
// Provides assert_invariant (embedded in production logic) plus property/contract
// test harness used by tests/test_contracts.rs. See docs on the PPT methodology.
pub mod invariant_ppt;
