//! Fixture that imports a forbidden production module.
//! This file should NOT compile — it exists to verify the dependency_policy test
//! can detect forbidden imports.
//!
//! To test: try to compile this file directly with rustc — it should fail
//! because airframe::inference is not a dependency of airframe-conformance.

// This import is FORBIDDEN for conformance code:
// use airframe::inference::InferencePipeline;

// This import is FORBIDDEN:
// use airframe::quant::dequantize;

// This import is FORBIDDEN:
// use airframe::attention::flash_attention;

// This import is FORBIDDEN:
// use airframe::capture::production::capture_layer;

// This import is ALLOWED (specification only):
// use airframe::capture::spec::PointIdentity;

// This import is ALLOWED (telemetry trait only):
// use airframe::capture::telemetry::CaptureTelemetry;

fn main() {
    // This fixture is not meant to run — it's a compile-time test.
    // The dependency_policy test verifies this file contains forbidden imports.
    println!("This fixture should not be compiled as part of airframe-conformance");
}
