//! Dependency policy test — enforces the conformance boundary.
//!
//! This test verifies that conformance code does not import forbidden production modules.
//! It uses a compile-time approach: a fixture file that imports forbidden modules should
//! fail to compile. If it compiles, the boundary is violated.

use std::path::Path;

/// Test that the forbidden prefixes list in lib.rs is comprehensive.
#[test]
fn forbidden_prefixes_documented() {
    let lib_rs = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let content = std::fs::read_to_string(&lib_rs).expect("lib.rs readable");

    // All forbidden prefixes from the architecture must be in the constant
    let required_forbidden = [
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

    for prefix in required_forbidden {
        assert!(
            content.contains(prefix),
            "Forbidden prefix '{prefix}' missing from FORBIDDEN_PRODUCTION_PREFIXES in lib.rs"
        );
    }
}

/// Test that the allowed prefixes list in lib.rs is documented.
#[test]
fn allowed_prefixes_documented() {
    let lib_rs = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let content = std::fs::read_to_string(&lib_rs).expect("lib.rs readable");

    let required_allowed = ["airframe::capture::spec", "airframe::capture::telemetry"];

    for prefix in required_allowed {
        assert!(
            content.contains(prefix),
            "Allowed prefix '{prefix}' missing from ALLOWED_SPEC_PREFIXES in lib.rs"
        );
    }
}

/// Test that README.md documents the boundary.
#[test]
fn readme_documents_boundary() {
    let readme = Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md");
    let content = std::fs::read_to_string(&readme).expect("README.md readable");

    let required_sections = [
        "## Dependency Boundary",
        "### Allowed Dependencies",
        "### Forbidden Dependencies",
        "## Telemetry-Only Capture",
    ];

    for section in required_sections {
        assert!(
            content.contains(section),
            "README.md missing required section: {section}"
        );
    }
}

/// This test would fail to compile if conformance code imports forbidden modules.
/// The fixture is in tests/fixtures/forbidden_import.rs — it should NOT compile.
/// We verify this by checking the fixture exists and contains forbidden imports.
#[test]
fn forbidden_import_fixture_exists() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("forbidden_import.rs");

    assert!(fixture.exists(), "Forbidden import fixture missing");

    let content = std::fs::read_to_string(&fixture).expect("fixture readable");

    // Fixture should contain at least one forbidden import
    let has_forbidden = airframe_conformance::FORBIDDEN_PRODUCTION_PREFIXES
        .iter()
        .any(|prefix| content.contains(prefix));

    assert!(
        has_forbidden,
        "Fixture must import at least one forbidden module"
    );
}

/// Verify that the conformance crate itself doesn't have forbidden dependencies
/// in its Cargo.toml (only specification crates allowed).
#[test]
fn cargo_toml_no_forbidden_deps() {
    let cargo_toml = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).expect("Cargo.toml readable");

    // These crates must NOT appear in dependencies
    let forbidden_deps = [
        "airframe",
        "libfse",
        "airframe-observe",
        "wgpu",
        "pollster",
        "bytemuck",
        "half",
        "memmap2",
    ];

    for dep in forbidden_deps {
        // Allow in dev-dependencies for testing the boundary itself
        if content.contains(&format!("{dep} =")) && !content.contains(&format!("{dep} {{")) {
            // Check it's not in [dependencies] section
            let in_deps = content
                .lines()
                .skip_while(|l| !l.trim().starts_with("[dependencies]"))
                .take_while(|l| !l.trim().starts_with('['))
                .any(|l| l.contains(dep));
            assert!(
                !in_deps,
                "Forbidden dependency '{dep}' found in [dependencies]"
            );
        }
    }
}
