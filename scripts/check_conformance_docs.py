#!/usr/bin/env python3
"""
Check conformance architecture documentation.

Validates that the conformance crate documentation correctly states:
1. The allowed dependency directions
2. The forbidden dependency directions
3. That production capture is exposed as telemetry only
"""

import sys
from pathlib import Path


CONFORMANCE_CRATE = Path(__file__).parent.parent / "crates" / "airframe-conformance"
README_FILE = CONFORMANCE_CRATE / "README.md"
LIB_FILE = CONFORMANCE_CRATE / "src" / "lib.rs"


REQUIRED_README_SECTIONS = [
    "## Dependency Boundary",
    "### Allowed Dependencies",
    "### Forbidden Dependencies",
    "## Telemetry-Only Capture",
    "## Architecture",
]

FORBIDDEN_PREFIXES = [
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
]

ALLOWED_PREFIXES = [
    "airframe::capture::spec",
    "airframe::capture::telemetry",
]


def check_readme() -> tuple[bool, list[str]]:
    """Check README.md has required sections."""
    errors = []
    if not README_FILE.exists():
        return False, [f"README.md not found at {README_FILE}"]

    content = README_FILE.read_text()
    for section in REQUIRED_README_SECTIONS:
        if section not in content:
            errors.append(f"Missing required section: {section}")

    # Check forbidden prefixes are documented
    for prefix in FORBIDDEN_PREFIXES:
        if prefix not in content:
            errors.append(f"Forbidden prefix not documented in README: {prefix}")

    # Check allowed prefixes are documented
    for prefix in ALLOWED_PREFIXES:
        if prefix not in content:
            errors.append(f"Allowed prefix not documented in README: {prefix}")

    return len(errors) == 0, errors


def check_lib_rs() -> tuple[bool, list[str]]:
    """Check lib.rs has the forbidden/allowed constants."""
    errors = []
    if not LIB_FILE.exists():
        return False, [f"lib.rs not found at {LIB_FILE}"]

    content = LIB_FILE.read_text()

    # Check FORBIDDEN_PRODUCTION_PREFIXES constant exists
    if "FORBIDDEN_PRODUCTION_PREFIXES" not in content:
        errors.append("FORBIDDEN_PRODUCTION_PREFIXES constant not found in lib.rs")

    # Check ALLOWED_SPEC_PREFIXES constant exists
    if "ALLOWED_SPEC_PREFIXES" not in content:
        errors.append("ALLOWED_SPEC_PREFIXES constant not found in lib.rs")

    # Check all forbidden prefixes are in the constant
    for prefix in FORBIDDEN_PREFIXES:
        if prefix not in content:
            errors.append(f"Forbidden prefix missing from lib.rs constant: {prefix}")

    # Check all allowed prefixes are in the constant
    for prefix in ALLOWED_PREFIXES:
        if prefix not in content:
            errors.append(f"Allowed prefix missing from lib.rs constant: {prefix}")

    # Check telemetry-only statement
    if "telemetry" not in content.lower():
        errors.append("Telemetry-only capture not mentioned in lib.rs")

    return len(errors) == 0, errors


def main() -> int:
    print("Checking conformance architecture documentation...")
    print()

    all_ok = True

    print("Checking README.md...")
    ok, errors = check_readme()
    if ok:
        print("  ✓ README.md: All required sections present")
    else:
        print("  ✗ README.md: Issues found")
        for err in errors:
            print(f"    - {err}")
        all_ok = False

    print()
    print("Checking src/lib.rs...")
    ok, errors = check_lib_rs()
    if ok:
        print("  ✓ lib.rs: All constants and documentation present")
    else:
        print("  ✗ lib.rs: Issues found")
        for err in errors:
            print(f"    - {err}")
        all_ok = False

    print()
    if all_ok:
        print("All documentation checks passed!")
        return 0
    else:
        print("Documentation checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
