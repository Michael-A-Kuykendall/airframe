#!/usr/bin/env python3
"""
Validate all conformance JSON schemas compile correctly.

This script validates that all JSON schema files in the airframe-conformance crate
are valid JSON Schema Draft 2020-12 and can be compiled by jsonschema.
"""

import json
import sys
from pathlib import Path
from jsonschema import Draft202012Validator


SCHEMA_DIR = (
    Path(__file__).parent.parent.parent / "crates" / "airframe-conformance" / "schemas"
)

SCHEMA_FILES = [
    "manifest.schema.json",
    "capture.schema.json",
    "declared_input.schema.json",
    "build_provenance.schema.json",
    "comparison.schema.json",
    "evidence.schema.json",
]


def validate_schema_file(schema_path: Path) -> tuple[bool, str]:
    """Validate a single schema file."""
    try:
        with open(schema_path, "r") as f:
            schema = json.load(f)

        # Check required fields
        if "$schema" not in schema:
            return False, f"Missing $schema field"
        if schema["$schema"] != "https://json-schema.org/draft/2020-12/schema":
            return False, f"Wrong $schema value: {schema['$schema']}"
        # $id is optional for local-only schemas (we don't host them remotely)
        if "title" not in schema:
            return False, f"Missing title field"

        # Try to compile with jsonschema
        Draft202012Validator.check_schema(schema)

        return True, "OK"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Schema compilation failed: {e}"


def main() -> int:
    print("Validating conformance JSON schemas...")
    print(f"Schema directory: {SCHEMA_DIR}")
    print()

    all_ok = True
    for schema_file in SCHEMA_FILES:
        schema_path = SCHEMA_DIR / schema_file
        if not schema_path.exists():
            print(f"  ✗ {schema_file}: FILE NOT FOUND")
            all_ok = False
            continue

        ok, msg = validate_schema_file(schema_path)
        status = "✓" if ok else "✗"
        print(f"  {status} {schema_file}: {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All schemas valid!")
        return 0
    else:
        print("Some schemas failed validation!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
