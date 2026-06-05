#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int(value):
    if value is None or value == "":
        return None
    return int(value)


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["Model"]: row for row in rows}


def check_field(actual_row, field, expected):
    actual = actual_row.get(field, "")
    if isinstance(expected, bool):
        return parse_bool(actual) == expected, actual
    if isinstance(expected, int):
        return parse_int(actual) == expected, actual
    return actual == expected, actual


def validate(manifest, rows):
    errors = []
    models = manifest.get("models", {})

    for model, expectations in models.items():
        if model not in rows:
            errors.append(f"missing model in CSV: {model}")
            continue

        row = rows[model]
        status = row.get("Status", "")
        if status != "PASS":
            errors.append(f"{model}: Status expected PASS, got {status}")

        for field, expected in expectations.items():
            ok, actual = check_field(row, field, expected)
            if not ok:
                errors.append(
                    f"{model}: field {field} expected {expected!r}, got {actual!r}"
                )

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate route-check CSV against manifest contract.")
    parser.add_argument("--manifest", required=True, help="Path to expected route manifest JSON")
    parser.add_argument("--csv", required=True, help="Path to route-check CSV")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    csv_path = Path(args.csv)

    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}")
        return 2
    if not csv_path.exists():
        print(f"csv not found: {csv_path}")
        return 2

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rows = load_csv_rows(csv_path)
    errors = validate(manifest, rows)

    if errors:
        print("ROUTE_MANIFEST_VALIDATION: FAIL")
        for err in errors:
            print(f" - {err}")
        return 1

    print("ROUTE_MANIFEST_VALIDATION: PASS")
    print(f"models_checked: {len(manifest.get('models', {}))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
