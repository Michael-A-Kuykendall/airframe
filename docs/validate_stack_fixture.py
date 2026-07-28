#!/usr/bin/env python3
"""OBS-0: validate docs/fixtures/stack_minimal.json against docs/stack.schema.v1.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = ROOT / "stack.schema.v1.json"
FIXTURE = ROOT / "fixtures" / "stack_minimal.json"


def main() -> int:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    # Prefer jsonschema if installed; else minimal structural checks matching OBS-0 AC.
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=fixture, schema=schema)
        print("OK: jsonschema validate passed")
        print(f"  schema={SCHEMA}")
        print(f"  fixture={FIXTURE}")
        return 0
    except ImportError:
        pass
    except Exception as e:
        print(f"FAIL: jsonschema validate: {e}", file=sys.stderr)
        return 1

    # Fallback: required fields + types (enough for offline gate without pip)
    errors: list[str] = []
    if fixture.get("schema") != "airframe.stack.v1":
        errors.append("schema const must be airframe.stack.v1")
    for key in ("engine", "prompt", "config", "tokens", "layers", "final", "decode"):
        if key not in fixture:
            errors.append(f"missing required key: {key}")
    cfg = fixture.get("config") or {}
    for key in ("arch", "n_layer", "n_embd", "n_head", "n_kv_head", "head_dim"):
        if key not in cfg:
            errors.append(f"config missing: {key}")
    ids = (fixture.get("tokens") or {}).get("ids")
    if not isinstance(ids, list) or len(ids) < 1:
        errors.append("tokens.ids must be non-empty array")
    layers = fixture.get("layers") or []
    if not layers:
        errors.append("layers must be non-empty for minimal fixture")
    else:
        r = layers[0].get("residual") or {}
        if "rms" not in r or "nan_count" not in r:
            errors.append("layers[0].residual needs rms + nan_count")
    logits = (fixture.get("final") or {}).get("logits") or {}
    top_k = logits.get("top_k") or []
    if len(top_k) < 1:
        errors.append("final.logits.top_k must be non-empty")
    dec = fixture.get("decode") or {}
    if dec.get("status") != "skipped":
        errors.append("decode.status should be skipped in minimal fixture")

    if errors:
        print("FAIL:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("OK: structural fallback validate passed (install jsonschema for full draft-2020-12)")
    print(f"  schema={SCHEMA}")
    print(f"  fixture={FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
