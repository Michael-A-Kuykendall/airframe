#!/usr/bin/env python3
"""Validate the mandatory conformance architecture decision record."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SPIKE = ROOT / "docs" / "conformance" / "ARCHITECTURE_SPIKE.md"

REQUIRED_TEXT = (
    "# Conformance Architecture Spike",
    "## Evidence",
    "## Decision",
    "## Invalidated Work",
    "## Replanned Bead Boundaries",
    "## Required Gates Before Code Resumes",
    "30bf8685ed4eb0a47f2b06229543327749904150",
    "docs/gguf.md",
    "src/ggml.c",
    "There is no tensor-size field.",
    "CONF-2 owns raw evidence only",
    "CONF-5 owns quant layout and exact spans",
    "Capture protocol boundary is unresolved and blocks CONF-1",
)


def main() -> int:
    if not SPIKE.is_file():
        print(f"missing architecture decision record: {SPIKE}")
        return 1

    contents = SPIKE.read_text(encoding="utf-8")
    missing = [text for text in REQUIRED_TEXT if text not in contents]
    if missing:
        print("architecture decision record is incomplete:")
        for text in missing:
            print(f"  missing: {text}")
        return 1

    print("conformance architecture decision record: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
