#!/usr/bin/env python3
"""Vision smoke test for shimmy_server_gpu with SHIMMY_MMPROJ_PATH set.

Uses real fixture images from fixtures/vision-samples/ (extracted from the
shimmy-private vision branch).  Each image has a human-verified description
and expected keyword list in MANIFEST.json.

Two test modes:
  --synthetic   Send a single all-zero 448×448 image (quick pipeline sanity)
  --fixtures    Send all MANIFEST images with keyword-validation (default)

Usage:
  python scripts/vision_smoke_test.py [--host HOST] [--port PORT] [--synthetic]

The server must already be running.  Start with e.g.:
  SHIMMY_MMPROJ_PATH=D:/shimmy-test-models/gguf_collection/minicpm-v-2.6/mmproj-model-f16.gguf \\
  LIBSHIMMY_MODEL_PATH=D:/shimmy-test-models/gguf_collection/minicpm-v-2.6/ggml-model-Q4_K_M.gguf \\
  cargo run --release --bin shimmy_server_gpu
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(SCRIPT_DIR, "..", "fixtures", "vision-samples")
MANIFEST_PATH = os.path.join(FIXTURES_DIR, "MANIFEST.json")

# ── helpers ───────────────────────────────────────────────────────────────────

def encode_image_file(path: str) -> tuple[str, int, int]:
    """Return (b64_string, height, width) by reading raw bytes + PNG header."""
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    # Read PNG dimensions from IHDR chunk (bytes 16-24)
    if raw[:4] == b"\x89PNG":
        import struct
        w = struct.unpack(">I", raw[16:20])[0]
        h = struct.unpack(">I", raw[20:24])[0]
    else:
        # Fallback — use manifest dims; caller should override
        w, h = 0, 0
    return b64, h, w


def post(host: str, port: int, payload: dict, label: str) -> dict:
    url = f"http://{host}:{port}/v1/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"  [{label}] HTTP {e.code}: {body[:200]}")
        return {}
    elapsed = time.monotonic() - t0
    result = json.loads(body)
    tok = result.get("tokens_generated", "?")
    stop = result.get("stop_reason", "?")
    print(f"  [{label}] {elapsed:.1f}s  stop={stop}  tokens={tok}")
    text = result.get("text", "")
    preview = text[:160].replace("\n", "\\n")
    print(f"  [response] {preview!r}")
    return result


def check_keywords(text: str, required: list[str], absent: list[str]) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures = []
    text_lower = text.lower()
    for kw in required:
        if kw.lower() not in text_lower:
            failures.append(f"MISSING expected keyword: {kw!r}")
    for kw in absent:
        if kw.lower() in text_lower:
            failures.append(f"UNEXPECTED keyword present: {kw!r}")
    return len(failures) == 0, failures


# ── test runners ──────────────────────────────────────────────────────────────

def run_synthetic(host: str, port: int) -> bool:
    """Minimal pipeline sanity: all-zero 448×448 image, expect non-empty text."""
    print("\n── SYNTHETIC (all-zero 448×448 image) ──")
    pixels = bytes(448 * 448 * 3)
    b64 = base64.b64encode(pixels).decode("ascii")
    result = post(host, port, {
        "prompt": "<image>Describe this image in one sentence.",
        "max_tokens": 40,
        "temperature": 0.0,
        "seed": 42,
        "image_payload": {"pixels_b64": b64, "h": 448, "w": 448},
    }, label="SYNTHETIC")
    ok = bool(result.get("text"))
    print(f"  RESULT: {'PASS' if ok else 'FAIL — empty response'}")
    return ok


def run_fixtures(host: str, port: int) -> bool:
    """Run all fixture images from MANIFEST.json with keyword validation."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results = []
    for entry in manifest["images"]:
        fname = entry["file"]
        path = os.path.join(FIXTURES_DIR, fname)
        if not os.path.exists(path):
            print(f"\n── {fname}: SKIP (file not found at {path})")
            results.append((fname, "SKIP", []))
            continue

        print(f"\n── {fname} ──")
        print(f"  prompt: {entry['smoke_prompt']!r}")

        b64, h, w = encode_image_file(path)
        # Fall back to manifest dims if PNG read failed
        if h == 0:
            h, w = entry["dims"]["h"], entry["dims"]["w"]

        result = post(host, port, {
            "prompt": entry["smoke_prompt"],
            "max_tokens": 80,
            "temperature": 0.0,
            "seed": 42,
            "image_payload": {"pixels_b64": b64, "h": h, "w": w},
        }, label=fname)

        text = result.get("text", "")
        if not text:
            print("  RESULT: FAIL — empty response")
            results.append((fname, "FAIL", ["Empty response"]))
            continue

        ok, failures = check_keywords(
            text,
            entry.get("expected_keywords", []),
            entry.get("expected_absent", []),
        )
        for f_ in failures:
            print(f"  !! {f_}")
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
        results.append((fname, "PASS" if ok else "FAIL", failures))

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Image':<45} {'Result'}")
    print("-" * 60)
    all_ok = True
    for fname, status, _ in results:
        print(f"{fname:<45} {status}")
        if status == "FAIL":
            all_ok = False
    print("=" * 60)
    return all_ok


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--synthetic", action="store_true",
                        help="Run synthetic all-zero image test only (no fixture files)")
    args = parser.parse_args()

    print(f"Vision smoke test → {args.host}:{args.port}")

    if args.synthetic:
        ok = run_synthetic(args.host, args.port)
    else:
        ok = run_fixtures(args.host, args.port)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

