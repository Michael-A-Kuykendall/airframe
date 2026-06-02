#!/usr/bin/env python3
"""Vision smoke test for shimmy_server_gpu with SHIMMY_MMPROJ_PATH set.

Sends two requests to a locally-running server:
  1. Text-only baseline — no image payload.
  2. Multimodal — a synthetic 448×448 all-zero RGB image base64-encoded in
     the image_payload field, with an <image> placeholder in the prompt.

Usage:
  python scripts/vision_smoke_test.py [--host HOST] [--port PORT]

The server must already be running.  Start it with e.g.:
  SHIMMY_MMPROJ_PATH=D:/shimmy-test-models/gguf_collection/minicpm-v-2.6/mmproj-model-f16.gguf \
  LIBSHIMMY_MODEL_PATH=D:/shimmy-test-models/gguf_collection/minicpm-v-2.6/ggml-model-Q4_K_M.gguf \
  cargo run --release --bin shimmy_server_gpu
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
import urllib.error

IMAGE_W = 448
IMAGE_H = 448

def synthetic_image_b64(h: int, w: int) -> str:
    """Return base64-encoded all-zero HWC u8 RGB bytes (h × w × 3)."""
    pixels = bytes(h * w * 3)
    return base64.b64encode(pixels).decode("ascii")

def post(host: str, port: int, payload: dict, label: str) -> dict:
    url = f"http://{host}:{port}/v1/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"[{label}] HTTP {e.code}: {body}")
        return {}
    elapsed = time.monotonic() - t0
    result = json.loads(body)
    print(f"[{label}] {elapsed:.1f}s  stop={result.get('stop_reason','?')}  tokens={result.get('tokens_generated','?')}")
    text = result.get("text", "")
    preview = text[:120].replace("\n", "\\n")
    print(f"         text: {preview!r}")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()

    print(f"Vision smoke test → {args.host}:{args.port}")
    print("=" * 60)

    # ── Case 1: text-only baseline ───────────────────────────────
    r1 = post(args.host, args.port, {
        "prompt": "Describe the sky in one sentence.",
        "max_tokens": 30,
        "temperature": 0.0,
        "seed": 42,
    }, label="TEXT-ONLY")

    ok1 = bool(r1.get("text"))

    # ── Case 2: synthetic all-zero image ────────────────────────
    img_b64 = synthetic_image_b64(IMAGE_H, IMAGE_W)
    print(f"\nSending synthetic {IMAGE_H}×{IMAGE_W} all-zero image "
          f"({IMAGE_H*IMAGE_W*3} bytes → {len(img_b64)} b64 chars)")
    r2 = post(args.host, args.port, {
        "prompt": "<image>What do you see in this image?",
        "max_tokens": 30,
        "temperature": 0.0,
        "seed": 42,
        "image_payload": {
            "pixels_b64": img_b64,
            "h": IMAGE_H,
            "w": IMAGE_W,
        },
    }, label="VISION-ZERO")

    ok2 = bool(r2.get("text"))

    # ── Summary ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"TEXT-ONLY  : {'PASS' if ok1 else 'FAIL'}")
    print(f"VISION-ZERO: {'PASS' if ok2 else 'FAIL'}")

    if not (ok1 and ok2):
        sys.exit(1)
    print("All smoke checks passed.")

if __name__ == "__main__":
    main()
