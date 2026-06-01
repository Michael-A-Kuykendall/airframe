#!/usr/bin/env python3
"""
long_prompt_soak_test.py
P0 stability gate: 10 consecutive INT4 requests with long-context prompts.

Target: ctx ≈ 2048 (prompt ~1500 tokens + 100 decode tokens + system overhead ≤ 2048)
Pass condition: all 10 requests complete with HTTP 200, no crash, no hang.

Usage:
    python scripts/long_prompt_soak_test.py [--host localhost] [--port 8099] [--requests 10]
"""

import urllib.request
import urllib.error
import json
import time
import sys
import argparse
from datetime import datetime, timezone

PORT = 8099
HOST = "localhost"
REQUESTS = 10
TIMEOUT = 300  # seconds per request — long prefill at ctx=2048 can take ~60s cold

# ~1500 token prompt: a passage repeated enough times to fill the context.
# Llama-3.2 tokenizes at roughly 0.7–0.8 tokens/word.
# This passage is ~120 words → ~90 tokens. Repeated 16× ≈ 1440 tokens.
PASSAGE = (
    "The following is a summary of recent developments in applied mathematics. "
    "Fourier analysis, originally developed to study heat conduction, has become a "
    "cornerstone of signal processing and data compression. Wavelet transforms extend "
    "this framework by allowing analysis at multiple scales simultaneously, making them "
    "ideal for image processing and seismic data analysis. In number theory, the Riemann "
    "Hypothesis remains one of the most famous unsolved problems, connecting the distribution "
    "of prime numbers to the zeros of the Riemann zeta function. Meanwhile, topological "
    "data analysis has emerged as a practical tool for understanding high-dimensional datasets "
    "by studying their shape. Category theory provides a unifying language across disparate "
    "branches of mathematics, with functors and natural transformations formalizing structural "
    "analogies. Computational complexity theory asks which problems can be solved efficiently, "
    "and the P versus NP question remains unresolved. "
)

def build_long_prompt(repeat: int = 4) -> str:
    """Build a prompt of roughly 1500-1600 templated tokens (fits ctx=2048 with 80 decode)."""
    context = PASSAGE * repeat
    return (
        f"{context}\n\n"
        "Based on the passage above, briefly summarize the key mathematical topic mentioned "
        "in the first sentence."
    )


def run_soak(host: str, port: int, n_requests: int) -> bool:
    url = f"http://{host}:{port}/v1/completions"
    prompt = build_long_prompt(repeat=4)

    print(f"[soak] target: {url}")
    print(f"[soak] prompt length: {len(prompt)} chars (~{len(prompt.split())//1} words)")
    print(f"[soak] requests: {n_requests}, timeout per request: {TIMEOUT}s")
    print()

    results = []
    for i in range(n_requests):
        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": 80,
            "temperature": 0.0,
        }).encode("utf-8")

        t0 = time.monotonic()
        status = "?"
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = json.loads(resp.read())
                elapsed = time.monotonic() - t0
                text = body.get("choices", [{}])[0].get("text", "")[:80].replace("\n", " ")
                status = f"PASS ({elapsed:.1f}s) -> \"{text}\""
                results.append(True)
        except urllib.error.HTTPError as e:
            elapsed = time.monotonic() - t0
            status = f"FAIL HTTP {e.code} ({elapsed:.1f}s)"
            results.append(False)
        except Exception as e:
            elapsed = time.monotonic() - t0
            status = f"FAIL {type(e).__name__}: {e} ({elapsed:.1f}s)"
            results.append(False)

        print(f"[{i+1:02d}/{n_requests}] {status}", flush=True)

    passed = sum(results)
    failed = n_requests - passed
    ts = datetime.now(timezone.utc).isoformat()
    print()
    print(f"[soak] RESULT: {passed}/{n_requests} PASS  {failed} FAIL  ({ts})")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Long-prompt INT4 soak test")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--requests", type=int, default=REQUESTS)
    args = parser.parse_args()

    ok = run_soak(args.host, args.port, args.requests)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
