#!/usr/bin/env python3
"""GPU decode speed + correctness battery for shimmy_server_gpu.

Runs text-only and vision cases from fixtures/vision-samples/MANIFEST.json,
repeats each --runs times, and prints a structured findings table.

Usage:
  python scripts/vision_bench.py [--host HOST] [--port PORT]
                                  [--runs N] [--max-tokens N]

The server must already be running with SHIMMY_MMPROJ_PATH set.
"""

import argparse
import base64
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime

# Ensure Unicode output works on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(SCRIPT_DIR, "..", "fixtures", "vision-samples")
MANIFEST_PATH = os.path.join(FIXTURES_DIR, "MANIFEST.json")

# ── text-only baseline cases ──────────────────────────────────────────────────

TEXT_CASES = [
    {
        "label": "arithmetic (2+2)",
        "prompt": "What is 2+2? Answer with just the number.",
        "expected_keywords": ["4"],
        "expected_absent": [],
    },
    {
        "label": "capitals (France)",
        "prompt": "What is the capital of France? One word only.",
        "expected_keywords": ["Paris"],
        "expected_absent": [],
    },
    {
        "label": "code snippet (hello world)",
        "prompt": "Write a one-line Python hello-world print statement.",
        "expected_keywords": ["print"],
        "expected_absent": [],
    },
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_image_b64(path: str) -> tuple[str, int, int]:
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    if raw[:4] == b"\x89PNG":
        w = struct.unpack(">I", raw[16:20])[0]
        h = struct.unpack(">I", raw[20:24])[0]
    else:
        w, h = 448, 448
    return b64, h, w


def post_completion(host: str, port: int, payload: dict) -> tuple[dict, float]:
    url = f"http://{host}:{port}/v1/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return {"error": f"HTTP {e.code}: {body[:200]}"}, time.monotonic() - t0
    except Exception as e:
        return {"error": str(e)}, time.monotonic() - t0
    wall = time.monotonic() - t0
    result = json.loads(body)
    choices = result.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        result["text"] = msg.get("content", choices[0].get("text", ""))
        result["stop_reason"] = choices[0].get("finish_reason", "?")
    return result, wall


def check_keywords(text: str, required: list, absent: list) -> tuple[bool, list]:
    failures = []
    tl = text.lower()
    for kw in required:
        if kw.lower() not in tl:
            failures.append(f"MISSING {kw!r}")
    for kw in absent:
        if kw.lower() in tl:
            failures.append(f"UNEXPECTED {kw!r}")
    return len(failures) == 0, failures


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


# ── per-case runner ───────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    label: str
    mode: str                  # "text" or "vision"
    runs: int
    ms_per_tok_samples: list[float] = field(default_factory=list)
    wall_samples: list[float] = field(default_factory=list)
    token_samples: list[int] = field(default_factory=list)
    kw_pass: bool = True
    kw_failures: list[str] = field(default_factory=list)
    last_response: str = ""
    errors: list[str] = field(default_factory=list)

    def mean_ms(self) -> float:
        return mean_std(self.ms_per_tok_samples)[0]

    def std_ms(self) -> float:
        return mean_std(self.ms_per_tok_samples)[1]

    def min_ms(self) -> float:
        return min(self.ms_per_tok_samples) if self.ms_per_tok_samples else 0.0

    def max_ms(self) -> float:
        return max(self.ms_per_tok_samples) if self.ms_per_tok_samples else 0.0

    def mean_wall(self) -> float:
        return sum(self.wall_samples) / len(self.wall_samples) if self.wall_samples else 0.0

    def mean_tokens(self) -> float:
        return sum(self.token_samples) / len(self.token_samples) if self.token_samples else 0.0

    def ok(self) -> bool:
        return self.kw_pass and not self.errors


def run_case(
    host: str, port: int, label: str, mode: str,
    payload_base: dict, runs: int,
    expected_keywords: list, expected_absent: list,
    verbose: bool,
) -> CaseResult:
    result = CaseResult(label=label, mode=mode, runs=runs)
    print(f"\n  [{label}]", flush=True)

    for i in range(runs):
        resp, wall = post_completion(host, port, payload_base)
        if "error" in resp:
            result.errors.append(resp["error"])
            print(f"    run {i+1}/{runs}  ERROR: {resp['error']}", flush=True)
            continue

        text = resp.get("text", "")
        n_tok = int(resp.get("tokens_generated", resp.get("usage", {}).get("completion_tokens", 0) or 0))
        ms_tok = (wall / n_tok * 1000.0) if n_tok > 0 else 0.0

        result.wall_samples.append(wall)
        result.token_samples.append(n_tok)
        result.ms_per_tok_samples.append(ms_tok)
        result.last_response = text

        stop = resp.get("stop_reason", "?")
        print(f"    run {i+1}/{runs}  wall={wall:.2f}s  tokens={n_tok}  ms/tok={ms_tok:.1f}  stop={stop}", flush=True)
        if verbose:
            preview = text[:120].replace("\n", "\\n")
            print(f"           response={preview!r}", flush=True)

    # keyword check on last response
    if result.last_response:
        result.kw_pass, result.kw_failures = check_keywords(
            result.last_response, expected_keywords, expected_absent
        )
        if not result.kw_pass:
            for f in result.kw_failures:
                print(f"    !! KW FAIL: {f}", flush=True)
        else:
            print(f"    KW: PASS", flush=True)

    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Airframe GPU decode battery")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8086, type=int)
    parser.add_argument("--runs", default=3, type=int, help="Repetitions per case")
    parser.add_argument("--max-tokens", default=40, type=int)
    parser.add_argument("--verbose", action="store_true", help="Print response text per run")
    args = parser.parse_args()

    print(f"╔{'═'*70}╗")
    print(f"║  AIRFRAME GPU DECODE BATTERY{'':41}║")
    print(f"║  host={args.host}:{args.port}  runs={args.runs}  max_tokens={args.max_tokens}  {datetime.now().strftime('%Y-%m-%d %H:%M')}{'':6}║")
    print(f"╚{'═'*70}╝")

    all_results: list[CaseResult] = []

    # ── text-only cases ──────────────────────────────────────────────────────
    print("\n── TEXT-ONLY BASELINE ──────────────────────────────────────────────")
    for tc in TEXT_CASES:
        payload = {
            "prompt": tc["prompt"],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": 42,
        }
        r = run_case(
            args.host, args.port,
            label=tc["label"], mode="text",
            payload_base=payload, runs=args.runs,
            expected_keywords=tc["expected_keywords"],
            expected_absent=tc["expected_absent"],
            verbose=args.verbose,
        )
        all_results.append(r)

    # ── vision cases from manifest ───────────────────────────────────────────
    print("\n── VISION CASES (MANIFEST) ─────────────────────────────────────────")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for entry in manifest["images"]:
        fname = entry["file"]
        fpath = os.path.join(FIXTURES_DIR, fname)
        if not os.path.exists(fpath):
            print(f"\n  [{fname}] SKIP — file not found")
            continue

        b64, h, w = load_image_b64(fpath)
        if h == 0:
            h, w = entry["dims"]["h"], entry["dims"]["w"]

        payload = {
            "prompt": entry["smoke_prompt"],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": 42,
            "image_payload": {"pixels_b64": b64, "h": h, "w": w},
        }
        r = run_case(
            args.host, args.port,
            label=fname, mode="vision",
            payload_base=payload, runs=args.runs,
            expected_keywords=entry.get("expected_keywords", []),
            expected_absent=entry.get("expected_absent", []),
            verbose=args.verbose,
        )
        all_results.append(r)

    # ── findings table ───────────────────────────────────────────────────────
    text_ms = [r.mean_ms() for r in all_results if r.mode == "text" and r.ms_per_tok_samples]
    vis_ms  = [r.mean_ms() for r in all_results if r.mode == "vision" and r.ms_per_tok_samples]
    overall_pass = all(r.ok() for r in all_results if r.ms_per_tok_samples)

    W = 80
    print(f"\n{'═'*W}")
    print(f"  STRUCTURED FINDINGS  (runs={args.runs}, max_tokens={args.max_tokens})")
    print(f"{'═'*W}")
    hdr = f"  {'Case':<34} {'Mode':<6} {'ms/tok':>7} {'±σ':>6} {'min':>6} {'max':>6}  {'toks':>4}  KW"
    print(hdr)
    print(f"  {'-'*76}")

    for r in all_results:
        if not r.ms_per_tok_samples:
            print(f"  {r.label:<34} {r.mode:<6} {'ERROR':>7}")
            continue
        kw_col = "PASS" if r.kw_pass else "FAIL"
        toks = f"{r.mean_tokens():.0f}"
        print(
            f"  {r.label:<34} {r.mode:<6} {r.mean_ms():>7.1f} {r.std_ms():>6.1f}"
            f" {r.min_ms():>6.1f} {r.max_ms():>6.1f}  {toks:>4}  {kw_col}"
        )

    print(f"  {'-'*76}")

    if text_ms:
        tm = sum(text_ms) / len(text_ms)
        print(f"  text-only mean:   {tm:>7.1f} ms/tok  ({len(text_ms)} cases)")
    if vis_ms:
        vm = sum(vis_ms) / len(vis_ms)
        print(f"  vision mean:      {vm:>7.1f} ms/tok  ({len(vis_ms)} cases)")
    if text_ms and vis_ms:
        overhead_pct = (vm - tm) / tm * 100.0
        sign = "+" if overhead_pct >= 0 else ""
        print(f"  vision overhead:  {sign}{overhead_pct:.1f}% vs text-only")

    kw_failures = [(r.label, r.kw_failures) for r in all_results if not r.kw_pass]
    if kw_failures:
        print(f"\n  KEYWORD FAILURES:")
        for label, failures in kw_failures:
            for f in failures:
                print(f"    {label}: {f}")

    errors = [(r.label, r.errors) for r in all_results if r.errors]
    if errors:
        print(f"\n  ERRORS:")
        for label, errs in errors:
            for e in errs:
                print(f"    {label}: {e}")

    verdict = "ALL PASS" if overall_pass else "FAILURES DETECTED"
    print(f"\n  VERDICT: {verdict}")
    print(f"{'═'*W}\n")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
