#!/usr/bin/env python3
"""
math_entropy_probe.py — diagnose exactly where the model's confidence
collapses during arithmetic generation.

Usage:
  1. Start the server with SHIMMY_MATH_BYPASS_DISABLE=1 so raw model output
     is captured (no forced tokens):
       $env:SHIMMY_MATH_BYPASS_DISABLE="1"; cargo run --release --bin shimmy_server_gpu

  2. In another terminal, run this script:
       python scripts/math_entropy_probe.py --log server_stderr.txt

  OR: pipe server stderr directly:
       cargo run ... 2>server_stderr.txt &
       python scripts/math_entropy_probe.py --log server_stderr.txt

  The script hits the server with a set of math problems, waits for the
  [TOKEN] lines in the server log, and prints a per-token confidence table
  showing exactly where entropy spikes — i.e. where the model is guessing.

Run against localhost:8080 by default.  Use --port to override.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request

PROBLEMS = [
    # (prompt, correct_answer, difficulty_label)
    ("What is 6 times 6? Reply with only the number.", "36", "easy-1d"),
    ("What is 9 times 9? Reply with only the number.", "81", "easy-1d"),
    ("What is 37 times 4? Reply with only the number.", "148", "medium-carry"),
    ("What is 48 times 52? Reply with only the number.", "2496", "hard-carry"),
    ("What is 77 times 77? Reply with only the number.", "5929", "hard-carry"),
    ("What is 127 plus 456? Reply with only the number.", "583", "add-carry"),
    ("What is 999 plus 1? Reply with only the number.", "1000", "add-carry-boundary"),
    ("What is 100 minus 37? Reply with only the number.", "63", "subtract"),
    ("What is 1000 minus 1? Reply with only the number.", "999", "subtract-large"),
    ("What is 198 divided by 2? Reply with only the number.", "99", "div"),
]

TOKEN_RE = re.compile(
    r"\[TOKEN\] Step (\d+): id=(\d+), text=(.*?), entropy=([\d.]+), max_prob=([\d.]+)"
)


def query(prompt: str, port: int, model: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 24,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


def tail_tokens(log_path: str, after_line: int) -> list[dict]:
    """Read [TOKEN] lines from log file starting after a known line index."""
    tokens = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < after_line:
                continue
            m = TOKEN_RE.search(line)
            if m:
                tokens.append({
                    "step": int(m.group(1)),
                    "id": int(m.group(2)),
                    "text": m.group(3),
                    "entropy": float(m.group(4)),
                    "max_prob": float(m.group(5)),
                })
    return tokens


def log_line_count(log_path: str) -> int:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default="TinyLlama-1.1B-Chat-v1.0.Q4_0")
    ap.add_argument("--log", required=True, help="Path to server stderr log file")
    args = ap.parse_args()

    print(f"{'Prompt':<52} {'Correct':>8} {'Got':>8}  {'Tok':>4}  Entropy trace")
    print("-" * 120)

    for prompt, correct, label in PROBLEMS:
        before = log_line_count(args.log)
        got = query(prompt, args.port, args.model)
        time.sleep(0.3)  # let log flush
        tokens = tail_tokens(args.log, before)

        # Only answer tokens (skip any prompt echoing)
        entropy_trace = "  ".join(
            f"{t['text'].strip() or '_':>3}:{t['entropy']:.2f}/{t['max_prob']:.2f}"
            for t in tokens[:8]
        )

        match = "✓" if got.strip() == correct else "✗"
        short_prompt = prompt[:50]
        print(f"{short_prompt:<52} {correct:>8} {got.strip():>8} {match}  {len(tokens):>3}t  {entropy_trace}")

    print()
    print("Format: token_text:entropy/max_prob per step")
    print("High entropy (>2.0) + low max_prob (<0.3) = model is guessing")


if __name__ == "__main__":
    main()
