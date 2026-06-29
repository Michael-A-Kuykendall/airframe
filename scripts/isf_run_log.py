#!/usr/bin/env python3
"""
ISF Run Logger — pipes shimmy server output through timing extraction.

Usage:
  SHIMMY_MAX_CTX=3000 SHIMMY_ROPE_SCALE=0.68 \
    ./target/release/shimmy.exe serve --model-path "..." --bind 127.0.0.1:11435 \
    2>&1 | python3 /c/Users/micha/repos/airframe/scripts/isf_run_log.py

Or redirect stderr to a log file and run this against it afterward:
  python3 isf_run_log.py < shimmy_stderr.txt

Outputs a summary CSV + prints to terminal.
"""

import sys
import re
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/c/Users/micha/repos/airframe/docs/internal/run-logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = LOG_DIR / f"run_{timestamp}.csv"
summary_path = LOG_DIR / f"summary_{timestamp}.txt"

runs = []
current_run = {}
decode_steps = []

lines = []
for line in sys.stdin:
    sys.stdout.write(line)  # pass through
    lines.append(line.strip())

    if "[ISF] generate_isf() called" in line:
        if current_run:
            current_run["decode_steps"] = decode_steps
            runs.append(current_run)
        m = re.search(r"prompt len=(\d+)", line)
        current_run = {
            "timestamp": timestamp,
            "prompt_len": int(m.group(1)) if m else 0,
            "token_count": 0,
            "unique_tokens": 0,
            "embed_time_s": 0.0,
            "prefill_time_s": 0.0,
            "total_time_s": 0.0,
            "tokens_generated": 0,
            "decode_steps_count": 0,
        }
        decode_steps = []

    elif "[ISF] tokenized:" in line:
        m = re.search(r"tokenized: (\d+)", line)
        if m:
            current_run["token_count"] = int(m.group(1))

    elif "[ISF] Batched" in line:
        m = re.search(r"Batched (\d+) embeddings \((\d+) unique tokens\) in ([\d.]+)s", line)
        if m:
            current_run["unique_tokens"] = int(m.group(2))
            current_run["embed_time_s"] = float(m.group(3))

    elif "[ISF-RULE] GPU prefill done" in line:
        m = re.search(r"done in ([\d.]+)s", line)
        if m:
            current_run["prefill_time_s"] = float(m.group(1))

    elif "[ISF-DECODE] step=" in line:
        m = re.search(r"step=(\d+) gpu_forward=([\d.]+)s", line)
        if m:
            decode_steps.append((int(m.group(1)), float(m.group(2))))

    elif "[ISF] run_to_fixpoint done" in line:
        m = re.search(r"done in ([\d.]+)s", line)
        if m:
            current_run["total_time_s"] = float(m.group(1))

    elif "[ISF] Done." in line:
        m = re.search(r"Generated (\d+) chars, (\d+) decode steps", line)
        if m:
            current_run["tokens_generated"] = int(m.group(2))
            current_run["decode_steps_count"] = int(m.group(2))

if current_run:
    current_run["decode_steps"] = decode_steps
    runs.append(current_run)

# Write CSV
with open(csv_path, "w") as f:
    f.write("timestamp,prompt_len,token_count,unique_tokens,embed_time_s,prefill_time_s,total_time_s,tokens_generated,avg_decode_ms,min_decode_ms,max_decode_ms\n")
    for r in runs:
        steps = r.get("decode_steps", [])
        if steps:
            times_ms = [t * 1000 for _, t in steps]
            avg_ms = sum(times_ms) / len(times_ms)
            min_ms = min(times_ms)
            max_ms = max(times_ms)
        else:
            avg_ms = min_ms = max_ms = 0.0
        f.write(f"{r['timestamp']},{r['prompt_len']},{r['token_count']},{r['unique_tokens']},"
                f"{r['embed_time_s']:.3f},{r['prefill_time_s']:.3f},{r['total_time_s']:.3f},"
                f"{r['tokens_generated']},{avg_ms:.1f},{min_ms:.1f},{max_ms:.1f}\n")

# Write summary
with open(summary_path, "w") as f:
    f.write(f"ISF Run Summary — {timestamp}\n")
    f.write("=" * 60 + "\n\n")
    for i, r in enumerate(runs):
        steps = r.get("decode_steps", [])
        times_ms = [t * 1000 for _, t in steps] if steps else [0]
        f.write(f"Run {i+1}:\n")
        f.write(f"  Prompt: {r['prompt_len']} chars → {r['token_count']} tokens ({r['unique_tokens']} unique)\n")
        f.write(f"  Embedding: {r['embed_time_s']:.3f}s\n")
        f.write(f"  Prefill:   {r['prefill_time_s']:.3f}s ({r['prefill_time_s']/max(r['token_count'],1)*1000:.1f}ms/tok)\n")
        f.write(f"  Decode:    {sum(times_ms)/1000:.2f}s total, avg={sum(times_ms)/len(times_ms):.0f}ms/tok, min={min(times_ms):.0f}ms, max={max(times_ms):.0f}ms\n")
        f.write(f"  Total:     {r['total_time_s']:.2f}s for {r['tokens_generated']} output tokens\n")
        f.write(f"  Throughput: {r['tokens_generated']/max(r['total_time_s'],0.001):.1f} tok/s\n\n")

print(f"\n[LOG] CSV: {csv_path}", file=sys.stderr)
print(f"[LOG] Summary: {summary_path}", file=sys.stderr)
