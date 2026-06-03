#!/usr/bin/env python3
"""
Needle-in-a-Haystack benchmark for the Airframe GPU server.

Tests context-window retrieval accuracy at multiple scales and document
depths, producing a results table that characterizes how F32 precision
holds up as context grows.

Usage:
    python scripts/needle_bench.py [--url URL] [--ctx 2048,4096,8192]
                                   [--depths 15,50,85] [--runs 1]
                                   [--out artifacts/needle_results.json]
                                   [--timeout 3600]

The server must be running. Set SHIMMY_MAX_CTX on server startup to at least
the largest ctx size you intend to test.

Example startup for tests up to 8192 tokens:
    SHIMMY_MAX_CTX=8192 ./target/release/shimmy_server_gpu.exe

The script exits with code 0 even on individual test failures so all tests run.
The final summary line exits 1 if any test failed.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

# ---------------------------------------------------------------------------
# Filler corpus — intentionally artificial, won't appear in training data
# as meaningful text.  Long enough to be sliced and repeated.
# ---------------------------------------------------------------------------
FILLER_PARAGRAPH = (
    "The Veldtwood Cartographic Archive holds seventeen thousand folio-sized "
    "charts catalogued under a hexadecimal accession system introduced in the "
    "archive's third decade.  Each chart is stored flat in acid-free interleaving "
    "tissue and filed by region code, then by survey year, then by the surveyor's "
    "alphabetical surname initial.  Visitors must submit a retrieval request by "
    "nine o'clock on working days; requests received after that hour are processed "
    "the following morning.  The reference reading room seats twenty-four and is "
    "kept at eighteen degrees Celsius and forty-five percent relative humidity to "
    "preserve the vellum originals.  Reproduction quality depends on the chart "
    "material: linen-backed paper charts may be photographed freely; vellum "
    "originals require written approval from the conservation officer before any "
    "imaging equipment is brought into the reading room.  A catalogue raisonné "
    "covering the first eight decades of acquisitions was published in three "
    "volumes and is available at the front desk.  The fourth volume, covering "
    "recent acquisitions, exists only in draft form and is accessible by "
    "appointment.  Researchers intending to visit for more than two consecutive "
    "days are asked to complete the extended-access registration form, which can "
    "be obtained from the archivist's office or downloaded from the secure "
    "institutional portal.  "
)

FILLER_UNIT_CHARS = len(FILLER_PARAGRAPH)
# Approximate tokens per character for BPE (conservative — keeps us under budget)
CHARS_PER_TOKEN = 4.2

# Prompt envelope overhead (system + question + template tokens)
# Measured empirically; keeps us safely under max_ctx
PROMPT_OVERHEAD_TOKENS = 120


def build_prompt(ctx_tokens: int, depth_frac: float, needle: str) -> tuple[str, int]:
    """
    Return (prompt_text, estimated_token_count).

    depth_frac: 0.0 = needle at start of filler, 1.0 = needle at end.
    """
    filler_budget_tokens = ctx_tokens - PROMPT_OVERHEAD_TOKENS
    filler_budget_chars = int(filler_budget_tokens * CHARS_PER_TOKEN)

    # Build enough filler to fill the budget
    repeats = math.ceil(filler_budget_chars / FILLER_UNIT_CHARS) + 1
    full_filler = FILLER_PARAGRAPH * repeats

    # Determine needle insertion position
    insert_char = int(depth_frac * filler_budget_chars)
    insert_char = max(0, min(insert_char, filler_budget_chars))

    # Needle sentence
    needle_sentence = f"\n\n[ARCHIVE NOTE: The special retrieval code for this collection is {needle}.]\n\n"

    before = full_filler[:insert_char]
    after = full_filler[insert_char:filler_budget_chars]
    filler_with_needle = before + needle_sentence + after

    system = "You are a precise research assistant. Answer exactly as instructed."
    user = (
        f"Below is a portion of a cartographic archive document.\n\n"
        f"{filler_with_needle}\n\n"
        f"What is the special retrieval code mentioned in the ARCHIVE NOTE above? "
        f"Reply with ONLY the code, no other words."
    )

    prompt = (
        f"<|system|>\n{system}</s>\n"
        f"<|user|>\n{user}</s>\n"
        f"<|assistant|>\n"
    )
    est_tokens = len(prompt) // CHARS_PER_TOKEN
    return prompt, int(est_tokens)


def run_inference(url: str, prompt: str, seed: int = 42, timeout_s: int = 300) -> dict:
    """Submit a completion request and block until the response arrives.

    Returns a dict with keys: status, text, tokens_generated.
    The server's non-streaming POST / returns a full OpenAI-compatible
    chat.completion response (choices[0].message.content) once done.
    """
    body = json.dumps({
        "prompt": prompt,
        "prompt_mode": "raw",
        "max_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())
    except Exception as e:
        raise RuntimeError(f"Inference request failed: {e}") from e
    # The server returns OpenAI-compatible chat.completion
    try:
        text = resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response format: {resp}") from e
    return {
        "status": "completed",
        "text": text,
        "tokens_generated": resp.get("usage", {}).get("completion_tokens", 0),
        "stop_reason": resp.get("choices", [{}])[0].get("finish_reason", ""),
    }


def extract_answer(result: dict) -> str:
    """Pull the model's text response out of a completed result."""
    if result.get("status") != "completed":
        return ""
    text = result.get("text", "")
    # Strip whitespace and take the first line
    # Strip spurious stop-token strings that leak into model output
    cleaned = text.strip().replace("</s>", "").replace("<|end|>", "").strip()
    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    return first_line


def check_pass(answer: str, needle: str) -> bool:
    """Pass if the answer contains the needle exactly, or if all
    alphanumeric segments of the needle appear in the answer in order.
    The latter tolerates minor model hallucinations like inserted characters."""
    import re
    if needle.upper() in answer.upper():
        return True
    # Fuzzy: split needle into tokens and check all are present in order
    parts = re.split(r'[-_]', needle.upper())
    haystack = answer.upper()
    pos = 0
    for part in parts:
        idx = haystack.find(part, pos)
        if idx == -1:
            return False
        pos = idx + len(part)
    return True


def wait_for_server(url: str, max_wait_s: int = 120) -> bool:
    """Block until server is responsive or max_wait_s expires."""
    probe = url.rstrip("/") + "/v1/models"
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(probe, timeout=5).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def run_benchmark(
    url: str,
    ctx_sizes: list[int],
    depths: list[float],
    runs: int,
    per_job_timeout: int,
    out_path: str,
) -> list[dict]:
    results = []
    total = len(ctx_sizes) * len(depths) * runs
    done = 0

    for ctx in ctx_sizes:
        for depth_pct in depths:
            depth = depth_pct / 100.0
            for run_idx in range(runs):
                done += 1
                # Vary needle per (ctx, depth, run) to prevent any cross-contamination
                needle = f"AIRFRAME-{ctx}-D{int(depth_pct):02d}-R{run_idx}"
                seed = int((ctx * 1000 + depth_pct * 10 + run_idx) % (2**31))

                prompt, est_tokens = build_prompt(ctx, depth, needle)
                label = f"[{done}/{total}] ctx={ctx} depth={depth_pct}% run={run_idx}"

                print(f"\n{label}")
                print(f"  Needle: {needle}")
                print(f"  Estimated tokens: {est_tokens}")

                t0 = time.time()
                try:
                    result = run_inference(url, prompt, seed=seed, timeout_s=per_job_timeout)
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  ERROR: {e}")
                    row = {
                        "ctx": ctx, "depth_pct": depth_pct, "run": run_idx,
                        "needle": needle, "est_tokens": est_tokens,
                        "status": "error", "error": str(e),
                        "answer": "", "pass": False,
                        "elapsed_s": round(elapsed, 1),
                        "tokens_generated": 0,
                    }
                    results.append(row)
                    continue

                elapsed = time.time() - t0
                status = result.get("status", "unknown")
                answer = extract_answer(result)
                passed = check_pass(answer, needle)
                tokens_gen = result.get("tokens_generated", 0)
                stop_reason = result.get("stop_reason", "")

                row = {
                    "ctx": ctx,
                    "depth_pct": depth_pct,
                    "run": run_idx,
                    "needle": needle,
                    "est_tokens": est_tokens,
                    "status": status,
                    "answer": answer,
                    "pass": passed,
                    "elapsed_s": round(elapsed, 1),
                    "tokens_generated": tokens_gen,
                    "stop_reason": stop_reason,
                }
                results.append(row)

                icon = "PASS" if passed else "FAIL"
                print(f"  {icon} | {elapsed:.0f}s | tokens_gen={tokens_gen} | stop={stop_reason}")
                print(f"  Answer: {repr(answer[:120])}")

                # Save incrementally so a crash doesn't lose progress
                _save_results(out_path, results)

    return results


def _save_results(out_path: str, results: list[dict]) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat(), "results": results}, f, indent=2)


def print_summary(results: list[dict]) -> bool:
    """Print a grid summary table. Returns True if all tests passed."""
    print("\n" + "=" * 72)
    print("NEEDLE BENCHMARK SUMMARY")
    print("=" * 72)

    ctx_sizes = sorted(set(r["ctx"] for r in results))
    depths = sorted(set(r["depth_pct"] for r in results))

    # Header
    hdr = f"{'ctx':>8} |"
    for d in depths:
        hdr += f"  {d:>3}%  |"
    hdr += "  total  | avg_s"
    print(hdr)
    print("-" * len(hdr))

    all_passed = True
    for ctx in ctx_sizes:
        row_results = [r for r in results if r["ctx"] == ctx]
        row = f"{ctx:>8} |"
        passed_total = 0
        total = 0
        elapsed_list = []
        for d in depths:
            cell = [r for r in row_results if r["depth_pct"] == d]
            if not cell:
                row += "   -   |"
                continue
            n_pass = sum(1 for r in cell if r["pass"])
            n_total = len(cell)
            passed_total += n_pass
            total += n_total
            elapsed_list.extend(r["elapsed_s"] for r in cell)
            marker = "OK" if n_pass == n_total else f"{n_pass}/{n_total}"
            row += f"  {marker:^5}|"
        avg_s = sum(elapsed_list) / len(elapsed_list) if elapsed_list else 0
        row += f"  {passed_total}/{total:>2}   | {avg_s:>5.0f}s"
        print(row)
        if passed_total < total:
            all_passed = False

    print("=" * 72)
    n_pass = sum(1 for r in results if r["pass"])
    n_total = len(results)
    print(f"Overall: {n_pass}/{n_total} passed")
    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Airframe needle-in-a-haystack benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8099", help="Server base URL")
    parser.add_argument("--ctx", default="2048,4096,8192",
                        help="Comma-separated context sizes to test (tokens)")
    parser.add_argument("--depths", default="15,50,85",
                        help="Comma-separated needle depths as percent (0-100)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per (ctx, depth) cell")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-job timeout in seconds (default 3600 = 1 hour)")
    parser.add_argument("--out", default="artifacts/needle_bench_results.json",
                        help="Output JSON path")
    parser.add_argument("--wait", type=int, default=30,
                        help="Seconds to wait for server to become ready")
    args = parser.parse_args()

    ctx_sizes = [int(x.strip()) for x in args.ctx.split(",")]
    depths = [float(x.strip()) for x in args.depths.split(",")]

    print(f"Airframe Needle Benchmark")
    print(f"  Server:  {args.url}")
    print(f"  Ctx:     {ctx_sizes}")
    print(f"  Depths:  {depths}%")
    print(f"  Runs:    {args.runs}")
    print(f"  Timeout: {args.timeout}s per job")
    print(f"  Output:  {args.out}")

    # Timing estimates so the user knows what they're in for
    print("\nEstimated prefill times (empirical RTX 3060 baseline):")
    for ctx in ctx_sizes:
        # t ≈ 10ms/token dequant + 3.5ms/token/layer × 22 layers
        dq = ctx * 0.010
        layer = ctx * 0.0035 * 22
        est = dq + layer
        print(f"  {ctx:>6} tokens: ~{est:.0f}s per test, "
              f"~{est * len(depths) * args.runs:.0f}s total for this ctx tier")

    print(f"\nWaiting for server at {args.url} ...")
    if not wait_for_server(args.url, max_wait_s=args.wait):
        print("ERROR: Server not responsive. Start shimmy_server_gpu first.")
        print("  SHIMMY_MAX_CTX=<max_ctx> ./target/release/shimmy_server_gpu.exe")
        return 1
    print("Server ready.\n")

    results = run_benchmark(
        url=args.url,
        ctx_sizes=ctx_sizes,
        depths=depths,
        runs=args.runs,
        per_job_timeout=args.timeout,
        out_path=args.out,
    )

    all_passed = print_summary(results)
    print(f"\nFull results saved to: {args.out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
