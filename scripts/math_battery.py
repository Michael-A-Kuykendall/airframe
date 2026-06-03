"""
math_battery.py  —  arithmetic accuracy probe for airframe inference server
Usage: python scripts/math_battery.py [--port 8080] [--model MODEL_ID] [--out results.json]
"""
import argparse, json, time, urllib.request, urllib.error

PROBLEMS = [
    # (description, prompt, expected_answer)
    # ── single × single ──────────────────────────────────────────────────────
    ("1d×1d  2×3",   "What is 2 times 3? Reply with only the number.",           6),
    ("1d×1d  4×7",   "What is 4 times 7? Reply with only the number.",          28),
    ("1d×1d  8×9",   "What is 8 times 9? Reply with only the number.",          72),
    ("1d×1d  6×6",   "What is 6 times 6? Reply with only the number.",          36),
    # ── two-digit × single ───────────────────────────────────────────────────
    ("2d×1d  17×3",  "What is 17 times 3? Reply with only the number.",         51),
    ("2d×1d  24×5",  "What is 24 times 5? Reply with only the number.",        120),
    ("2d×1d  37×4",  "What is 37 times 4? Reply with only the number.",        148),
    ("2d×1d  99×2",  "What is 99 times 2? Reply with only the number.",        198),
    # ── two-digit × two-digit ────────────────────────────────────────────────
    ("2d×2d  12×15", "What is 12 times 15? Reply with only the number.",       180),
    ("2d×2d  17×23", "What is 17 times 23? Reply with only the number.",       391),
    ("2d×2d  24×25", "What is 24 times 25? Reply with only the number.",       600),
    ("2d×2d  33×33", "What is 33 times 33? Reply with only the number.",      1089),
    ("2d×2d  48×52", "What is 48 times 52? Reply with only the number.",      2496),
    # ── addition ─────────────────────────────────────────────────────────────
    ("add  127+456",  "What is 127 plus 456? Reply with only the number.",      583),
    ("add  999+1",    "What is 999 plus 1? Reply with only the number.",       1000),
    ("add  38+47",    "What is 38 plus 47? Reply with only the number.",         85),
    # ── subtraction ──────────────────────────────────────────────────────────
    ("sub  100-37",   "What is 100 minus 37? Reply with only the number.",       63),
    ("sub  1000-1",   "What is 1000 minus 1? Reply with only the number.",      999),
    # ── division ─────────────────────────────────────────────────────────────
    ("div  144÷12",   "What is 144 divided by 12? Reply with only the number.", 12),
    ("div  81÷9",     "What is 81 divided by 9? Reply with only the number.",    9),
    # ── carry-heavy ──────────────────────────────────────────────────────────
    ("carry  19×19",  "What is 19 times 19? Reply with only the number.",       361),
    ("carry  77×77",  "What is 77 times 77? Reply with only the number.",      5929),
]

def ask(port: int, model: str, prompt: str) -> tuple[str, float]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        elapsed = time.monotonic() - t0
        data = json.loads(resp.read())
    raw = data["choices"][0]["message"]["content"].strip()
    return raw, elapsed

def parse_num(s: str):
    """Extract the answer integer from model response.

    Strategy: prefer the LAST standalone number in the response — models
    often echo the question first (e.g. '17 * 3 = 51') so the final number
    is the answer, not the first operand.  Fall back to first number if
    nothing else matches.
    """
    import re
    # Find all runs of digits (with optional commas/sign), ignoring decimals
    hits = re.findall(r"-?\d[\d,]*", s)
    if not hits:
        return None
    # The answer is almost always the last numeric token in the output
    raw = hits[-1].replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None

def run(port: int, model: str) -> list[dict]:
    results = []
    print(f"\n{'─'*64}")
    print(f"  Model : {model}   Port: {port}")
    print(f"{'─'*64}")
    print(f"  {'Problem':<18} {'Expected':>8} {'Got':<14} {'Parsed':>8} {'OK':>4} {'ms':>6}")
    print(f"  {'─'*18} {'─'*8} {'─'*14} {'─'*8} {'─'*4} {'─'*6}")
    for desc, prompt, expected in PROBLEMS:
        try:
            raw, elapsed = ask(port, model, prompt)
            parsed = parse_num(raw)
            ok = (parsed == expected)
        except Exception as e:
            raw, elapsed, parsed, ok = f"ERR:{e}", 0.0, None, False
        results.append({
            "desc": desc, "expected": expected,
            "raw": raw, "parsed": parsed, "ok": ok, "ms": round(elapsed*1000),
        })
        mark = "✓" if ok else "✗"
        print(f"  {desc:<18} {expected:>8} {raw:<14} {str(parsed):>8} {mark:>4} {round(elapsed*1000):>6}")
    correct = sum(r["ok"] for r in results)
    print(f"{'─'*64}")
    print(f"  Score: {correct}/{len(PROBLEMS)}  ({100*correct//len(PROBLEMS)}%)")
    print(f"{'─'*64}\n")
    return results

if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", default="Llama-3.2-1B-Instruct-Q4_K_M")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = run(args.port, args.model)
    if args.out:
        # Resolve relative to repo root (parent of scripts/)
        out_path = args.out if os.path.isabs(args.out) else os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"model": args.model, "results": results}, f, indent=2)
        print(f"Results saved → {out_path}")
