#!/usr/bin/env python3
"""
vault seed_all.py
─────────────────
Runs vault_seed.exe on every GGUF in the collection directory.
Skips models that already have a seed file unless --force is passed.
Reports results, then runs import_seeds.py automatically.

Usage:
    python vault/scripts/seed_all.py [gguf_dir] [--force]

Defaults:
    gguf_dir = D:/shimmy-test-models/gguf_collection
"""

import subprocess
import sys
import os
import glob
import time

VAULT_SEED_EXE = "./target/release/vault_seed.exe"
SEEDS_DIR      = "vault/seeds"
IMPORT_SCRIPT  = "vault/scripts/import_seeds.py"
VAULT_DB       = "vault/vault.duckdb"

# Models we skip for now (too large for casual batch — run manually)
SKIP_PATTERNS = [
    "Mixtral-8x7B",
    "Qwen2.5-Coder-32B",
    "qwen2.5-coder-32b",
]

def should_skip(filename):
    for pattern in SKIP_PATTERNS:
        if pattern.lower() in filename.lower():
            return True
    return False


def main():
    gguf_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/shimmy-test-models/gguf_collection"
    force    = "--force" in sys.argv

    gguf_files = sorted(glob.glob(os.path.join(gguf_dir, "*.gguf")))
    if not gguf_files:
        print(f"No GGUF files found in {gguf_dir}")
        sys.exit(1)

    os.makedirs(SEEDS_DIR, exist_ok=True)

    print(f"Found {len(gguf_files)} GGUF files")
    print(f"Seeds dir: {SEEDS_DIR}")
    print(f"Force regenerate: {force}")
    print()

    results = []

    for gguf_path in gguf_files:
        filename = os.path.basename(gguf_path)
        stem     = os.path.splitext(filename)[0]
        seed_out = os.path.join(SEEDS_DIR, f"{stem}.json")

        if should_skip(filename):
            print(f"[SKIP-LARGE] {filename}")
            results.append((filename, "skip_large", 0))
            continue

        if os.path.exists(seed_out) and not force:
            print(f"[SKIP-EXISTS] {filename} → {seed_out}")
            results.append((filename, "skip_exists", 0))
            continue

        mb = os.path.getsize(gguf_path) // 1048576
        print(f"[SEEDING] {filename} ({mb} MB) ...", flush=True)
        t0 = time.time()

        proc = subprocess.run(
            [VAULT_SEED_EXE, gguf_path, seed_out],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - t0

        if proc.returncode == 0:
            # Extract summary line from stderr
            summary = [l for l in proc.stderr.splitlines() if l.startswith("✅")]
            print(f"  OK ({elapsed:.0f}s) — {summary[0] if summary else 'done'}")
            results.append((filename, "ok", elapsed))
        else:
            # Print last few lines of stderr for diagnosis
            tail = proc.stderr.splitlines()[-5:]
            print(f"  FAIL (exit={proc.returncode}):")
            for line in tail:
                print(f"    {line}")
            results.append((filename, "fail", elapsed))

    # Summary
    print()
    print("═" * 60)
    ok    = [r for r in results if r[1] == "ok"]
    fail  = [r for r in results if r[1] == "fail"]
    skip  = [r for r in results if r[1].startswith("skip")]
    print(f"Seeds generated : {len(ok)}")
    print(f"Skipped         : {len(skip)}")
    print(f"Failed          : {len(fail)}")
    if fail:
        print("Failed models:")
        for f, _, _ in fail:
            print(f"  {f}")
    print()

    if len(ok) == 0 and len(fail) == 0:
        print("Nothing new to import.")
        return

    # Run import
    print("Running import ...")
    proc = subprocess.run(
        ["python", IMPORT_SCRIPT, SEEDS_DIR, VAULT_DB],
        text=True,
    )
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
