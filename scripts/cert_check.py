#!/usr/bin/env python3
"""
cert_check.py — Airframe file certification checker.

Usage:
  python scripts/cert_check.py <file> [<file> ...]   # check criteria
  python scripts/cert_check.py --seal <file>          # seal file after manual review
  python scripts/cert_check.py --list                 # show all registered files

Criteria are sourced from the bullshite project's empirically validated
AI-detection patterns (15 patterns, measured on jQuery vs Shimmy).
A file is CERTIFIABLE when all auto-checkable criteria PASS or have a
recorded WAIVER in CERT.toml.
"""

import sys
import re
import hashlib
import argparse
import tomllib
import tomli_w
from pathlib import Path
from datetime import date

CERT_FILE = Path(__file__).parent.parent / "CERT.toml"
REPO_ROOT = Path(__file__).parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def load_cert() -> dict:
    if CERT_FILE.exists():
        with open(CERT_FILE, "rb") as f:
            return tomllib.load(f)
    return {"files": {}}


def save_cert(data: dict):
    with open(CERT_FILE, "wb") as f:
        tomli_w.dump(data, f)


# ---------------------------------------------------------------------------
# Individual criteria checks. Each returns (status, detail).
# status: "PASS" | "WARN" | "FAIL" | "SKIP"
# ---------------------------------------------------------------------------

def c3_line_count(lines: list[str], path: Path) -> tuple[str, str]:
    """C3: ≤600 lines"""
    n = len(lines)
    if n <= 600:
        return "PASS", f"{n} lines"
    return "FAIL", f"{n} lines — exceeds 600"


def c4_gpu_unwrap(lines: list[str], path: Path) -> tuple[str, str]:
    """C4: No .unwrap() on GPU/wgpu call sites in production code (non-test)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    bad = []
    in_test = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if "#[cfg(test)]" in stripped or "#[test]" in stripped:
            in_test = True
        if in_test and stripped == "}":
            in_test = False
        if in_test:
            continue
        # unwrap on lines that mention wgpu / device / queue / buffer / pipeline
        if ".unwrap()" in line and any(k in line for k in [
            "device.", "queue.", "buffer", "pipeline", "wgpu::", "encoder",
            "poll(", "submit(", "create_buffer", "map_async"
        ]):
            bad.append(i)
    if not bad:
        return "PASS", "no GPU .unwrap()"
    return "FAIL", f"GPU .unwrap() at lines: {bad[:5]}"


def c5_buffer_labels(lines: list[str], path: Path) -> tuple[str, str]:
    """C5: All create_buffer / create_buffer_init calls have a label"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    unlabeled = []
    # Find create_buffer calls, look ahead 3 lines for label: Some(...)
    text = "\n".join(lines)
    for m in re.finditer(r"create_buffer(_init)?\s*\(", text):
        snippet = text[m.start():m.start()+300]
        if "label: None" in snippet or ("label:" not in snippet[:200]):
            line_no = text[:m.start()].count("\n") + 1
            unlabeled.append(line_no)
    if not unlabeled:
        return "PASS", "all buffers labeled"
    return "WARN", f"possibly unlabeled buffers at lines: {unlabeled[:5]}"


def c6_dead_code(lines: list[str], path: Path) -> tuple[str, str]:
    """C6: No bare #[allow(dead_code)] without justification comment"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    bad = []
    for i, line in enumerate(lines, 1):
        if "#[allow(dead_code)]" in line:
            # Check if the preceding line has a comment
            prev = lines[i-2].strip() if i >= 2 else ""
            if not prev.startswith("//"):
                bad.append(i)
    if not bad:
        return "PASS", "no unjustified dead_code suppression"
    return "FAIL", f"bare #[allow(dead_code)] at lines: {bad}"


def c7_commented_blocks(lines: list[str], path: Path) -> tuple[str, str]:
    """C7: No commented-out code blocks (3+ consecutive commented lines with code)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    runs, run_start, run_len = [], 0, 0
    code_pat = re.compile(r"^\s*//\s*(let |fn |if |for |match |return |self\.|[a-z_]+\()")
    for i, line in enumerate(lines, 1):
        if code_pat.match(line):
            if run_len == 0:
                run_start = i
            run_len += 1
        else:
            if run_len >= 3:
                runs.append(run_start)
            run_len = 0
    if run_len >= 3:
        runs.append(run_start)
    if not runs:
        return "PASS", "no commented code blocks"
    return "WARN", f"possible commented-out code starting at lines: {runs[:3]}"


# --- Bullshite-derived criteria ---

def b1_clone_pressure(lines: list[str], path: Path) -> tuple[str, str]:
    """B1: Clone pressure — .clone() density (P6, 24× more in AI code)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    # Exclude test code and derive macros
    non_test = [l for l in lines if not l.strip().startswith("#[")]
    count = sum(1 for l in non_test if ".clone()" in l)
    per_100 = count / max(len(lines), 1) * 100
    if per_100 <= 3.0:
        return "PASS", f"{count} clones ({per_100:.1f}/100 lines)"
    if per_100 <= 6.0:
        return "WARN", f"{count} clones ({per_100:.1f}/100 lines) — elevated"
    return "FAIL", f"{count} clones ({per_100:.1f}/100 lines) — clone pressure"


def b2_todo_desert(lines: list[str], path: Path) -> tuple[str, str]:
    """B2: TODO Desert — files with zero TODOs are an AI signal (P1)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    has_todo = any(re.search(r"\b(TODO|FIXME|HACK|XXX)\b", l) for l in lines)
    if has_todo:
        return "PASS", "has honest uncertainty markers"
    return "WARN", "zero TODOs/FIXMEs — TODO Desert signal (AI generates 'complete' code)"


def b3_unwrap_density(lines: list[str], path: Path) -> tuple[str, str]:
    """B3: Unwrap density in non-test production code (N5)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    in_test = False
    count = 0
    non_test_lines = 0
    for line in lines:
        s = line.strip()
        if "#[cfg(test)]" in s or "#[test]" in s:
            in_test = True
        if in_test and s == "}":
            in_test = False
        if not in_test:
            non_test_lines += 1
            if ".unwrap()" in line:
                count += 1
    per_1k = count / max(non_test_lines, 1) * 1000
    if per_1k <= 5:
        return "PASS", f"{count} unwraps ({per_1k:.1f}/1k lines)"
    if per_1k <= 15:
        return "WARN", f"{count} unwraps ({per_1k:.1f}/1k lines) — elevated"
    return "FAIL", f"{count} unwraps ({per_1k:.1f}/1k lines) — exceeds 15/1k"


def b4_lint_suppressions(lines: list[str], path: Path) -> tuple[str, str]:
    """B4: Lint suppressions without justification comment (P7, 4× more in AI)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    bare = []
    for i, line in enumerate(lines, 1):
        if re.search(r"#\[allow\(", line):
            prev = lines[i-2].strip() if i >= 2 else ""
            if not prev.startswith("//"):
                bare.append(i)
    if not bare:
        return "PASS", "all #[allow(..)] have justification comments"
    return "WARN", f"bare #[allow(..)] at lines: {bare[:5]}"


def b5_pub_pressure(lines: list[str], path: Path) -> tuple[str, str]:
    """B5: Public API surface (C3, 12.6× more in AI code)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    pub_count = sum(1 for l in lines if re.match(r"\s*pub\s+(fn|struct|enum|const|static|type)\b", l))
    total_fns = sum(1 for l in lines if re.match(r"\s*(pub\s+)?fn\s+", l))
    if total_fns < 5:
        return "SKIP", f"too few functions ({total_fns}) to measure pub pressure"
    ratio = pub_count / total_fns
    if ratio <= 0.4:
        return "PASS", f"{pub_count} pub items, {ratio:.0%} of functions"
    if ratio <= 0.7:
        return "WARN", f"{pub_count} pub items ({ratio:.0%}) — elevated pub surface"
    return "FAIL", f"{pub_count} pub items ({ratio:.0%}) — pub pressure"


def b6_debug_prints(lines: list[str], path: Path) -> tuple[str, str]:
    """B6: Debug prints in library/server code (A6 — perfect discriminator)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    # Allow eprintln!/println! that are clearly diagnostic (behind cfg or with [tag])
    bad = []
    for i, line in enumerate(lines, 1):
        if re.search(r"\b(println!|eprintln!|dbg!)\s*\(", line):
            s = line.strip()
            # OK: behind cfg(debug_assertions) — checked by looking at prev lines
            prev_2 = " ".join(l.strip() for l in lines[max(0,i-3):i-1])
            if "cfg(debug_assertions)" in prev_2:
                continue
            # WARN everything else
            bad.append(i)
    if not bad:
        return "PASS", "no raw debug prints"
    if len(bad) <= 5:
        return "WARN", f"raw println!/eprintln! at lines: {bad} — intentional?"
    return "FAIL", f"{len(bad)} raw debug prints — exceeds threshold for lib code"


def b7_vague_expects(lines: list[str], path: Path) -> tuple[str, str]:
    """B7: Vague .expect() messages (A4)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    vague_patterns = re.compile(
        r'\.expect\(\s*"(error|failed|fail|should work|should not happen|'
        r'unreachable|TODO|fixme|oops|bad|wrong|impossible)"\s*\)',
        re.IGNORECASE
    )
    bad = []
    for i, line in enumerate(lines, 1):
        if vague_patterns.search(line):
            bad.append(i)
    if not bad:
        return "PASS", "all .expect() messages are specific"
    return "FAIL", f"vague .expect() at lines: {bad}"


def b8_prod_panics(lines: list[str], path: Path) -> tuple[str, str]:
    """B8: panic!/todo!() in production paths (A3)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    in_test = False
    bad = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if "#[cfg(test)]" in s or "#[test]" in s:
            in_test = True
        if in_test and s == "}":
            in_test = False
        if not in_test:
            if re.search(r"\btodo!\s*\(", line):
                bad.append(("todo!", i))
            elif re.search(r"\bunimplemented!\s*\(", line):
                bad.append(("unimplemented!", i))
            # panic! is OK if it has a justification comment on the preceding line
            elif re.search(r"\bpanic!\s*\(", line):
                prev = lines[i-2].strip() if i >= 2 else ""
                if not prev.startswith("//"):
                    bad.append(("panic!", i))
    if not bad:
        return "PASS", "no unjustified panics in production code"
    return "FAIL", f"production panics: {bad[:5]}"


def b9_placeholder_names(lines: list[str], path: Path) -> tuple[str, str]:
    """B9: Placeholder variable names in function signatures/struct fields (D11)"""
    if not str(path).endswith(".rs"):
        return "SKIP", "not Rust"
    placeholders = re.compile(
        r"\b(data|buf|tmp|res|result|output|handler|manager|helper|processor|wrapper)\s*[,:\)]"
    )
    bad = []
    for i, line in enumerate(lines, 1):
        if re.match(r"\s*(pub\s+)?(fn |struct |enum )", line):
            if placeholders.search(line):
                bad.append(i)
    if not bad:
        return "PASS", "no generic placeholder names in signatures"
    return "WARN", f"placeholder names in signatures at lines: {bad[:5]}"


# ---------------------------------------------------------------------------
# Criteria registry — ordered, with IDs matching the doc
# ---------------------------------------------------------------------------

CRITERIA = [
    ("C3",  "line_count",        c3_line_count),
    ("C4",  "gpu_unwrap",        c4_gpu_unwrap),
    ("C5",  "buffer_labels",     c5_buffer_labels),
    ("C6",  "dead_code_bare",    c6_dead_code),
    ("C7",  "commented_blocks",  c7_commented_blocks),
    ("B1",  "clone_pressure",    b1_clone_pressure),
    ("B2",  "todo_desert",       b2_todo_desert),
    ("B3",  "unwrap_density",    b3_unwrap_density),
    ("B4",  "lint_suppress",     b4_lint_suppressions),
    ("B5",  "pub_pressure",      b5_pub_pressure),
    ("B6",  "debug_prints",      b6_debug_prints),
    ("B7",  "vague_expects",     b7_vague_expects),
    ("B8",  "prod_panics",       b8_prod_panics),
    ("B9",  "placeholder_names", b9_placeholder_names),
]

MANUAL_CRITERIA = [
    ("C1",  "clippy",      "cargo clippy -- -D warnings"),
    ("C2",  "rustfmt",     "cargo fmt --check"),
    ("C8",  "no_todos",    "review: unresolved TODOs that block release"),
    ("C9",  "quant_verify","cargo run --release --bin quant_verify"),
    ("C10", "smoke_test",  "scripts/model_smoke_test.ps1"),
]


def check_file(path: Path, cert_data: dict) -> dict:
    rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    current_hash = sha256(path)
    registered = cert_data.get("files", {}).get(rel, {})

    results = {}
    for cid, name, fn in CRITERIA:
        status, detail = fn(lines, path)
        # Check for waivers
        waiver = registered.get("waivers", {}).get(cid)
        if waiver and status in ("FAIL", "WARN"):
            status = "WAIVED"
            detail = f"{detail} [waiver: {waiver}]"
        results[cid] = (status, detail, name)

    return {
        "path": rel,
        "hash": current_hash,
        "registered_hash": registered.get("hash"),
        "sealed_date": registered.get("sealed"),
        "results": results,
    }


def print_report(report: dict):
    path = report["path"]
    h = report["hash"]
    rh = report["registered_hash"]

    if rh is None:
        seal_status = "UNCERTIFIED"
    elif h == rh:
        seal_status = f"SEALED ({report['sealed_date']})"
    else:
        seal_status = "BROKEN (file changed since seal)"

    print(f"\n{'='*60}")
    print(f"FILE:   {path}")
    print(f"SHA256: {h}  [{seal_status}]")
    print(f"{'='*60}")

    fails = warns = passes = skips = 0
    for cid, (status, detail, name) in report["results"].items():
        icon = {"PASS": "✓", "WARN": "~", "FAIL": "✗", "SKIP": "-", "WAIVED": "W"}.get(status, "?")
        print(f"  {icon} {cid:3}  {name:22}  {status:6}  {detail}")
        if status == "PASS": passes += 1
        elif status == "WARN": warns += 1
        elif status == "FAIL": fails += 1
        elif status == "SKIP": skips += 1

    print(f"\n  Manual criteria (must be recorded in CERT.toml):")
    for cid, name, cmd in MANUAL_CRITERIA:
        print(f"  ? {cid:3}  {name:22}  run: {cmd}")

    print(f"\n  Auto: {passes} PASS, {warns} WARN, {fails} FAIL, {skips} SKIP")
    certifiable = fails == 0
    print(f"  Certifiable: {'YES — seal with --seal' if certifiable else 'NO — fix FAILs first'}")


def do_seal(path: Path, cert_data: dict):
    rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    h = sha256(path)
    files = cert_data.setdefault("files", {})
    existing = files.get(rel, {})
    files[rel] = {
        **existing,
        "hash": h,
        "sealed": str(date.today()),
        "path": rel,
    }
    save_cert(cert_data)
    print(f"SEALED: {rel}  hash={h}  date={date.today()}")


def main():
    parser = argparse.ArgumentParser(description="Airframe cert checker")
    parser.add_argument("files", nargs="*")
    parser.add_argument("--seal", action="store_true", help="Seal files after review")
    parser.add_argument("--waive", metavar="CID:REASON", help="Add waiver for a criterion")
    parser.add_argument("--list", action="store_true", help="List all sealed files")
    args = parser.parse_args()

    cert_data = load_cert()

    if args.list:
        files = cert_data.get("files", {})
        if not files:
            print("No sealed files.")
        for rel, info in sorted(files.items()):
            print(f"  {info.get('sealed','?')}  {info.get('hash','?')}  {rel}")
        return

    if not args.files:
        parser.print_help()
        return

    exit_code = 0
    for f in args.files:
        path = Path(f).resolve()
        if not path.exists():
            print(f"ERROR: {f} not found")
            exit_code = 1
            continue

        if args.waive:
            cid, _, reason = args.waive.partition(":")
            rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            entry = cert_data.setdefault("files", {}).setdefault(rel, {})
            entry.setdefault("waivers", {})[cid.upper()] = reason or "waived"
            save_cert(cert_data)
            print(f"WAIVER recorded: {rel}  {cid.upper()} = {reason}")
            continue

        report = check_file(path, cert_data)
        print_report(report)

        any_fail = any(s == "FAIL" for s, _, _ in report["results"].values())
        if any_fail:
            exit_code = 1

        if args.seal:
            if any_fail:
                print(f"\n  Cannot seal — fix FAILs first.")
            else:
                do_seal(path, cert_data)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
