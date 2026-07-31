#!/usr/bin/env python3
"""Generate the README Supported Models table from the certification ledger.

Reads certified entries from cert/math_ledger.duckdb (or ledger.sqlite fallback)
and prints the exact markdown table used in the shimmy README. This is the
single source of truth for what models Shimmy advertises as certified.

Usage:
    python scripts/cert/generate_models_table.py
    python scripts/cert/generate_models_table.py --write-shimmy
        # rewrites the table block in shimmy/README.md + docs/SUPPORTED_MODELS.md

The --write-shimmy mode updates the README in-place so the advertised table can
never drift from the ledger. Requires the ledger to be reachable at the given
--ledger path.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path


def _print_utf8(text: str) -> None:
    """Write text to stdout as UTF-8 bytes (Windows-safe, no reconfigure())."""
    sys.stdout.buffer.write(text.encode("utf-8"))


WS_ROOT = Path(__file__).resolve().parents[3]  # airframe-workspace/


def _connect(ledger: Path):
    try:
        import duckdb  # type: ignore

        con = duckdb.connect(str(ledger), read_only=True)
        return con, "duckdb"
    except Exception:
        import sqlite3

        con = sqlite3.connect(str(ledger))
        return con, "sqlite"


def load_certified(con, backend: str) -> list[tuple[str, str, str]]:
    sql = """
        SELECT model_id, family, quant
        FROM math_runs
        WHERE certified = 1
        ORDER BY family, model_id, quant
    """
    if backend == "duckdb":
        return [tuple(r) for r in con.execute(sql).fetchall()]
    return [tuple(r) for r in con.execute(sql).fetchall()]


# Curated family ordering for the README table (insertion order).
FAMILY_ORDER = [
    "llama",
    "qwen3",
    "qwen2",
    "qwen3.5",
    "phi3",
    "phi2",
    "gemma2",
    "gemma4",
    "deepseek-r1",
    "ministral",
    "starcoder2",
]


def pretty_model(model_id: str) -> str:
    for suf in ("-q4-k-m", "-q5-k-m", "-q6-k", "-q4-0", "-q8-0", "-f16", "-f32"):
        if model_id.endswith(suf):
            return model_id[: -len(suf)]
    return model_id


FAMILY_DISPLAY = {
    "deepseek-r1": "DeepSeek-R1",
    "gemma2": "Gemma-2",
    "gemma4": "Gemma-4",
    "llama": "Llama",
    "ministral": "Ministral",
    "phi2": "Phi-2",
    "phi3": "Phi-3",
    "qwen2": "Qwen2",
    "qwen3": "Qwen3",
    "qwen3.5": "Qwen3.5",
    "starcoder2": "StarCoder2",
}

# model_id (quant suffix stripped) -> (display name, HuggingFace GGUF page)
MODEL_DISPLAY: dict[str, tuple[str, str]] = {
    "deepseek-r1-0528-qwen3-8b": (
        "DeepSeek-R1-0528-Qwen3-8B",
        "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-8B-GGUF",
    ),
    "gemma-2-2b-it": (
        "Gemma-2-2B-it",
        "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF",
    ),
    "gemma-2-9b-it": (
        "Gemma-2-9B-it",
        "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF",
    ),
    "gemma-4-E4B": ("Gemma-4-E4B", "https://huggingface.co/google/gemma-4-E4B-it-GGUF"),
    "gemma-4-12B-coder": (
        "Gemma-4-12B-coder",
        "https://huggingface.co/google/gemma-4-12B-coder-GGUF",
    ),
    "tinyllama-1.1b": (
        "TinyLlama-1.1B-Chat",
        "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    ),
    "llama-3.2-1b-instruct": (
        "Llama-3.2-1B-Instruct",
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
    ),
    "llama-3.2-3b-instruct": (
        "Llama-3.2-3B-Instruct",
        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
    ),
    "ministral-3-14b": (
        "Ministral-3-14B-Reasoning",
        "https://huggingface.co/bartowski/Ministral-3-14B-Reasoning-GGUF",
    ),
    "phi-2": ("Phi-2", "https://huggingface.co/TheBloke/phi-2-GGUF"),
    "phi-3.5-mini": (
        "Phi-3.5-mini-Instruct",
        "https://huggingface.co/microsoft/Phi-3.5-mini-instruct-gguf",
    ),
    "phi3-mini-4k-instruct-q4": (
        "Phi-3-mini-4k-Instruct",
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf",
    ),
    "qwen2-0.5b-instruct": (
        "Qwen2-0.5B-Instruct",
        "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF",
    ),
    "qwen2-1.5b-instruct": (
        "Qwen2-1.5B-Instruct",
        "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF",
    ),
    "qwen2-7b-instruct": (
        "Qwen2-7B-Instruct",
        "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF",
    ),
    "qwen3-0.6b": ("Qwen3-0.6B", "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF"),
    "qwen3-1.7b": ("Qwen3-1.7B", "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF"),
    "qwen3-4b": ("Qwen3-4B", "https://huggingface.co/Qwen/Qwen3-4B-GGUF"),
    "qwen3-4b-thinking": (
        "Qwen3-4B-Thinking",
        "https://huggingface.co/Qwen/Qwen3-4B-Thinking-GGUF",
    ),
    "qwen3-8b": ("Qwen3-8B", "https://huggingface.co/Qwen/Qwen3-8B-GGUF"),
    "qwen3.5-9b": ("Qwen3.5-9B", "https://huggingface.co/Qwen/Qwen3.5-9B-GGUF"),
    "starcoder2-3b": (
        "StarCoder2-3B",
        "https://huggingface.co/second-state/StarCoder2-3B-GGUF",
    ),
}


def build_table(rows: list[tuple[str, str, str]]) -> str:
    """Return the markdown table as rendered in the shimmy README."""
    groups: OrderedDict[str, OrderedDict[str, list[str]]] = OrderedDict()
    for model_id, family, quant in rows:
        family_group = groups.setdefault(family, OrderedDict())
        model = pretty_model(model_id)
        family_group.setdefault(model, [])
        if quant not in family_group[model]:
            family_group[model].append(quant)

    ordered: OrderedDict[str, OrderedDict[str, list[str]]] = OrderedDict()
    for fam in FAMILY_ORDER:
        if fam in groups:
            ordered[fam] = groups[fam]
    for fam in groups:
        if fam not in ordered:
            ordered[fam] = groups[fam]
    groups = ordered

    lines = [
        "| Family | Model | Quants |",
        "|---|---|---|",
    ]
    for family, models in groups.items():
        display = FAMILY_DISPLAY.get(family, family)
        for i, (model, quants) in enumerate(models.items()):
            family_cell = f"**{display}**" if i == 0 else ""
            name, url = MODEL_DISPLAY.get(model, (model, ""))
            model_cell = f"[{name}]({url})" if url else name
            lines.append(f"| {family_cell} | {model_cell} | {' · '.join(quants)} |")
    return "\n".join(lines) + "\n"


def intro_line(rows: list[tuple[str, str, str]]) -> str:
    """The one-line summary shown above the table (counts derived from the ledger)."""
    families = len({r[1] for r in rows})
    return (
        f"**{families} model families · {len(rows)} certified model/quant combinations** — "
        "every model below passes Shimmy's 5-gate GPU math verification pipeline "
        "(dequant, structural peel, numerical, decode≡prefill, logits) against the "
        "certification ledger. GGUF files load as-is; no recompilation, no hardcoded "
        "per-model constants."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=WS_ROOT / "cert" / "math_ledger.duckdb",
        help="path to math ledger (duckdb or sqlite)",
    )
    parser.add_argument(
        "--write-shimmy",
        action="store_true",
        help="rewrite the table block in shimmy/README.md and docs/SUPPORTED_MODELS.md",
    )
    args = parser.parse_args()

    con, backend = _connect(args.ledger)
    rows = load_certified(con, backend)
    table = build_table(rows)

    if not args.write_shimmy:
        _print_utf8(table + "\n")
        return 0

    shimmy = WS_ROOT / "shimmy"
    targets = [shimmy / "README.md", shimmy / "docs" / "SUPPORTED_MODELS.md"]
    for target in targets:
        text = target.read_text(encoding="utf-8")
        marker = "| Family | Model | Quants |"
        if marker not in text:
            print(f"ERROR: marker not found in {target}", file=sys.stderr)
            return 1
        start = text.index(marker)
        # Replace the contiguous run of table rows starting at the marker.
        lines = text[start:].splitlines(keepends=True)
        end_off = 0
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith("| ")
                or stripped == "|---|"
                or stripped.startswith("|---|---")
            ):
                end_off += len(line)
            else:
                break
        new_block = table + "\n"
        text = text[:start] + new_block + text[start + end_off :]
        target.write_text(text, encoding="utf-8")
        print(f"updated {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
