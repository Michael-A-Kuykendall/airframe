#!/usr/bin/env python3
"""
vault_verify — Vault-driven frontier_compare verification.

For each vault model with oracles, scans artifacts/ for matching frontier_compare
trace files, computes algebraic formula signatures (GPU vs CPU), compares against
vault oracle RMS values, and populates inference_formulas + formula_comparisons.

Usage:
    python vault/scripts/vault_verify.py                          # scan artifacts/ + live vault
    python vault/scripts/vault_verify.py --trace <path>           # compare a single trace against vault
    python vault/scripts/vault_verify.py --model <model_id>       # verify a specific model
    python vault/scripts/vault_verify.py --run                    # also run frontier_compare for missing traces
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb

# ─── Config ───────────────────────────────────────────────────────────────────

VAULT_DB = Path(__file__).resolve().parent.parent.parent / "vault" / "vault.duckdb"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
FRONTIER_BIN = Path(__file__).resolve().parent.parent.parent / "target" / "release" / "frontier_compare.exe"
GGUF_DIR = Path("D:/shimmy-test-models/gguf_collection")
EPS = 1e-12
LOG2_FOLD_FAIL_THRESHOLD = 2.0  # fail if mean log2-fold > 2.0

# Models with oracles — metadata from vault (must match vault model rows)
MODELS: Dict[int, dict] = {
    2:  {"name": "Llama 3.2 1B Q4_K_M",  "n_layers": 16, "gguf": "Llama-3.2-1B-Instruct-Q4_K_M.gguf"},
    3:  {"name": "Llama 3.2 1B Q6_K",    "n_layers": 16, "gguf": "Llama-3.2-1B-Instruct-Q6_K.gguf"},
    4:  {"name": "Llama 3.2 3B",         "n_layers": 28, "gguf": "Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
    7:  {"name": "Qwen3 1.7B",           "n_layers": 28, "gguf": "Qwen3-1.7B-Q4_K_M.gguf"},
    9:  {"name": "Qwen3 8B",             "n_layers": 32, "gguf": "Qwen3-8B-Q4_K_M.gguf"},
    10: {"name": "TinyLlama Q4_0",       "n_layers": 22, "gguf": "TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"},
    11: {"name": "DeepSeek Coder",       "n_layers": 30, "gguf": "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"},
    12: {"name": "DeepSeek LLM",         "n_layers": 32, "gguf": "deepseek-llm-7b-chat.Q4_K_M.gguf"},
    18: {"name": "Qwen2 0.5B",           "n_layers": 24, "gguf": "qwen2-0_5b-instruct-q4_k_m.gguf"},
    19: {"name": "Qwen2 1.5B",           "n_layers": 28, "gguf": "qwen2-1_5b-instruct-q4_k_m.gguf"},
    20: {"name": "Qwen2 7B",             "n_layers": 28, "gguf": "qwen2-7b-instruct-q4_k_m.gguf"},
    22: {"name": "TinyLlama Q6_K",       "n_layers": 22, "gguf": "TinyLlama-1.1B-Chat-v1.0.Q6_K.gguf"},
}

# Known trace file → vault model_id mapping
TRACE_MODEL_MAP: Dict[str, int] = {
    # TinyLlama Q4_0 traces (22 layers)
    "tinyllama_fc.json": 10,
    "tinyllama_frontier.json": 10,
    "tinyllama_hi.json": 10,
    "tinyllama_hi_v2.json": 10,
    "tinyllama_postclean.json": 10,
    # Llama 3.2 1B Q4_K_M traces (16 layers)
    "llama32_1b_after_scale_fix.json": 2,
    "llama32_1b_table.json": 2,
    "llama32_baseline.json": 2,
    "llama32_clean2.json": 2,
    "llama32_current.json": 2,
    "llama32_current2.json": 2,
    "llama32_patch1.json": 2,
    "llama32_patch2.json": 2,
    "llama32_patch3.json": 2,
    "llama32_patch4.json": 2,
    "llama32_patch5.json": 2,
    "llama32_patch6.json": 2,
    "llama32_patch7.json": 2,
    "llama32_patch8.json": 2,
    "llama32_postclean.json": 2,
    "llama32_revert_test.json": 2,
    "llama32_test_p3.json": 2,
    "llama32_v1all.json": 2,
    "llama32_v1ffn.json": 2,
    # Qwen3 0.6B traces (28 layers, no vault oracles — metadata only)
    "qwen3_06b_frontier.json": 6,
    "qwen3_06b_table.json": 6,
    "qwen3_06b_v2.json": 6,
    # Qwen3 1.7B traces (28 layers)
    "qwen3_17b_table.json": 7,
    # Other traces (layer count only, no vault model match)
    "test2.json": -1,
    "tiny_table.json": -1,
    "tinyllama_fixed.json": -1,
    "tinyllama_layer_dump.json": -1,
    "tinyllama_layer_dump_fixed.json": -1,
    "tinyllama_v024.json": -1,
}



# ─── Formula types ────────────────────────────────────────────────────────────

@dataclass
class LayerFormula:
    output_energy: float        # RMS of output hidden state
    post_attn_energy: float     # RMS of post_attn
    ffn_energy: float           # RMS of FFN output
    residual_gain: float        # output_rms / post_attn_rms
    ffn_gain: float             # ffn_rms / post_attn_rms
    qk_balance: float           # Q_rms / K_rms
    kv_mean_gap: float          # |K_mean - V_mean|
    has_nan: bool = False
    has_inf: bool = False


@dataclass
class LayerComparison:
    layer_idx: int
    oracle_rms: float           # RMS from vault oracle
    cpu_rms: float              # RMS from frontier_compare CPU
    gpu_rms: float              # RMS from frontier_compare GPU
    gpu_oracle_log2fold: float  # |log2(gpu_rms / oracle_rms)|
    cpu_gpu_log2fold: float     # |log2(cpu_rms / gpu_rms)|
    cpu_formula: LayerFormula
    gpu_formula: LayerFormula
    formula_log2fold: float     # mean of metric log2 folds


# ─── Helpers ──────────────────────────────────────────────────────────────────

def log2_fold(a: float, b: float) -> float:
    return abs(math.log2((abs(a) + EPS) / (abs(b) + EPS)))


def safe_div(a: float, b: float) -> float:
    return a / (b if abs(b) > EPS else EPS)


def compute_formula(layer: dict, side: str = "cpu") -> LayerFormula:
    """Compute LayerFormula from a frontier_compare layer using CPU or GPU stats."""
    rms_key = f"{side}_rms"
    nf_key = f"{side}_non_finite"

    def get_rms(comp: dict) -> float:
        return float(comp.get(rms_key, 0.0) or 0.0)

    def get_nf(comp: dict) -> int:
        return int(comp.get(nf_key, 0) or 0)

    q_rms = get_rms(layer["q"])
    k_rms = get_rms(layer["k"])
    v_rms = get_rms(layer["v"])
    post_rms = get_rms(layer["post_attn"])
    ffn_rms = get_rms(layer["ffn_out"])
    out_rms = get_rms(layer["output"])

    # frontier_compare traces don't have mean values — set kv_mean_gap to 0
    has_nf = get_nf(layer["output"]) > 0

    return LayerFormula(
        output_energy=out_rms,
        post_attn_energy=post_rms,
        ffn_energy=ffn_rms,
        residual_gain=safe_div(out_rms, post_rms),
        ffn_gain=safe_div(ffn_rms, post_rms),
        qk_balance=safe_div(q_rms, k_rms),
        kv_mean_gap=0.0,
        has_nan=has_nf,
        has_inf=has_nf,
    )


def formula_metrics(f: LayerFormula) -> Dict[str, float]:
    return {
        "output_energy": f.output_energy,
        "post_attn_energy": f.post_attn_energy,
        "ffn_energy": f.ffn_energy,
        "residual_gain": f.residual_gain,
        "ffn_gain": f.ffn_gain,
        "qk_balance": f.qk_balance,
        "kv_mean_gap": f.kv_mean_gap,
    }


def formula_log2fold(golden: LayerFormula, candidate: LayerFormula) -> float:
    """Mean log2-fold divergence between two formulas."""
    gm = formula_metrics(golden)
    cm = formula_metrics(candidate)
    folds = [log2_fold(cm[k], gm[k]) for k in gm]
    return statistics.fmean(folds) if folds else 0.0


# ─── Trace matching ───────────────────────────────────────────────────────────

def identify_trace_model(trace_path: Path) -> Optional[int]:
    """Match a trace file to a vault model_id using TRACE_MODEL_MAP or layer count."""
    fname = trace_path.name
    if fname in TRACE_MODEL_MAP:
        mid = TRACE_MODEL_MAP[fname]
        if mid >= 1:
            return mid
        return None  # explicitly unmapped (-1)

    # Fallback: match by layer count if unambiguous
    try:
        with open(trace_path) as f:
            d = json.load(f)
        n_layers = len(d.get("layers", []))
    except Exception:
        return None

    # Count how many models with oracles have this layer count
    matches = [mid for mid, info in MODELS.items() if info["n_layers"] == n_layers]
    return matches[0] if len(matches) == 1 else None


def find_traces_for_model(model_id: int, artifacts_dir: Path) -> List[Path]:
    """Find trace files that match a model by TRACE_MODEL_MAP or layer count."""
    traces = []
    for f in sorted(artifacts_dir.glob("*.json")):
        if f.name.startswith("model_smoke"):
            continue
        mid = identify_trace_model(f)
        if mid == model_id:
            traces.append(f)
    return traces


# ─── Vault queries ────────────────────────────────────────────────────────────

def get_vault_oracles(con: duckdb.DuckDBPyConnection, model_id: int) -> Dict[int, float]:
    """Get layer RMS values from vault oracles for a model."""
    rows = con.execute("""
        SELECT layer_idx, expected_rms FROM layer_oracles
        WHERE model_id = ? AND operation = 'layer_output' AND layer_idx >= 0
        ORDER BY layer_idx
    """, [model_id]).fetchall()
    return {int(r[0]): float(r[1] or 0.0) for r in rows}


# ─── Layer diagnostics ────────────────────────────────────────────────────────

def ensure_layer_diags_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create layer_diags table if it does not exist (migration 003)."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS layer_diags (
            id              INTEGER PRIMARY KEY,
            model_id        INTEGER NOT NULL REFERENCES models(id),
            layer_idx       INTEGER NOT NULL,
            q_quant         INTEGER NOT NULL DEFAULT 0,
            k_quant         INTEGER NOT NULL DEFAULT 0,
            v_quant         INTEGER NOT NULL DEFAULT 0,
            ffn_gate_quant  INTEGER NOT NULL DEFAULT 0,
            ffn_down_quant  INTEGER NOT NULL DEFAULT 0,
            ffn_up_quant    INTEGER NOT NULL DEFAULT 0,
            attn_out_quant  INTEGER NOT NULL DEFAULT 0,
            v_offset        BIGINT NOT NULL DEFAULT 0,
            q_offset        BIGINT NOT NULL DEFAULT 0,
            k_offset        BIGINT NOT NULL DEFAULT 0,
            ffn_gate_offset BIGINT NOT NULL DEFAULT 0,
            ffn_down_offset BIGINT NOT NULL DEFAULT 0,
            ffn_up_offset   BIGINT NOT NULL DEFAULT 0,
            ffn_kind        INTEGER NOT NULL DEFAULT 0,
            qkv_layout      INTEGER NOT NULL DEFAULT 0,
            qk_norm         INTEGER NOT NULL DEFAULT 0,
            post_norm       INTEGER NOT NULL DEFAULT 0,
            layer_norm      INTEGER NOT NULL DEFAULT 0,
            batch_count     INTEGER NOT NULL DEFAULT 1,
            source_trace    VARCHAR,
            created_at      TIMESTAMP DEFAULT NOW(),
            UNIQUE(model_id, layer_idx, source_trace)
        )
    """)


def insert_layer_diags(con: duckdb.DuckDBPyConnection, model_id: int, trace_path: Path) -> int:
    """Upsert layer diagnostics from a frontier_compare trace into vault."""
    with open(trace_path) as f:
        trace = json.load(f)

    diags = trace.get("layer_diags", [])
    if not diags:
        return 0

    source = trace_path.name
    count = 0
    max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM layer_diags").fetchone()[0]

    for d in diags:
        li = int(d["layer_idx"])
        existing = con.execute(
            "SELECT id FROM layer_diags WHERE model_id=? AND layer_idx=? AND source_trace=?",
            [model_id, li, source]
        ).fetchone()
        if existing:
            con.execute("""
                UPDATE layer_diags SET
                    q_quant=?, k_quant=?, v_quant=?,
                    ffn_gate_quant=?, ffn_down_quant=?, ffn_up_quant=?, attn_out_quant=?,
                    v_offset=?, q_offset=?, k_offset=?,
                    ffn_gate_offset=?, ffn_down_offset=?, ffn_up_offset=?,
                    ffn_kind=?, qkv_layout=?, qk_norm=?, post_norm=?, layer_norm=?, batch_count=?
                WHERE id=?
            """, [
                d["q_quant"], d["k_quant"], d["v_quant"],
                d["ffn_gate_quant"], d["ffn_down_quant"], d["ffn_up_quant"], d["attn_out_quant"],
                d["v_offset"], d["q_offset"], d["k_offset"],
                d["ffn_gate_offset"], d["ffn_down_offset"], d["ffn_up_offset"],
                d["ffn_kind"], d["qkv_layout"], d["qk_norm"], d["post_norm"], d["layer_norm"], d["batch_count"],
                existing[0]
            ])
        else:
            max_id += 1
            con.execute("""
                INSERT INTO layer_diags (
                    id, model_id, layer_idx,
                    q_quant, k_quant, v_quant,
                    ffn_gate_quant, ffn_down_quant, ffn_up_quant, attn_out_quant,
                    v_offset, q_offset, k_offset,
                    ffn_gate_offset, ffn_down_offset, ffn_up_offset,
                    ffn_kind, qkv_layout, qk_norm, post_norm, layer_norm, batch_count,
                    source_trace
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?
                )
            """, [
                max_id, model_id, li,
                d["q_quant"], d["k_quant"], d["v_quant"],
                d["ffn_gate_quant"], d["ffn_down_quant"], d["ffn_up_quant"], d["attn_out_quant"],
                d["v_offset"], d["q_offset"], d["k_offset"],
                d["ffn_gate_offset"], d["ffn_down_offset"], d["ffn_up_offset"],
                d["ffn_kind"], d["qkv_layout"], d["qk_norm"], d["post_norm"], d["layer_norm"], d["batch_count"],
                source
            ])
        count += 1

    return count


# ─── Core verification ────────────────────────────────────────────────────────

def verify_trace(con: duckdb.DuckDBPyConnection, trace_path: Path, model_id: int) -> List[LayerComparison]:
    """Compare a frontier_compare trace against vault oracles."""
    info = MODELS.get(model_id)
    if not info:
        print(f"  Unknown model_id={model_id}, skipping")
        return []

    with open(trace_path) as f:
        trace = json.load(f)

    layers = trace.get("layers", [])
    oracles = get_vault_oracles(con, model_id)

    comparisons: List[LayerComparison] = []
    for layer in layers:
        li = int(layer["layer_idx"])

        # Compute CPU and GPU formulas
        cpu_f = compute_formula(layer, "cpu")
        gpu_f = compute_formula(layer, "gpu")

        # RMS values
        cpu_rms = float(layer.get("output", {}).get("cpu_rms", 0.0) or 0.0)
        gpu_rms = float(layer.get("output", {}).get("gpu_rms", 0.0) or 0.0)
        oracle_rms = oracles.get(li, 0.0)

        # Log2-fold divergences
        gpu_oracle_fold = log2_fold(gpu_rms, oracle_rms) if oracle_rms > 0 else 999.0
        cpu_gpu_fold = log2_fold(cpu_rms, gpu_rms) if cpu_rms > 0 or gpu_rms > 0 else 0.0
        formula_fold = formula_log2fold(cpu_f, gpu_f)

        comparisons.append(LayerComparison(
            layer_idx=li,
            oracle_rms=oracle_rms,
            cpu_rms=cpu_rms,
            gpu_rms=gpu_rms,
            gpu_oracle_log2fold=gpu_oracle_fold,
            cpu_gpu_log2fold=cpu_gpu_fold,
            cpu_formula=cpu_f,
            gpu_formula=gpu_f,
            formula_log2fold=formula_fold,
        ))

    return comparisons


def insert_formulas(
    con: duckdb.DuckDBPyConnection,
    model_id: int,
    source: str,
    comparisons: List[LayerComparison],
    side: str,
) -> int:
    """Insert formula rows into inference_formulas table."""
    # Clear existing rows for this model + source
    con.execute("DELETE FROM inference_formulas WHERE model_id = ? AND source = ?", [model_id, source])

    # Get next id
    max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM inference_formulas").fetchone()[0]

    count = 0
    for i, c in enumerate(comparisons):
        f = c.cpu_formula if side == "cpu" else c.gpu_formula
        try:
            con.execute("""
                INSERT INTO inference_formulas
                    (id, model_id, source, layer_idx, position, input_token_id,
                     output_energy, post_attn_energy, ffn_energy,
                     residual_gain, ffn_gain, qk_balance, kv_mean_gap,
                     has_nan, has_inf, notes)
                VALUES (?, ?, ?, ?, 1, NULL,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?)
            """, [
                max_id + 1 + i, model_id, source, c.layer_idx,
                f.output_energy, f.post_attn_energy, f.ffn_energy,
                f.residual_gain, f.ffn_gain, f.qk_balance, f.kv_mean_gap,
                f.has_nan, f.has_inf,
                f"vault_verify from {side} side of frontier_compare trace",
            ])
            count += 1
        except Exception as e:
            print(f"  Error inserting formula for layer {c.layer_idx}: {e}")
    return count


def insert_comparison(
    con: duckdb.DuckDBPyConnection,
    model_id: int,
    comparisons: List[LayerComparison],
    trace_path: Path,
) -> bool:
    """Insert a formula_comparisons row and return pass/fail."""
    if not comparisons:
        return False

    folds = [c.formula_log2fold for c in comparisons if c.formula_log2fold < 100]
    if not folds:
        mean_score = 999.0
        median_score = 999.0
        max_score = 999.0
    else:
        mean_score = statistics.fmean(folds)
        median_score = statistics.median(folds)
        max_score = max(folds)

    n_layers = len(comparisons)
    passed = mean_score < LOG2_FOLD_FAIL_THRESHOLD

    first_nan = None
    for c in comparisons:
        if c.gpu_formula.has_nan or c.cpu_formula.has_nan:
            first_nan = c.layer_idx
            break

    # Clear existing comparison for this model
    con.execute("DELETE FROM formula_comparisons WHERE model_id = ? AND golden_source = 'frontier_cpu' AND candidate_source = 'frontier_gpu'", [model_id])

    max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM formula_comparisons").fetchone()[0]

    con.execute("""
        INSERT INTO formula_comparisons
            (id, model_id, golden_source, candidate_source,
             mean_layer_score, median_layer_score, max_layer_score,
             n_layer_points, mean_top1_logit_fold, shared_steps,
             first_nan_layer, first_nan_source,
             passed, threshold, notes)
        VALUES (?, ?, 'frontier_cpu', 'frontier_gpu',
                ?, ?, ?,
                ?, NULL, ?,
                ?, 'frontier_gpu',
                ?, ?, ?)
    """, [
        max_id + 1, model_id,
        mean_score, median_score, max_score,
        n_layers, n_layers,
        first_nan,
        passed, LOG2_FOLD_FAIL_THRESHOLD,
        f"vault_verify from {trace_path.name}",
    ])

    return passed


def print_report(comparisons: List[LayerComparison], model_name: str, trace_name: str, passed: bool):
    """Print a human-readable verification report."""
    status = "PASS" if passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"  {model_name:40s}  [{status}]")
    print(f"  Trace: {trace_name}")
    print(f"{'='*60}")
    print(f"  {'Layer':>5s}  {'OracleRMS':>9s}  {'CPU_RMS':>8s}  {'GPU_RMS':>8s}  "
          f"{'GvO_l2':>6s}  {'CvG_l2':>6s}  {'FmL2':>5s}  {'NaN':>4s}")
    print(f"  {'-'*58}")

    for c in comparisons:
        nan_flag = "N" if c.gpu_formula.has_nan else "."
        print(f"  {c.layer_idx:5d}  {c.oracle_rms:9.6f}  {c.cpu_rms:8.6f}  {c.gpu_rms:8.6f}  "
              f"{c.gpu_oracle_log2fold:6.2f}  {c.cpu_gpu_log2fold:6.2f}  {c.formula_log2fold:5.2f}  {nan_flag:>4s}")

    folds = [c.formula_log2fold for c in comparisons if c.formula_log2fold < 100]
    if folds:
        print(f"\n  Mean log2-fold: {statistics.fmean(folds):.4f}  "
              f"Median: {statistics.median(folds):.4f}  "
              f"Max: {max(folds):.4f}")
    print()


# ─── Frontier_compare runner ─────────────────────────────────────────────────

def run_frontier_compare(model_id: int, output_path: Path) -> bool:
    """Run frontier_compare for a model and write output to path."""
    info = MODELS.get(model_id)
    if not info:
        return False

    gguf_path = GGUF_DIR / info["gguf"]
    if not gguf_path.exists():
        print(f"  GGUF not found: {gguf_path}")
        return False

    if not FRONTIER_BIN.exists():
        print(f"  frontier_compare binary not found: {FRONTIER_BIN}")
        return False

    cmd = [
        str(FRONTIER_BIN),
        "--model", str(gguf_path),
        "--prompt", "hi",
        "--max-ctx", "2048",
        "--output", str(output_path),
    ]
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  frontier_compare failed (code={result.returncode}):")
            print(f"    stderr: {result.stderr[:500]}")
            return False
        print(f"  Done ({os.path.getsize(output_path)} bytes)")
        return True
    except subprocess.TimeoutExpired:
        print("  frontier_compare timed out (120s)")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vault-driven frontier_compare verification")
    p.add_argument("--trace", help="Path to a single frontier_compare trace JSON")
    p.add_argument("--model", type=int, help="Verify a specific model by vault ID")
    p.add_argument("--run", action="store_true", help="Run frontier_compare for missing traces")
    p.add_argument("--db", default=str(VAULT_DB), help="Path to vault DuckDB database")
    p.add_argument("--artifacts", default=str(ARTIFACTS_DIR), help="Path to artifacts directory")
    return p


def main() -> int:
    args = build_parser().parse_args()

    db_path = Path(args.db)
    artifacts_dir = Path(args.artifacts)

    if not db_path.exists():
        print(f"Vault DB not found: {db_path}")
        return 1

    con = duckdb.connect(str(db_path))

    if args.trace:
        # Single trace mode
        trace_path = Path(args.trace)
        if not trace_path.exists():
            print(f"Trace not found: {trace_path}")
            return 1

        model_id = args.model or identify_trace_model(trace_path)
        if model_id is None:
            print(f"Could not identify model for trace: {trace_path.name}")
            print("Use --model <id> to specify vault model ID, or add to TRACE_MODEL_MAP")
            return 1

        info = MODELS.get(model_id)
        if info is None:
            print(f"Unknown model_id={model_id} (not in MODELS dict)")
            return 1
        comparisons = verify_trace(con, trace_path, model_id)
        if not comparisons:
            print(f"No oracle data found for model_id={model_id}")
            return 1

        # Insert formulas
        n_cpu = insert_formulas(con, model_id, "frontier_cpu", comparisons, "cpu")
        n_gpu = insert_formulas(con, model_id, "frontier_gpu", comparisons, "gpu")
        passed = insert_comparison(con, model_id, comparisons, trace_path)
        # Import layer diagnostics
        ensure_layer_diags_table(con)
        n_diags = insert_layer_diags(con, model_id, trace_path)
        con.commit()

        print_report(comparisons, info["name"], trace_path.name, passed)
        print(f"  Inserted: {n_cpu} CPU formulas + {n_gpu} GPU formulas + {n_diags} layer_diags")
        print(f"  Verdict: {'PASS' if passed else 'FAIL'} (threshold={LOG2_FOLD_FAIL_THRESHOLD})")
        return 0 if passed else 2

    # Batch mode: scan artifacts for all models with oracles
    print("=" * 60)
    print("  VAULT VERIFY — Batch Scan")
    print(f"  DB: {db_path}")
    print(f"  Artifacts: {artifacts_dir}")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for model_id in sorted(MODELS.keys()):
        info = MODELS[model_id]
        traces = find_traces_for_model(model_id, artifacts_dir)

        if not traces:
            print(f"\n  [{model_id}] {info['name']:35s}  no traces found ({info['n_layers']} layers)")
            total_skipped += 1
            continue

        # Use the most recent trace
        trace_path = traces[-1]
        print(f"\n  [{model_id}] {info['name']:35s}  using {trace_path.name}")

        comparisons = verify_trace(con, trace_path, model_id)
        if not comparisons:
            print(f"    No oracle data — skipping")
            total_skipped += 1
            continue

        n_cpu = insert_formulas(con, model_id, "frontier_cpu", comparisons, "cpu")
        n_gpu = insert_formulas(con, model_id, "frontier_gpu", comparisons, "gpu")
        passed = insert_comparison(con, model_id, comparisons, trace_path)
        # Import layer diagnostics
        ensure_layer_diags_table(con)
        n_diags = insert_layer_diags(con, model_id, trace_path)
        con.commit()

        folds = [c.formula_log2fold for c in comparisons if c.formula_log2fold < 100]
        mean_fold = statistics.fmean(folds) if folds else 999.0

        status = "PASS" if passed else "FAIL"
        print(f"    Layers: {len(comparisons):2d}  Mean L2: {mean_fold:.4f}  Diags: {n_diags} [{status}]")

        if passed:
            total_passed += 1
        else:
            total_failed += 1

    con.close()

    print(f"\n{'='*60}")
    print(f"  Summary: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    print(f"{'='*60}")

    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
