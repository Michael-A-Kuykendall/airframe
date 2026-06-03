#!/usr/bin/env python3
"""
Formula-lens diff for Airframe inference trace packages.

This script treats each traced tensor statistic as a compact algebraic signature
and compares a candidate trace against a golden trace.

Why this exists:
- Raw token/layer logs are noisy for root-cause work.
- Ratio/energy style formulas make divergence obvious quickly.
- We can rank "where shape changed most" across (phase, step, layer).

Usage:
    python scripts/trace_formula_diff.py \
      --golden artifacts/debug/phi2_nan_hunt/trace_A.json \
      --candidate artifacts/debug/phi2_nan_hunt/trace_B.json \
      --top 25
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EPS = 1e-9


@dataclass(frozen=True)
class LayerKey:
    phase: str
    step: int
    layer: int


@dataclass
class LayerFormula:
    output_energy: float
    post_attn_energy: float
    ffn_energy: float
    residual_gain: float
    ffn_gain: float
    qk_balance: float
    kv_mean_gap: float
    output_to_ffn_absmax_ratio: float


@dataclass
class TokenFormula:
    top1_logit: float
    top2_logit: float
    top12_margin: float


@dataclass
class LayerDiff:
    key: LayerKey
    score: float
    metric_deltas: Dict[str, float]


def safe_div(a: float, b: float) -> float:
    return a / (b if abs(b) > EPS else EPS)


def log2_fold(a: float, b: float) -> float:
    return abs(math.log2((abs(a) + EPS) / (abs(b) + EPS)))


def load_trace(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_layer_formula(layer: Dict) -> LayerFormula:
    q = layer["q"]["stats"]
    k = layer["k"]["stats"]
    v = layer["v"]["stats"]
    post_attn = layer["post_attn"]["stats"]
    ffn_out = layer["ffn_out"]["stats"]
    out = layer["output"]["stats"]

    output_energy = float(out["std_dev"])
    post_attn_energy = float(post_attn["std_dev"])
    ffn_energy = float(ffn_out["std_dev"])

    return LayerFormula(
        output_energy=output_energy,
        post_attn_energy=post_attn_energy,
        ffn_energy=ffn_energy,
        residual_gain=safe_div(output_energy, post_attn_energy),
        ffn_gain=safe_div(ffn_energy, post_attn_energy),
        qk_balance=safe_div(float(q["std_dev"]), float(k["std_dev"])),
        kv_mean_gap=abs(float(k["mean"]) - float(v["mean"])),
        output_to_ffn_absmax_ratio=safe_div(float(out["abs_max"]), float(ffn_out["abs_max"])),
    )


def build_token_formula(token_step: Dict) -> TokenFormula:
    topk = token_step.get("logits_topk") or []
    if not topk:
        return TokenFormula(0.0, 0.0, 0.0)
    top1 = float(topk[0]["logit"])
    top2 = float(topk[1]["logit"]) if len(topk) > 1 else top1
    return TokenFormula(top1_logit=top1, top2_logit=top2, top12_margin=top1 - top2)


def iter_token_steps(pkg: Dict, phase_filter: Optional[str]) -> Iterable[Tuple[str, Dict]]:
    for phase_name in ("prefill", "decode"):
        if phase_filter and phase_filter != phase_name:
            continue
        for step in pkg.get(f"{phase_name}_steps", []):
            yield phase_name, step


def extract_formulas(pkg: Dict, phase_filter: Optional[str]) -> Tuple[Dict[LayerKey, LayerFormula], Dict[Tuple[str, int], TokenFormula]]:
    layer_map: Dict[LayerKey, LayerFormula] = {}
    token_map: Dict[Tuple[str, int], TokenFormula] = {}

    for phase, step in iter_token_steps(pkg, phase_filter):
        step_idx = int(step["step_index"])
        token_map[(phase, step_idx)] = build_token_formula(step)
        for layer in step.get("layers", []):
            key = LayerKey(phase=phase, step=step_idx, layer=int(layer["layer_idx"]))
            layer_map[key] = build_layer_formula(layer)

    return layer_map, token_map


def formula_to_dict(f: LayerFormula) -> Dict[str, float]:
    return {
        "output_energy": f.output_energy,
        "post_attn_energy": f.post_attn_energy,
        "ffn_energy": f.ffn_energy,
        "residual_gain": f.residual_gain,
        "ffn_gain": f.ffn_gain,
        "qk_balance": f.qk_balance,
        "kv_mean_gap": f.kv_mean_gap,
        "output_to_ffn_absmax_ratio": f.output_to_ffn_absmax_ratio,
    }


def compare_layers(golden: Dict[LayerKey, LayerFormula], candidate: Dict[LayerKey, LayerFormula]) -> List[LayerDiff]:
    shared = sorted(set(golden.keys()) & set(candidate.keys()), key=lambda k: (k.phase, k.step, k.layer))
    diffs: List[LayerDiff] = []

    for key in shared:
        g = formula_to_dict(golden[key])
        c = formula_to_dict(candidate[key])

        metric_deltas: Dict[str, float] = {}
        for name, g_val in g.items():
            c_val = c[name]
            metric_deltas[name] = log2_fold(c_val, g_val)

        score = statistics.fmean(metric_deltas.values())
        diffs.append(LayerDiff(key=key, score=score, metric_deltas=metric_deltas))

    return sorted(diffs, key=lambda d: d.score, reverse=True)


def compare_tokens(golden: Dict[Tuple[str, int], TokenFormula], candidate: Dict[Tuple[str, int], TokenFormula]) -> Dict[str, float]:
    shared = sorted(set(golden.keys()) & set(candidate.keys()))
    if not shared:
        return {"shared_steps": 0.0}

    margin_folds = []
    top1_folds = []
    for key in shared:
        g = golden[key]
        c = candidate[key]
        margin_folds.append(log2_fold(c.top12_margin, g.top12_margin))
        top1_folds.append(log2_fold(c.top1_logit, g.top1_logit))

    return {
        "shared_steps": float(len(shared)),
        "median_top12_margin_fold_log2": statistics.median(margin_folds),
        "mean_top12_margin_fold_log2": statistics.fmean(margin_folds),
        "median_top1_logit_fold_log2": statistics.median(top1_folds),
        "mean_top1_logit_fold_log2": statistics.fmean(top1_folds),
    }


def summarize_layer_diffs(diffs: Sequence[LayerDiff]) -> Dict[str, float]:
    if not diffs:
        return {"shared_layer_points": 0.0}
    scores = [d.score for d in diffs]
    return {
        "shared_layer_points": float(len(diffs)),
        "max_score": max(scores),
        "median_score": statistics.median(scores),
        "mean_score": statistics.fmean(scores),
    }


def format_phase_breakout(diffs: Sequence[LayerDiff]) -> str:
    buckets: Dict[str, List[float]] = {"prefill": [], "decode": []}
    for d in diffs:
        buckets.setdefault(d.key.phase, []).append(d.score)

    lines = ["Phase Divergence (mean log2-fold score):"]
    for phase in ("prefill", "decode"):
        vals = buckets.get(phase, [])
        if vals:
            lines.append(f"  {phase:7s}  mean={statistics.fmean(vals):.4f}  median={statistics.median(vals):.4f}  n={len(vals)}")
    return "\n".join(lines)


def print_top_diffs(diffs: Sequence[LayerDiff], top_n: int) -> None:
    print("Top Divergent Layer Points:")
    for d in diffs[:top_n]:
        md = d.metric_deltas
        print(
            f"  {d.key.phase:7s} step={d.key.step:4d} layer={d.key.layer:2d} score={d.score:.4f} "
            f"res_gain={md['residual_gain']:.3f} ffn_gain={md['ffn_gain']:.3f} "
            f"qk_bal={md['qk_balance']:.3f} outE={md['output_energy']:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare Airframe traces using formula-style layer signatures.")
    p.add_argument("--golden", required=True, help="Path to golden trace JSON")
    p.add_argument("--candidate", required=True, help="Path to candidate trace JSON")
    p.add_argument("--phase", choices=["prefill", "decode"], default=None, help="Optional phase filter")
    p.add_argument("--top", type=int, default=20, help="How many top divergent layer points to print")
    p.add_argument("--json-out", default=None, help="Optional output file for machine-readable report")
    return p


def main() -> int:
    args = build_parser().parse_args()

    golden_path = Path(args.golden)
    candidate_path = Path(args.candidate)
    if not golden_path.exists():
        raise SystemExit(f"Golden trace not found: {golden_path}")
    if not candidate_path.exists():
        raise SystemExit(f"Candidate trace not found: {candidate_path}")

    golden_pkg = load_trace(golden_path)
    candidate_pkg = load_trace(candidate_path)

    golden_layers, golden_tokens = extract_formulas(golden_pkg, args.phase)
    candidate_layers, candidate_tokens = extract_formulas(candidate_pkg, args.phase)

    layer_diffs = compare_layers(golden_layers, candidate_layers)
    layer_summary = summarize_layer_diffs(layer_diffs)
    token_summary = compare_tokens(golden_tokens, candidate_tokens)

    print("=== Airframe Trace Formula Diff ===")
    print(f"golden   : {golden_path}")
    print(f"candidate: {candidate_path}")
    print(f"phase    : {args.phase or 'all'}")
    print()
    print("Layer Summary:")
    for k, v in layer_summary.items():
        print(f"  {k:26s} {v:.6f}")
    print()
    print("Token Logit Summary:")
    for k, v in token_summary.items():
        print(f"  {k:26s} {v:.6f}")
    print()
    print(format_phase_breakout(layer_diffs))
    print()
    print_top_diffs(layer_diffs, args.top)

    if args.json_out:
        out = {
            "golden": str(golden_path),
            "candidate": str(candidate_path),
            "phase": args.phase or "all",
            "layer_summary": layer_summary,
            "token_summary": token_summary,
            "top_diffs": [
                {
                    "phase": d.key.phase,
                    "step": d.key.step,
                    "layer": d.key.layer,
                    "score": d.score,
                    "metric_deltas": d.metric_deltas,
                }
                for d in layer_diffs[: args.top]
            ],
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
