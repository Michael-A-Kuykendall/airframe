#!/usr/bin/env python3
"""
llama_formula_side_by_side.py

Builds a compact "llama.cpp canonical equations vs Airframe runtime math" report
from a trace JSON and optional formula-diff JSON.

Usage:
  python scripts/llama_formula_side_by_side.py \
    --trace artifacts/debug/starcoder2_probe_diag2_chat/trace_*.json \
    --formula-json artifacts/debug/starcoder2_probe_diag2_chat/raw_vs_chat_formula.json \
    --out artifacts/debug/starcoder2_probe_diag2_chat/llama_formula_side_by_side.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


LLAMA_CANONICAL: Dict[str, Dict[str, Any]] = {
    "starcoder2": {
        "source_refs": [
            "ggml-org/llama.cpp: src/llama-graph.cpp (LLM_NORM -> ggml_norm)",
            "ggml-org/llama.cpp: src/models/* attention path (Q,K,V -> RoPE -> softmax(QK^T/sqrt(d))V)",
        ],
        "equations": {
            "norm_type": "LayerNorm",
            "norm_formula": "y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta",
            "attn_formula": "Attn(Q,K,V) = softmax((QK^T)/sqrt(d_k))V",
            "rope_formula": "Q_rope = RoPE(Q), K_rope = RoPE(K)",
            "qk_norm": "disabled by default for StarCoder2",
            "post_norm": "disabled",
        },
    },
    "gpt2": {
        "source_refs": [
            "ggml-org/llama.cpp: src/llama-graph.cpp (LLM_NORM -> ggml_norm)",
        ],
        "equations": {
            "norm_type": "LayerNorm",
            "norm_formula": "y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta",
            "attn_formula": "Attn(Q,K,V) = softmax((QK^T)/sqrt(d_k))V",
            "rope_formula": "not used",
            "qk_norm": "disabled",
            "post_norm": "disabled",
        },
    },
    "phi": {
        "source_refs": [
            "ggml-org/llama.cpp: src/llama-graph.cpp (LLM_NORM_RMS -> ggml_rms_norm)",
        ],
        "equations": {
            "norm_type": "RMSNorm",
            "norm_formula": "y = x / sqrt(mean(x^2) + eps) * gamma (+ optional beta)",
            "attn_formula": "Attn(Q,K,V) = softmax((QK^T)/sqrt(d_k))V",
            "rope_formula": "Q_rope = RoPE(Q), K_rope = RoPE(K)",
            "qk_norm": "model-dependent",
            "post_norm": "disabled",
        },
    },
    "qwen3": {
        "source_refs": [
            "ggml-org/llama.cpp: src/models/qwen3.cpp (RoPE on Q/K, optional QK norm)",
            "ggml-org/llama.cpp: src/llama-graph.cpp (RMS norm path for RMS families)",
        ],
        "equations": {
            "norm_type": "RMSNorm",
            "norm_formula": "y = x / sqrt(mean(x^2) + eps) * gamma (+ optional beta)",
            "attn_formula": "Attn(Q,K,V) = softmax((QK^T)/sqrt(d_k))V",
            "rope_formula": "Q_rope = RoPE(Q), K_rope = RoPE(K)",
            "qk_norm": "enabled for Qwen3",
            "post_norm": "disabled",
        },
    },
}


def load_json(path_or_glob: str) -> Dict[str, Any]:
    matches = sorted(glob.glob(path_or_glob))
    if matches:
        path = matches[-1]
    else:
        path = path_or_glob
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_for_arch(arch: str) -> Dict[str, Any]:
    key = arch.lower()
    if key in LLAMA_CANONICAL:
        return LLAMA_CANONICAL[key]
    if "starcoder" in key:
        return LLAMA_CANONICAL["starcoder2"]
    if "gpt" in key:
        return LLAMA_CANONICAL["gpt2"]
    if "phi" in key:
        return LLAMA_CANONICAL["phi"]
    if "qwen3" in key:
        return LLAMA_CANONICAL["qwen3"]
    return {
        "source_refs": ["ggml-org/llama.cpp: src/llama-graph.cpp"],
        "equations": {
            "norm_type": "RMSNorm",
            "norm_formula": "y = x / sqrt(mean(x^2) + eps) * gamma",
            "attn_formula": "Attn(Q,K,V) = softmax((QK^T)/sqrt(d_k))V",
            "rope_formula": "Q_rope = RoPE(Q), K_rope = RoPE(K)",
            "qk_norm": "disabled",
            "post_norm": "disabled",
        },
    }


def yesno(v: bool) -> str:
    return "yes" if v else "no"


def infer_norm_name(layer_norm_enabled: int) -> str:
    return "LayerNorm" if layer_norm_enabled else "RMSNorm"


def build_checks(trace: Dict[str, Any], expected: Dict[str, Any]) -> List[Dict[str, str]]:
    exp = expected["equations"]
    observed_norm = infer_norm_name(int(trace.get("layer_norm_enabled", 0)))
    observed_qk = bool(trace.get("qk_norm_enabled", 0))
    observed_post = bool(trace.get("post_norm_enabled", 0))

    checks: List[Dict[str, str]] = []

    checks.append(
        {
            "name": "norm_type",
            "expected": exp["norm_type"],
            "observed": observed_norm,
            "status": "PASS" if exp["norm_type"] == observed_norm else "MISMATCH",
        }
    )

    qk_expected_enabled = "enabled" in exp["qk_norm"].lower()
    checks.append(
        {
            "name": "qk_norm_enabled",
            "expected": yesno(qk_expected_enabled),
            "observed": yesno(observed_qk),
            "status": "PASS" if qk_expected_enabled == observed_qk else "MISMATCH",
        }
    )

    post_expected_enabled = "enabled" in exp["post_norm"].lower()
    checks.append(
        {
            "name": "post_norm_enabled",
            "expected": yesno(post_expected_enabled),
            "observed": yesno(observed_post),
            "status": "PASS" if post_expected_enabled == observed_post else "MISMATCH",
        }
    )

    return checks


def summarize_formula(formula: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not formula:
        return {}
    layer = formula.get("layer_summary", {})
    token = formula.get("token_summary", {})
    top = formula.get("top_diffs", [])
    return {
        "mean_score": layer.get("mean_score"),
        "median_score": layer.get("median_score"),
        "max_score": layer.get("max_score"),
        "token_top1_logit_fold": token.get("mean_top1_logit_fold_log2"),
        "top_diffs": top[:6],
    }


def render_report(trace: Dict[str, Any], expected: Dict[str, Any], formula_summary: Dict[str, Any]) -> str:
    arch = str(trace.get("model_arch", "unknown"))
    eq = expected["equations"]
    checks = build_checks(trace, expected)

    lines: List[str] = []
    lines.append(f"# Llama Formula Side-by-Side ({arch})")
    lines.append("")
    lines.append("## Canonical (llama.cpp)")
    for ref in expected.get("source_refs", []):
        lines.append(f"- {ref}")
    lines.append("")
    lines.append("### Equation Set")
    lines.append(f"- Norm: {eq['norm_type']}")
    lines.append(f"- Norm formula: {eq['norm_formula']}")
    lines.append(f"- Attention: {eq['attn_formula']}")
    lines.append(f"- RoPE: {eq['rope_formula']}")
    lines.append(f"- QK norm: {eq['qk_norm']}")
    lines.append(f"- Post norm: {eq['post_norm']}")
    lines.append("")
    lines.append("## Airframe Observed")
    lines.append(f"- model_arch: {trace.get('model_arch')}")
    lines.append(f"- prompt_mode: {trace.get('prompt_mode')}")
    lines.append(f"- prompt_renderer_mode: {trace.get('prompt_renderer_mode')}")
    lines.append(f"- prompt_renderer_family: {trace.get('prompt_renderer_family')}")
    lines.append(f"- prompt_template_source: {trace.get('prompt_template_source')}")
    lines.append(f"- templated_prompt: {repr(trace.get('templated_prompt', ''))}")
    lines.append(f"- norm_eps: {trace.get('norm_eps')}")
    lines.append(f"- layer_norm_enabled: {trace.get('layer_norm_enabled')}")
    lines.append(f"- qk_norm_enabled: {trace.get('qk_norm_enabled')}")
    lines.append(f"- post_norm_enabled: {trace.get('post_norm_enabled')}")
    lines.append(f"- packed_quant_type: {hex(int(trace.get('packed_quant_type', 0)))}")
    lines.append("")
    lines.append("## Consistency Checks")
    for c in checks:
        lines.append(f"- {c['name']}: expected={c['expected']} observed={c['observed']} status={c['status']}")
    lines.append("")

    if formula_summary:
        lines.append("## Formula Divergence")
        lines.append(f"- layer_mean_score: {formula_summary.get('mean_score')}")
        lines.append(f"- layer_median_score: {formula_summary.get('median_score')}")
        lines.append(f"- layer_max_score: {formula_summary.get('max_score')}")
        lines.append(f"- token_mean_top1_logit_fold_log2: {formula_summary.get('token_top1_logit_fold')}")
        top = formula_summary.get("top_diffs") or []
        if top:
            lines.append("")
            lines.append("### Top Divergent Points")
            for row in top:
                lines.append(
                    "- "
                    f"phase={row.get('phase')} step={row.get('step')} layer={row.get('layer')} "
                    f"score={row.get('score')} "
                    f"res_gain={row.get('metric_deltas', {}).get('residual_gain')} "
                    f"ffn_gain={row.get('metric_deltas', {}).get('ffn_gain')} "
                    f"qk_balance={row.get('metric_deltas', {}).get('qk_balance')} "
                    f"output_energy={row.get('metric_deltas', {}).get('output_energy')}"
                )
    lines.append("")
    lines.append("## Decision")
    mismatch_count = sum(1 for c in checks if c["status"] != "PASS")
    if mismatch_count == 0:
        lines.append("- Math-routing equations match canonical expectations for this model family.")
        lines.append("- Remaining quality issues likely live deeper than high-level equation routing (e.g., kernel math details, tensor interpretation, or logits path).")
    else:
        lines.append(f"- Found {mismatch_count} high-level equation-routing mismatches that should be fixed before deeper surgery.")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build llama.cpp vs Airframe formula side-by-side report")
    parser.add_argument("--trace", required=True, help="Trace JSON path or glob")
    parser.add_argument("--formula-json", help="Optional formula diff JSON path or glob")
    parser.add_argument("--out", required=True, help="Output markdown path")
    args = parser.parse_args()

    trace = load_json(args.trace)
    formula = load_json(args.formula_json) if args.formula_json else None

    expected = expected_for_arch(str(trace.get("model_arch", "unknown")))
    formula_summary = summarize_formula(formula)
    report = render_report(trace, expected, formula_summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"Wrote side-by-side report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
