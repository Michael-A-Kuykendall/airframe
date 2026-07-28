#!/usr/bin/env python3
"""Judge PLAN vs PEEL (+ optional quant_verify log) → reds.json + REPORT.md.

No GPU. Exit 0 iff zero REDs. See CERT_REGIMEN.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/cert_reds.py` without install
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cert_plan import plan_from_stack  # noqa: E402


def parse_quant_verify(log_text: str) -> list[dict[str, Any]]:
    """Extract per-type PASS/FAIL from quant_verify log."""
    reds: list[dict[str, Any]] = []
    # [quant_verify] Q4_K  (type 12) — ...
    #   FAIL  max_err=...
    #   PASS  max_err=...
    current_type: str | None = None
    type_re = re.compile(
        r"\[quant_verify\]\s+(\S+)\s+\(type\s+(\d+)\)"
    )
    fail_re = re.compile(r"FAIL\s+max_err=([0-9.eE+-]+)")
    pass_re = re.compile(r"PASS\s+max_err=")
    skip_re = re.compile(r"not present in model, skipping")
    for line in log_text.splitlines():
        m = type_re.search(line)
        if m:
            current_type = m.group(1)
            continue
        if current_type and skip_re.search(line):
            current_type = None
            continue
        if current_type and fail_re.search(line):
            err = fail_re.search(line).group(1)
            reds.append(
                {
                    "code": f"DEQUANT.{current_type}",
                    "severity": "error",
                    "detail": f"GPU vs quant_formula max_err={err}",
                    "plan": "match quant_formula within tolerance",
                    "peel": f"FAIL max_err={err}",
                }
            )
            current_type = None
            continue
        if current_type and pass_re.search(line):
            current_type = None
            continue
    if "FAILED — one or more quant types" in log_text and not reds:
        reds.append(
            {
                "code": "DEQUANT.UNKNOWN",
                "severity": "error",
                "detail": "quant_verify reported FAILED but no type lines parsed",
                "plan": "ALL PASS",
                "peel": "FAILED",
            }
        )
    return reds


def judge_peel(plan: dict[str, Any], stack: dict[str, Any]) -> list[dict[str, Any]]:
    reds: list[dict[str, Any]] = []
    required = plan.get("stage_counts") or {}
    layers = stack.get("layers") or []
    n_layer = int(plan.get("n_layer") or 0)

    if n_layer and len(layers) != n_layer:
        reds.append(
            {
                "code": "PEEL.layer_count",
                "severity": "error",
                "detail": f"peel has {len(layers)} layers, plan n_layer={n_layer}",
                "plan": n_layer,
                "peel": len(layers),
            }
        )

    # Config consistency: head_dim / dim_q in dump config
    cfg = stack.get("config") or {}
    if cfg:
        hd = cfg.get("head_dim")
        nh = cfg.get("n_head")
        if hd is not None and nh is not None:
            expect_q = int(nh) * int(hd)
            if expect_q != plan["dim_q"]:
                reds.append(
                    {
                        "code": "PLAN.internal",
                        "severity": "error",
                        "detail": "plan dim_q inconsistent with config",
                        "plan": plan["dim_q"],
                        "peel": expect_q,
                    }
                )

    for layer in layers:
        li = layer.get("layer_idx", "?")
        stages = layer.get("stages") or {}
        # residual nans
        for key in ("residual", "residual_out", "residual_in"):
            block = layer.get(key)
            if isinstance(block, dict) and block.get("nan_count", 0):
                reds.append(
                    {
                        "code": f"L{li}.{key}.nan",
                        "severity": "error",
                        "detail": f"nan_count={block.get('nan_count')}",
                        "plan": 0,
                        "peel": block.get("nan_count"),
                    }
                )

        for name, expect_count in required.items():
            if name == "residual_in":
                # may live top-level residual_in
                block = layer.get("residual_in")
                if block is None:
                    # optional if stages don't include it
                    continue
            else:
                block = stages.get(name)
            if block is None:
                reds.append(
                    {
                        "code": f"L{li}.{name}.missing",
                        "severity": "error",
                        "detail": "required stage absent from peel",
                        "plan": f"count={expect_count} sampled=real",
                        "peel": None,
                    }
                )
                continue
            if not isinstance(block, dict):
                reds.append(
                    {
                        "code": f"L{li}.{name}.shape",
                        "severity": "error",
                        "detail": "stage not an object",
                        "plan": "object with count/sampled",
                        "peel": type(block).__name__,
                    }
                )
                continue
            got = block.get("count")
            if got is not None and int(got) != int(expect_count):
                reds.append(
                    {
                        "code": f"L{li}.{name}.count",
                        "severity": "error",
                        "detail": "stage element count mismatch",
                        "plan": expect_count,
                        "peel": got,
                    }
                )
            sampled = block.get("sampled")
            if sampled is not None and sampled != "real":
                reds.append(
                    {
                        "code": f"L{li}.{name}.sampled",
                        "severity": "error",
                        "detail": "stage not real capture",
                        "plan": "real",
                        "peel": sampled,
                    }
                )
            if block.get("nan_count", 0):
                reds.append(
                    {
                        "code": f"L{li}.{name}.nan",
                        "severity": "error",
                        "detail": f"nan_count={block.get('nan_count')}",
                        "plan": 0,
                        "peel": block.get("nan_count"),
                    }
                )

    # Final logits
    final = stack.get("final") or {}
    logits = final.get("logits") or {}
    if isinstance(logits, dict):
        if logits.get("nan_count", 0):
            reds.append(
                {
                    "code": "FINAL.logits.nan",
                    "severity": "error",
                    "detail": f"nan_count={logits.get('nan_count')}",
                    "plan": 0,
                    "peel": logits.get("nan_count"),
                }
            )
        ln = logits.get("len")
        if ln is not None and int(ln) <= 0:
            reds.append(
                {
                    "code": "FINAL.logits.len",
                    "severity": "error",
                    "detail": "empty logits",
                    "plan": ">0",
                    "peel": ln,
                }
            )

    return reds


def build_report(
    plan: dict[str, Any],
    reds: list[dict[str, Any]],
    *,
    family_id: str | None,
    warnings: list[str],
) -> str:
    lines = [
        f"# Cert REPORT — {family_id or plan.get('family_id') or '?'}",
        "",
        f"**MATH:** {'GREEN' if not reds else 'RED'} ({len(reds)} reds)",
        "",
        "## Plan summary",
        "",
        f"- arch={plan.get('arch')} layers={plan.get('n_layer')}",
        f"- n_embd={plan.get('n_embd')} head_dim={plan.get('head_dim')} dim_q={plan.get('dim_q')} dim_kv={plan.get('dim_kv')}",
        f"- ffn_dim={plan.get('ffn_dim')} qk_norm={plan.get('qk_norm')} **norm_slots={plan.get('norm_slots')}**",
        f"- rope_base={plan.get('rope_base')} rms_eps={plan.get('rms_eps')}",
        "",
    ]
    if warnings:
        lines.append("## Warnings (not auto-RED)")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("## Reds")
    lines.append("")
    if not reds:
        lines.append("_none_")
    else:
        lines.append("| code | plan | peel | detail |")
        lines.append("|------|------|------|--------|")
        for r in reds:
            lines.append(
                f"| `{r['code']}` | {r.get('plan')} | {r.get('peel')} | {r.get('detail')} |"
            )
    lines.append("")
    lines.append("## Authority")
    lines.append("")
    lines.append("- PLAN: GGUF config + family invariants (`cert_plan.py`)")
    lines.append("- DEQUANT: `airframe_observe::quant_formula` via quant_verify")
    lines.append("- PEEL: product `stack_dump_gpu` stages")
    lines.append("- External engines: never sole GREEN")
    lines.append("")
    return "\n".join(lines)


def run_judge(
    stack: dict[str, Any],
    *,
    family_id: str | None = None,
    quant_log: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    plan = plan_from_stack(stack, family_id=family_id)
    reds = judge_peel(plan, stack)
    if quant_log:
        reds.extend(parse_quant_verify(quant_log))
    reds.sort(key=lambda r: r["code"])
    report = build_report(
        plan, reds, family_id=family_id, warnings=list(plan.get("warnings") or [])
    )
    return plan, reds, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PLAN vs PEEL → reds.json")
    ap.add_argument("stack_json", type=Path, help="airframe.stack.v1 peel JSON")
    ap.add_argument("--plan-out", type=Path, default=None)
    ap.add_argument("--reds-out", type=Path, required=True)
    ap.add_argument("--report-out", type=Path, default=None)
    ap.add_argument("--quant-log", type=Path, default=None)
    ap.add_argument("--family-id", default=None)
    args = ap.parse_args(argv)

    stack = json.loads(args.stack_json.read_text(encoding="utf-8"))
    quant_text = None
    if args.quant_log and args.quant_log.is_file():
        quant_text = args.quant_log.read_text(encoding="utf-8", errors="replace")

    plan = plan_from_stack(stack, family_id=args.family_id)
    reds = judge_peel(plan, stack)
    if quant_text:
        reds.extend(parse_quant_verify(quant_text))
    reds.sort(key=lambda r: r["code"])

    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    payload = {
        "schema": "airframe.reds.v1",
        "family_id": args.family_id or plan.get("family_id"),
        "math_ok": len(reds) == 0,
        "n_reds": len(reds),
        "reds": reds,
        "plan_summary": {
            "norm_slots": plan.get("norm_slots"),
            "dim_q": plan.get("dim_q"),
            "dim_kv": plan.get("dim_kv"),
            "ffn_dim": plan.get("ffn_dim"),
            "qk_norm": plan.get("qk_norm"),
            "n_layer": plan.get("n_layer"),
        },
    }
    args.reds_out.parent.mkdir(parents=True, exist_ok=True)
    args.reds_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = build_report(
        plan, reds, family_id=args.family_id, warnings=list(plan.get("warnings") or [])
    )
    report_path = args.report_out or (args.reds_out.parent / "REPORT.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"MATH={'GREEN' if payload['math_ok'] else 'RED'} n_reds={payload['n_reds']}")
    print(f"wrote {args.reds_out}")
    print(f"wrote {report_path}")
    for r in reds[:20]:
        print(f"  RED {r['code']}: plan={r.get('plan')} peel={r.get('peel')}")
    if len(reds) > 20:
        print(f"  ... +{len(reds) - 20} more")
    return 0 if payload["math_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
