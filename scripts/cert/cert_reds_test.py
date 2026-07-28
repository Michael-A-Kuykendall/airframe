#!/usr/bin/env python3
"""Regression tests for cert_plan / cert_reds (no GPU).

Run: python scripts/cert_reds_test.py
Exit 0 = all pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cert_plan import plan_from_config, plan_from_stack  # noqa: E402
from cert_reds import judge_peel, parse_quant_verify, run_judge  # noqa: E402

FIX = ROOT / "fixtures"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  OK  {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


def main() -> int:
    print("=== cert_plan / cert_reds regression ===")

    # 1) qk_norm ⇒ norm_slots 6
    plan = plan_from_config(
        {
            "arch": "qwen3",
            "n_layer": 36,
            "n_embd": 2560,
            "n_head": 32,
            "n_kv_head": 8,
            "head_dim": 128,
            "qk_norm": True,
            "ffn_dim": 9728,
            "rope_base": 1e6,
        },
        family_id="qwen3-test",
    )
    check("norm_slots_6_when_qk_norm", plan["norm_slots"] == 6, str(plan["norm_slots"]))
    check("dim_q_4096", plan["dim_q"] == 4096, str(plan["dim_q"]))
    check("dim_kv_1024", plan["dim_kv"] == 1024, str(plan["dim_kv"]))
    check("q_count_in_stages", plan["stage_counts"]["q"] == 4096)
    check("attn_ctx_is_dim_q", plan["stage_counts"]["attn_ctx"] == 4096)

    # 2) no qk_norm ⇒ 4 slots
    plan4 = plan_from_config(
        {
            "arch": "llama",
            "n_layer": 2,
            "n_embd": 2048,
            "n_head": 32,
            "n_kv_head": 8,
            "head_dim": 64,
            "qk_norm": False,
            "ffn_dim": 5632,
        }
    )
    check("norm_slots_4_no_qk", plan4["norm_slots"] == 4, str(plan4["norm_slots"]))
    check("no_q_norm_stage", "q_norm" not in plan4["stage_counts"])

    # 3) good peel → zero peel reds
    ok = json.loads((FIX / "peel_ok_minimal.json").read_text(encoding="utf-8"))
    plan_ok = plan_from_stack(ok, family_id="fixture-ok")
    reds_ok = judge_peel(plan_ok, ok)
    check("peel_ok_zero_reds", len(reds_ok) == 0, str(reds_ok))

    # 4) bad q.count (2560 vs 4096) → RED L0.q.count
    bad = json.loads((FIX / "peel_bad_q_count.json").read_text(encoding="utf-8"))
    plan_bad = plan_from_stack(bad)
    reds_bad = judge_peel(plan_bad, bad)
    codes = {r["code"] for r in reds_bad}
    check("bad_q_count_red", "L0.q.count" in codes, str(codes))
    check("bad_attn_ctx_count_red", "L0.attn_ctx.count" in codes, str(codes))

    # 5) missing stage
    missing = json.loads(json.dumps(ok))
    del missing["layers"][0]["stages"]["v"]
    reds_m = judge_peel(plan_from_stack(missing), missing)
    check(
        "missing_v_red",
        any(r["code"] == "L0.v.missing" for r in reds_m),
        str([r["code"] for r in reds_m]),
    )

    # 6) quant_verify parse
    qlog = (FIX / "quant_verify_fail_snippet.log").read_text(encoding="utf-8")
    qreds = parse_quant_verify(qlog)
    check("dequant_q4k_red", any(r["code"] == "DEQUANT.Q4_K" for r in qreds), str(qreds))
    check("dequant_no_false_f32", not any(r["code"] == "DEQUANT.F32" for r in qreds))

    # 7) full judge with quant log on ok peel still reds on dequant
    _, reds_j, _ = run_judge(ok, family_id="x", quant_log=qlog)
    check("combined_dequant_red", any(r["code"] == "DEQUANT.Q4_K" for r in reds_j))
    check("combined_not_math_ok", len(reds_j) > 0)

    # 8) under-dispatch class: attn_ctx used dim not dim_q — already covered by bad fixture

    print()
    if failures:
        print(f"FAILED {len(failures)}: {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
