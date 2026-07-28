#!/usr/bin/env python3
"""Build PLAN (law) from an airframe.stack.v1 peel config + family invariants.

No GPU. Pure math / header rules. See CERT_REGIMEN.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def plan_from_config(cfg: dict[str, Any], *, family_id: str | None = None) -> dict[str, Any]:
    """Derive PLAN from stack dump `config` object (GGUF → ModelSpec fields)."""
    n_embd = int(cfg["n_embd"])
    n_head = int(cfg["n_head"])
    n_kv = int(cfg.get("n_kv_head") or cfg.get("n_head_kv") or n_head)
    head_dim = int(cfg["head_dim"])
    n_layer = int(cfg["n_layer"])
    arch = str(cfg.get("arch") or "").lower()
    qk_norm = bool(cfg.get("qk_norm", False))
    # Qwen3 always has QK-norm even if a stale dump omitted the flag
    if arch in ("qwen3",) or arch.startswith("qwen3"):
        qk_norm = True

    ffn_dim = cfg.get("ffn_dim") or cfg.get("ff_dim")
    if ffn_dim is None:
        # not always in stack config today — leave null and let peel supply when checking gate only
        ffn_dim = None
    else:
        ffn_dim = int(ffn_dim)

    dim_q = n_head * head_dim
    dim_kv = n_kv * head_dim
    norm_slots = 6 if qk_norm else 4
    rope_base = cfg.get("rope_base")
    rope_dim = cfg.get("rope_dim", head_dim)
    rms_eps = cfg.get("rms_eps")

    stage_counts: dict[str, int] = {
        "residual_in": n_embd,
        "attn_norm": n_embd,
        "q": dim_q,
        "k": dim_kv,
        "v": dim_kv,
        "attn_ctx": dim_q,
        "attn_residual": n_embd,
        "ffn_norm": n_embd,
        "ffn_residual": n_embd,
    }
    if qk_norm:
        stage_counts["q_norm"] = dim_q
        stage_counts["k_norm"] = dim_kv
    if ffn_dim is not None:
        stage_counts["ffn_gate"] = ffn_dim
        stage_counts["ffn_up"] = ffn_dim

    warnings: list[str] = []
    # Family rope expectations (warn, not hard red — architecture signal)
    if arch.startswith("qwen") and rope_base is not None:
        try:
            rb = float(rope_base)
            if rb < 1e5:
                warnings.append(
                    f"rope_base={rb} looks like Llama-scale; Qwen family usually 1e6"
                )
        except (TypeError, ValueError):
            pass
    if dim_q != n_embd:
        # Legal for Qwen3 (4096 vs 2560); record so tools cannot "fix" to n_embd
        warnings.append(
            f"dim_q ({dim_q}) != n_embd ({n_embd}) — attn must use dim_q, not n_embd"
        )

    return {
        "schema": "airframe.plan.v1",
        "family_id": family_id,
        "arch": arch,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_kv_head": n_kv,
        "head_dim": head_dim,
        "dim_q": dim_q,
        "dim_kv": dim_kv,
        "ffn_dim": ffn_dim,
        "qk_norm": qk_norm,
        "norm_slots": norm_slots,
        "rope_base": rope_base,
        "rope_dim": rope_dim,
        "rms_eps": rms_eps,
        "stage_counts": stage_counts,
        "required_stages": list(stage_counts.keys()),
        "warnings": warnings,
        "source": "stack.config + family invariants",
    }


def plan_from_stack(stack: dict[str, Any], *, family_id: str | None = None) -> dict[str, Any]:
    cfg = stack.get("config") or {}
    plan = plan_from_config(cfg, family_id=family_id)
    # Prefer ffn_dim from first layer ffn_gate.count if config omitted it
    if plan["ffn_dim"] is None:
        layers = stack.get("layers") or []
        if layers:
            st = (layers[0].get("stages") or {}).get("ffn_gate") or {}
            c = st.get("count")
            if isinstance(c, int) and c > 0:
                plan["ffn_dim"] = c
                plan["stage_counts"]["ffn_gate"] = c
                plan["stage_counts"]["ffn_up"] = c
                if "ffn_gate" not in plan["required_stages"]:
                    plan["required_stages"].extend(["ffn_gate", "ffn_up"])
    return plan


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build PLAN from airframe.stack.v1 JSON")
    p.add_argument("stack_json", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--family-id", default=None)
    args = p.parse_args(argv)
    stack = json.loads(args.stack_json.read_text(encoding="utf-8"))
    plan = plan_from_stack(stack, family_id=args.family_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"wrote {args.out} norm_slots={plan['norm_slots']} dim_q={plan['dim_q']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
