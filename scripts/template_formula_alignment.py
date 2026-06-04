#!/usr/bin/env python3
"""
Template + formula alignment helper.

Goal:
- Compare Airframe trace prompt-renderer metadata against a known-good
  llama.cpp-style family inference from the rendered prompt text.
- Optionally include formula-lens JSON summary to prioritize impactful mismatches.

This script is intentionally lightweight and offline: it only reads artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_llama_family(prompt: str) -> str:
    # Mirrors key heuristics from llama.cpp llm_chat_detect_template at a coarse level.
    if "<|im_start|>" in prompt:
        if "<|im_sep|>" in prompt:
            return "phi4"
        if "<end_of_utterance>" in prompt:
            return "smolvlm"
        return "chatml"
    if "<|start_header_id|>" in prompt and "<|end_header_id|>" in prompt:
        return "llama3"
    if "<|header_start|>" in prompt and "<|header_end|>" in prompt:
        return "llama4"
    if "<start_of_turn>" in prompt and "<end_of_turn>" in prompt:
        return "gemma"
    if "<|user|>" in prompt and "<|endoftext|>" in prompt:
        return "zephyr"
    if "<|assistant|>" in prompt and "<|end|>" in prompt:
        return "phi3"
    if "USER:" in prompt and "ASSISTANT:" in prompt:
        return "vicuna"
    if "### Instruction:" in prompt and "<|EOT|>" in prompt:
        return "deepseek"
    if "<|START_OF_TURN_TOKEN|>" in prompt and "<|USER_TOKEN|>" in prompt:
        return "command-r"
    if "[gMASK]<sop>" in prompt:
        return "chatglm4"
    if "[gMASK]sop" in prompt:
        return "chatglm3"
    if "<用户>" in prompt or "<AI>" in prompt:
        return "minicpm"
    if "<｜User｜>" in prompt and "<｜Assistant｜>" in prompt:
        return "deepseek3"
    if "<|start_of_role|>" in prompt and "<|end_of_role|>" in prompt:
        return "granite"
    if "<role>ASSISTANT</role>" in prompt:
        return "bailing"
    if "<|begin|>" in prompt and "<|content|>" in prompt and "<|end|>" in prompt:
        return "solar-open"
    return "completion_or_unknown"


def map_airframe_family_to_llama_hint(airframe_family: str) -> str:
    mapping = {
        "TinyLlama": "chatml_or_tinyllama_variant",
        "ChatML": "chatml",
        "Llama3": "llama3",
        "Gemma2": "gemma",
        "MiniCpmV": "minicpm",
        "Completion": "completion_or_unknown",
    }
    return mapping.get(airframe_family, "unknown")


def pick_formula_mean_score(formula: Dict[str, Any]) -> Optional[float]:
    summary = formula.get("summary")
    if isinstance(summary, dict):
        score = summary.get("mean_score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def evaluate(trace: Dict[str, Any], formula_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    renderer_mode = trace.get("prompt_renderer_mode", "none")
    renderer_family = trace.get("prompt_renderer_family")
    template_source = trace.get("prompt_template_source", "unknown")
    prompt_mode = trace.get("prompt_mode", "unknown")
    templated_prompt = trace.get("templated_prompt", "")

    inferred_llama = infer_llama_family(templated_prompt)
    inferred_from_airframe = (
        map_airframe_family_to_llama_hint(renderer_family) if renderer_family else "none"
    )

    formula_mean = pick_formula_mean_score(formula_json) if formula_json else None

    mismatch = False
    if renderer_mode == "family" and inferred_from_airframe not in {"unknown", inferred_llama, "chatml_or_tinyllama_variant"}:
        mismatch = True

    severity = "info"
    if mismatch and formula_mean is not None and formula_mean >= 0.2:
        severity = "high"
    elif mismatch:
        severity = "medium"

    return {
        "prompt_mode": prompt_mode,
        "renderer_mode": renderer_mode,
        "renderer_family": renderer_family,
        "template_source": template_source,
        "llama_inferred_family": inferred_llama,
        "airframe_family_llama_hint": inferred_from_airframe,
        "formula_mean_score": formula_mean,
        "mismatch": mismatch,
        "severity": severity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare template routing with llama-style inference and formula score")
    parser.add_argument("--trace", required=True, help="Path to InferenceTracePackage JSON")
    parser.add_argument("--formula-json", help="Optional formula diff JSON output")
    parser.add_argument("--json-out", help="Optional output JSON path")
    args = parser.parse_args()

    trace = load_json(Path(args.trace))
    formula = load_json(Path(args.formula_json)) if args.formula_json else None

    report = evaluate(trace, formula)

    print("Template/Formula Alignment Report")
    print(f"  prompt_mode:          {report['prompt_mode']}")
    print(f"  renderer_mode:        {report['renderer_mode']}")
    print(f"  renderer_family:      {report['renderer_family']}")
    print(f"  template_source:      {report['template_source']}")
    print(f"  llama_inferred:       {report['llama_inferred_family']}")
    print(f"  airframe_hint:        {report['airframe_family_llama_hint']}")
    print(f"  formula_mean_score:   {report['formula_mean_score']}")
    print(f"  mismatch:             {report['mismatch']}")
    print(f"  severity:             {report['severity']}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
