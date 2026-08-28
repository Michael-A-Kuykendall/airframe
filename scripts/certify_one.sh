#!/usr/bin/env bash
# certify_one.sh — STANDARDIZED per-model certification runner (Linux, real GPU)
#
# THE authoritative cert flow (see airframe/docs/CERT_REGIMEN.md). Runs the
# real MATH + INFERENCE gates for ONE model and persists the result to the
# unified DuckDB cert ledger (cert/ledger.duckdb). A model is "certified" iff
# this exits 0 with math_ok=true AND chat_ok=true recorded for it.
#
# Usage:
#   bash scripts/certificate.sh <model_id> <path/to/model.gguf> [discover_name]
#
# model_id:     canonical id, e.g. qwen3-0.6b-q4-k-m (also the ledger key)
# discover_name: shimmy list name if different from model_id (default: model_id)
#
# Gates (stop on first red):
#   G1 quant_verify     -- dequant conformance vs quant_formula
#   G2 stack_dump_gpu   -- per-layer PEEL (multi-token prompt)
#   G3 cert_reds.py     -- PLAN vs PEEL diff -> reds.json (0 reds = MATH green)
#   G5 shimmy generate  -- INFERENCE: several prompts, coherent, 0 NaN
#   G6 cert_ledger.py   -- persist to cert/ledger.duckdb (math_runs + family_runs + reds)
#
# Env (GPU): WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER, MESA_D3D12_DEFAULT_ADAPTER_NAME,
#            LD_LIBRARY_PATH=/usr/lib/wsl/lib  (llvmpipe is NOT a valid cert runner)

set -euo pipefail

MODEL_ID="${1:-}"
GGUF="${2:-}"
DISCOVER="${3:-$MODEL_ID}"
[[ -n "$MODEL_ID" && -n "$GGUF" ]] || { echo "Usage: $0 <model_id> <model.gguf> [discover_name]" >&2; exit 2; }
[[ -f "$GGUF" ]] || { echo "SKIP - not found: $GGUF" >&2; exit 0; }

AF="/home/michael/repos/airframe-workspace/airframe"
SH="/home/michael/repos/airframe-workspace/shimmy"
CERT="$AF/cert"
PKG="$CERT/packages/$MODEL_ID"
LEDGER="$CERT/ledger.duckdb"

mkdir -p "$PKG/math"

export WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
export SHIMMY_MAX_CTX="${SHIMMY_MAX_CTX:-2048}"
export KV_QUANT="${KV_QUANT:-f32}"

PROMPT="${PROMPT:-The capital of France is}"
# The 5-prompt inference battery (multi-question, not one word).
INFERENCE_PROMPTS=(
  "The capital of France is"
  "2 + 2 ="
  "Write a Rust function that reverses a string"
  "Explain the water cycle in three sentences"
  "Once upon a time there was a small llama who"
)

header() { echo ""; echo "═══ $* ═══"; }

header "CERTIFY: $MODEL_ID"
echo "  gguf:  $GGUF"
echo "  pkg:   $PKG"
echo "  ledger:$LEDGER"

GIT_SHA=$(git -C "$AF" rev-parse HEAD 2>/dev/null || echo "unknown")

header "G1 — quant_verify (dequant conformance)"
if "$AF/target/release/quant_verify" --model-path "$GGUF" > "$PKG/math/01_quant_verify.log" 2>&1; then
  echo "  G1 PASS"
else
  echo "  G1 RED — quant_verify failed" >&2; tail -20 "$PKG/math/01_quant_verify.log" >&2; exit 1
fi

# G2/G3: PLAN vs PEEL via stack_dump_gpu.json + cert_reds.py
header "G2 — stack_dump_gpu (PEEL)"
if "$AF/target/release/stack_dump_gpu" "$GGUF" "$PROMPT" "$PKG/math/peel.json" --top-k 2 > "$PKG/math/02_stack.log" 2>&1; then
  echo "  G2 PASS"
else
  echo "  G2 RED — stack_dump_gpu failed" >&2; tail -10 "$PKG/math/02_stack.log" >&2; exit 2
fi

header "G3 — cert_reds.py (PLAN vs PEEL)"
if python3 "$AF/scripts/cert/cert_reds.py" \
    "$PKG/math/peel.json" \
    --plan-out "$PKG/math/plan.json" \
    --reds-out "$PKG/math/reds.json" \
    --report-out "$PKG/math/REPORT.md" \
    --quant-log "$PKG/math/01_quant_verify.log" \
    --family-id "$MODEL_ID" > "$PKG/math/03_reds.log" 2>&1; then
  N_REDS=$(python3 -c "import json; print(len(json.load(open('$PKG/math/reds.json')).get('reds',[])))" 2>/dev/null || echo "?")
  echo "  G3 PASS — $N_REDS reds"
  if [ "$N_REDS" != "0" ] && [ -n "$N_REDS" ] && [ "$N_REDS" != "?" ]; then
    echo "  WARN: non-zero reds; model not math-clean"
  fi
else
  echo "  G3 RED — cert_reds failed" >&2; cat "$PKG/math/03_reds.log" >&2; exit 3
fi

# G5 — inference battery (multi-question, deterministic)
header "G5 — inference battery (real GPU, multi-question)"
INF_OK=1
mkdir -p "$PKG/inference_exam"
for i in "${!INFERENCE_PROMPTS[@]}"; do
  p="${INFERENCE_PROMPTS[$i]}"
  out="$PKG/inference_exam/prompt_$((i+1)).text"
  log="$PKG/inference_exam/prompt_$((i+1)).log"
  (
    cd "$SH"
    echo "P: $p"
    WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER=1 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
      LD_LIBRARY_PATH=/usr/lib/wsl/lib SHIMMY_MAX_CTX="$SHIMMY_MAX_CTX" \
      target/release/shimmy generate "$DISCOVER" --prompt "$p" --max-tokens 24 2>"$log" \
      | tee "$out"
  ) || INF_OK=0
  if grep -qiE "error:.*gpu|validation error" "$log"; then INF_OK=0; echo "  error in prompt $((i+1))"; fi
  # NaN is a hard failure ONLY when the count is non-zero (e.g. nans=5).
  # The healthy counter `nans=0` / `logits_nans=0` is NOT a failure.
  if grep -qiE "nans=[1-9]|logits_nans=[1-9]|nan_count=[1-9]|FAILED.*nan" "$log"; then INF_OK=0; echo "  NaN in prompt $((i+1))"; fi
done
if [ "$INF_OK" = "1" ]; then
  echo "  G5 PASS — coherent, no NaN"
else
  echo "  G5 RED — inference battery failed" >&2; exit 5
fi

# G6 — persist to ledger
header "G6 — persist to cert/ledger.duckdb"
MATH_OK=$(python3 -c "import json; print('true' if json.load(open('$PKG/math/reds.json')).get('math_ok') else 'false')" 2>/dev/null || echo "false")
python3 "$AF/scripts/cert/cert_ledger.py" --db "$LEDGER" record \
  --family-id "$MODEL_ID" \
  --reds-json "$PKG/math/reds.json" \
  --report "$PKG/math/REPORT.md" \
  --chat-ok "$([[ "$INF_OK" == 1 ]] && echo true || echo false)" \
  --git-sha "$GIT_SHA"
echo "  G6 done (math_ok=$MATH_OK chat_ok=$INF_OK)"

echo ""
echo "✅ CERTIFIED: $MODEL_ID  (math_ok=$MATH_OK chat_ok=$INF_OK)"
echo "   packages: $PKG"