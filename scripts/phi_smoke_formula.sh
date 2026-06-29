#!/usr/bin/env bash
# Standard Phi smoke + formula validation harness.
#
# Usage:
#   bash scripts/phi_smoke_formula.sh
#
# Optional env overrides:
#   MODEL_PATH       GGUF path (default: D:/shimmy-test-models/gguf_collection/phi-2.Q4_K_M.gguf)
#   BASE_URL         Server URL (default: http://127.0.0.1:8080)
#   SHIMMY_PORT      Server port (default: 8080)
#   STARTUP_TIMEOUT  Seconds to wait for readiness (default: 180)
#   REQUEST_TIMEOUT  Curl timeout per request in seconds (default: 60)
#   PROFILE          Debug profile name (default: phi2_smoke_formula)
#   MAX_TOKENS_A     Max tokens for math prompt (default: 16)
#   MAX_TOKENS_B     Max tokens for capital prompt (default: 24)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-D:/shimmy-test-models/gguf_collection/phi-2.Q4_K_M.gguf}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
SHIMMY_PORT="${SHIMMY_PORT:-8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"
PROFILE="${PROFILE:-phi2_smoke_formula}"
MAX_TOKENS_A="${MAX_TOKENS_A:-16}"
MAX_TOKENS_B="${MAX_TOKENS_B:-24}"

LOG_PATH="/tmp/${PROFILE}.log"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    taskkill //F //IM shimmy_server_gpu.exe 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "== Phi smoke harness =="
echo "profile: $PROFILE"
echo "model:   $MODEL_PATH"
echo "base:    $BASE_URL"

taskkill //F //IM shimmy_server_gpu.exe 2>/dev/null || true

env -u SHIMMY_FSE_REJECT_PATTERNS \
    -u SHIMMY_FSE_REJECT_PATTERNS_PATH \
    SHIMMY_DEBUG_PROFILE="$PROFILE" \
    RUST_BACKTRACE=1 \
    SHIMMY_PORT="$SHIMMY_PORT" \
    LIBSHIMMY_MODEL_PATH="$MODEL_PATH" \
    ./target/release/shimmy_server_gpu.exe > "$LOG_PATH" 2>&1 &
SERVER_PID=$!

ready_url="$BASE_URL/api/repro/queue"
ready=0
for ((i=0; i<STARTUP_TIMEOUT; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server exited before readiness"
        tail -40 "$LOG_PATH" || true
        exit 1
    fi
    if curl -fsS -m 2 "$ready_url" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done

if [[ "$ready" -ne 1 ]]; then
    echo "FAIL: readiness timeout after ${STARTUP_TIMEOUT}s"
    tail -40 "$LOG_PATH" || true
    exit 1
fi

echo "MATH_PROMPT:"
curl -fsS -m "$REQUEST_TIMEOUT" \
    -X POST "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"phi-2\",\"prompt\":\"Math: 2+2=\",\"prompt_mode\":\"raw\",\"max_tokens\":$MAX_TOKENS_A,\"temperature\":0}" || true
echo

echo "CAPITAL_PROMPT:"
curl -fsS -m "$REQUEST_TIMEOUT" \
    -X POST "$BASE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"phi-2\",\"prompt\":\"The capital of France is\",\"prompt_mode\":\"raw\",\"max_tokens\":$MAX_TOKENS_B,\"temperature\":0.2}" || true
echo

candidate="$(ls -1t "artifacts/debug/${PROFILE}"/trace_*.json 2>/dev/null | head -1 || true)"
if [[ -z "$candidate" ]]; then
    echo "FAIL: no trace artifact found under artifacts/debug/${PROFILE}"
    exit 1
fi

set +e
python scripts/trace_formula_diff.py \
    --golden-bank artifacts/debug/phi2_nan_hunt/golden_bank.json \
    --candidate "$candidate" \
    --top 20 \
    --fail-threshold 0.65 \
    --json-out "artifacts/debug/${PROFILE}/formula_diff_vs_bank.json"
FORMULA_EXIT=$?
set -e

echo
echo "PREFILL_SANITY:"
rg -n "PREFILL_SANITY" "$LOG_PATH" | tail -10 || true

echo
echo "TOKEN_TAIL:"
rg -n "\[TOKEN\]|Metric Violation" "$LOG_PATH" | tail -30 || true

echo
echo "Artifacts:"
echo "  trace:      $candidate"
echo "  formula:    artifacts/debug/${PROFILE}/formula_diff_vs_bank.json"
echo "  server log: $LOG_PATH"

if [[ "$FORMULA_EXIT" -ne 0 ]]; then
    echo "RESULT: formula gate failed (exit=$FORMULA_EXIT)"
    exit "$FORMULA_EXIT"
fi

echo "RESULT: smoke harness passed"