#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${1:-}"
PROFILE_BASE="${2:-}"
PROMPT_TEXT="${3:-}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
SHIMMY_PORT="${SHIMMY_PORT:-8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"

if [[ -z "$MODEL_PATH" || -z "$PROFILE_BASE" || -z "$PROMPT_TEXT" ]]; then
    echo "usage: bash scripts/prompt_mode_formula_probe.sh <model_path> <profile_base> <prompt_text>" >&2
    exit 1
fi

run_mode() {
    local mode="$1"
    local profile="${PROFILE_BASE}_${mode}"
    local server_log="/tmp/${profile}.server.log"
    local req_file="/tmp/${profile}.request.json"
    local resp_file="/tmp/${profile}.response.json"
    local pid=""

    rm -rf "artifacts/debug/${profile}"
    taskkill //F //IM shimmy_server_gpu.exe >/dev/null 2>&1 || true

    env -u SHIMMY_FSE_REJECT_PATTERNS \
        -u SHIMMY_FSE_REJECT_PATTERNS_PATH \
        SHIMMY_DEBUG_PROFILE="$profile" \
        SHIMMY_TRACE_LIGHT_MODE=1 \
        SHIMMY_PORT="$SHIMMY_PORT" \
        RUST_BACKTRACE=1 \
        LIBSHIMMY_MODEL_PATH="$MODEL_PATH" \
        ./target/release/shimmy_server_gpu.exe >"$server_log" 2>&1 &
    pid=$!

    for i in $(seq 1 "$STARTUP_TIMEOUT"); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "SERVER_EXIT profile=$profile" >&2
            tail -40 "$server_log" >&2 || true
            return 1
        fi
        if curl -fsS -m 2 "$BASE_URL/api/repro/queue" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if [[ "$mode" == "chat" ]]; then
        python - "$PROMPT_TEXT" "$MAX_TOKENS" "$TEMPERATURE" <<'PY' > "$req_file"
import json
import sys

print(json.dumps({
    'model': 'local',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': int(sys.argv[2]),
    'temperature': float(sys.argv[3]),
    'stream': False,
}))
PY
        curl -fsS -m 120 \
            -X POST "$BASE_URL/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            --data-binary @"$req_file" > "$resp_file"
        python - "$resp_file" <<'PY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    data = json.load(handle)
print(data['choices'][0]['message']['content'])
PY
    else
        python - "$PROMPT_TEXT" "$MAX_TOKENS" "$TEMPERATURE" <<'PY' > "$req_file"
import json
import sys

print(json.dumps({
    'model': 'local',
    'prompt': sys.argv[1],
    'prompt_mode': 'raw',
    'max_tokens': int(sys.argv[2]),
    'temperature': float(sys.argv[3]),
}))
PY
        curl -fsS -m 120 \
            -X POST "$BASE_URL/v1/completions" \
            -H 'Content-Type: application/json' \
            --data-binary @"$req_file" > "$resp_file"
        python - "$resp_file" <<'PY'
import json
import sys

with open(sys.argv[1], 'r', encoding='utf-8') as handle:
    data = json.load(handle)
choice = data['choices'][0]
print(choice.get('text') or choice.get('message', {}).get('content', ''))
PY
    fi

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    echo "--- ${profile} log markers ---"
    rg -n '\[ARCH_REGISTRY\]|\[ARCH_TENSOR|PromptRenderer|thinking|chat_template' "$server_log" || true
    echo "--- ${profile} trace ---"
    ls -1t "artifacts/debug/${profile}"/trace_*.json | head -1
    echo
}

echo "=== ${PROFILE_BASE} chat ==="
run_mode chat

echo "=== ${PROFILE_BASE} raw ==="
run_mode raw

chat_trace="$(ls -1t "artifacts/debug/${PROFILE_BASE}_chat"/trace_*.json | head -1)"
raw_trace="$(ls -1t "artifacts/debug/${PROFILE_BASE}_raw"/trace_*.json | head -1)"

echo "=== Formula diff: raw vs chat (${PROFILE_BASE}) ==="
python scripts/trace_formula_diff.py \
    --golden "$raw_trace" \
    --candidate "$chat_trace" \
    --top 12 \
    --json-out "artifacts/debug/${PROFILE_BASE}_chat/raw_vs_chat_formula.json"