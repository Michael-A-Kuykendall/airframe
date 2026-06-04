#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SERVER_BIN="${SERVER_BIN:-$REPO_ROOT/target/release/shimmy_server_gpu.exe}"
MODEL_DIR="${MODEL_DIR:-D:/shimmy-test-models/gguf_collection}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
SHIMMY_PORT="${SHIMMY_PORT:-8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/model_smoke}"
PROFILE_PREFIX="${PROFILE_PREFIX:-matrix_smoke}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_GEMMA="${RUN_GEMMA:-1}"
RUN_INT4="${RUN_INT4:-0}"
RUN_LARGE_INT4="${RUN_LARGE_INT4:-0}"
RUN_MATH="${RUN_MATH:-0}"
SSE_CHECK="${SSE_CHECK:-0}"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$OUTPUT_DIR/smoke_${TIMESTAMP}.log"
CSV_FILE="$OUTPUT_DIR/smoke_${TIMESTAMP}.csv"
TMP_DIR="$(mktemp -d)"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    taskkill //F //IM shimmy_server_gpu.exe 2>/dev/null || true
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

log() {
    local line="[$(date +%H:%M:%S)] $*"
    echo "$line" | tee -a "$LOG_FILE"
}

csv_escape() {
    python - "$1" <<'PY'
import csv
import io
import sys

value = sys.argv[1]
buf = io.StringIO()
writer = csv.writer(buf, lineterminator='')
writer.writerow([value])
print(buf.getvalue())
PY
}

append_result() {
    local model="$1"
    local result="$2"
    local detail="$3"
    printf '%s,%s,%s\n' "$(csv_escape "$model")" "$(csv_escape "$result")" "$(csv_escape "$detail")" >> "$CSV_FILE"
}

json_field() {
    local file="$1"
    local path="$2"
    python - "$file" "$path" <<'PY'
import json
import sys

file_path, path = sys.argv[1], sys.argv[2]
with open(file_path, 'r', encoding='utf-8') as handle:
    data = json.load(handle)

value = data
for raw_part in path.split('.'):
    if raw_part.endswith(']'):
        name, index = raw_part[:-1].split('[')
        if name:
            value = value[name]
        value = value[int(index)]
    else:
        value = value[raw_part]

if value is None:
    print('')
elif isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
PY
}

wait_ready() {
    local ready_url="$BASE_URL/api/repro/queue"
    local waited=0
    while (( waited < STARTUP_TIMEOUT )); do
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            return 1
        fi
        if curl -fsS -m 2 "$ready_url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        ((waited+=1))
    done
    return 1
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    taskkill //F //IM shimmy_server_gpu.exe 2>/dev/null || true
}

start_server() {
    local model_path="$1"
    local profile_name="$2"
    local kv_quant="${3:-}"

    stop_server

    local server_log="$TMP_DIR/${profile_name}.server.log"
    if [[ -n "$kv_quant" ]]; then
        env -u SHIMMY_FSE_REJECT_PATTERNS \
            -u SHIMMY_FSE_REJECT_PATTERNS_PATH \
            RUST_BACKTRACE=1 \
            SHIMMY_PORT="$SHIMMY_PORT" \
            SHIMMY_DEBUG_PROFILE="$profile_name" \
            LIBSHIMMY_MODEL_PATH="$model_path" \
            SHIMMY_KV_QUANT="$kv_quant" \
            "$SERVER_BIN" > "$server_log" 2>&1 &
    else
        env -u SHIMMY_FSE_REJECT_PATTERNS \
            -u SHIMMY_FSE_REJECT_PATTERNS_PATH \
            RUST_BACKTRACE=1 \
            SHIMMY_PORT="$SHIMMY_PORT" \
            SHIMMY_DEBUG_PROFILE="$profile_name" \
            LIBSHIMMY_MODEL_PATH="$model_path" \
            "$SERVER_BIN" > "$server_log" 2>&1 &
    fi
    SERVER_PID=$!

    if ! wait_ready; then
        log "FAIL  startup -- $model_path"
        tail -40 "$server_log" | tee -a "$LOG_FILE" || true
        return 1
    fi

    return 0
}

run_completion() {
    local prompt="$1"
    local prompt_mode="$2"
    local max_tokens="$3"
    local temperature="$4"
    local out_json="$5"
    local body

    body="$(python - "$prompt" "$prompt_mode" "$max_tokens" "$temperature" <<'PY'
import json
import sys

prompt, prompt_mode, max_tokens, temperature = sys.argv[1:5]
payload = {
    'model': 'local',
    'prompt': prompt,
    'prompt_mode': prompt_mode,
    'max_tokens': int(max_tokens),
    'temperature': float(temperature),
}
print(json.dumps(payload))
PY
)"

    curl -fsS -m "$REQUEST_TIMEOUT" \
        -X POST "$BASE_URL/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "$body" > "$out_json"
}

run_chat_math() {
    local prompt="$1"
    local out_json="$2"
    local body

    body="$(python - "$prompt" <<'PY'
import json
import sys

payload = {
    'model': 'local',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': 16,
    'temperature': 0.0,
    'stream': False,
}
print(json.dumps(payload))
PY
)"

    curl -fsS -m 30 \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$body" > "$out_json"
}

probe_sse() {
    local out_file="$TMP_DIR/sse.txt"
    local body_file="$TMP_DIR/sse_body.json"

    python - <<'PY' > "$body_file"
import json

payload = {
    'model': 'local',
    'messages': [{'role': 'user', 'content': 'Say: hi'}],
    'max_tokens': 4,
    'temperature': 0.0,
    'stream': True,
}
print(json.dumps(payload))
PY

    if curl -sS -N -m 30 \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "@$body_file" > "$out_file" 2>&1; then
        if rg -q '^data: ' "$out_file"; then
            log "      SSE stream: OK"
        else
            log "      SSE stream: WARNING -- no 'data: ' events received"
        fi
    else
        log "      SSE stream: WARNING -- curl stream probe failed"
    fi
}

run_math_suite() {
    local -a math_prompts=(
        'What is 2 + 2?|4'
        'What is 7 * 8?|56'
        'What is 100 / 4?|25'
        'What is 15 - 7?|8'
        'What is 3 to the power 3?|27'
    )
    local pass_count=0
    local fail_count=0

    log '=== Math interception tests ==='
    for entry in "${math_prompts[@]}"; do
        local prompt="${entry%%|*}"
        local expect="${entry##*|}"
        local out_json="$TMP_DIR/math.json"
        if run_chat_math "$prompt" "$out_json"; then
            local text
            text="$(json_field "$out_json" 'choices[0].message.content')"
            if [[ "$text" == *"$expect"* ]]; then
                log "PASS  math '$prompt' -> '$text'"
                ((pass_count+=1))
            else
                log "FAIL  math '$prompt' -> '$text'"
                ((fail_count+=1))
            fi
        else
            log "FAIL  math '$prompt' -> request error"
            ((fail_count+=1))
        fi
    done
    log "Math: PASS=$pass_count FAIL=$fail_count"
    [[ "$fail_count" -eq 0 ]]
}

run_case() {
    local model_file="$1"
    local expect_word="$2"
    local prompt="$3"
    local prompt_mode="$4"
    local max_tokens="$5"
    local temperature="$6"
    local label="$7"
    local kv_quant="${8:-}"

    local model_path="$MODEL_DIR/$model_file"
    if [[ ! -f "$model_path" ]]; then
        log "SKIP  $label$model_file (not found)"
        append_result "$label$model_file" 'SKIP' 'file not found'
        return 0
    fi

    local profile_name="${PROFILE_PREFIX}_$(echo "$model_file" | tr '/ .' '___')"
    if [[ -n "$kv_quant" ]]; then
        profile_name+="_${kv_quant}"
    fi

    log "START ${label}${model_file}"
    if ! start_server "$model_path" "$profile_name" "$kv_quant"; then
        append_result "$label$model_file" 'FAIL' 'startup failure'
        stop_server
        return 1
    fi

    local out_json="$TMP_DIR/response.json"
    if ! run_completion "$prompt" "$prompt_mode" "$max_tokens" "$temperature" "$out_json"; then
        log "FAIL  ${label}${model_file} -- request error"
        append_result "$label$model_file" 'FAIL' 'request error'
        stop_server
        return 1
    fi

    local text finish_reason
    text="$(json_field "$out_json" 'choices[0].text')"
    finish_reason="$(json_field "$out_json" 'choices[0].finish_reason')"
    if [[ -z "$text" ]]; then
        log "FAIL  ${label}${model_file} -- empty response"
        append_result "$label$model_file" 'FAIL' 'empty response'
        stop_server
        return 1
    fi

    local result='WEAK'
    if [[ -z "$expect_word" || "$text" == *"$expect_word"* ]]; then
        result='PASS'
    fi
    log "$result  ${label}${model_file} -- finish_reason=$finish_reason response: ${text:0:120}"
    append_result "$label$model_file" "$result" "$text"

    if [[ "$SSE_CHECK" == '1' ]]; then
        probe_sse
    fi
    if [[ "$RUN_MATH" == '1' && -z "$kv_quant" ]]; then
        run_math_suite || true
    fi

    stop_server
    sleep 2
    return 0
}

printf 'Model,Result,Detail\n' > "$CSV_FILE"

log '=== Airframe bash model smoke matrix ==='
log "Branch target: $(git branch --show-current) @ $(git rev-parse --short HEAD)"
log "Server bin : $SERVER_BIN"
log "Model dir  : $MODEL_DIR"
log "Base URL   : $BASE_URL"

declare -a BASELINE_MODELS=(
    'TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf|Paris|The capital of France is|raw|32|0.0'
    'Llama-3.2-1B-Instruct-Q4_K_M.gguf|Paris|The capital of France is|raw|32|0.0'
    'Llama-3.2-3B-Instruct-Q4_K_M.gguf|Paris|The capital of France is|raw|32|0.0'
    'phi-2.Q4_K_M.gguf|Paris|The capital of France is|raw|32|0.2'
    'starcoder2-3b-Q4_K_M.gguf|def |def hello_world():|raw|32|0.0'
    'gpt2.Q4_K_M.gguf||The capital of France is|raw|32|0.0'
    'Qwen3-0.6B-Q4_K_M.gguf|Paris|The capital of France is|raw|32|0.0'
)

declare -a GEMMA_MODELS=(
    'gemma-2-2b-it-Q4_K_M.gguf|Paris|The capital of France is|raw|32|0.0'
)

declare -a LARGE_MODELS=(
    'deepseek-llm-7b-chat.Q4_K_M.gguf|Paris|The capital of France is|raw|32|0.0'
    'deepseek-coder-6.7b-instruct.Q4_K_M.gguf|def |def hello_world():|raw|32|0.0'
    'qwen2-7b-instruct-q4_k_m.gguf|Paris|The capital of France is|raw|32|0.0'
)

if [[ "$RUN_BASELINE" == '1' ]]; then
    for entry in "${BASELINE_MODELS[@]}"; do
        IFS='|' read -r model expect prompt prompt_mode max_tokens temperature <<< "$entry"
        run_case "$model" "$expect" "$prompt" "$prompt_mode" "$max_tokens" "$temperature" ''
    done
fi

if [[ "$RUN_GEMMA" == '1' ]]; then
    for entry in "${GEMMA_MODELS[@]}"; do
        IFS='|' read -r model expect prompt prompt_mode max_tokens temperature <<< "$entry"
        run_case "$model" "$expect" "$prompt" "$prompt_mode" "$max_tokens" "$temperature" ''
    done
fi

if [[ "$RUN_INT4" == '1' ]]; then
    for entry in "${BASELINE_MODELS[@]}"; do
        IFS='|' read -r model expect prompt prompt_mode max_tokens temperature <<< "$entry"
        run_case "$model" "$expect" "$prompt" "$prompt_mode" "$max_tokens" "$temperature" '[INT4] ' 'int4'
    done
    for entry in "${GEMMA_MODELS[@]}"; do
        IFS='|' read -r model expect prompt prompt_mode max_tokens temperature <<< "$entry"
        run_case "$model" "$expect" "$prompt" "$prompt_mode" "$max_tokens" "$temperature" '[INT4] ' 'int4'
    done
fi

if [[ "$RUN_LARGE_INT4" == '1' ]]; then
    for entry in "${LARGE_MODELS[@]}"; do
        IFS='|' read -r model expect prompt prompt_mode max_tokens temperature <<< "$entry"
        run_case "$model" "$expect" "$prompt" "$prompt_mode" "$max_tokens" "$temperature" '[INT4] ' 'int4'
    done
fi

pass_count="$(rg -c ',PASS,' "$CSV_FILE" || true)"
weak_count="$(rg -c ',WEAK,' "$CSV_FILE" || true)"
fail_count="$(rg -c ',FAIL,' "$CSV_FILE" || true)"
skip_count="$(rg -c ',SKIP,' "$CSV_FILE" || true)"

log '=== Summary ==='
log "PASS: ${pass_count:-0} WEAK: ${weak_count:-0} FAIL: ${fail_count:-0} SKIP: ${skip_count:-0}"
log "Results written to: $CSV_FILE"

if [[ "${fail_count:-0}" -gt 0 ]]; then
    exit 1
fi

exit 0