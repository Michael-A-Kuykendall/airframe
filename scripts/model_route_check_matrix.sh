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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/artifacts/route_check}"
PROFILE_PREFIX="${PROFILE_PREFIX:-route_check}"
ROUTE_CHECK_STRICT="${ROUTE_CHECK_STRICT:-1}"
ROUTE_CHECK_FAIL_ON_WARN="${ROUTE_CHECK_FAIL_ON_WARN:-1}"
ROUTE_V2_LAYER_PARAMS="${ROUTE_V2_LAYER_PARAMS:-1}"

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$OUTPUT_DIR/route_check_${TIMESTAMP}.log"
CSV_FILE="$OUTPUT_DIR/route_check_${TIMESTAMP}.csv"
TMP_DIR="$(mktemp -d)"
SERVER_PID=""

wait_for_pid_exit() {
    local pid="$1"
    local timeout_s="$2"
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        if (( waited >= timeout_s )); then
            return 1
        fi
        sleep 1
        ((waited+=1))
    done
    return 0
}

terminate_pid() {
    local pid="$1"
    local timeout_s="${2:-10}"
    if [[ -z "$pid" ]]; then
        return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    kill "$pid" 2>/dev/null || true
    if wait_for_pid_exit "$pid" "$timeout_s"; then
        return 0
    fi

    taskkill //F //PID "$pid" >/dev/null 2>&1 || true
    wait_for_pid_exit "$pid" 5 || true
    return 0
}

cleanup() {
    terminate_pid "$SERVER_PID" 10 || true
    taskkill //F //IM shimmy_server_gpu.exe >/dev/null 2>&1 || true
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

log() {
    local line="[$(date +%H:%M:%S)] $*"
    echo "$line" | tee -a "$LOG_FILE"
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
    terminate_pid "$SERVER_PID" 10 || true
    SERVER_PID=""
    taskkill //F //IM shimmy_server_gpu.exe >/dev/null 2>&1 || true
}

start_server() {
    local model_path="$1"
    local profile_name="$2"
    local server_log="$3"

    stop_server

    env -u SHIMMY_FSE_REJECT_PATTERNS \
        -u SHIMMY_FSE_REJECT_PATTERNS_PATH \
        RUST_BACKTRACE=1 \
        SHIMMY_PORT="$SHIMMY_PORT" \
        SHIMMY_ROUTE_CHECK_STRICT="$ROUTE_CHECK_STRICT" \
        SHIMMY_ROUTE_CHECK_FAIL_ON_WARN="$ROUTE_CHECK_FAIL_ON_WARN" \
        SHIMMY_ROUTE_V2_LAYER_PARAMS="$ROUTE_V2_LAYER_PARAMS" \
        SHIMMY_DEBUG_PROFILE="$profile_name" \
        LIBSHIMMY_MODEL_PATH="$model_path" \
        "$SERVER_BIN" > "$server_log" 2>&1 &
    SERVER_PID=$!

    if ! wait_ready; then
        return 1
    fi

    return 0
}

extract_route_json() {
    local server_log="$1"
    local out_json="$2"
    python - "$server_log" "$out_json" <<'PY'
import json
import sys

log_path, out_path = sys.argv[1], sys.argv[2]
route = None
with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        if '[ROUTE_CHECK]' in line:
            payload = line.split('[ROUTE_CHECK]', 1)[1].strip()
            route = json.loads(payload)
            break

if route is None:
    raise SystemExit(1)

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(route, f)
PY
}

append_csv_row() {
    local model="$1"
    local status="$2"
    local route_json="$3"
    local notes="$4"

    python - "$CSV_FILE" "$model" "$status" "$route_json" "$notes" <<'PY'
import csv
import json
import sys

csv_path, model, status, route_json_path, notes = sys.argv[1:6]
route = {}
if route_json_path:
    with open(route_json_path, 'r', encoding='utf-8') as f:
        route = json.load(f)

hard_errors = route.get('hard_errors') or []
warnings = route.get('warnings') or []

row = [
    model,
    status,
    route.get('route_version', ''),
    route.get('route_digest', ''),
    route.get('arch', ''),
    route.get('prompt_renderer_mode', ''),
    route.get('prompt_renderer_family', ''),
    route.get('prompt_template_source', ''),
    route.get('norm_mode', ''),
    route.get('qkv_layout', ''),
    route.get('ffn_mode', ''),
    len(hard_errors),
    len(warnings),
    route.get('strict_mode_pass', False),
    notes,
]

with open(csv_path, 'a', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(row)
PY
}

run_case() {
    local model_file="$1"
    local model_path="$MODEL_DIR/$model_file"
    local profile_name="${PROFILE_PREFIX}_$(echo "$model_file" | tr '/ .' '___')"
    local server_log="$TMP_DIR/${profile_name}.server.log"
    local route_json="$TMP_DIR/${profile_name}.route.json"

    if [[ ! -f "$model_path" ]]; then
        log "SKIP  $model_file (not found)"
        append_csv_row "$model_file" "SKIP" "" "file not found"
        return 0
    fi

    log "START $model_file"

    if ! start_server "$model_path" "$profile_name" "$server_log"; then
        log "FAIL  $model_file -- startup failure"
        tail -40 "$server_log" | tee -a "$LOG_FILE" || true
        append_csv_row "$model_file" "FAIL" "" "startup failure"
        stop_server
        return 0
    fi

    if extract_route_json "$server_log" "$route_json"; then
        local counts
        counts="$(python - "$route_json" <<'PY'
import json
import sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    d = json.load(f)
print(f"hard_errors={len(d.get('hard_errors') or [])} warnings={len(d.get('warnings') or [])} arch={d.get('arch')} prompt_mode={d.get('prompt_renderer_mode')} ffn={d.get('ffn_mode')} qkv={d.get('qkv_layout')}")
PY
)"
        log "PASS  $model_file -- $counts"
        append_csv_row "$model_file" "PASS" "$route_json" ""
    else
        log "FAIL  $model_file -- missing ROUTE_CHECK"
        tail -80 "$server_log" | tee -a "$LOG_FILE" || true
        append_csv_row "$model_file" "FAIL" "" "missing ROUTE_CHECK"
    fi

    stop_server
    sleep 2
}

cat > "$CSV_FILE" <<'CSV'
Model,Status,RouteVersion,RouteDigest,Arch,PromptMode,PromptFamily,TemplateSource,NormMode,QkvLayout,FfnMode,HardErrors,Warnings,StrictPass,Notes
CSV

log '=== Airframe route check matrix ==='
log "Branch target: $(git branch --show-current) @ $(git rev-parse --short HEAD)"
log "Server bin : $SERVER_BIN"
log "Model dir  : $MODEL_DIR"
log "Base URL   : $BASE_URL"
log "Strict flags: SHIMMY_ROUTE_CHECK_STRICT=$ROUTE_CHECK_STRICT SHIMMY_ROUTE_CHECK_FAIL_ON_WARN=$ROUTE_CHECK_FAIL_ON_WARN SHIMMY_ROUTE_V2_LAYER_PARAMS=$ROUTE_V2_LAYER_PARAMS"

models=(
    'TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf'
    'Llama-3.2-1B-Instruct-Q4_K_M.gguf'
    'Llama-3.2-3B-Instruct-Q4_K_M.gguf'
    'phi-2.Q4_K_M.gguf'
    'starcoder2-3b-Q4_K_M.gguf'
    'gpt2.Q4_K_M.gguf'
    'Qwen3-0.6B-Q4_K_M.gguf'
    'gemma-2-2b-it-Q4_K_M.gguf'
)

for model in "${models[@]}"; do
    run_case "$model"
done

pass_count="$(rg -c ',PASS,' "$CSV_FILE" || true)"
fail_count="$(rg -c ',FAIL,' "$CSV_FILE" || true)"
skip_count="$(rg -c ',SKIP,' "$CSV_FILE" || true)"

log '=== Summary ==='
log "PASS: ${pass_count:-0} FAIL: ${fail_count:-0} SKIP: ${skip_count:-0}"
log "Results written to: $CSV_FILE"

if [[ "${fail_count:-0}" -gt 0 ]]; then
    exit 1
fi

exit 0
