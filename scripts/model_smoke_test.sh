#!/usr/bin/env bash
# Airframe model smoke test
#
# Usage:
#   bash scripts/model_smoke_test.sh [--include-large] [--test-int4] [--test-math]
#
# Override defaults with env vars:
#   MODEL_DIR        directory containing .gguf files
#   BASE_URL         server base URL
#   SERVER_BIN       path to shimmy_server_gpu binary
#   SHIMMY_PORT      port the server listens on (default 8080)
#   STARTUP_TIMEOUT  seconds to wait for server ready (default 180)
#   REQUEST_TIMEOUT  seconds for each inference call (default 180)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Path helpers ──────────────────────────────────────────────────────────────
# Git Bash on Windows uses /d/foo paths internally.
# Windows .exe binaries need D:\foo paths via env vars.

to_bash_path() {
    # D:\foo\bar  or  D:/foo/bar  ->  /d/foo/bar
    local p="${1//\\//}"
    if [[ "$p" =~ ^([a-zA-Z]):(/.*)$ ]]; then
        echo "/${BASH_REMATCH[1],,}${BASH_REMATCH[2]}"
    else
        echo "$p"
    fi
}

to_win_path() {
    # /d/foo/bar  ->  D:\foo\bar  (passed to Windows .exe via env)
    if command -v cygpath > /dev/null 2>&1; then
        cygpath -w "$1"
    else
        local p="$1"
        if [[ "$p" =~ ^/([a-zA-Z])/(.*)$ ]]; then
            echo "${BASH_REMATCH[1]^^}:\\${BASH_REMATCH[2]//\//\\}"
        else
            echo "$p"
        fi
    fi
}

# ── Defaults ──────────────────────────────────────────────────────────────────
SERVER_BIN="${SERVER_BIN:-$REPO_ROOT/target/release/shimmy_server_gpu.exe}"
MODEL_DIR="$(to_bash_path "${MODEL_DIR:-D:/shimmy-test-models/gguf_collection}")"
BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
OUTPUT_DIR="$(to_bash_path "${OUTPUT_DIR:-$REPO_ROOT/artifacts/model_smoke}")"
SHIMMY_PORT="${SHIMMY_PORT:-8080}"
INCLUDE_LARGE=0
TEST_INT4=0
TEST_MATH=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)       MODEL_DIR="$(to_bash_path "$2")"; shift 2 ;;
        --base-url)        BASE_URL="$2";                    shift 2 ;;
        --startup-timeout) STARTUP_TIMEOUT="$2";             shift 2 ;;
        --request-timeout) REQUEST_TIMEOUT="$2";             shift 2 ;;
        --include-large)   INCLUDE_LARGE=1;                  shift   ;;
        --test-int4)       TEST_INT4=1;                      shift   ;;
        --test-math)       TEST_MATH=1;                      shift   ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done

# ── Output setup ──────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_DIR/smoke_${TIMESTAMP}.log"
CSV_FILE="$OUTPUT_DIR/smoke_${TIMESTAMP}.csv"

printf 'Model,Result,Detail\n' > "$CSV_FILE"

PASS_COUNT=0; FAIL_COUNT=0; WEAK_COUNT=0; SKIP_COUNT=0; LIMIT_COUNT=0
CSV_ROWS=()

log() {
    local line="[$(date '+%H:%M:%S')] $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$LOG_FILE"
}

record() {
    local model="$1" result="$2" detail="${3:-}"
    CSV_ROWS+=("\"${model//\"/\"\"}\",\"$result\",\"${detail//\"/\"\"}\"")
    case "$result" in
        PASS)  PASS_COUNT=$((PASS_COUNT+1))   ;;
        FAIL)  FAIL_COUNT=$((FAIL_COUNT+1))   ;;
        WEAK)  WEAK_COUNT=$((WEAK_COUNT+1))   ;;
        SKIP)  SKIP_COUNT=$((SKIP_COUNT+1))   ;;
        LIMIT) LIMIT_COUNT=$((LIMIT_COUNT+1)) ;;
    esac
}

# ── Server lifecycle ──────────────────────────────────────────────────────────
SERVER_PID=""
SERVER_LOG=""

start_server() {
    local model_path_bash="$1"
    local kv_quant="${2:-}"
    local win_path
    win_path="$(to_win_path "$model_path_bash")"
    SERVER_LOG="$(mktemp)"
    LIBSHIMMY_MODEL_PATH="$win_path" \
    SHIMMY_PORT="$SHIMMY_PORT" \
    SHIMMY_KV_QUANT="$kv_quant" \
        "$SERVER_BIN" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
    if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
        rm -f "$SERVER_LOG"
        SERVER_LOG=""
    fi
}

server_alive() {
    [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null
}

wait_server_ready() {
    local ready_url="$BASE_URL/api/repro/queue"
    local i
    for (( i=0; i<STARTUP_TIMEOUT; i++ )); do
        if ! server_alive; then return 1; fi
        if curl -sf -m 2 "$ready_url" > /dev/null 2>&1; then return 0; fi
        sleep 1
    done
    return 1
}

trap 'stop_server' EXIT INT TERM

# ── HTTP helpers ──────────────────────────────────────────────────────────────
chat_request() {
    # Sends a non-streaming chat completion. Prints raw JSON or empty on error.
    local prompt="$1"
    local max_tokens="${2:-32}"
    local body
    body="$(jq -cn --arg p "$prompt" --argjson n "$max_tokens" \
        '{model:"local",messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0.0,stream:false}')"
    curl -sf -m "$REQUEST_TIMEOUT" \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" 2>/dev/null || true
}

extract_content() {
    # Read JSON on stdin, return .choices[0].message.content
    jq -r '.choices[0].message.content // empty' 2>/dev/null || true
}

check_models_endpoint() {
    local count
    count="$(curl -sf -m 5 "$BASE_URL/v1/models" 2>/dev/null \
        | jq '.data | length' 2>/dev/null)" || count=0
    [[ "${count:-0}" -gt 0 ]]
}

check_sse() {
    local body out
    body="$(jq -cn '{model:"local",messages:[{role:"user",content:"Say: hi"}],max_tokens:4,temperature:0.0,stream:true}')"
    out="$(curl -sf -N -m 15 \
        -X POST "$BASE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" 2>&1)" || out=""
    if printf '%s\n' "$out" | grep -q '^data: '; then
        log "      SSE stream: OK"
    else
        log "      SSE stream: WARNING -- no 'data: ' events received"
    fi
}

# ── Math bypass tests ─────────────────────────────────────────────────────────
# Each entry: "question|expected_numeric_answer"
MATH_TESTS=(
    "What is 2 + 2?|4"
    "What is 7 * 8?|56"
    "What is 100 / 4?|25"
    "What is 15 - 7?|8"
    "What is 3 to the power 3?|27"
    "What is 17 * 13?|221"
    "What is 144 / 12?|12"
    "What is 99 - 37?|62"
    "Calculate 8 + 8|16"
    "What is 2 * 2 * 2?|8"
)

math_answer_correct() {
    # Only check the part of the response that comes after the last '='.
    # This prevents a false PASS when the expected number appears in the
    # echoed question rather than in the computed answer.
    # Example: "144 / 12 = 1.33" must NOT match expected "12".
    local text="$1" expect="$2" region
    if printf '%s' "$text" | grep -q '='; then
        region="$(printf '%s' "$text" | sed 's/.*=\s*//')"
    else
        region="$text"
    fi
    # Whole-word match: "8" must not match inside "56" or "28"
    printf '%s' "$region" | grep -qw "$expect"
}

run_math_tests() {
    log ""
    log "=== Math interception tests ==="
    local math_pass=0 math_fail=0
    local entry prompt expect t_start t_ms resp text tag note short
    for entry in "${MATH_TESTS[@]}"; do
        prompt="${entry%%|*}"
        expect="${entry##*|}"
        t_start="$(date +%s%3N 2>/dev/null || echo "$(( $(date +%s) * 1000 ))")"
        resp="$(chat_request "$prompt" 16)"
        t_ms="$(( $(date +%s%3N 2>/dev/null || echo "$(( $(date +%s) * 1000 ))") - t_start ))"
        text="$(printf '%s' "$resp" | extract_content)"
        if math_answer_correct "$text" "$expect"; then
            tag="PASS"; math_pass=$((math_pass+1))
        else
            tag="FAIL"; math_fail=$((math_fail+1))
        fi
        if (( t_ms < 3000 )); then note="fast=${t_ms}ms (bypassed)"
        else                       note="slow=${t_ms}ms (model path?)"; fi
        short="${text:0:50}"
        log "$tag  math '$prompt' -> '$short' [$note]"
    done
    log "Math: PASS=${math_pass}  FAIL=${math_fail}  Total=${#MATH_TESTS[@]}"
    return $(( math_fail > 0 ? 1 : 0 ))
}

# ── Core per-model test ───────────────────────────────────────────────────────
run_model_test() {
    local modelfile="$1"
    local expect="$2"      # keyword the response must contain; empty = any non-empty response
    local prompt="$3"
    local label="${4:-}"   # "INT4" when running the KV quant pass, empty otherwise
    local kv_quant="${5:-}"
    local display="${label:+[${label}] }${modelfile}"
    local model_path="$MODEL_DIR/$modelfile"

    if [[ ! -f "$model_path" ]]; then
        log "SKIP  $display (not found at $model_path)"
        record "$display" "SKIP" "file not found"
        return
    fi

    log "START $display"
    start_server "$model_path" "$kv_quant"

    if ! wait_server_ready; then
        local detail
        if server_alive; then
            detail="startup timeout after ${STARTUP_TIMEOUT}s"
        else
            detail="server process exited -- last lines of server log follow"
            log "FAIL  $display -- $detail"
            if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]]; then
                tail -20 "$SERVER_LOG" | while IFS= read -r line; do
                    log "      $line"
                done
            fi
            record "$display" "FAIL" "$detail"
            stop_server
            return
        fi
        log "FAIL  $display -- $detail"
        record "$display" "FAIL" "$detail"
        stop_server
        return
    fi

    if check_models_endpoint; then
        log "      /v1/models: OK"
    else
        log "      /v1/models: WARNING -- endpoint missing or empty"
    fi

    local resp text
    resp="$(chat_request "$prompt" 32)"
    text="$(printf '%s' "$resp" | extract_content)"

    if [[ -z "$text" ]]; then
        log "FAIL  $display -- empty response"
        record "$display" "FAIL" "empty response"
        stop_server; sleep 2
        return
    fi

    local result
    if [[ -z "$expect" ]] || printf '%s' "$text" | grep -qF "$expect"; then
        result="PASS"
    else
        result="WEAK"
    fi
    log "$result  $display -- response: ${text:0:80}"
    record "$display" "$result" "$text"

    check_sse

    # Math bypass tests run inline while the server is up.
    # Only run on the primary pass (not the INT4 re-run) to avoid doubling runtime.
    if [[ "$TEST_MATH" -eq 1 && -z "$label" ]]; then
        run_math_tests || true
    fi

    stop_server
    sleep 2
}

# ── Model lists ───────────────────────────────────────────────────────────────
# Format: "filename|expected_keyword|prompt"
# expected_keyword: a word/phrase the model response must contain to count as PASS.
#   Leave empty (two pipes "||") only when the model has no reliable completion
#   style (e.g. gpt2 base model -- we just verify it produces any non-empty output).

VERIFIED_MODELS=(
    "TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf|Paris|The capital of France is"
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf|Paris|The capital of France is"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf|Paris|The capital of France is"
    "phi-2.Q4_K_M.gguf|Paris|The capital of France is"
    "starcoder2-3b-Q4_K_M.gguf|def |def hello_world():"
    "gpt2.Q4_K_M.gguf||The capital of France is"
    "Qwen3-0.6B-Q4_K_M.gguf|Paris|The capital of France is"
)

# gemma-2 has an output weight tensor just over 2 GB.
# This model confirms the large-tensor split fix works end-to-end.
LARGE_TENSOR_MODELS=(
    "gemma-2-2b-it-Q4_K_M.gguf|Paris|The capital of France is"
)

# 7B+ models -- only run with --include-large (needs ~4-6 GB VRAM headroom)
BIG_MODELS=(
    "deepseek-llm-7b-chat.Q4_K_M.gguf|Paris|The capital of France is"
    "deepseek-coder-6.7b-instruct.Q4_K_M.gguf|def |def hello_world():"
    "qwen2-7b-instruct-q4_k_m.gguf|Paris|The capital of France is"
)

# Models we deliberately skip -- architecture not supported yet, or wrong branch.
# Recorded as LIMIT in the results without starting a server.
# Remove an entry only when the blocking issue is fixed.
KNOWN_LIMIT_MODELS=(
    "Phi-3.5-mini-instruct.Q4_K_M.gguf|Fused QKV not yet supported"
    "phi3-mini-4k-instruct-q4.gguf|Fused QKV not yet supported"
    "LFM2.5-VL-1.6B/LFM2.5-VL-1.6B-Q4_0.gguf|Vision model -- deferred to vision branch"
    "minicpm-v-2.6/ggml-model-Q4_K_M.gguf|Vision model -- deferred to vision branch"
)

# ── Main ──────────────────────────────────────────────────────────────────────
log "=== Airframe model smoke test ==="
log "Models dir : $MODEL_DIR"
log "Server bin : $SERVER_BIN"
log "Base URL   : $BASE_URL"
log ""

# Record known-limit models up front -- no server start needed
for entry in "${KNOWN_LIMIT_MODELS[@]}"; do
    mf="${entry%%|*}"
    reason="${entry##*|}"
    if [[ -f "$MODEL_DIR/$mf" ]]; then
        log "LIMIT $mf -- $reason"
        record "$mf" "LIMIT" "KNOWN LIMIT: $reason"
    else
        log "LIMIT $mf -- $reason (not on disk)"
        record "$mf" "LIMIT" "KNOWN LIMIT (not present): $reason"
    fi
done
[[ ${#KNOWN_LIMIT_MODELS[@]} -gt 0 ]] && log ""

# Build the test list for this run
TEST_MODELS=( "${VERIFIED_MODELS[@]}" "${LARGE_TENSOR_MODELS[@]}" )
(( INCLUDE_LARGE )) && TEST_MODELS+=( "${BIG_MODELS[@]}" )

# Primary pass
for entry in "${TEST_MODELS[@]}"; do
    mf="${entry%%|*}"
    rest="${entry#*|}"
    kw="${rest%%|*}"
    prompt="${rest#*|}"
    run_model_test "$mf" "$kw" "$prompt"
done

# INT4 KV pass -- same models again with SHIMMY_KV_QUANT=int4
if (( TEST_INT4 )); then
    log ""
    log "=== INT4 KV pass (SHIMMY_KV_QUANT=int4) ==="
    int4_models=( "${VERIFIED_MODELS[@]}" "${LARGE_TENSOR_MODELS[@]}" )
    for entry in "${int4_models[@]}"; do
        mf="${entry%%|*}"
        rest="${entry#*|}"
        kw="${rest%%|*}"
        prompt="${rest#*|}"
        run_model_test "$mf" "$kw" "$prompt" "INT4" "int4"
    done
fi

# ── Summary ───────────────────────────────────────────────────────────────────
log ""
log "=== Summary ==="
total=$(( PASS_COUNT + WEAK_COUNT + FAIL_COUNT + SKIP_COUNT + LIMIT_COUNT ))
log "PASS: $PASS_COUNT  WEAK: $WEAK_COUNT  FAIL: $FAIL_COUNT  SKIP: $SKIP_COUNT  LIMIT: $LIMIT_COUNT  Total: $total"
(( TEST_INT4 )) && log "  (INT4 KV pass counts are included above)"
(( TEST_MATH )) && log "  (math bypass results are logged inline above)"

for row in "${CSV_ROWS[@]}"; do
    printf '%s\n' "$row" >> "$CSV_FILE"
done
log "Results : $CSV_FILE"
log "Log     : $LOG_FILE"

if (( FAIL_COUNT > 0 )); then
    log "SMOKE TEST: FAILED ($FAIL_COUNT failures)"
    exit 1
fi
log "SMOKE TEST: PASSED"
exit 0
