#!/usr/bin/env bash
# rope_ladder_test.sh — needle-in-haystack context ladder for airframe GPU server
# Runs entirely in Git Bash on Windows. No PowerShell required.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
SERVER_EXE="${SERVER_EXE:-./target/release/shimmy_server_gpu.exe}"
RESULTS_OUT="${RESULTS_OUT:-artifacts/rope_ladder_results.json}"
SERVER_START_SEC="${SERVER_START_SEC:-90}"
POLL_SEC="${POLL_SEC:-180}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
ONLY_LABEL="${ONLY_LABEL:-}"

SECRET="XYLOPHONE-7743"
QUESTION="What is the secret code mentioned earlier in this document? Reply with ONLY the code, nothing else."

# Filler sentence ~15 tokens / rep
FILLER_SENTENCE="The mountain range extends across the northern hemisphere, providing habitat for numerous species of birds and mammals. "

mkdir -p artifacts

# ---------------------------------------------------------------------------
kill_server() {
    taskkill //F //IM shimmy_server_gpu.exe >/dev/null 2>&1 || true
    sleep 2
}

start_server() {
    local max_ctx="$1"
    local rope_scale="$2"
    kill_server
    echo "  Starting server: ctx=${max_ctx} scale=${rope_scale}"
    SHIMMY_MAX_CTX="$max_ctx" SHIMMY_ROPE_SCALE="$rope_scale" \
        "$SERVER_EXE" >"artifacts/rope_ladder_server_stdout.txt" 2>"artifacts/rope_ladder_server_stderr.txt" &
    echo $! > artifacts/rope_ladder_server.pid
}

wait_ready() {
    local max_sec="$1"
    for i in $(seq 1 "$max_sec"); do
        if curl -sf "${BASE_URL}/api/repro/queue" >/dev/null 2>&1; then
            echo "  Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    return 1
}

submit_job() {
    local prompt_text="$1"
    # Large prompts exceed Windows command-line length limits if passed via -d.
    local prompt_file
    local body_file
    local job_id
    prompt_file=$(mktemp)
    body_file=$(mktemp)
    printf '%s' "$prompt_text" > "$prompt_file"
    jq -Rs \
        '{task:"needle", prompt:., prompt_mode:"raw", max_tokens:32, temperature:0.0, top_p:1.0, seed:42, stream:false}' \
        < "$prompt_file" > "$body_file"

    if job_id=$(curl -sf -X POST "${BASE_URL}/" \
        -H "Content-Type: application/json" \
        --data-binary "@${body_file}" | jq -r '.job_id'); then
        rm -f "$prompt_file"
        rm -f "$body_file"
        printf '%s' "$job_id"
        return 0
    fi

    rm -f "$prompt_file"
    rm -f "$body_file"
    return 1
}

poll_job() {
    local job_id="$1"
    local max_sec="$2"
    for i in $(seq 1 "$max_sec"); do
        local json
        json=$(curl -sf "${BASE_URL}/api/repro/job-status?job_id=${job_id}" 2>/dev/null || echo '{}')
        local status
        status=$(printf '%s' "$json" | jq -r '.status // ""')
        if [[ "$status" == "completed" || "$status" == "failed" ]]; then
            printf '%s' "$json"
            return 0
        fi
        sleep 1
    done
    echo '{}'
    return 1
}

build_needle_prompt() {
    local filler_tokens="$1"
    # Approximate 4 chars per token for this all-ASCII filler
    local filler_chars=$(( filler_tokens * 4 ))
    local filler_reps=$(( (filler_chars / ${#FILLER_SENTENCE}) + 1 ))
    local filler=""
    for i in $(seq 1 "$filler_reps"); do filler+="$FILLER_SENTENCE"; done
    filler="${filler:0:$filler_chars}"

    printf '%s' "<|system|>
You are a helpful assistant.</s>
<|user|>
This is a long document for testing context retention.

SECRET CODE: ${SECRET}

${filler}

${QUESTION}</s>
<|assistant|>
"
}

# ---------------------------------------------------------------------------
# Ladder: label, max_ctx, rope_scale, filler_tokens
# rope_scale=auto means 2048/max_ctx
declare -a LABELS=(    "2048-baseline" "4096-linear"  "8192-linear"  "16384-linear" "32768-linear" )
declare -a CTX=(       2048            4096           8192           16384          32768          )
declare -a SCALES=(    "1.0"           "auto"         "auto"         "auto"         "auto"         )
declare -a FILLER=(    1600            3600           7500           15000          30000          )

# ---------------------------------------------------------------------------
results_json="["
first_result=true
ladder_broken=false

for idx in "${!LABELS[@]}"; do
    label="${LABELS[$idx]}"
    max_ctx="${CTX[$idx]}"
    scale_raw="${SCALES[$idx]}"
    filler="${FILLER[$idx]}"

    if [[ -n "$ONLY_LABEL" && "$label" != "$ONLY_LABEL" ]]; then
        continue
    fi

    if [[ "$scale_raw" == "auto" ]]; then
        # python/awk float division
        rope_scale=$(awk "BEGIN{printf \"%.6f\", 2048.0/$max_ctx}")
    else
        rope_scale="$scale_raw"
    fi

    echo ""
    echo "=== RUNG: ${label} (ctx=${max_ctx} scale=${rope_scale}) ==="

    start_server "$max_ctx" "$rope_scale"

    if ! wait_ready "$SERVER_START_SEC"; then
        echo "  ERROR: server did not start in ${SERVER_START_SEC}s"
        kill_server
        entry=$(jq -n \
            --arg lbl "$label" --argjson ctx "$max_ctx" \
            --arg sc "$rope_scale" --argjson ft "$filler" \
            '{label:$lbl, max_ctx:$ctx, rope_scale:$sc, filler_target:$ft,
              pass:false, found_keyword:false, response_text:null, error:"server_start_timeout"}')
        [[ "$first_result" == "true" ]] && first_result=false || results_json+=","
        results_json+="$entry"
        ladder_broken=true
        if [[ "$STOP_ON_FAIL" == "1" ]]; then
            break
        fi
        continue
    fi

    # Build prompt and log approximate size
    prompt=$(build_needle_prompt "$filler")
    prompt_chars=${#prompt}
    approx_tokens=$(( prompt_chars / 4 ))
    echo "  Prompt: ~${approx_tokens} tokens  (${prompt_chars} chars)"

    rung_pass=false
    found_keyword=false
    response_text=""
    rung_err=""

    job_id=$(submit_job "$prompt" 2>/dev/null || true)
    if [[ -z "$job_id" ]]; then
        rung_err="submit_failed"
    else
        echo "  Job: $job_id"
        status_json=$(poll_job "$job_id" "$POLL_SEC" || echo '{}')
        job_status=$(printf '%s' "$status_json" | jq -r '.status // ""')
        if [[ "$job_status" == "completed" ]]; then
            response_text=$(printf '%s' "$status_json" | jq -r '.result.text // ""')
            if [[ "$response_text" == *"$SECRET"* ]]; then
                found_keyword=true
                rung_pass=true
            fi
        elif [[ "$job_status" == "failed" ]]; then
            rung_err="job_failed:$(printf '%s' "$status_json" | jq -r '.error // ""')"
        else
            rung_err="poll_timeout_or_empty"
        fi
    fi

    if [[ "$rung_pass" == "true" ]]; then
        echo "  Result: PASS  found=${found_keyword}  response='${response_text}'"
    else
        echo "  Result: FAIL  found=${found_keyword}  err=${rung_err}  response='${response_text}'"
    fi

    entry=$(jq -n \
        --arg lbl "$label" --argjson ctx "$max_ctx" \
        --arg sc "$rope_scale" --argjson ft "$filler" \
        --argjson approx "$approx_tokens" \
        --argjson pass "$([ "$rung_pass" = true ] && echo true || echo false)" \
        --argjson found "$([ "$found_keyword" = true ] && echo true || echo false)" \
        --arg resp "$response_text" \
        --arg err "$rung_err" \
        '{label:$lbl, max_ctx:$ctx, rope_scale:$sc, filler_target:$ft,
          approx_prompt_tokens:$approx, pass:$pass, found_keyword:$found,
          response_text:$resp, error:$err}')

    [[ "$first_result" == "true" ]] && first_result=false || results_json+=","
    results_json+="$entry"

    kill_server

    if [[ "$found_keyword" != "true" ]]; then
        echo "  Retrieval lost at ${label} — stopping ladder."
        ladder_broken=true
        if [[ "$STOP_ON_FAIL" == "1" ]]; then
            break
        fi
    fi
done

results_json+="]"

# ---------------------------------------------------------------------------
echo ""
echo "=== SUMMARY ==="
printf '%s' "$results_json" | jq -r '.[] | "\(if .pass then "[PASS]" else "[FAIL]" end)  \(.label)  ctx=\(.max_ctx)  scale=\(.rope_scale)  filler_tokens=~\(.filler_target)  found=\(.found_keyword)"'

printf '%s' "$results_json" | jq '.' > "$RESULTS_OUT"
echo ""
echo "Results: $RESULTS_OUT"
