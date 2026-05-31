#!/usr/bin/env bash
# battery_test.sh — 4-question scored run across every supported GGUF
# Runs entirely in the foreground. Watch it live.

set -euo pipefail

BASE="/d/shimmy-test-models/gguf_collection"
BIN="/c/Users/micha/repos/airframe/target/release/shimmy_server_gpu.exe"
PORT=8099

# Kill any leftover server on our port
taskkill //F //IM shimmy_server_gpu.exe 2>/dev/null || true
sleep 1

declare -A SCORES

# ── helpers ──────────────────────────────────────────────────────────────────

wait_ready() {
    for i in $(seq 1 30); do
        curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
        sleep 2
        echo "    waiting... (${i})"
    done
    return 1
}

ask() {
    local PROMPT="$1" EXPECT="$2" TAG="$3"
    local SUBMIT JID ANSWER STATUS

    # Server is synchronous: blocks until generation done. Use 120s timeout.
    SUBMIT=$(curl -sf -m 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":10}")

    ANSWER=$(echo "$SUBMIT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['choices'][0]['message']['content'].strip())
except Exception:
    print('')
" 2>/dev/null)

    local GOT
    GOT=$(echo "$ANSWER" | tr '\n' ' ' | head -c 80)

    if echo "$GOT" | grep -qi "$EXPECT"; then
        printf "      %-6s ✅  \"%s\"\n" "[$TAG]" "$GOT"
        echo 1
    else
        printf "      %-6s ❌  expected \"%s\" — got \"%s\"\n" "[$TAG]" "$EXPECT" "$GOT"
        echo 0
    fi
}

battery() {
    local TOTAL=0 R
    R=$(ask "What is 7 plus 8? Reply with the number only."             "15"    "math")
    TOTAL=$((TOTAL + $(echo "$R" | tail -1)))
    R=$(ask "What is the capital of France? Reply with one word only."  "Paris" "fact")
    TOTAL=$((TOTAL + $(echo "$R" | tail -1)))
    R=$(ask "Repeat this word exactly and nothing else: PING"           "PING"  "echo")
    TOTAL=$((TOTAL + $(echo "$R" | tail -1)))
    R=$(ask "Is fire hot or cold? Reply with one word only."            "hot"   "logic")
    TOTAL=$((TOTAL + $(echo "$R" | tail -1)))
    echo "$TOTAL"
}

test_model() {
    local GGUF="$1" LABEL="$2"

    echo ""
    echo "════════════════════════════════════════════════"
    printf "  MODEL: %s\n" "$LABEL"
    echo "════════════════════════════════════════════════"

    echo "  [1/3] Starting server..."
    LIBSHIMMY_MODEL_PATH="$GGUF" SHIMMY_PORT=$PORT SHIMMY_MAX_CTX=2048 "$BIN" &
    SRV_PID=$!
    echo "        PID=$SRV_PID  port=$PORT"

    echo "  [2/3] Waiting for server ready..."
    if ! wait_ready; then
        echo "  ⚠️  TIMEOUT — server did not become ready"
        kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null || true
        SCORES["$LABEL"]="TIMEOUT"
        return
    fi

    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "  ⚠️  Server crashed during startup"
        SCORES["$LABEL"]="CRASH"
        return
    fi

    echo "  [3/3] Running 4-question battery..."
    local TOTAL
    TOTAL=$(battery)
    TOTAL=$(echo "$TOTAL" | tail -1)

    echo ""
    printf "  RESULT: %s/4\n" "$TOTAL"
    SCORES["$LABEL"]="$TOTAL/4"

    echo "  Stopping server..."
    kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null || true
    sleep 2
}

# ── model list ───────────────────────────────────────────────────────────────

test_model "$BASE/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"        "TinyLlama-1.1B    Q4_0"
test_model "$BASE/Llama-3.2-1B-Instruct-Q4_K_M.gguf"         "Llama-3.2-1B      Q4_K_M"
test_model "$BASE/Llama-3.2-3B-Instruct-Q4_K_M.gguf"         "Llama-3.2-3B      Q4_K_M"
test_model "$BASE/gemma-2-2b-it-Q4_K_M.gguf"                 "Gemma-2-2B        Q4_K_M"
test_model "$BASE/phi-2.Q4_K_M.gguf"                         "Phi-2             Q4_K_M"
test_model "$BASE/phi3-mini-4k-instruct-q4.gguf"             "Phi-3-Mini-4K     Q4"
test_model "$BASE/Phi-3.5-mini-instruct.Q4_K_M.gguf"         "Phi-3.5-Mini      Q4_K_M"
test_model "$BASE/starcoder2-3b-Q4_K_M.gguf"                 "StarCoder2-3B     Q4_K_M"
test_model "$BASE/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"  "DeepSeek-Coder-6.7B Q4_K_M"
test_model "$BASE/deepseek-llm-7b-chat.Q4_K_M.gguf"          "DeepSeek-LLM-7B   Q4_K_M"
test_model "$BASE/qwen2-7b-instruct-q4_k_m.gguf"             "Qwen2-7B          Q4_K_M"

# ── final table ──────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════"
echo "  FINAL BATTERY RESULTS  (math / fact / echo / logic)"
echo "════════════════════════════════════════════════"
for m in "${!SCORES[@]}"; do
    printf "  %-38s %s\n" "$m" "${SCORES[$m]}"
done | sort
echo "  GPT-2 Q4_K_M                           SKIP (different tensor layout)"
echo "════════════════════════════════════════════════"
echo "  Done."
