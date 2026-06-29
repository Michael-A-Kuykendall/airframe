#!/usr/bin/env bash
# cert_model.sh — test a single GGUF model end-to-end
# Usage: bash scripts/cert_model.sh /path/to/model.gguf
# Exit: 0=PASS  1=FAIL/CRASH  2=OOM

set -euo pipefail

GGUF="${1:-}"
SHIMMY="/c/Users/micha/repos/shimmy/target/debug/shimmy.exe"
MAX_TOKENS=20

[[ -z "$GGUF" ]] && { echo "Usage: $0 <path>" >&2; exit 1; }
[[ ! -f "$GGUF" ]] && { echo "SKIP - not found: $GGUF"; exit 0; }

SIZE_MB=$(( $(wc -c < "$GGUF") / 1048576 ))
LABEL=$(basename "$GGUF" .gguf)

(( SIZE_MB > 2000 )) && { echo "OOM  - ${SIZE_MB}MB > 2GB cap"; exit 2; }

echo "Testing: $LABEL (${SIZE_MB} MB)"

# Kill stale shimmy
taskkill /F /IM shimmy.exe 2>/dev/null || true
sleep 3

# Free port
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
URL="http://127.0.0.1:${PORT}"
echo "  Port: $PORT"

# Start shimmy
export SHIMMY_MAX_CTX=2048
"$SHIMMY" serve --bind "127.0.0.1:${PORT}" --model-path "$GGUF" \
  >/tmp/shimmy_stdout.txt 2>/tmp/shimmy_stderr.txt &
SHIMMY_PID=$!
echo "  PID: $SHIMMY_PID"

cleanup() { kill "$SHIMMY_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Poll /health — 2s initial delay then poll every 1s
sleep 2
READY=0
for i in $(seq 2 90); do
  kill -0 "$SHIMMY_PID" 2>/dev/null || {
    echo "CRASH - died at ${i}s"
    tail -3 /tmp/shimmy_stderr.txt 2>/dev/null
    exit 1
  }
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 1 "$URL/health" 2>/dev/null || echo "000")
  [[ "$CODE" == "200" ]] && { READY=1; echo "  Ready after ${i}s"; break; }
  sleep 1
done

(( READY == 0 )) && { echo "TIMEOUT - not ready after 90s"; exit 1; }

# Get model list via jq
MODELS_JSON=$(curl -s "$URL/v1/models" 2>/dev/null)
echo "  Available: $(echo "$MODELS_JSON" | jq -r '.data[].id' | tr '\n' ' ')"

# Select: exact > contains label > non-phi > first
MODEL=$(echo "$MODELS_JSON" | jq -r --arg l "$LABEL" '.data[].id | select(. == $l)' | head -1)
[[ -z "$MODEL" ]] && MODEL=$(echo "$MODELS_JSON" | jq -r --arg l "$LABEL" '.data[].id | select(contains($l))' | head -1)
[[ -z "$MODEL" ]] && MODEL=$(echo "$MODELS_JSON" | jq -r '.data[].id | select(contains("phi") | not)' | head -1)
[[ -z "$MODEL" ]] && MODEL=$(echo "$MODELS_JSON" | jq -r '.data[0].id')

echo "  Using: $MODEL"

# Inference — write JSON to temp file to avoid shell quoting issues
TMPBODY=$(mktemp)
jq -n --arg m "$MODEL" --argjson t "$MAX_TOKENS" \
  '{"model":$m,"messages":[{"role":"user","content":"hi"}],"stream":false,"max_tokens":$t}' \
  > "$TMPBODY"

RESP=$(curl -s --max-time 120 -X POST "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary "@$TMPBODY" 2>/dev/null)
rm -f "$TMPBODY"

CONTENT=$(echo "$RESP" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")
FINISH=$(echo  "$RESP" | jq -r '.choices[0].finish_reason // "?"'     2>/dev/null || echo "?")

if [[ -z "$CONTENT" ]]; then
  ERR=$(echo "$RESP" | jq -r '.error.message // .error // "no response"' 2>/dev/null || echo "$RESP")
  echo "FAIL - $ERR"
  tail -3 /tmp/shimmy_stderr.txt 2>/dev/null
  exit 1
fi

if (( ${#CONTENT} < 2 )); then
  echo "WEAK - '${CONTENT}' (finish=${FINISH})"
  exit 1
fi

echo "PASS - '${CONTENT}' (finish=${FINISH})"
exit 0
