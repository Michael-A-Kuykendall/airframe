# Lambda A100 — Shimmy v2.0 Validation Agent Brief

**Instance:** 150.136.92.11 — A100-SXM4-40GB, 40 960 MiB VRAM, 466 GB disk  
**Working dir:** `~/airframe`  
**Branch:** `lambda-vulkan-gpu-fix` (already checked out)  
**Binary:** `~/airframe/target/release/shimmy_server_gpu` (already compiled)  
**Models dir:** `~/models/`  

---

## Current repo state

```
git log --oneline -3
fff71ab feat(gpu): Q4K/Q6K shader support, multi-model pipeline expansion
cbfb9bd fix(gpu): online softmax, chat messages, prompt_tokens
5a73474 fix(vulkan): bypass staging belt for large buffer uploads
```

The server starts and runs inference. Two models are present:
- `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` — **VALIDATED** (answers "2+2=4")
- `gemma-2-2b-it-Q4_K_M.gguf` — Q4K/Q6K pipelines compile and run, but chat template is wrong

---

## Task 1 — Fix Gemma-2 chat template (code patch, ~2 min)

The messages-to-prompt conversion at line ~587 of `src/bin/shimmy_server_gpu.rs` always
uses ChatML (`<|im_start|>`). Gemma-2 needs `<start_of_turn>` format. `spec.arch_string()`
is available in scope at that site.

Find this block (around line 583):

```rust
            if let Some(ref messages) = inference_req.messages {
                    // Build a ChatML prompt from the messages array.
                    let mut parts = String::new();
                    for msg in messages {
                        parts.push_str(&format!(
                            "<|im_start|>{}\n{}<|im_end|>\n",
                            msg.role, msg.content
                        ));
                    }
                    parts.push_str("<|im_start|>assistant\n");
```

Replace with:

```rust
            if let Some(ref messages) = inference_req.messages {
                    let mut parts = String::new();
                    let is_gemma = spec.arch_string().contains("gemma");
                    for msg in messages {
                        if is_gemma {
                            parts.push_str(&format!(
                                "<start_of_turn>{}\n{}<end_of_turn>\n",
                                msg.role, msg.content
                            ));
                        } else {
                            parts.push_str(&format!(
                                "<|im_start|>{}\n{}<|im_end|>\n",
                                msg.role, msg.content
                            ));
                        }
                    }
                    if is_gemma {
                        parts.push_str("<start_of_turn>model\n");
                    } else {
                        parts.push_str("<|im_start|>assistant\n");
                    }
```

Then rebuild:
```bash
source ~/.cargo/env
cd ~/airframe
cargo build --release --bin shimmy_server_gpu 2>&1 | grep -E "^error|Finished|Compiling airframe"
```

---

## Task 2 — Download additional models (~5 min each)

466 GB disk free. Download these for cross-model Q4K validation:

```bash
cd ~/models

# Llama-3.2-1B — primary v2.0 target model, Q4K
wget -q --show-progress "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Llama-3.2-3B — Q4K, 1.9 GB
wget -q --show-progress "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# StarCoder2-3B — code generation, Q4K
wget -q --show-progress "https://huggingface.co/bartowski/starcoder2-3b-GGUF/resolve/main/starcoder2-3b-Q4_K_M.gguf"
```

---

## Task 3 — Per-model smoke test sequence

For each model, run this block (substitute model path and test prompt):

```bash
# Kill any running server
pkill -9 shimmy_server_gpu 2>/dev/null; sleep 2

# Start server
MODEL=/home/ubuntu/models/<MODEL_FILE>.gguf
LIBSHIMMY_MODEL_PATH=$MODEL \
SHIMMY_PORT=8080 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
RUST_LOG=wgpu=error \
nohup ~/airframe/target/release/shimmy_server_gpu > /tmp/server_test.log 2>&1 &

# Wait for ready signal
for i in $(seq 1 60); do
  grep -q "Async listener spawned" /tmp/server_test.log && break
  sleep 2
done
grep -E "ModelSpec|use_q4k|Output head|listener" /tmp/server_test.log

# Submit inference
RESP=$(curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"What is 2+2? Answer in one line."}],"max_tokens":24,"temperature":0.0}')
JOB_ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job: $JOB_ID"

# Poll result
for i in $(seq 1 24); do
  sleep 5
  S=$(curl -s "http://127.0.0.1:8080/api/repro/job-status?job_id=$JOB_ID")
  STATE=$(echo "$S" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
  echo "[$i] $STATE"
  if [ "$STATE" = "completed" ] || [ "$STATE" = "failed" ]; then
    echo "$S" | python3 -m json.tool
    break
  fi
done
```

### Models to test in order:

| # | Model file | Expected | Notes |
|---|-----------|----------|-------|
| 1 | `gemma-2-2b-it-Q4_K_M.gguf` | `"2 + 2 = 4"` or equivalent | After template fix; Q6K output head path |
| 2 | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | `"2 + 2 = 4"` | Primary v2.0 target; Q4K |
| 3 | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | `"2 + 2 = 4"` | Bigger Llama, Q4K |
| 4 | `starcoder2-3b-Q4_K_M.gguf` | Code-style completion | Completion model, use `prompt` field not `messages` |
| 5 | `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | `"2 + 2 = 4"` | Regression baseline; already validated |

### Pass criteria for each model:
- Server starts without panic
- `[HTTP] Async listener spawned` appears in log
- Job reaches `status: completed`
- `result.text` contains a correct or coherent answer
- `prompt_tokens` and `completion_tokens` (or `tokens_generated`) are non-zero in result

---

## Task 4 — Full OpenAI API schema check (on any working model)

The v2.0 release requires valid OpenAI-compat response structure. Run this and confirm all fields:

```bash
# With server running
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Say: hello"}],"max_tokens":8,"temperature":0.0}' \
  | python3 -m json.tool
```

The queued response must have: `job_id`

After polling to completion, the job-status result must have:
- `result.text` — non-empty string
- `result.prompt_tokens` — integer > 0
- `result.tokens_generated` or `completion_tokens` — integer > 0
- `result.stop_reason` — `"eos"` or `"max_tokens"`

---

## Task 5 — quant_verify on TinyLlama (CPU vs GPU dequant agreement)

This only works on models ≤ 2 GB due to a buffer binding limit on smaller GPUs. A100 can handle it.

```bash
source ~/.cargo/env
cd ~/airframe
LIBSHIMMY_MODEL_PATH=/home/ubuntu/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
cargo run --release --bin quant_verify 2>&1 | tail -20
```

**Pass:** all tensor types print `OK`, no `MISMATCH`.

Also run on Llama-3.2-1B (771 MB) once downloaded:

```bash
LIBSHIMMY_MODEL_PATH=/home/ubuntu/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
cargo run --release --bin quant_verify 2>&1 | tail -20
```

---

## Task 6 — Multi-turn chat test (Llama-3.2-1B or Gemma-2)

Tests context accumulation across turns — required for v2.0 interactive use:

```bash
# With Llama-3.2-1B server running
# Turn 1
JOB1=$(curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"My name is Alice. Remember it."}],"max_tokens":16,"temperature":0.0}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

sleep 30
curl -s "http://127.0.0.1:8080/api/repro/job-status?job_id=$JOB1" | python3 -m json.tool

# Turn 2
JOB2=$(curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"My name is Alice. Remember it."},{"role":"assistant","content":"Hello Alice, I will remember your name."},{"role":"user","content":"What is my name?"}],"max_tokens":16,"temperature":0.0}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

sleep 30
curl -s "http://127.0.0.1:8080/api/repro/job-status?job_id=$JOB2" | python3 -m json.tool
```

**Pass:** Turn 2 response contains "Alice".

---

## Task 7 — Commit results and push to the lambda remote

After all tests:

```bash
cd ~/airframe
# Record results in a quick note
cat >> /tmp/v2_validation_results.txt << 'EOF'
# Shimmy v2.0 Lambda Validation Results
# Date: $(date)
# Instance: A100-SXM4-40GB
EOF

git add -A
git commit -m "test(v2): Lambda A100 validation results — Gemma-2 template fix + multi-model smoke

Models tested: [fill in which passed/failed]
quant_verify: [pass/fail]
Multi-turn: [pass/fail]"
```

The local Windows machine will pull this branch via the lambda remote.

---

## Priority order if time is short

1. **Task 1** — Gemma-2 template fix + retest (most important new result)
2. **Task 3, model #2** — Llama-3.2-1B (primary v2.0 model, first Q4K Llama test)
3. **Task 5** — quant_verify TinyLlama (Vulkan CPU/GPU agreement)
4. **Task 4** — API schema check
5. **Task 3, model #3** — Llama-3.2-3B
6. **Task 6** — Multi-turn
7. **Task 3, model #4** — StarCoder2

---

*Shut down the instance after Task 7 commit is done.*
