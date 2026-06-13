# Airframe A100 Test Plan — NVIDIA A100 80 GB SXM
<!-- Deferred test regimen for high-VRAM hardware. -->
<!-- Run AFTER the local RTX 3060 plan is fully signed off. -->
<!-- These tests cannot run locally due to VRAM constraints or require -->
<!-- longer sustained inference that would cause TDR on consumer hardware. -->
<!-- Last updated: 2026-06-03 -->

---

## Why A100

| Constraint | RTX 3060 12 GB | A100 80 GB |
|---|---|---|
| 7B model F32 KV at ctx 4096 | OOM | ✅ |
| 13B model at any KV | OOM | ✅ |
| 70B model Q4_K_M | OOM | ✅ |
| Multi-model soak (run 3 sequentially, keep logs) | Risk of TDR | ✅ |
| Long-context needle bench (8K+ tokens) | Risk of TDR | ✅ |
| gemma-2-9b+ | OOM | ✅ |
| Mistral-7B / Mixtral (if added later) | Tight | ✅ |

**Assumption:** The A100 machine runs the same `shimmy_server_gpu` binary built on that host with `cargo build --release --bin shimmy_server_gpu`.  No architecture-specific changes are expected — the wgpu backend adapts to the device at runtime.

---

## Models Needed on A100 (not on local machine)

### Already on local machine — copy or re-download

| File | Size | Notes |
|---|---|---|
| `TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf` | 608 MB | Baseline regression — bring for comparison |
| `gemma-2-2b-it-Q4_K_M.gguf` | 1629 MB | Blob fix regression — re-verify on A100 |
| `deepseek-llm-7b-chat.Q4_K_M.gguf` | 4028 MB | 7B F32 KV full ctx test |
| `deepseek-coder-6.7b-instruct.Q4_K_M.gguf` | 3893 MB | 7B coding model F32 full ctx |
| `qwen2-7b-instruct-q4_k_m.gguf` | 4466 MB | 7B Qwen2 F32 full ctx |

### Download before A100 session

| Priority | File | Source (HuggingFace) | Size (est.) | Reason |
|---|---|---|---|---|
| P0 | `gemma-2-9b-it-Q4_K_M.gguf` | `bartowski/gemma-2-9b-it-GGUF` | ~6 GB | Next Gemma-2 blob tier above 2B |
| P0 | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | ~5 GB | Llama-3.1 (not 3.2) arch coverage |
| P0 | `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` | `TheBloke/Mistral-7B-Instruct-v0.3-GGUF` | ~4.4 GB | Mistral GQA heads — different KV shape |
| P1 | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | `Qwen/Qwen2.5-7B-Instruct-GGUF` | ~4.7 GB | Qwen2.5 vs Qwen2 arch delta |
| P1 | `Llama-3.1-70B-Instruct-Q4_K_M.gguf` | `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF` | ~41 GB | 70B — requires A100 |
| P2 | `deepseek-llm-67b-chat.Q4_K_M.gguf` | `TheBloke/deepseek-llm-67b-chat-GGUF` | ~40 GB | 67B DeepSeek — requires A100 |
| P2 | `gemma-2-27b-it-Q4_K_M.gguf` | `bartowski/gemma-2-27b-it-GGUF` | ~17 GB | 27B Gemma-2 — requires A100 |

---

## Pre-flight Checklist (A100 Host)

- [ ] Cargo / Rust toolchain installed on A100 host
- [ ] wgpu 0.20+ compiles on A100 driver (Vulkan or CUDA via wgpu?)  
  **Note:** wgpu on Linux typically uses Vulkan. Confirm `WGPU_BACKEND=vulkan` or let wgpu auto-select.
- [ ] Binary builds clean:
  ```bash
  cargo build --release --bin shimmy_server_gpu 2>&1 | tail -5
  ```
- [ ] All P0 models downloaded to a local fast-NVMe path (not network mount)
- [ ] `nvidia-smi` shows 80 GB available before each phase
- [ ] Port 8086 free on the A100 host
- [ ] `scripts/model_smoke_test.ps1` available — or port to bash if host is Linux-only (see §Linux Porting below)

---

## Phase A1 — Regression Baseline (same models as local Phase 1)

**Goal:** Prove the binary produces identical results on A100 as on RTX 3060.  
If this fails, there is an architecture-specific bug in the wgpu backend.

```powershell
$env:SHIMMY_PORT    = "8086"
$env:SHIMMY_MAX_CTX = "4096"
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086"
```

- [ ] TinyLlama-1.1B: PASS
- [ ] Llama-3.2-1B: PASS
- [ ] Llama-3.2-3B: PASS
- [ ] phi-2: PASS
- [ ] starcoder2-3b: PASS
- [ ] gpt2: PASS
- [ ] Qwen3-0.6B: PASS
- [ ] gemma-2-2b: PASS (blob fix)

---

## Phase A2 — 7B Models, F32 KV, Full Context

**Goal:** 7B models that were INT4-only on local hardware, now run with F32 KV and full ctx 4096.  
This is the primary new capability test on A100.

```powershell
$env:SHIMMY_PORT    = "8086"
$env:SHIMMY_MAX_CTX = "4096"   # full context, no INT4 constraint
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086" -IncludeLarge
```

- [ ] deepseek-llm-7b F32 ctx4096: PASS
- [ ] deepseek-coder-6.7b F32 ctx4096: PASS
- [ ] qwen2-7b F32 ctx4096: PASS
- [ ] minicpm-v-2.6 (text path) F32 ctx4096: PASS

**Bonus — 8K context (if binary supports it):**

```powershell
$env:SHIMMY_MAX_CTX = "8192"
# re-run IncludeLarge
```

- [ ] deepseek-llm-7b F32 ctx8192: PASS / OOM / TDR  ← record actual result

---

## Phase A3 — New Model Tier (9B–13B)

**Goal:** First ever test of models too large for RTX 3060.

| Model | Test prompt | Expected keyword | ctx |
|---|---|---|---|
| `gemma-2-9b-it-Q4_K_M.gguf` | "The capital of France is" | "Paris" | 2048 |
| `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | "The capital of France is" | "Paris" | 4096 |
| `Mistral-7B-Instruct-v0.3.Q4_K_M.gguf` | "The capital of France is" | "Paris" | 4096 |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | "The capital of France is" | "Paris" | 4096 |

Run each manually:

```bash
export SHIMMY_PORT=8086
export SHIMMY_MAX_CTX=4096
export LIBSHIMMY_MODEL_PATH=/path/to/<model>.gguf
./target/release/shimmy_server_gpu &
# wait for ready
curl -s -X POST http://127.0.0.1:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":20}' \
  | jq .choices[0].message.content
kill %1
```

- [ ] gemma-2-9b: PASS / FAIL / OOM ← record
- [ ] Meta-Llama-3.1-8B: PASS / FAIL / OOM ← record
- [ ] Mistral-7B-Instruct-v0.3: PASS / FAIL / OOM ← record
- [ ] Qwen2.5-7B: PASS / FAIL / OOM ← record

---

## Phase A4 — INT4 KV Cross-test on 7B+

**Goal:** INT4 KV (TurboShimmy) on models that needed it locally — verify no regression at full A100 VRAM headroom.

```powershell
$env:SHIMMY_KV_QUANT = "int4"
$env:SHIMMY_MAX_CTX  = "4096"
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086" -IncludeLarge -TestInt4
Remove-Item Env:SHIMMY_KV_QUANT
```

- [ ] All 7B INT4 from Phase A2 still pass (INT4 KV should not degrade correctness)

---

## Phase A5 — 70B Models (if downloaded)

**Goal:** Prove the binary can load and run a 70B model.  
**This is purely a capability test — not a correctness benchmark.**

```bash
export LIBSHIMMY_MODEL_PATH=/path/to/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf
export SHIMMY_MAX_CTX=2048
export SHIMMY_KV_QUANT=int4
./target/release/shimmy_server_gpu &
# wait up to 10 minutes for load
curl -s -X POST http://127.0.0.1:8086/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":10}' \
  | jq .choices[0].message.content
```

- [ ] 70B loads without crash or OOM
- [ ] Returns a coherent response (keyword not required — just non-garbage tokens)
- [ ] Record time-to-first-token (TTFT)

---

## Phase A6 — Needle Bench (Long Context)

**Goal:** Validate context recall at 2K, 4K, 8K for all tiers.  
Uses `scripts/needle_bench.py`.

```bash
# Server running with ctx 8192
python scripts/needle_bench.py \
  --ctx 2048,4096,8192 \
  --depths 10,25,50,75,90 \
  --runs 3 \
  --timeout 2400 \
  --out artifacts/needle_bench_a100_full.json
```

- [ ] 2K ctx: recall ≥ 90% at all depths
- [ ] 4K ctx: recall ≥ 80% at all depths
- [ ] 8K ctx: recall ≥ 70% at all depths

---

## Phase A7 — Soak Test (Stability)

**Goal:** 7B model, 50 sequential requests, no TDR, no crash, no memory leak.

```bash
# Use deepseek-7b or Llama-3.1-8B
# Run the soak script from scripts/battery_test.sh
# or adapt model_smoke_test to send 50 requests to one model session
```

- [ ] 50/50 requests complete
- [ ] `nvidia-smi` VRAM stable (not monotonically growing)
- [ ] No GPU timeout / TDR error in log

---

## Linux Porting Note (if A100 host is Linux-only)

The smoke test is a PowerShell `.ps1` script. On Linux you have two options:

1. **Install pwsh:** `apt install powershell` or `snap install powershell --classic` — the script runs as-is.
2. **Port to bash:** The script is straightforward; the main loops, `curl`, and `jq` equivalents exist in bash. If porting is needed, create `scripts/model_smoke_test.sh` — do not delete the `.ps1`.

---

## Sign-off Criteria (A100 Session Complete)

- [ ] All Phase A1 regression checks match local RTX 3060 results
- [ ] All Phase A2 7B F32 models PASS
- [ ] Phase A3 results recorded (even if some FAIL — they become known-issue tickets)
- [ ] Phase A4 INT4 cross-test passes
- [ ] Phase A5 70B load test completed (PASS or documented OOM)
- [ ] Phase A6 needle bench results saved to `artifacts/`
- [ ] Any new failures filed as GitHub issues before the next release
