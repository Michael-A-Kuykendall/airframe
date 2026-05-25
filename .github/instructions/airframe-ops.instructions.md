---
applyTo: "**"
---

# Airframe GPU Server — Operational Conventions

Use this file to resolve ALL terminal naming, env vars, readiness checks, and flow sequencing
for Airframe server and smoke test operations. Do not invent bespoke approaches.

---

## Repo Architecture — Obfuscation Boundary

```
airframe/                     ← PRIVATE repo (GPU engine, WGSL shaders, inference)
└── shimmy_integration/       ← submodule → shimmy-private repo
      Cargo.toml              ← airframe = { path = "../" }
```

**Rules — never violate these:**
- `airframe` is PRIVATE. It must never appear as a public repo, public branch, or public submodule.
- `shimmy_integration` (`shimmy-private`) is the public-facing product repo.
- Airframe is wired in as a **path dep pointing to `../`** — the parent directory.
  This means airframe source never lives inside the shimmy repo tree.
- **Do NOT add a nested `airframe/` submodule inside `shimmy_integration`.** This was done
  by a previous AI session as an OpenClaw artifact and has been removed. Do not recreate it.
- For CI on a clean machine: `git clone <private airframe> ../airframe` before
  `cargo build --features airframe`. The path dep resolves to that checkout.
- For local dev: `../` already resolves to the airframe workspace. No extra steps needed.
- `shimmy_integration` has only ONE remote: `private` → `shimmy-private`. Never push to `origin`.
  `shimmy_integration` has NO `origin` remote. Do not add one.

---

## Terminal Names

| Purpose | Terminal name | Shell |
|---------|--------------|-------|
| GPU inference server (background) | `gpu-server` | PowerShell |
| Smoke / validation scripts | `smoke-test` | PowerShell |
| Build operations | `build` | bash |
| File inspection / git | `inspect` | bash |

**Rule:** Never use the same terminal for a running server and for sending test commands.
If `gpu-server` is running a server, open `smoke-test` for everything else.

---

## Starting the GPU Server

**Env vars** (must be set before or alongside the process):

| Variable | Purpose | Default |
|----------|---------|---------|
| `LIBSHIMMY_MODEL_PATH` | Path to the GGUF file to load | `D:\shimmy-test-models\gguf_collection\TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf` |
| `SHIMMY_PORT` | HTTP listen port | `8080` |
| `SHIMMY_MAX_CTX` | Override context window (optional) | reads from GGUF |
| `RUST_BACKTRACE` | Set to `1` for crash traces | unset |

**NOT** `SHIMMY_BASE_GGUF` — that is the Shimmy provider env var, not the server binary var.

**Binary:** `target\release\shimmy_server_gpu.exe`

**Manual start (PowerShell, gpu-server terminal):**
```powershell
$env:LIBSHIMMY_MODEL_PATH = "D:\shimmy-test-models\gguf_collection\<model>.gguf"
$env:SHIMMY_PORT = "8080"
$env:RUST_BACKTRACE = "1"
.\target\release\shimmy_server_gpu.exe
```

**Via VS Code task:** "Run Airframe GPU Server" (uses TinyLlama default)
**Via VS Code task with 8K context:** "Run Airframe GPU Server (8K ctx)"

**Ready signal (stderr):**
```
[HTTP] Async listener spawned on 0.0.0.0:8080
```

---

## Readiness Check

The correct readiness endpoint is:
```
GET http://127.0.0.1:8080/api/repro/queue
```

**NOT** `/v1/models`, `/health`, or `/ping` — those do not exist on this server.

Polling pattern (PowerShell):
```powershell
for ($i = 0; $i -lt 120; $i++) {
    try {
        $null = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8080/api/repro/queue" -TimeoutSec 2
        break
    } catch { Start-Sleep -Seconds 1 }
}
```

---

## Inference API

```
POST http://127.0.0.1:8080/v1/chat/completions
Content-Type: application/json

{
  "model": "local",
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 64,
  "temperature": 0.0,
  "stream": false
}
```

Response is OpenAI-compatible. Extract text via `$response.choices[0].message.content`.

---

## Model Inventory

Local collection: `D:\shimmy-test-models\gguf_collection\`

### Verified (quant_verify + inference smoke tested, RTX 3060 12 GB)

| File | Arch | Size | Native ctx |
|------|------|------|-----------|
| `TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf` | llama | 609 MB | 2048 |
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | llama | 771 MB | 131072 |
| `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | llama | 1.9 GB | 131072 |
| `phi-2.Q4_K_M.gguf` | phi2 | 1.7 GB | 2048 |
| `gemma-2-2b-it-Q4_K_M.gguf` | gemma2 | 1.6 GB | 8192 |
| `starcoder2-3b-Q4_K_M.gguf` | starcoder2 | 1.8 GB | 16384 |
| `gpt2.Q4_K_M.gguf` | gpt2 | 108 MB | n/a — completion model, not instruction |

### Present but NOT inference-validated locally

| File | Reason |
|------|--------|
| `Phi-3.5-mini-instruct.Q4_K_M.gguf` | Fused QKV (`attn_qkv.weight`) — server will crash |
| `phi3-mini-4k-instruct-q4.gguf` | Same fused QKV issue |
| `deepseek-coder-6.7b-instruct.Q4_K_M.gguf` | 3.9 GB — needs VRAM budget confirmation |
| `deepseek-llm-7b-chat.Q4_K_M.gguf` | 4.0 GB — same |
| `qwen2-7b-instruct-q4_k_m.gguf` | 4.4 GB — same; Qwen2 arch not explicitly validated |
| `LFM2.5-VL-1.6B/LFM2.5-VL-1.6B-Q4_0.gguf` | Multimodal — unknown arch |

Do not attempt to run Phi-3 or Phi-3.5 through the server without code changes.

---

## Per-Model Smoke Test

**Script:** `scripts\model_smoke_test.ps1`

Run from `smoke-test` terminal. The script manages its own server lifecycle (start, wait, request, kill). Do NOT have `gpu-server` terminal running a server when invoking this script — port conflict on 8080.

```powershell
cd C:\Users\micha\repos\airframe
powershell -ExecutionPolicy Bypass -File scripts\model_smoke_test.ps1
```

With 7B models included:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\model_smoke_test.ps1 -IncludeLarge
```

Results written to `artifacts\model_smoke\smoke_<timestamp>.log` and `.csv`.

**VS Code task:** "Run Model Smoke Test"

---

## quant_verify

Validates GPU vs CPU dequant agreement for all tensor types in a GGUF.

```powershell
# In build terminal:
$env:LIBSHIMMY_MODEL_PATH = "D:\shimmy-test-models\gguf_collection\<model>.gguf"
cargo run --release --bin quant_verify
```

**Limitation:** Models > ~2.1 GB will fail with a buffer binding error on RTX 3060.
Only run quant_verify on models ≤ 2 GB (TinyLlama, Llama-3.2-1B, Llama-3.2-3B skips because 1.9 GB is right at the limit — use TinyLlama or Llama-3.2-1B as the safe targets).

---

## End-to-End Validation Flows (VS Code Tasks)

Use these tasks rather than hand-rolling the flows:

| Task | What it does |
|------|-------------|
| `U: Validate Airframe Default Path` | Starts server → Short SHA check → Long Story check |
| `U: Validate Card Shadow Path` | Starts server → Card smoke (shadow mode) |
| `U: Validate Card On Path` | Starts server → Card smoke (on mode) |
| `U: Smoke Shimmy Provider Path` | Starts server → Starts Shimmy provider → Provider smoke |
| `U: Needle Smoke (2K, default server)` | Starts server → Needle bench at 2K ctx |
| `Run Model Smoke Test` | PowerShell smoke test across all verified models |

---

## Verbose Output Rule

Never redirect server or script output to files during local dev or smoke testing.
All stdout/stderr must flow to the terminal. Silent failures are unacceptable —
if a process crashes, panics, or OOMs, that output must be visible immediately.
File redirection is only appropriate in fully verified CI pipelines, never during bring-up or debugging.

---

## No-Duplicate-Server Rule

Before starting a server, always verify port 8080 is free:
```powershell
netstat -ano | findstr :8080
```
If occupied, kill the PID shown or use `terminal-tools_cancelCommand` on `gpu-server`.
Do NOT start a second server instance — the second one will crash silently and tests will hit the first server's loaded model.

---

## Model Acceptance Protocol

A model is only "supported" once all applicable gates pass. Use `.github/prompts/model-onboarding.prompt.md` as the step-by-step execution guide.

**Five gates — run in order:**

| Gate | Name | Requirement | Skip condition |
|------|------|-------------|---------------|
| 1 | `quant_verify` | All tensor types print `OK`, no `MISMATCH` | Model > 2.0 GB (buffer binding limit on RTX 3060) |
| 2 | Smoke entry | Entry added to `$VerifiedModels` in `model_smoke_test.ps1` | None |
| 3 | Smoke test | Model prints `PASS` or `WEAK` in `model_smoke_test.ps1` | None |
| 4 | API schema | `id`, `choices[0].message.content`, `usage.prompt_tokens`, `usage.completion_tokens` all present | None |
| 5 | Reasoning mode | Both `/no_think` and `/think` return coherent responses | Non-reasoning models |

After all gates: update model inventory in this file, `MODEL_EXPANSION.md`, and `RELEASE_STATUS.md`.

**Known non-starters (do not attempt without prior code fixes):**
- `gemma-2-2b-it-Q4_K_M.gguf` — output head 2.19 GB > WebGPU 2 GB limit (needs output head chunking)
- `Phi-3.5-mini-instruct.Q4_K_M.gguf` / `phi3-mini-4k-instruct-q4.gguf` — fused QKV tensors, server panics
- Any Qwen3 model — missing QK norm shader + output head buffer limit

---

## Build

Server binary must be up to date before smoke testing:
```bash
# build terminal:
cargo build --release --bin shimmy_server_gpu
```

Check binary freshness: if `target/release/shimmy_server_gpu.exe` is older than the most recent `.rs` change in `src/`, rebuild.

---

## New Model Debug Checklist

Three structured log tags are emitted on every server startup and first inference. Grep these to
diagnose new model failures before wasting time on template or code investigations.

### `[ARCH_REGISTRY]` — emitted on model load

```
[ARCH_REGISTRY] arch=qwen3  layers=28  vocab=151936  kv_heads=8  head_dim=128
```

Followed by `[ARCH_TENSOR_MISSING]` for any absent required tensors and `[ARCH_TENSOR_UNEXPECTED]`
for tensors that signal a likely arch mismatch (e.g. fused `attn_qkv.weight` on a Qwen3 model).

**Action:** If `[ARCH_TENSOR_MISSING] REQUIRED` appears, the model is missing a tensor the server
needs for this arch. It will run but produce garbage or crash. Check the GGUF file and the
server's arch detection logic in `ModelArch::from`.

### `[VRAM_BUDGET]` — emitted after KV cache init

```
[VRAM_BUDGET] kv_cache=8960 MB  output_head=593 MB  tracked_total=9553 MB  limit=10500 MB
[VRAM_BUDGET] WARN: tracked allocations (9553 MB) exceed limit (10500 MB).
[VRAM_BUDGET] KV cache is dominant: ctx=40960  layers=28  kv_heads=8  head_dim=128
[VRAM_BUDGET] Suggest: set SHIMMY_MAX_CTX=896 to bring KV cache within budget.
```

**Action when WARN appears:** The model's native context is too large for available VRAM.
Set `SHIMMY_MAX_CTX` to the suggested value. Example: Qwen3-0.6B native ctx=40960 needs 8960 MB
KV cache alone; setting `SHIMMY_MAX_CTX=4096` cuts it to ~896 MB.

Override the default limit (10500 MB for RTX 3060) with `SHIMMY_VRAM_LIMIT_MB`.

### `[PREFILL_SANITY]` — emitted after first prefill

```
[PREFILL_SANITY] arch=qwen3  ctx=40960  kv_cache=8960MB  top1_prob=0.0012  ppl_est=3245.44  norm=1202.68  WARN:high_ppl -- likely VRAM pressure; try lower SHIMMY_MAX_CTX
```

| `ppl_est` range | Meaning |
|----------------|---------|
| < 50 | OK — model is working correctly |
| 50–500 | ELEVATED — may work; monitor first few tokens |
| > 500 | WARN — almost always VRAM pressure (garbage numerics), not template or arch issue |

**Diagnosis rule:** PPL > 500 at step 0 = VRAM pressure. Do NOT investigate templates, tokenizer,
or chat format until VRAM budget is confirmed clean. Check `[VRAM_BUDGET]` first.

**What `[REDO] Metric Violation` means:** Every generation step where `ppl > 500 || norm > 1e5`
falls back to greedy. This is the per-token self-heal. When you see it on every step, the entire
KV cache is corrupted — VRAM is the cause.

### Typical new model failure sequence

```
[VRAM_BUDGET] WARN ...                       ← 1. Budget exceeded on load
[PREFILL_SANITY] ... ppl_est=3245 ... WARN   ← 2. Numerics are garbage immediately
[REDO] Metric Violation (PPL=3245...) ...    ← 3. Per-token fallback fires every step
→ WEAK result in smoke test with random multilingual tokens
```

**Fix:** Lower `SHIMMY_MAX_CTX` until `[VRAM_BUDGET]` shows no WARN and `[PREFILL_SANITY]`
shows `ppl_est < 50`.

