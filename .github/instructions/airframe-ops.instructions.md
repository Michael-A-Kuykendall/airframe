---
applyTo: "**"
---

# Airframe GPU Server — Operational Conventions

Use this file to resolve ALL terminal naming, env vars, readiness checks, and flow sequencing
for Airframe server and smoke test operations. Do not invent bespoke approaches.

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

## No-Duplicate-Server Rule

Before starting a server, always verify port 8080 is free:
```powershell
netstat -ano | findstr :8080
```
If occupied, kill the PID shown or use `terminal-tools_cancelCommand` on `gpu-server`.
Do NOT start a second server instance — the second one will crash silently and tests will hit the first server's loaded model.

---

## Build

Server binary must be up to date before smoke testing:
```bash
# build terminal:
cargo build --release --bin shimmy_server_gpu
```

Check binary freshness: if `target/release/shimmy_server_gpu.exe` is older than the most recent `.rs` change in `src/`, rebuild.
