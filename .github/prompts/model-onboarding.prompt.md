---
mode: agent
description: Step-by-step regimen for adding and validating a new GGUF model in Airframe. Run this before calling any model "supported".
---

# Model Onboarding Regimen

Use this prompt to onboard a new GGUF model file into Airframe. Complete all gates in order. A model is only "supported" once all applicable gates pass.

**Model to onboard:** (specify filename, e.g. `Qwen3-8B-Instruct-Q4_K_M.gguf`)
**GGUF arch string:** (from `general.architecture` in the GGUF metadata — e.g. `qwen3`, `llama`, `phi2`)
**Type:** instruction-tuned | completion | code | reasoning (affects prompt choice)
**Location:** `D:\shimmy-test-models\gguf_collection\<filename>`

---

## Gate 1 — quant_verify (GPU dequant conformance)

Only run if model is **≤ 2.0 GB** (RTX 3060 buffer binding limit). Skip and note "size-exempt" for larger models.

```powershell
# In build terminal (bash):
$env:LIBSHIMMY_MODEL_PATH = "D:\shimmy-test-models\gguf_collection\<filename>"
cargo run --release --bin quant_verify
```

**Pass:** All tensor types print `OK`. No `MISMATCH` lines.
**Fail:** Any `MISMATCH` → stop, do not proceed. Open a `quant_verify` bug.
**Size-exempt:** Model > 2 GB → skip this gate, proceed to Gate 2.

---

## Gate 2 — Add smoke test entry

Open `scripts\model_smoke_test.ps1` and add an entry to `$VerifiedModels`:

```powershell
@("<filename>", "<expected_keyword>", "<prompt_text>")
```

Prompt / keyword guidance by model type:

| Type | Prompt | Expected keyword |
|------|--------|-----------------|
| Instruction-tuned (general) | `"The capital of France is"` | `"Paris"` |
| Code model | `"def hello_world():"` | `"def "` |
| Completion (no instruction template) | `"The capital of France is"` | `""` (any non-empty output passes) |
| Reasoning (Qwen3 etc.) | `"The capital of France is /no_think"` | `"Paris"` |

**Rule:** Temperature is always `0.0` in smoke tests. The keyword must be reliably present at temp=0.

---

## Gate 3 — Run model smoke test

Verify port 8080 is free first:

```powershell
netstat -ano | findstr :8080
```

Then run:

```powershell
cd C:\Users\micha\repos\airframe
powershell -ExecutionPolicy Bypass -File scripts\model_smoke_test.ps1
```

**Pass:** The new model entry prints `PASS` or `WEAK` (for completion models with empty keyword).
**Fail:** `FAIL` — diagnose from terminal output. Common causes: startup timeout, empty response, wrong keyword.

---

## Gate 4 — API schema spot-check

After the smoke test, run one manual request and verify the response shape:

```powershell
$env:LIBSHIMMY_MODEL_PATH = "D:\shimmy-test-models\gguf_collection\<filename>"
$env:SHIMMY_PORT = "8080"
$env:RUST_BACKTRACE = "1"
.\target\release\shimmy_server_gpu.exe &

# Wait for readiness...
$r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/v1/chat/completions" `
    -ContentType "application/json" `
    -Body '{"model":"local","messages":[{"role":"user","content":"Hi"}],"max_tokens":16,"temperature":0.0,"stream":false}'

# Verify required fields:
$r.id              # must be non-empty string
$r.choices[0].message.content  # must be non-empty string
$r.usage.prompt_tokens         # must be int > 0
$r.usage.completion_tokens     # must be int > 0
```

**Pass:** All four fields present and non-zero.
**Fail:** Missing field → file an API schema bug, do not mark model as supported.

Kill server after:
```powershell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8080).OwningProcess
```

---

## Gate 5 — Reasoning mode gate (reasoning models only)

Only applies to models with thinking mode (Qwen3, QwQ, DeepSeek-R1, etc.).

Test both modes:
```
"The capital of France is /no_think"  → expect "Paris" (fast path, no <think> block)
"What is 2+2? /think"                 → expect response contains reasoning or "4"
```

**Pass:** Both modes return coherent responses. `/no_think` must NOT contain a `<think>` block.

---

## Post-Gates — Update documentation

Once all applicable gates pass:

1. **`scripts\model_smoke_test.ps1`** — entry already added in Gate 2. Confirm it's in `$VerifiedModels` (not just tested once).

2. **`.github\instructions\airframe-ops.instructions.md`** — add a row to the "Verified" table under Model Inventory:
   ```
   | `<filename>` | <arch> | <size> | <native ctx> |
   ```

3. **`shimmy_integration\docs\MODEL_EXPANSION.md`** — add row to Supported Model Architectures table (if new arch).

4. **`RELEASE_STATUS.md`** — note the addition under the current version.

---

## Known-Failure Reference

Do NOT run these through the server without prior code fixes:

| File | Failure mode |
|------|-------------|
| `gemma-2-2b-it-Q4_K_M.gguf` | Output head = 2.19 GB — exceeds WebGPU 2 GB buffer limit. Fix: output head chunking (roadmap). |
| `Phi-3.5-mini-instruct.Q4_K_M.gguf` | Fused QKV tensor (`attn_qkv.weight`) — server panics on load. Fix: fused QKV split (roadmap). |
| `phi3-mini-4k-instruct-q4.gguf` | Same fused QKV issue. |
| Any Qwen3 model | Missing QK norm shader + output head buffer limit. Fix: Qwen3 dense sprint (roadmap). |
