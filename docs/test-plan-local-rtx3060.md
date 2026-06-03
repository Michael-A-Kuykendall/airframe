# Airframe Local Test Plan — RTX 3060 12 GB
<!-- Auditable step-by-step regimen. Check off each step as you go. -->
<!-- Branch target: feat/gpu-large-model (post-merge with master) -->
<!-- Last updated: 2026-06-03 -->

---

## Hardware & Environment Baseline

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 3060 12 GB |
| VRAM budget | ~11.5 GB usable (driver overhead ~0.5 GB) |
| Test port | 8086 (never conflict with a running 8080 session) |
| Model dir | `D:\shimmy-test-models\gguf_collection` |
| Binary | `target\release\shimmy_server_gpu.exe` |
| Readiness probe | `GET http://127.0.0.1:8086/api/repro/queue` |

**Rule:** Kill any running server before starting the next.  Never run two instances simultaneously — the second crashes silently.

---

## Model Inventory (on disk, 2026-06-03)

| File | Size | Quant | Arch | Local runnable? | Notes |
|---|---|---|---|---|---|
| `TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf` | 608 MB | Q4_0 | LLaMA | ✅ F32+INT4 | Verified |
| `Llama-3.2-1B-Instruct-Q4_K_M.gguf` | 770 MB | Q4_K_M | LLaMA | ✅ F32+INT4 | Verified |
| `Qwen3-0.6B-Q4_K_M.gguf` | 378 MB | Q4_K_M | Qwen2 | ✅ F32+INT4 | Verified |
| `gpt2.Q4_K_M.gguf` | 107 MB | Q4_K_M | GPT-2 | ✅ F32+INT4 | Verified |
| `phi-2.Q4_K_M.gguf` | 1706 MB | Q4_K_M | Phi-2 | ✅ F32+INT4 | Verified |
| `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | 1925 MB | Q4_K_M | LLaMA | ✅ F32+INT4 | Verified |
| `starcoder2-3b-Q4_K_M.gguf` | 1763 MB | Q4_K_M | Starcoder2 | ✅ F32+INT4 | Verified |
| `gemma-2-2b-it-Q4_K_M.gguf` | 1629 MB | Q4_K_M | Gemma-2 | ✅ F32+INT4 | **Blob fix gate** — output head = 2.19 GB, needs feat/gpu-large-model blob split |
| `deepseek-coder-6.7b-instruct.Q4_K_M.gguf` | 3893 MB | Q4_K_M | DeepSeek | ✅ INT4 only | ~8 GB VRAM loaded; too tight for F32 KV at ctx 4K |
| `deepseek-llm-7b-chat.Q4_K_M.gguf` | 4028 MB | Q4_K_M | DeepSeek | ✅ INT4 only | Same constraint |
| `qwen2-7b-instruct-q4_k_m.gguf` | 4466 MB | Q4_K_M | Qwen2 | ✅ INT4 only | Same constraint |
| `minicpm-v-2.6/ggml-model-Q4_K_M.gguf` | 4464 MB | Q4_K_M | Qwen2 | ✅ INT4 only | Text-only path; mmproj NOT loaded |
| `Phi-3.5-mini-instruct.Q4_K_M.gguf` | 2282 MB | Q4_K_M | Phi-3 | ❌ LIMIT | Fused QKV — single attn tensor covers Q+K+V, not split |
| `phi3-mini-4k-instruct-q4.gguf` | 2282 MB | Q4_K_M | Phi-3 | ❌ LIMIT | Same fused QKV blocker |
| `LFM2.5-VL-1.6B/LFM2.5-VL-1.6B-Q4_0.gguf` | 663 MB | Q4_0 | LFM | ❌ LIMIT | Vision/multimodal — deferred to feat/vision-multimodal |

**Models needing download before testing:**

| Priority | Model | Source | Reason |
|---|---|---|---|
| P0 | None — all target models present | — | — |
| Nice to have | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | HF `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | Validate Qwen2.5 arch separately from Qwen3 |
| Nice to have | `gemma-2-9b-it-Q4_K_M.gguf` | HF `bartowski/gemma-2-9b-it-GGUF` | Larger Gemma-2 blob test — defer to A100 |

---

## Pre-flight Checklist

- [ ] **P0** Confirm you are on `feat/gpu-large-model` branch
  ```bash
  git branch --show-current
  # must print: feat/gpu-large-model
  ```
- [ ] **P1** Confirm master has been merged into `feat/gpu-large-model` (merge step not yet done as of 2026-06-03)
  ```bash
  git log --oneline feat/gpu-large-model..master | wc -l
  # must print: 0  (zero commits ahead on master)
  ```
- [ ] **P2** Build release binary
  ```bash
  cargo build --release --bin shimmy_server_gpu 2>&1 | tail -5
  # must end with: Compiling / Finished — zero errors
  ```
- [ ] **P3** Binary exists and is fresh
  ```bash
  ls -lh target/release/shimmy_server_gpu.exe
  ```
- [ ] **P4** No server already running on port 8086
  ```bash
  netstat -an | grep 8086
  # must be empty
  ```
- [ ] **P5** Smoke test script parses clean
  ```powershell
  $err = $null; $tok = $null
  [System.Management.Automation.Language.Parser]::ParseFile(
    (Resolve-Path 'scripts\model_smoke_test.ps1'), [ref]$tok, [ref]$err)
  $err.Count  # must be 0
  ```

---

## Phase 1 — Verified Models, F32 KV (Regression Baseline)

**Goal:** Confirm the 7 historically passing models still pass after all master merges.  
**Branch:** `feat/gpu-large-model` (post-merge)  
**INT4:** off  
**Expected result:** 7 PASS, 0 FAIL

```powershell
$env:SHIMMY_PORT = "8086"
$env:SHIMMY_MAX_CTX = "4096"
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086"
```

**Pass criteria:**

- [ ] TinyLlama-1.1B: PASS — contains "Paris"
- [ ] Llama-3.2-1B: PASS — contains "Paris"
- [ ] Llama-3.2-3B: PASS — contains "Paris"
- [ ] phi-2: PASS — contains "Paris"
- [ ] starcoder2-3b: PASS — contains "def "
- [ ] gpt2: PASS — non-empty response
- [ ] Qwen3-0.6B: PASS — contains "Paris"
- [ ] SSE streaming: **no WARNING lines** (fix this bug before marking phase complete — see §SSE Fix below)

**Record:** `artifacts/model_smoke/smoke_<timestamp>.csv`

---

## Phase 2 — Blob Fix Gate (gemma-2-2b)

**Goal:** Confirm the >2 GB blob split + GPU lm_head u32 word-offset fix actually works.  
This is the primary deliverable of `feat/gpu-large-model`.  
**Branch:** `feat/gpu-large-model` (post-merge)  
**INT4:** off first, then on

### 2a — F32 KV

```powershell
$env:SHIMMY_PORT   = "8086"
$env:SHIMMY_MAX_CTX = "2048"   # tighter ctx to stay inside 12 GB
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086"
# gemma-2-2b should now PASS instead of LIMIT
```

- [ ] gemma-2-2b-it: **PASS** — contains "Paris" — PROMOTED from LIMIT

### 2b — INT4 KV

```powershell
$env:SHIMMY_PORT      = "8086"
$env:SHIMMY_MAX_CTX   = "4096"
$env:SHIMMY_KV_QUANT  = "int4"
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086" -TestInt4
Remove-Item Env:SHIMMY_KV_QUANT
```

- [ ] gemma-2-2b-it INT4: PASS — contains "Paris"

---

## Phase 3 — INT4 KV Regression (All Verified Models)

**Goal:** Confirm TurboShimmy INT4 KV (from master, `bab91dd`) is not broken by the blob-split merge.  
**Branch:** `feat/gpu-large-model` (post-merge)

```powershell
$env:SHIMMY_PORT     = "8086"
$env:SHIMMY_MAX_CTX  = "4096"
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086" -TestInt4
```

**Pass criteria (INT4 pass for each):**

- [ ] TinyLlama-1.1B INT4: PASS
- [ ] Llama-3.2-1B INT4: PASS
- [ ] Llama-3.2-3B INT4: PASS
- [ ] phi-2 INT4: PASS
- [ ] starcoder2-3b INT4: PASS
- [ ] gpt2 INT4: PASS
- [ ] Qwen3-0.6B INT4: PASS
- [ ] gemma-2-2b INT4: PASS (first time ever tested)

---

## Phase 4 — Math Interception Smoke (MathBypassControl)

**Goal:** Verify the evalexpr CAS bypass is active and routing correctly.  
**Prerequisite:** MathBypassControl is on master (`79c8aa2` + `55248d1`) — only available post-merge.

```powershell
$env:SHIMMY_PORT = "8086"
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086" -TestMath
```

**Spot-check by hand (one model, e.g. Llama-3.2-1B):**

| Prompt | Expected | Latency gate |
|---|---|---|
| `What is 7 * 8?` | `56` | < 500 ms (bypassed) |
| `What is 2 + 2?` | `4` | < 500 ms |
| `What is 144 / 12?` | `12` | < 500 ms |

- [ ] All 10 math cases PASS
- [ ] Fast latency observed (bypassed path, not model inference)

---

## Phase 5 — 7B Models, INT4 KV

**Goal:** First-ever test of 7B models with INT4 KV on this hardware.  
**VRAM budget:** ~10.5 GB loaded + INT4 KV → fits within 11.5 GB usable.  
**ctx:** 2048 (conservative — 4096 may OOM on 7B with F32 KV)

```powershell
$env:SHIMMY_PORT    = "8086"
$env:SHIMMY_MAX_CTX = "2048"
.\scripts\model_smoke_test.ps1 -BaseUrl "http://127.0.0.1:8086" -IncludeLarge -TestInt4
```

**Pass criteria:**

- [ ] deepseek-coder-6.7b INT4: PASS — contains "def "
- [ ] deepseek-llm-7b INT4: PASS — contains "Paris"
- [ ] qwen2-7b INT4: PASS — contains "Paris"
- [ ] minicpm-v-2.6 (text path) INT4: PASS — contains "Paris"

**If any FAIL:** check GPU memory with `nvidia-smi` during run, reduce ctx to 1024 and retry.

---

## Phase 6 — Full Matrix Run (Single Command)

**Goal:** One definitive run that covers everything for the release sign-off.  
**Only run after Phases 1–5 all pass individually.**

```powershell
$env:SHIMMY_PORT    = "8086"
$env:SHIMMY_MAX_CTX = "2048"
.\scripts\model_smoke_test.ps1 `
    -BaseUrl "http://127.0.0.1:8086" `
    -IncludeLarge `
    -TestInt4 `
    -TestMath
```

- [ ] Summary line: `PASS: N  WEAK: 0  FAIL: 0`
- [ ] CSV archived to `artifacts/model_smoke/smoke_<timestamp>.csv`
- [ ] Log archived to `artifacts/model_smoke/smoke_<timestamp>.log`

---

## Known Blockers (do not run, document only)

| Model | Blocker | Resolution path |
|---|---|---|
| `Phi-3.5-mini-instruct.Q4_K_M.gguf` | Fused QKV: single tensor covers Q+K+V — airframe expects split tensors | Implement fused-QKV split in tensor loader |
| `phi3-mini-4k-instruct-q4.gguf` | Same as above | Same fix |
| `LFM2.5-VL-1.6B` | Vision/multimodal architecture | `feat/vision-multimodal` only, never merge to master |

---

## SSE Streaming Fix (Pre-release P0)

Every test run to date shows `WARNING -- no 'data: ' events received` on **all models**.  
This means `stream: true` is silently broken for all users.

**Location:** `src/bin/shimmy_server_gpu/server_inference.rs` — `stream_tx` path (~line 814)

Steps to diagnose:

1. Start server, send manual streaming request:
   ```bash
   curl -N -X POST http://127.0.0.1:8086/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"local","messages":[{"role":"user","content":"Count to 5"}],"stream":true,"max_tokens":30}'
   ```
2. Observe: are `data: {...}` chunks returned, or is the response silent?
3. Check `use_stream` flag — is `stream_tx` being wired through to the token loop?

- [ ] Root cause identified
- [ ] Fix implemented and committed
- [ ] SSE test in smoke script shows no WARNING

---

## Release Gate Checklist (after all phases pass)

- [ ] All Phase 1–6 checks marked ✅
- [ ] SSE streaming fixed
- [ ] CHANGELOG `[Unreleased]` → `[0.2.2]` with release date
- [ ] Version bump in `Cargo.toml`: `version = "0.2.2"`
- [ ] `cargo test` passes (no regressions in unit tests)
- [ ] `git tag v0.2.2 && git push origin v0.2.2`
- [ ] `cargo publish -p airframe`
- [ ] Update shimmy dependency to `airframe = "0.2.2"` and publish shimmy
