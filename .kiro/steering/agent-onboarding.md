# Agent Onboarding — Airframe + Shimmy (Read This First)

**You are working on a pure-Rust WebGPU GGUF inference engine.**  
Read this entire file before touching any code. It will save you days.

---

## The Stack

```
shimmy/          ← HTTP server product (OpenAI/Ollama-compat API)
  └── depends on airframe via path dep (../airframe)
airframe/        ← GPU inference engine (wgpu + WGSL compute shaders)
  └── crates/airframe_observe/  ← ISF reactive fabric (D0/FSE)
shimmytok/       ← Tokenizer (C:\Users\micha\repos\shimmytok)
dzero-cas/       ← d0-engine (reactive graph engine)
```

**Shell**: Always bash (Cygwin). Paths use `/c/Users/micha/...`. Never PowerShell syntax.

---

## One-Liner Tests (Run These First)

### FASTEST: shimmy generate (no server needed — direct inference, single command)
```bash
# From shimmy directory — loads model, runs prompt, prints output, exits
cd /c/Users/micha/repos/shimmy
SHIMMY_MAX_CTX=3000 SHIMMY_ROPE_SCALE=0.68 ./target/release/shimmy.exe generate \
  "TinyLlama-1.1B-Chat-v1.0.Q4_0" "hi" --max-tokens 20 \
  --model-path "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"
```
Expected: prints a few words and exits. No server, no curl, no log file needed.

### DeepSeek generate one-liner
```bash
SHIMMY_MAX_CTX=4096 SHIMMY_ROPE_SCALE=0.5 ./target/release/shimmy.exe generate \
  "deepseek-coder-6.7b-instruct.Q4_K_M" "write hello world in python" --max-tokens 60 \
  --model-path "D:/shimmy-test-models/gguf_collection/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
```

### Full smoke test (uses test_model.ps1 — starts/stops server automatically)
```bash
powershell.exe -ExecutionPolicy Bypass -File "C:/Users/micha/repos/airframe/scripts/test_model.ps1" -ModelPath "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"
```
Expected: `PASS - 'Hello' (finish=stop)`

### Start persistent server (when you need curl testing or streaming)
```bash
SHIMMY_MAX_CTX=3000 SHIMMY_ROPE_SCALE=0.68 ./target/release/shimmy.exe serve \
  --model-path "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf" \
  --bind 127.0.0.1:11435 2>&1 | tee /tmp/shimmy_isf_run.log
```

### Check ISF log (after any run)
```bash
cat /tmp/shimmy_isf_run.log | grep -E "ISF-RULE|ISF-R4|hidden_rms|complete,"
```

### Build airframe
```bash
cd /c/Users/micha/repos/airframe && cargo build --release --features isf
```

### Build shimmy
```bash
cmd.exe /C "taskkill /F /IM shimmy.exe" 2>&1
cd /c/Users/micha/repos/shimmy && cargo build --release
```

### Export full code to markdown (for cloud review)
```bash
cd /c/Users/micha/repos/airframe && bash scripts/export_code_to_md.sh > docs/internal/code-export-$(date +%Y-%m-%d).md
```

---

## Current Branch State (as of 2026-06-17)

| Repo | Branch | Status |
|------|--------|--------|
| airframe | `feat/phase4-pingpong-activation` | Active dev — clean, Q4K debug pipeline fix committed |
| shimmy | `fix/template-apply-raw-prompt` | Active dev — clean |

**Always `git status` before starting.**

**CRITICAL — DO NOT REGRESS (learned 2026-06-17):**
- `run_layer_with_cache_debug` in `pipeline/layer.rs` MUST dispatch Q4K pipelines when `(params.quant_type & 0xFF) == 12`. This was the primary source of all-zero GPU Q/K/V in frontier_compare.
- `frontier_compare` MUST derive `quant_type` from model metadata, never hardcode 0.
- `dequantize_embeddings` in `frontier_compare` MUST use `run_dequant_any_hot`, not `run_dequant_request` (which is Q4_0 only).
- See `docs/internal/handoff-q4k-debug-pipeline-fix.md` for full root cause analysis.

---

## Model Status

| Model | Size | Status | Best For | Ollama Name |
|-------|------|--------|----------|-------------|
| TinyLlama-1.1B-Chat-v1.0.Q4_0 | 608MB | ✅ PASS | **Fastest regression test** | `tinyllama-chat` |
| Llama-3.2-1B-Instruct-Q4_K_M | 771MB | ❓ untested | Quick coding test | `llama32-1b` |
| **Llama-3.2-3B-Instruct-Q4_K_M** | 1.9GB | ❓ untested | **Best for aider** (3B, instruct, fits comfortably) | `llama32-3b` |
| Qwen3-1.7B-Q4_K_M | 1.2GB | ❓ untested | Coding + reasoning | `qwen3-1.7b` |
| Qwen3-4B-Q4_K_M | 2.4GB | ❓ untested | Stronger coding | `qwen3-4b` |
| deepseek-coder-6.7b-instruct.Q4_K_M | 3.89GB | ⚠️ broken output | Target model — Q4K math bug | `deepseek-coder-6.7b` |
| Qwen3-8B-Q4_K_M | 4.7GB | ❓ untested | Largest tested | `qwen3-8b` |
| Qwen3-0.6B-Q4_K_M | 379MB | ❌ GPU NaN | — | `qwen3-0.6b` |
| Starcoder2-3B | 1.68GB | ❌ arch panic | — | `starcoder2-3b` |
| gpt2 | 108MB | ❌ arch panic | — | `gpt2` |

**Recommended test sequence when validating a patch:**
1. `shimmy generate tinyllama-chat "hi"` — fastest smoke test
2. `shimmy generate llama32-3b "write hello world in python"` — 3B sanity check
3. `shimmy generate deepseek-coder-6.7b "write hello world in python"` — target model

All models are in `D:/shimmy-test-models/gguf_collection/` and registered in Ollama as the names above.

---

## Architecture in 60 Seconds

1. **GGUF file** → mmap'd into a single GPU `StorageBuffer` (the "bindless blob")
2. **BindlessMetadata** parses tensor offsets/types at startup — no per-inference file I/O
3. **PreflightResources** extracts RoPE table + norm weights into separate GPU buffers
4. **run_full_model_with_cache_state()** in `inference.rs` is the main dispatch loop:
   - For each layer: AttnNorm → QKV → QKNorm → AttnOut → AttnProj → FFNNorm → FFNProj → FFNDown
   - QKV uses pre-built per-chunk bind groups (no write_buffer, Phase 4a Step 5)
   - TDR: force yield every 16 QKV chunks + every layer boundary (batch>1), conditional (batch=1)
5. **ISF (Inference Saturation Fabric)** in `crates/airframe_observe/src/isf.rs` drives the generate loop:
   - `PromptToken` → `EmbeddingRequest` → `EmbeddingReady` → `PrefillBatchReady` → `PrefillComplete` → `DecodeStep` → `GenerationHalt`
   - TDR facts: `DispatchCompleted` → `TdrRiskHigh` → `YieldNow`
6. **Shimmy** wraps GpuRuntime, applies chat templates, serves OpenAI/Ollama API

---

## Key Files

| File | What It Does |
|------|-------------|
| `src/backend/bindless/pipeline/inference.rs` | THE main dispatch loop (~900 LOC) |
| `src/backend/bindless/sh_layer_q4k.wgsl` | Q4_K_M layer kernels (WGSL) |
| `src/backend/bindless/sh_layer_v1.wgsl` | Q4_0/Q8_0/F16 layer kernels (WGSL) |
| `src/runtime/gpu.rs` | GpuRuntime::load() + generate_isf() |
| `src/backend/tdr.rs` | TdrScheduler (extracted from inference.rs) |
| `crates/airframe_observe/src/isf.rs` | ISF rules + ISFState |
| `crates/airframe_observe/src/facts.rs` | InferenceFact enum + keys |
| `shimmy/src/api.rs` | HTTP handlers, template application |
| `shimmy/src/engine/airframe.rs` | AirframeEngine → GpuRuntime bridge |
| `shimmy/src/templates.rs` | TemplateFamily renders (ChatML, Llama3, TinyLlama, DeepSeekCoder) |
| `shimmy/src/model_registry.rs` | infer_template() by model name |

---

## The One Remaining Bug (DeepSeek)

DeepSeek 6.7B Q4_K_M produces garbage output. All infrastructure is correct:
- KV cache positions correct (`current_pos=72` at decode step 0 ✅)
- `kv_increment` called N times (N=prompt_len ✅)
- `hidden_rms=17.1` finite, `logits_nans=0` ✅

**Root cause**: Q4K attention shader (`main_attn_out` in `sh_layer_q4k.wgsl`) produces wrong attention score distribution for this model's weight pattern.

**Fix path**: Compare `main_attn_out` WGSL formula algebraically against llama.cpp's `ggml_vec_dot_q4_K` + attention math. The delta will be the bug.

**Reference**: `docs/internal/local_ollama/attn_out_diagnostic_plan.md` has the diagnostic plan.

---

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `SHIMMY_MAX_CTX` | 8192 | Context window cap |
| `SHIMMY_ROPE_SCALE` | 1.0 | YaRN RoPE scaling factor |
| `SHIMMY_PREFILL_CHUNK` | 1 | Tokens per QKV chunk |
| `SHIMMY_TDR_BUDGET_MS` | 1400 | TDR yield budget (Windows) |
| `AIRFRAME_TRACE_PREFILL_LAYERS` | 0 | Log per-layer activation stats |
| `AIRFRAME_LOG_TDR_POLLS` | 0 | Log yield count per forward pass |
| `AIRFRAME_PINGPONG_ACTIVATION` | 0 | Enable ping-pong activation buffers |

---

## Patent Notice

FSE (Fused Semantic Execution) and D0 Saturation Fabric are pending patents by Michael A. Kuykendall. All ISF/fabric code must carry the patent notice comment. Do not remove or alter patent notices.

---

## Docs Index

| File | Purpose |
|------|---------|
| `.kiro/steering/agent-onboarding.md` | **THIS FILE** — start here |
| `.kiro/steering/inference-testing.md` | Detailed testing procedures |
| `.kiro/steering/fse-d0-lens.md` | FSE + D0 architecture concepts |
| `.kiro/steering/current-work-directive.md` | Active work priorities |
| `docs/internal/code-export-2026-06-15.md` | Full source dump (72K lines) |
| `docs/internal/handoff-2026-06-15.md` | Latest technical handoff |
| `docs/internal/airframe_inference_redesign_report.md` | Full redesign plan (cloud-authored) |
| `docs/internal/phase4a-pingpong-plan.md` | Phase 4a execution plan |
| `docs/internal/session-log-export-2026-06-15.md` | Session history + git log |
