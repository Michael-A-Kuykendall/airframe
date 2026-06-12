# Inference Testing & Certification — Airframe/Shimmy

## One-Command Model Test

```powershell
# From C:\Users\micha\repos\airframe\
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\micha\repos\airframe\scripts\test_model.ps1" -ModelPath "D:\shimmy-test-models\gguf_collection\<model>.gguf"
```

What it does: kills all shimmy, random port, fresh server, waits for /health with 2s initial delay, 
selects model by partial name match (avoids phi auto-discovery trap), sends prompt "hi", reports PASS/FAIL/OOM.

**One model at a time. Wait for it to finish. Never run two.**

---

## Known Pitfalls (all learned the hard way)

### 1. Prompt matters — use "hi"
- `"Say hello in exactly 3 words."` → TinyLlama returns `"` (hits stop token early)
- `"hi"` → returns "Hello" cleanly
- Always use short prompts with no quote characters

### 2. phi-3-mini auto-discovery trap
- shimmy auto-discovers `phi-3-mini-4k-instruct-q4-k-m` from `~/.shimmy/` config
- It always appears in the model list regardless of which GGUF you load
- Script handles this: exact match > partial filename match > first non-phi model
- Always verify "Using:" line in output — if it says phi, the test is invalid

### 3. Stale server false positives
- `Ready after 0s` means you hit a PREVIOUS server, not the one you just started
- Script adds 2s delay before first health poll to prevent this
- Script kills ALL shimmy processes at start and waits 3s for ports to release

### 4. layer_dump_gpu is NOT the same code path as production
- `layer_dump_gpu` binary uses manual BindlessPipeline calls — different from GpuRuntime
- It will show NaN even when production inference works
- Use `frontier_compare` for diagnostic traces, not `layer_dump_gpu`

### 5. Branch discipline for hotfixes
- Always branch hotfix from the LATEST release branch, not from v0.2.2-clean
- Chain: v0.2.3 → v0.2.4 should have been v0.2.3 → branch fix → v0.2.4
- We had to create v0.2.5 rollup because v0.2.3 and v0.2.4 were independent branches

---

## Model Status (as of v0.2.5 / 2026-06-12)

Run: `powershell.exe -ExecutionPolicy Bypass -File "C:\Users\micha\repos\airframe\scripts\test_model.ps1" -ModelPath "D:\shimmy-test-models\gguf_collection\<model>.gguf"`

| Model | Size | Expected |
|---|---|---|
| TinyLlama-1.1B-Chat-v1.0.Q4_0 | 608MB | ✅ PASS |
| tinyllama-1.1b-chat-v1.0.Q6_K | 862MB | run to verify |
| Llama-3.2-1B-Instruct-Q4_K_M | 770MB | run to verify |
| Llama-3.2-3B-Instruct-Q4_K_M | 1887MB | run to verify |
| Qwen3-0.6B-Q4_K_M | 378MB | GPU NaN (QKV layer 0) |
| qwen2-0_5b-instruct-q4_k_m | 379MB | run to verify |
| gemma-2-2b-it-Q4_K_M | 1629MB | run to verify |
| phi-2.Q4_K_M | 1706MB | run to verify |
| starcoder2-3b-Q4_K_M | 1681MB | arch routing panic |
| gpt2.Q4_K_M | 108MB | arch routing panic |
| Any model >2GB | — | 💾 OOM (wgpu 2GB cap) |

---

## Diagnosing a Failure

### Step 1: Read stderr
```
C:\Users\micha\AppData\Local\Temp\shimmy_err.txt
```

### Step 2: Map error to fix

| Error | Root Cause | Fix |
|---|---|---|
| `output.weight type not found` | Output head quant type not handled | `airframe/src/runtime/gpu.rs` match arm |
| `MissingTensor { name: "output.weight" }` | Tied embeddings — no separate output.weight | `frontier_compare.rs` fallback (fixed in v0.2.5) |
| `ShapeMismatch { tensor: "kv_head_dims" }` | CpuKvCache using n_embd/n_head instead of head_dim | `frontier_compare.rs` (fixed in v0.2.5) |
| `Parent device is lost` + `Validation Error` | GPU shader bug for this arch | Use frontier_compare to get layer trace |
| `metadata.rs:XXX panic` | Arch not implemented in bindless pipeline | Needs arch routing work |
| model returns single garbage char | GPU NaN in QKV from layer 0 | Investigate WGSL shader path |

---

## Formula Comparison Workflow (for GPU NaN investigation)

```bash
# 1. Run frontier_compare to get GPU vs CPU trace
./target/debug/frontier_compare.exe \
  --model "D:/shimmy-test-models/gguf_collection/<model>.gguf" \
  --prompt "Hello" \
  --output artifacts/<model>_trace.json

# 2. Check which layer/tensor first goes NaN
python -c "
import json
d = json.load(open('artifacts/<model>_trace.json'))
for l in d['layers'][:5]:
    print(f'Layer {l[\"layer_idx\"]}: Q gpu_non_finite={l[\"q\"][\"gpu_non_finite\"]}')
"

# 3. Full formula diff vs CPU golden
python scripts/trace_formula_diff.py \
  --candidate artifacts/<model>_trace.json \
  --golden artifacts/<model>_cpu_golden.json \
  --top 10
```

Vault golden data available for Qwen3 1.7B, Qwen3 8B, TinyLlama (see DuckDB vault).

---

## Rebuild Procedure

```bash
# After any airframe change:
cd /c/Users/micha/repos/airframe && cargo build
# Rebuild shimmy (uses path dep):
cd /c/Users/micha/repos/shimmy && cargo build
# Then cert:
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\micha\repos\airframe\scripts\test_model.ps1" -ModelPath "<path>"
```

---

## Hotfix Branch Discipline

```
release/v0.2.X-clean  (stable base)
    └── fix/description  (hotfix branch)
            ↓ CI green
        release/v0.2.(X+1)  (ALWAYS based on previous release, not v0.2.2-clean)
            ↓ push private + public
```

Never branch a hotfix from v0.2.2-clean if there are already newer releases.
