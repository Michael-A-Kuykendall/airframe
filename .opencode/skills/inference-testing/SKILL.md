---
name: inference-testing
description: Testing airframe GPU inference — smoke tests, frontier_compare, vault comparison. Use for running one-liner frontier_compare or shimmy generate inference tests.
---

# Inference Testing Skill

## One-Liner Smoke Tests (run these first, always)

### Fastest — TinyLlama Q4_0 (baseline, must always pass)
```powershell
cd C:\Users\micha\repos\airframe
.\target\release\frontier_compare.exe --model "D:\shimmy-test-models\gguf_collection\TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf" --prompt "hi" --max-ctx 2048 --output artifacts\tinyllama_smoke.json
# Expected: layer 0 output MAE < 0.01, no NaN anywhere
```

### Llama-3.2-1B Q4_K_M (Q4K regression test)
```powershell
.\target\release\frontier_compare.exe --model "D:\shimmy-test-models\gguf_collection\Llama-3.2-1B-Instruct-Q4_K_M.gguf" --prompt "hi" --max-ctx 4096 --output artifacts\llama32_smoke.json
# Expected: layer 0 MAE < 0.0001, layer 1 MAE < 0.0001, no NaN on layers 0-1
# Layer 2+ NaN is a known frontier_compare debug issue, not production
```

### Read smoke results
```powershell
python -c @"
import json
d = json.load(open('artifacts/llama32_smoke.json'))
for l in d['layers'][:4]:
    print(f'layer {l["layer_idx"]}: output MAE={l["output"]["mean_abs_err"]}  post_attn MAE={l["post_attn"]["mean_abs_err"]}')
print('logits nf=', d['logits']['gpu_non_finite'])
"@
```

## Vault Comparison (ground truth)

### Query vault for a model's expected layer output RMS
```powershell
duckdb vault/vault.duckdb "SELECT o.layer_idx, o.expected_rms, o.expected_nan FROM layer_oracles o JOIN models m ON o.model_id = m.id WHERE m.name LIKE '%Llama%1B%' OR m.name LIKE '%TinyLlama%' ORDER BY m.name, o.layer_idx LIMIT 20;"
```

### Compare GPU output against vault
```powershell
python -c @"
import json
gpu = json.load(open('artifacts/llama32_smoke.json'))
vault = {0:0.044914,1:0.054706,2:0.077304,3:0.086485}  # from vault query above
for l in gpu['layers'][:4]:
    idx = l['layer_idx']
    g = l['output'].get('gpu_rms') or 0.0
    v = vault.get(idx, 0)
    pct = abs(g-v)/v*100 if v else 0
    print(f'layer {idx}: vault={v:.5f} gpu={g:.5f} diff={pct:.1f}%')
"@
```

### List all vault models and oracle counts
```powershell
duckdb vault/vault.duckdb "SELECT m.name, m.quant, COUNT(o.id) as oracles FROM models m LEFT JOIN layer_oracles o ON m.id = o.model_id GROUP BY m.name, m.quant ORDER BY oracles DESC;"
```

## Shimmy Generate Test (end-to-end with template)
```powershell
cd C:\Users\micha\repos\shimmy
$env:SHIMMY_BASE_GGUF = "D:\shimmy-test-models\gguf_collection\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
$env:SHIMMY_MAX_CTX = "4096"
$env:SHIMMY_ROPE_SCALE = "0.5"
.\target\release\shimmy.exe generate "tinyllama-1.1b" --prompt "write hello world in python" --max-tokens 60 2>&1 | Select-String -NotMatch "^\[Metadata\]|^\[Preflight\]|^\[DIAG\]|^\[ISF"
# Expected: coherent Python code, Llama3 instruct format
```

### TinyLlama baseline (quickest sanity check)
```powershell
cd C:\Users\micha\repos\shimmy
$env:SHIMMY_BASE_GGUF = "D:\shimmy-test-models\gguf_collection\TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf"
$env:SHIMMY_MAX_CTX = "3000"
$env:SHIMMY_ROPE_SCALE = "0.68"
.\target\release\shimmy.exe generate "tinyllama-1.1b" --prompt "hi" --max-tokens 20 2>&1 | Select-String -NotMatch "^\[Metadata\]|^\[Preflight\]|^\[DIAG\]|^\[ISF"
# Expected: a few words, no garbage, exits cleanly
```

## Build Commands
```powershell
# Airframe lib only (fast check)
cd C:\Users\micha\repos\airframe
cargo build --release --bin frontier_compare

# Full airframe release
cargo build --release

# Shimmy (after airframe changes)
cd C:\Users\micha\repos\shimmy
cargo build --release
```

## What Good Looks Like
| Test | Pass Threshold |
|------|---------------|
| TinyLlama frontier_compare layer 0 | MAE < 0.01 |
| Llama-3.2-1B layer 0 | MAE < 0.001 |
| Any model logits gpu_non_finite | 0 |
| Vault delta for any tested model | < 50% (Q4K quantization noise is normal) |

## Known Issues (do not investigate unless assigned)
- frontier_compare layer 2+ NaN on Llama-3.2: debug path only, production unaffected
- [DIAG]/[ISF-TDR] stderr noise: cosmetic, does not affect inference correctness
