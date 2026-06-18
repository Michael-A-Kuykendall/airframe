---
name: inference-testing
description: Airframe GPU inference testing — smoke tests, vault-driven verification, formula comparison, algebraic debugging.
---

# Inference Testing Skill

## CRITICAL: Vault-First Debugging

**Never dive into raw vectors or step-by-step arithmetic.** Every debugging session MUST start with:

1. Query the vault for the model's golden oracle data
2. Run frontier_compare to get GPU trace
3. Compare at the **formula level** (RMS, energies, gains)
4. Find the first divergence → that is the only thing to fix
5. Re-measure, confirm improvement, done

## Available Scripts

| Script | Purpose |
|--------|---------|
| `scripts/trace_formula_diff.py` | Compare two traces using log2-fold on algebraic signatures |
| `scripts/llama_formula_side_by_side.py` | Arch-specific formula against llama.cpp canonical equations |
| `scripts/template_formula_alignment.py` | Template formula alignment |
| `scripts/phi_smoke_formula.sh` | Phi formula smoke test |
| `scripts/prompt_mode_formula_probe.sh` | Prompt mode formula probe |

## One-Liner Smoke Tests

### TinyLlama Q4_0 (baseline — must always pass)
```powershell
cd C:\Users\micha\repos\airframe
.\target\release\frontier_compare.exe --model "D:\shimmy-test-models\gguf_collection\TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf" --prompt "hi" --max-ctx 2048 --output artifacts\tinyllama_smoke.json
```

### TinyLlama Q6_K (test target for output head Q6_K fix)
```powershell
.\target\release\frontier_compare.exe --model "D:\shimmy-test-models\gguf_collection\tinyllama-1.1b-chat-v1.0.Q6_K.gguf" --prompt "hi" --max-ctx 2048 --output artifacts\tinyllama_q6k_trace.json
```

### Read smoke results (check for first bad layer)
```powershell
python -c @"
import json
d=json.load(open('artifacts/tinyllama_q6k_trace.json'))
print('layer | layer_mae | logits_nf')
print('-'*40)
for l in d.get('layers',[]):
    o=l.get('output',{}); print(f'{l[\"layer_idx\"]:5d} | {o.get(\"mean_abs_err\",0):.6f} | {o.get(\"gpu_non_finite\",0)}')
print(f'logits gpu_non_finite={d.get(\"logits\",{}).get(\"gpu_non_finite\",\"?\")}')
"@
```

## Vault-Driven Comparison Workflow

### 1. Query vault for golden oracles
```powershell
duckdb vault/vault.duckdb "SELECT o.layer_idx, o.expected_rms, o.expected_nan FROM layer_oracles o JOIN models m ON o.model_id = m.id WHERE m.name LIKE '%TinyLlama%' AND m.quant = 'q6_k' ORDER BY o.layer_idx;"
```

### 2. Compare GPU output vs vault per layer
```powershell
python -c @"
import json, duckdb
con = duckdb.connect('vault/vault.duckdb')
rows = con.execute(\"\"\"
  SELECT o.layer_idx, o.expected_rms FROM layer_oracles o
  JOIN models m ON o.model_id = m.id
  WHERE m.name LIKE '%TinyLlama%' AND m.quant = 'q6_k'
  ORDER BY o.layer_idx
\"\"\").fetchall()
vault = {r[0]: r[1] for r in rows}
gpu = json.load(open('artifacts/tinyllama_q6k_trace.json'))
print('layer | vault_rms | gpu_rms | diff%')
for l in gpu['layers']:
    idx=l['layer_idx']; v=vault.get(idx); g=l['output'].get('gpu_rms',0)
    if v: print(f'{idx:4d} | {v:.5f} | {g:.5f} | {abs(g-v)/v*100:.1f}%')
"@
```

### 3. Formula diff (algebraic signatures)
```powershell
python scripts/trace_formula_diff.py --candidate artifacts/tinyllama_q6k_trace.json --golden artifacts/tinyllama_q4k_cpu_golden.json --top 10
```

### 4. Cross-validate against Candle
```powershell
python vault/scripts/vault_certify.py vault/vault.duckdb vault/seeds/candle
```

## Build Commands

```powershell
cd C:\Users\micha\repos\airframe
cargo build --release --bin frontier_compare   # Fast: trace binary only
cargo build --release                          # Full release
```

## Pass Thresholds

| Test | Threshold |
|------|-----------|
| TinyLlama layer 0 output MAE | < 0.01 |
| Llama-3.2-1B layer 0 MAE | < 0.001 |
| Any model logits gpu_non_finite | 0 |
| Vault RMS delta | < 50% (Q4K quantization noise expected) |
| Formula log2-fold divergence | < 2.0 mean |

## Known Issues (Do Not Investigate Unless Assigned)

- frontier_compare layer 2+ NaN on Llama-3.2: debug path only (airframe-mbc)
- [DIAG]/[ISF-TDR] stderr noise: cosmetic (airframe-6ex)
- Sh_layer_q4k.wgsl was deleted 2026-06-17 — do not recreate
