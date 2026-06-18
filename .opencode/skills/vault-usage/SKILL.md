---
name: vault-usage
description: Golden reference vault for inference correctness — algebraic formula signatures, two-source verification (CPU + Candle), vault-driven debugging.
---

# Vault Usage — Algebraic Formula Verification System

The vault (`vault/vault.duckdb`) is the ground truth. Every debugging session starts here, never with eyeballing raw output.

## Core Methodology (Analytical Algebra Approach)

**Do not dive into raw vectors or step-by-step arithmetic.** Use the algebraic formula approach:

1. Every layer output is compressed into **dimensionless ratios** (energies = std_dev, gains = ratios, qk_balance)
2. Compare GPU vs CPU golden at the **formula level**, not element-by-element
3. The first divergence point at the formula level IS the smoking gun
4. Fix only there, re-measure, confirm

## Schema (Full)

```sql
-- Models: one row per (name, quant) combination
models (id, name, arch, quant, n_layers, n_embd, ff_dim, n_vocab, ...)

-- Layer oracles: golden CPU traces per layer/operation/position
layer_oracles (id, model_id, layer_idx, operation, position, expected_rms, expected_nan, first20, checksum)

-- Inference formulas: algebraic signature fingerprints per (model, source, layer)
inference_formulas (id, model_id, source, layer_idx, position,
  output_energy, post_attn_energy, ffn_energy,
  residual_gain, ffn_gain, qk_balance, kv_mean_gap,
  has_nan, has_inf)

-- Formula comparisons: aggregate divergence scores (golden vs candidate)
formula_comparisons (id, model_id, golden_source, candidate_source,
  mean_layer_score, median_layer_score, max_layer_score,
  first_nan_layer, passed, threshold)

-- Verification runs: CI/CD integration
verification_runs (id, model_id, run_type, passed, rms_diff_avg, ...)

-- Cross-validations: vault CPU oracles vs Candle probe
cross_validations (id, model_id, layer_idx, operation,
  airframe_rms, candle_rms, delta_rms, pass)
```

## Step 1: Find your model in the vault

```powershell
duckdb vault/vault.duckdb "SELECT id, name, quant, n_layers, COUNT(o.id) as oracles FROM models m LEFT JOIN layer_oracles o ON m.id = o.model_id GROUP BY m.id, m.name, m.quant, m.n_layers ORDER BY oracles DESC;"
```

## Step 2: Get layer oracles for a specific model

```powershell
# Replace pattern with your model name fragment
duckdb vault/vault.duckdb "SELECT o.layer_idx, o.operation, o.expected_rms, o.expected_nan, o.expected_inf FROM layer_oracles o JOIN models m ON o.model_id = m.id WHERE m.name LIKE '%TinyLlama%' ORDER BY o.layer_idx, o.operation;"
```

## Step 3: Run frontier_compare to get GPU trace

```powershell
# For TinyLlama Q6_K (or replace with your model)
cd C:\Users\micha\repos\airframe
.\target\release\frontier_compare.exe --model "D:\shimmy-test-models\gguf_collection\tinyllama-1.1b-chat-v1.0.Q6_K.gguf" --prompt "hi" --max-ctx 4096 --output artifacts\tinyllama_q6k_trace.json
```

## Step 4: Compare GPU layer outputs against vault golden (RMS comparison)

```powershell
python -c @"
import json, duckdb

# Get vault oracles
con = duckdb.connect('vault/vault.duckdb')
rows = con.execute("""
  SELECT o.layer_idx, o.expected_rms
  FROM layer_oracles o
  JOIN models m ON o.model_id = m.id
  WHERE m.name LIKE '%TinyLlama%' AND m.quant = 'q6_k'
  ORDER BY o.layer_idx
""").fetchall()
vault = {r[0]: r[1] for r in rows}

# Load GPU output
gpu = json.load(open('artifacts/tinyllama_q6k_trace.json'))

print('layer | vault_rms | gpu_rms   | diff%   | status')
for l in gpu['layers']:
    idx = l['layer_idx']
    v = vault.get(idx)
    g = l['output'].get('gpu_rms') or 0.0
    nf = l['output'].get('gpu_non_finite') or 0
    if v and v > 0:
        pct = abs(g-v)/v*100
        status = 'OK' if pct < 15 else ('WARN' if pct < 50 else 'FAIL')
        nf_flag = ' NaN!' if nf > 0 else ''
        print(f'  {idx:2d}  | {v:.5f}  | {g:.5f}  | {pct:.1f}%  {status}{nf_flag}')
    else:
        print(f'  {idx:2d}  | --no-vault-- | {g:.5f}')
"@
```

## Step 5: Formula-level analysis (algebraic signatures)

Use dimensionless ratios to find WHERE the divergence starts:

```powershell
python scripts/trace_formula_diff.py --candidate artifacts/tinyllama_q6k_trace.json --golden artifacts/tinyllama_q6k_cpu_golden.json --top 10
```

Key formula metrics:
- `residual_gain` = output_energy / post_attn_energy (should match golden)
- `ffn_gain` = ffn_energy / post_attn_energy
- `qk_balance` = std_dev(Q) / std_dev(K)
- `kv_mean_gap` = |mean(K) - mean(V)|

If metrics diverge by more than 2x log2-fold, that layer/operation is the root cause.

## Step 6: Read seed JSON directly (for first20 values and checksums)

```powershell
python -c @"
import json
seed = json.load(open('vault/seeds/tinyllama-1.1b-chat-v1.0.Q6_K.json'))
for o in seed['oracles'][:5]:
    print(f'layer={o[\"layer_idx\"]} op={o[\"operation\"]} rms={o[\"rms\"]} first4={o.get(\"first20\",[])[:4]}')
"@
```

## Step 7: Cross-validate against Candle (second golden source)

```powershell
python vault/scripts/vault_certify.py vault/vault.duckdb vault/seeds/candle
# Rows marked 'certified' = both sources agree
# Rows marked 'disputed' = sources disagree (needs investigation)
```

## Seed a new model into the vault

```powershell
cd C:\Users\micha\repos\airframe
cargo build --release --bin vault_seed
python vault/scripts/seed_all.py "D:\shimmy-test-models\gguf_collection"
python vault/scripts/import_seeds.py vault/seeds vault/vault.duckdb
duckdb vault/vault.duckdb "SELECT COUNT(*) FROM models; SELECT COUNT(*) FROM layer_oracles;"
```

## Diagnostic Pattern: First Bad Layer

When GPU output diverges, Vault isolates the exact first layer:

```powershell
python -c @"
import json
gpu = json.load(open('artifacts/trace.json'))
# From vault query
vault = {0: 0.045, 1: 0.055, 2: 0.077}
for l in gpu['layers']:
    idx = l['layer_idx']
    g = l['output'].get('gpu_rms') or 0.0
    nf = l['output']['gpu_non_finite']
    v = vault.get(idx, 0)
    if nf > 0 or (v and abs(g-v)/v > 0.5):
        print(f'FIRST BAD LAYER: {idx}  gpu_rms={g:.4f}  vault={v:.4f}  nf={nf}')
        # Inspect Q/K/V/FFN stats at this layer
        for component in ['q', 'k', 'v', 'post_attn', 'ffn_out']:
            s = l.get(component, {})
            print(f'  {component}: rms_diff={s.get(\"mean_abs_err\",\"?\")}  nf={s.get(\"gpu_non_finite\",\"?\")}')
        break
"@
```

## Known Vault Issues

- Some seed files have `"quant": "unknown"` — fix by editing the JSON directly before import
- layer_oracles only has `layer_output` for most models (no per-operation breakdown yet)
- Q6_K seed for TinyLlama has 22 layer_output oracles (layers 0-21)

## FSE/D0 Fact Emission Pattern

Every vault query or comparison emits structured facts:
- `VaultOracle { model_id, layer, expected_rms }`
- `LayerOutput { layer, gpu_rms, nan_count }`
- `FormulaDelta { layer, residual_gain_diff_log2, ffn_gain_diff_log2 }`

Rules derive: `FirstBadLayer`, `DivergenceSeverity`, `QkNormMismatch`
Consequents drive: targeted fix, re-verification, issue update.

For implementation, extend `airframe_observe` facts or use directly via Python/CLI.
