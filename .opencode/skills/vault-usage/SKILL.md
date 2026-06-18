---
name: vault-usage
description: Using the golden reference vault (vault/vault.duckdb) to verify inference correctness against stored oracles.
---

# Vault Usage Skill

The vault (`vault/vault.duckdb`) is the ground truth. Every debugging session should start here, not with eyeballing output.

## Schema
```sql
models        (id, name, quant, n_layers, gguf_path, file_size_bytes)
layer_oracles (id, model_id, layer_idx, operation, position, expected_rms, expected_nan, first20, checksum)
verification_runs  -- currently empty, should record every test run
```

## Step 1: Find your model in the vault
```powershell
duckdb vault/vault.duckdb "SELECT id, name, quant, n_layers, COUNT(o.id) as oracles FROM models m LEFT JOIN layer_oracles o ON m.id = o.model_id GROUP BY m.id, m.name, m.quant, m.n_layers ORDER BY oracles DESC;"
```

## Step 2: Get layer oracles for a specific model
```powershell
# Replace '%Llama%1B%' with your model pattern
duckdb vault/vault.duckdb "SELECT o.layer_idx, o.operation, o.expected_rms, o.expected_nan FROM layer_oracles o JOIN models m ON o.model_id = m.id WHERE m.name LIKE '%Llama%1B%' ORDER BY o.layer_idx, o.operation;"
```

## Step 3: Run frontier_compare to get GPU values
```powershell
cd C:\Users\micha\repos\airframe
.\target\release\frontier_compare.exe --model "D:\shimmy-test-models\gguf_collection\<model>.gguf" --prompt "hi" --max-ctx 4096 --output artifacts\<model>_trace.json
```

## Step 4: Compare GPU output against vault
```powershell
python -c @"
import json, duckdb

# Get vault oracles
con = duckdb.connect('vault/vault.duckdb')
rows = con.execute("SELECT layer_idx, expected_rms FROM layer_oracles o JOIN models m ON o.model_id = m.id WHERE m.name LIKE '%YourModel%' ORDER BY layer_idx").fetchall()
vault = {r[0]: r[1] for r in rows}

# Load GPU output
gpu = json.load(open('artifacts/yourmodel_trace.json'))

print('layer | vault_rms | gpu_rms   | diff%')
for l in gpu['layers']:
    idx = l['layer_idx']
    v = vault.get(idx)
    g = l['output'].get('gpu_rms') or 0.0
    if v:
        pct = abs(g-v)/v*100
        status = 'OK' if pct < 15 else ('WARN' if pct < 50 else 'FAIL')
        print(f'  {idx:2d}  | {v:.5f}  | {g:.5f}  | {pct:.1f}%  {status}')
"@
```

## Step 5: Read seed JSON directly (when vault query insufficient)
Seed JSONs at `vault/seeds/<model>.json` have `rms` and `first20` values per oracle:
```powershell
python -c @"
import json
seed = json.load(open('vault/seeds/Llama-3.2-1B-Instruct-Q4_K_M.json'))
for o in seed['oracles'][:5]:
    print(f'layer={o[\"layer_idx\"]} op={o[\"operation\"]} rms={o[\"rms\"]} first4={o.get(\"first20\",[])[:4]}')
"@
```

## Seed a new model into the vault
```powershell
# Build vault_seed
cd C:\Users\micha\repos\airframe
cargo build --release --bin vault_seed

# Generate seeds (runs CPU forward pass)
python vault/scripts/seed_all.py "D:\shimmy-test-models\gguf_collection"

# Import into vault DB
python vault/scripts/import_seeds.py vault/seeds vault/vault.duckdb

# Verify
duckdb vault/vault.duckdb "SELECT COUNT(*) FROM models; SELECT COUNT(*) FROM layer_oracles;"
```

## Diagnostic pattern: first bad layer
When GPU output is wrong, use vault to find the exact first layer that diverges:
```powershell
python -c @"
import json
gpu = json.load(open('artifacts/trace.json'))
vault_rms = {0: 0.045, 1: 0.055}  # from vault query
for l in gpu['layers']:
    idx = l['layer_idx']
    g = l['output'].get('gpu_rms') or 0.0
    nf = l['output']['gpu_non_finite']
    v = vault_rms.get(idx, 0)
    if nf > 0 or (v and abs(g-v)/v > 0.5):
        print(f'FIRST BAD LAYER: {idx}  gpu_rms={g:.4f}  vault={v:.4f}  nf={nf}')
        break
"@
```
