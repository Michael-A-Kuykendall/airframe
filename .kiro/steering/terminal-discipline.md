# Terminal & Process Discipline

## The Rule

Before starting ANY background process, call `list_processes` to see what is already running.
Never start a duplicate. Stop what you don't need immediately after it completes.

## Process Budget for This Repo

| Slot | Purpose | Start with | Stop when |
|------|---------|------------|-----------|
| downloads | HF model downloads | `control_pwsh_process start` | Download complete — stop immediately |
| seed_sweep | vault seed generation | `control_pwsh_process start` | Sweep complete — stop immediately |

Maximum background processes at any time: **2** (one download + one build/seed job).

## Cleanup Protocol

After ANY background process completes:
1. Call `get_process_output` to confirm completion and check for errors
2. Call `control_pwsh_process stop <terminalId>` immediately
3. Never leave finished processes running

## Before Starting a Background Process

1. `list_processes` — check what's running
2. If same command + same cwd already running → reuse it, don't start another
3. Stop any stale processes before starting a new one in the same slot

## Vault Seed Sweep Checklist

```
[ ] list_processes → confirm no seed sweep already running
[ ] vault_seed.exe built with cargo build --release --bin vault_seed
[ ] seeds dir exists at vault/seeds/
[ ] Run: python vault/scripts/seed_all.py "D:/shimmy-test-models/gguf_collection"
[ ] Wait for completion, check output
[ ] stop the process terminal
[ ] Run: python vault/scripts/import_seeds.py vault/seeds vault/vault.duckdb
[ ] Verify: duckdb vault/vault.duckdb "SELECT COUNT(*) FROM models; SELECT COUNT(*) FROM layer_oracles;"
```

## Warning Signs

| Symptom | Cause | Fix |
|---------|-------|-----|
| Multiple terminals with same command | Duplicate processes | Stop all but one |
| Process output shows 0 seeds | Wrong gguf_dir path | Check path, re-run |
| Import fails NOT NULL | Schema uses INTEGER PK without sequence | Use nextval subquery in INSERT |
| Seed fails with WeightMissing | Arch not fully supported | Partial seed OK — model metadata captured |
