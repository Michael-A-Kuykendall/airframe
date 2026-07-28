# Cert Regimen (authoritative)

**Goal:** One command → full **RED ledger** of math/plan vs product peel.  
**Not a goal:** Match llama/candle as the pass bar. External engines are optional flashlights only.

## Two surfaces

| Surface | Source | Role |
|---------|--------|------|
| **PLAN** | GGUF-derived config + family invariants + `quant_formula` | Law |
| **PEEL** | Product `stack_dump_gpu` / quant_verify GPU | Truth under test |

**RED** = PLAN ≠ PEEL, or PLAN incomplete, or dequant fails vs `quant_formula`.

## Two checkboxes per family (both required for “certified”)

| Box | Meaning | Evidence |
|-----|---------|----------|
| **MATH** | Red ledger empty (or only waived with written reason) | `reds.json` + `REPORT.md` |
| **CHAT** | Multi-prompt generate coherent (not one-word only) | `chat_smoke.log` |

Neither substitutes for the other.

## One command

```bat
scripts\certify_math.bat <family-id> <gguf> ["multi-token prompt"]
```

Produces:

```text
cert/packages/<family-id>/
  plan.json          # derived plan (counts, slots, rope, dims)
  peel.json          # copy/link of airframe.stack.v1 peel
  quant_verify.log
  reds.json          # only failures + summary
  REPORT.md          # human table
  chat_smoke.log     # optional second phase
```

Exit code: `0` only if MATH box green (zero unwaived REDs).

## PLAN rules (fastidious — this is where 4 vs 6 lives)

From stack/config (GGUF → ModelSpec → dump `config`):

| Field | Rule |
|-------|------|
| `dim_q` | `n_head * head_dim` |
| `dim_kv` | `n_kv_head * head_dim` |
| `ffn_dim` | from config / GGUF |
| `norm_slots` | **6 if `qk_norm` else 4** |
| Stage counts | `q`/`q_norm`=`dim_q`, `k`/`v`/`k_norm`=`dim_kv`, `attn_ctx`=`dim_q`, `ffn_gate`/`ffn_up`=`ffn_dim`, residuals=`n_embd` |
| `rope_base` | recorded; Qwen family expected `1e6` (warn if arch qwen* and base≈1e4) |
| head_dim | must be consistent: `n_embd` may **≠** `dim_q` (Qwen3); never force `head_dim = n_embd/n_head` if dump says otherwise |

## PEEL checks (per layer, required stages)

For every layer in peel:

- each required stage present with `sampled == "real"`
- `count` matches PLAN
- `nan_count == 0`
- residual rms finite

Final logits: `nan_count == 0`, `len > 0`.

## Dequant checks

Parse `quant_verify` log:

- each present type must PASS vs `quant_formula`
- FAIL → `RED DEQUANT.<type>` with max_err

## Ledger (tracking)

`cert/ledger.duckdb` (or `cert/ledger.sqlite` fallback):

- `family_runs(family_id, ts, git_sha, math_ok, chat_ok, n_reds, report_path)`
- `reds(run_id, code, detail)`

Check off MATH/CHAT when both true for latest run.

## External engines

Optional column only. **No GREEN solely because another engine matched.**

## Regression tests (no GPU)

```bat
python scripts/cert_reds_test.py
```

Fixtures under `scripts/fixtures/cert/` prove:

- wrong `q.count` → RED
- missing stage → RED  
- correct peel → peel section green
- `qk_norm` → plan.norm_slots == 6

## Order of work

1. Lock this machine + tests (this doc + scripts).  
2. Qwen3: run math cert → fix REDs → chat smoke → check both boxes.  
3. Replay machine on every claimed family; ledger checkoffs.
