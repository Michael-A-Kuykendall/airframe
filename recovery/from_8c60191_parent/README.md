# Recovered tools (parent of purge `8c60191`)

Extracted for the **Family Onboarding Factory**. These files are **not** compiled by default.

| File | Origin | Reuse plan |
|------|--------|------------|
| `frontier_compare.rs` | `git show 8c60191^:src/bin/frontier_compare.rs` | **Bead F1:** GPU-only cut → per-tensor + residual dump; L2/L3 columns = candle/llama JSON, not vault |
| `vault_seed.rs` | `git show 8c60191^:src/bin/vault_seed.rs` | Schema/shape reference only; **not** cert authority |
| `FILE_LIST.txt` | family + ops/reference paths | Full Airframe-CPU restore only if explicitly required |

## Do not

- Restore vault DuckDB as certification authority.  
- Compile frontier as-is without either (a) restoring CPU stack + fixing Qwen head_dim, or (b) stripping CPU deps.

## Restore cost if full CPU path wanted

See `FILE_LIST.txt` + `FAMILY_FACTORY.md` §4. Roughly 3.5k+ LOC under `src/family` and `src/ops/reference`.
