---
name: airframe-vault
description: Use when working with the Airframe golden vault (DuckDB), the airframe_observe observability crate, per-layer oracle/invariant comparison, or certifying GPU inference against CPU/candle references. Covers vault.duckdb schema, the ObservationSession fact/observer API, the seed/verify/certify scripts, and the inspect→reference→probe→compare→certify methodology. Reach for this when the task is "what does the vault say", "how do I compare layers", "how do facts flow", or "how do I certify a model" — as opposed to airframe-inference which answers "run tool X on model Y".
---

# Airframe Golden Vault & Observability Methodology

This is the **data workbench** skill. It explains where ground truth lives
(`vault.duckdb`), how runtime captures are produced (`airframe_observe`),
and the repeatable loop for proving a GPU inference change is correct. It pairs
with the `airframe-inference` skill (which lists the tool invocations).

## 0. Where the database actually lives (read this first)

- **Canonical path:** `airframe/vault/vault.duckdb`.
- In a fresh checkout this path is a **133-byte git-LFS pointer**, not the real
  DB. The real 11,284,480-byte object is cached by git-LFS at:
  `C:/Users/micha/repos/airframe/.git/lfs/objects/36/e7/36e7beaf0ee87887ebe508465de72d8d9ceaaefcd8097b8c1805a8fa6e373359`
- **Materialize for local work** (do NOT commit the 11 MB blob — keep the LFS
  pointer in git):
  ```bash
  cp "C:/Users/micha/repos/airframe/.git/lfs/objects/36/e7/36e7beaf0ee87887ebe508465de72d8d9ceaaefcd8097b8c1805a8fa6e373359" airframe/vault/vault.duckdb
  ```
- **Query it** with DuckDB 1.4.4 (installed for Python) or the `duckdb` CLI:
  ```bash
  python -c "import duckdb; c=duckdb.connect('airframe/vault/vault.duckdb', read_only=True); print(c.execute('SELECT id,name,arch,quant,n_layers FROM models ORDER BY id').fetchall())"
  ```
- A stray empty/garbage file named like an absolute path
  (`Usersmicharepos...vault.duckdb`) at the workspace root is an orphaned
  *empty* DuckDB — it has zero tables and must be deleted, not used.

## 1. Schema (11 tables)

| Table | Rows* | Purpose |
|-------|------|---------|
| `models` | 22 | One row per `(name, quant)`. Columns: `arch, quant, n_layers, n_heads, n_heads_kv, head_dim, n_embd, ff_dim, n_vocab, n_ctx, rope_base, rope_scale, rope_dim, rms_eps, has_qk_norm, attn_logit_softcap, final_logit_softcap, expert_count, gguf_path, file_size, file_sha256, oracle_git_commit, notes`. |
| `layer_oracles` | 322 | **Golden per-layer traces.** `layer_idx` (INT, -1 = embedding), `operation` ('layer_output'\|'final_logits'), `position`, `expected_rms`, `expected_max`, `expected_nan`, `expected_inf`, `checksum BIGINT`, `cpu_blob_path`, `cpu_blob_hash`. Unique `(model_id,layer_idx,operation,position)`. |
| `inference_formulas` | 176 | Quantization-stable per-layer invariants. `source` ('airframe_cpu'\|'airframe_gpu'\|'llama_cpp'\|'candle'), `layer_idx`, `output_energy, post_attn_energy, ffn_energy, residual_gain, ffn_gain, qk_balance, kv_mean_gap`, `has_nan, has_inf`. Survive precision differences — use for cross-backend comparison. |
| `formula_comparisons` | 4 | `golden_source` vs `candidate_source`, `mean_layer_score, median_layer_score, max_layer_score`, `threshold` (DEFAULT **2.0**), `passed`. |
| `layer_diags` | 0** | Per-layer tensor offsets/quants/batch_count from a `frontier_compare` trace. |
| `tensor_metadata` | 0** | Per-tensor dims/offsets. |
| `verification_runs` | 0 | CI run log: `passed, rms_diff_avg/max, nan_count, first_fail_layer`. |
| `sync_log` | 0 | LFS sync bookkeeping. |
| `vault_config` | 5 | `sync_mode=git_lfs`, `conflict_resolution=local_wins_for_oracles`, `rms_threshold=1e-5`, `max_abs_threshold=1e-4`, `ci_backend=airframe_gpu`. |
| `schema_version` | 3 | v1 base + v2 (`inference_formulas`/`formula_comparisons`/`temp_buffer_audit`) + v3 (`layer_diags`). |
| `temp_buffer_audit` | 22 (VIEW) | Flags `temp_buffer_size` underallocation: `correct = n_embd + n_heads*head_dim + n_heads_kv*head_dim*2 + ff_dim*2`. Query this when GPU output is NaN/garbage. |
| `cross_validations` | runtime | Created by `vault_certify.py` (NOT in schema.sql). `airframe_rms, candle_rms, delta_rms, pass`. |

\* counts as of materialization; ** empty = no trace imported yet for that model.

## 2. Read-to-run SQL recipes

```sql
-- All models
SELECT id, name, arch, quant, n_layers, head_dim, n_embd FROM models ORDER BY id;

-- Golden layer_output oracles for model <ID> (position >= 0)
SELECT layer_idx, position, expected_rms, checksum
FROM layer_oracles
WHERE model_id = <ID> AND operation='layer_output' AND layer_idx >= 0
ORDER BY layer_idx;

-- Quantization-stable invariants (GPU source, for a model)
SELECT layer_idx, output_energy, residual_gain, ffn_gain, qk_balance, has_nan
FROM inference_formulas
WHERE model_id = <ID> AND source='airframe_gpu' ORDER BY layer_idx;

-- Certification / pass status
SELECT m.name, fc.passed, fc.mean_layer_score, fc.max_layer_score, fc.threshold
FROM formula_comparisons fc JOIN models m ON fc.model_id=m.id
WHERE fc.golden_source='frontier_cpu' AND fc.candidate_source='frontier_gpu';

-- Candle dispute report (rows that fail candle certify)
SELECT m.name, cv.layer_idx, cv.operation, cv.airframe_rms, cv.candle_rms, cv.delta_rms
FROM cross_validations cv JOIN models m ON cv.model_id=m.id
WHERE cv.pass=false ORDER BY cv.delta_rms DESC;

-- Temp-buffer underallocation (common NaN root cause)
SELECT name, n_embd, (n_embd+n_heads*head_dim+n_heads_kv*head_dim*2+ff_dim*2) AS correct_temp
FROM temp_buffer_audit;
```

## 3. The `airframe_observe` crate (capture side)

Crate root: `airframe/crates/airframe_observe/src/`. The live system is the
**dzero-backed** implementation in `observers.rs` wired through `session.rs`.
(The simple struct observers in `observer.rs` are re-exported but NOT wired in
— ignore them for methodology.)

### 3.1 Facts (`facts.rs`) — the alpha-key dispatch
`InferenceFact` variants each map to a numeric `AlphaKey` via `alpha_key_of()`
(dzero routes a fact to every rule registered on that key in one lookup).
Key facts:

| Fact | Alpha key | Fields | Meaning |
|-------|-----------|---------|---------|
| `LayerOutput` | 1 | `layer_idx, position, rms_bits:u32, checksum:i64` | Hidden state after layer N at pos P. |
| `FinalLogits` | 2 | `position, rms_bits, checksum` | Final logits (candle compare point). |
| `PerTensorOutput` | 9 | `layer_idx, position` + 6×`rms_bits`/`checksum` for `q,k,v,post,ffn,output` | Per-kernel stats — use to find the broken sub-kernel. |
| `OutputToken` | 3 | `step, token_id` | Decoded token. |
| `DispatchTiming` | 10 | `layer, kernel:KernelKind, elapsed_ms` | Kernel timing (TDR input). |
| `EmbeddingRequest` | 19 | `token_id` | FSE dedups once per unique token. |
| `PromptToken`(12), `PrefillComplete`(15), `DecodeStep`(16) | — | ISF-loop signals | — |

Helpers: `rms(&[f32])->f32`, `checksum(&[f32])->i64`, `f32_to_bits`/`bits_to_f32`.

### 3.2 `ObservationSession` (`session.rs`) — the API
Build once, emit per layer, `saturate()`, then `drain()`.

| Method | Signature | Notes |
|--------|-----------|-------|
| `new()` | `-> Self` | builds the reactive graph |
| `register_vault_oracle()` | `-> &VaultOracleObserver` | rule on key 1 → accumulates `OracleCapture{layer_idx,position,rms,checksum}` |
| `register_candle_compare()` | `-> &CandleCompareObserver` | rule on key 2 → `LogitCapture{position,rms,checksum}` |
| `emit_layer_output(l, p, &[f32])` | computes rms+checksum, fires key 1 | |
| `emit_final_logits(p, &[f32])` | fires key 2 | |
| `emit_per_tensor_output(l, p, &q,&k,&v,&post,&ffn,&output)` | 8 args, fires key 9 | |
| `saturate()` | `-> RunResult` | runs graph to fixpoint (observers accumulate) |
| `vault_oracle().drain()` / `candle_compare().take()` | pull captures AFTER saturate | |
| `register_tdr_scheduler(budget_ms)` / `should_yield()` / `reset_tdr()` | TDR yield control | |

`LayerStabilityObserver` (also key 1) independently derives `LayerOutputStable`
if `0 < rms < 1000 && !nan && !inf`. A single `LayerOutput` fact thus feeds
both the oracle capture and the stability check.

### 3.3 The GPU capture sink (what the probes ACTUALLY use)

`airframe_observe` exposes BOTH a rich dzero `ObservationSession` (§3.1–3.2)
AND a lightweight global-sink capture used by the GPU probes. **The GPU
probes do NOT call `emit_per_tensor_output` on the session** — they push into
the simpler sinks below, gated by `feature = "isf"` + env var. Do not go
looking for `ObservationSession` calls in `gpu.rs`/`inference.rs`; the
capture path is:

- `inference.rs` (gated `#[cfg(feature="isf")]`):
  - `CapturedLayer` sink — `set_invariant_capture_sink` / `clear_*` /
    `invariant_capture_sink_mut`. `emit_layer_capture(...)` reads back the
    post-layer activation and pushes `{layer_idx, position, rms, checksum}`.
    **This is wired** (called from the layer loop in `run_full_model_with_cache_state`).
  - `CapturedPerTensor` sink — `set_invariant_ptensor_capture_sink` /
    `clear_*` / `invariant_ptensor_capture_sink_mut`. `emit_ptensor_capture(...)`
    pushes `{layer_idx, position, q/k/v/post/ffn/output rms+checksum}`.
    **This is wired into `run_layer_with_cache_debug`** (which already
    readbacks q/k/v/post-attn/ffn/output — zero extra GPU work).
- Gate: both are no-ops unless `AIRFRAME_CAPTURE_INVARIANT=1` AND a sink is
  installed. `invariant_probe` sets the env + installs both sinks;
  `frontier_compare` (built with `--features isf`) installs the ptensor sink
  and now serializes it as `captured_per_tensor` in its JSON output.
- **Where per-tensor comes from:** the *production* forward
  (`run_layer_with_cache`) merges q/k/v/post/ffn into one encoder and does
  NOT keep them separate, so per-tensor is only available via the **debug
  layer path** (`run_layer_with_cache_debug`, used by `frontier_compare`).
  `invariant_probe`'s production `generate_isf` therefore emits
  layer-output + final-logits; its `per_tensor` field is populated only when
  a debug forward drives it (follow-up). For per-kernel localization, run
  `frontier_compare` — that is the tool that already separates q/k/v/post/ffn/output.

### 3.4 Env / feature gating
- `AIRFRAME_CAPTURE_INVARIANT=1` — `invariant_probe` sets this internally to
  install a global sink (`set_invariant_capture_sink`). Capture is gated so it
  is a no-op in normal inference.
- `SHIMMY_TDR_BUDGET_MS` — override TDR budget. Default **1400 ms Windows,
  30000 ms elsewhere** (`isf.rs`).
- `VAULT_DB=<path>` — vault location for certify/verify scripts (default
  `vault/vault.duckdb`).

## 4. The toolchain → who writes/reads the vault

| Binary / script | Produces | Lands in | Drives |
|------|-----------|----------|--------|
| `vault_seed <gguf> [out.json]` | per-layer CPU RMS + first20 + final logits seed JSON | `vault/seeds/<stem>.json` | `import_seeds.py` → `models` + `layer_oracles` |
| `seed_all.py [gguf_dir]` | runs `vault_seed` over a model dir, then imports | `vault.duckdb` | bulk population |
| `import_seeds.py` | idempotent upsert of seeds | `models`,`layer_oracles` | — |
| `invariant_probe <gguf> <name>` | GPU `LayerOutput`+`FinalLogits` (RMS/checksum) as JSON/captures | stdout / `CapturedLayer` | PPT cage capture side |
| `layer_dump_gpu <gguf> "<prompt>" <out.json>` | full GPU hidden states per layer | JSON | algebraic compare |
| `frontier_compare --model --prompt --output` | per-tensor Q/K/V/post/ffn/output CPU-vs-GPU + final logits top-k | trace JSON | `vault_verify.py` → `inference_formulas`,`formula_comparisons`,`layer_diags` |
| `vault_verify.py [--trace|--model|--run]` | reads trace JSON, matches vault, computes log2-fold divergences | `inference_formulas`,`formula_comparisons` | certification (`LOG2_FOLD_FAIL_THRESHOLD=2.0`) |
| `vault_certify.py` | compares `layer_oracles` vs candle seeds | `cross_validations` | tolerated `layer_output: Δrms<0.01`, `final_logits: Δrms<1.0` |
| `quant_verify --model-path <gguf>` | dequant max/mean abs error per quant type | stdout, exit 0/1 | dequant sanity (tol 1e-2) |
| `tests/test_invariants.rs` | CI cage | asserts GPU capture vs `layer_oracles` | **MUST run `--test-threads=1`** (global invariant log) |

## 5. The methodology (the loop)

Repeat for every model/change under test:

1. **Inspect.** `vault_seed <gguf> /dev/null 2>&1 | head -30` — confirm arch/quant/n_layers; check size vs VRAM (RTX 3060 = 12 GB; >2 GB needs multi-buffer; >6 GB hits the 3-blob cap).
2. **CPU reference (ground truth).** `vault_seed <gguf> cpu.json` → import via `import_seeds.py` if the model isn't in `models` yet. This populates `layer_oracles`.
3. **GPU probe.** `invariant_probe <gguf> <name>` (or `layer_dump_gpu`) → capture per-layer RMS/checksum. The first divergent layer localizes the bug.
4. **Per-tensor pin (only if step 3 diverges).** `frontier_compare` → `inference_formulas`/`layer_diags`. Use `PerTensorOutput` (key 9) to find the exact broken kernel (Q/K/V / attn-out / ffn).
5. **Certify.** `vault_verify.py` (log2-fold ≤ 2.0 ⇒ pass) and/or `vault_certify.py` (candle cross-check). Then `cargo test -p airframe --test test_invariants -- --test-threads=1`.
6. **End-to-end.** `shimmy generate <name> --prompt "Hello" --max-tokens 32` to confirm coherent text.

Expected behavior when the GPU path is currently broken: step 3/5 **will FAIL on
the first divergent layer** — that is the intended localization, not a harness
defect.

## 6. Gotchas (burned before — do not re-litigate blindly)

- **Golden fixture hazard.** For Qwen3 "Hello" = token **9707 with NO BOS**.
  `vault_seed`/`invariant_probe` HARDCODE `hello_id=15043` + BOS — so CPU
  `oracles[i]` ↔ GPU `layers[i+1]` is an **off-by-one**. GPU `layers[0]` =
  embedding (no CPU counterpart); CPU `oracles[-1]` = final logits (RMS 4.78,
  not comparable to GPU last hidden state). Always use the identical tokenizer
  fixture across tools or the comparison is invalid.
- **LFS pointer vs materialized DB** (§0). Querying the 133-byte pointer yields
  empty tables — materialize first.
- **Two observer APIs.** Only `observers.rs` (dzero-backed) is wired; the
  `observer.rs` structs are dead ends. Document/use the former.
- **`cross_validations` is runtime-created** by `vault_certify.py`, not in
  `schema.sql` — include it in queries but don't expect it from a fresh import.
- **`temp_buffer_audit`** is the first thing to check on NaN/garbage GPU output.
- **`layer_dump_gpu` still hardcodes `batch_count: 0`** (unlike `frontier_compare`
  which uses `1`) — this kills all QKV invocations on the clean v0.2.8 path.
- **TDR budget** differs by OS; set `SHIMMY_TDR_BUDGET_MS` if long kernels time out.
- **Write outputs to a file and read verbatim** — never pipe GPU test output
  through `grep|head|tail` (silently masks failures).
- **Kill stale GPU procs** before re-running: `taskkill //f //im shimmy_server_gpu.exe`
  (double-slash in MSYS — a single `/` is mangled into a path).

## 7. When to use which skill

- **airframe-inference** — "run `vault_seed`/`layer_dump_gpu`/`frontier_compare`
  on model X", env vars, quick tool table, end-to-end certify command.
- **airframe-vault (this one)** — "what does the vault say about model X", "how
  do I compare layers / find the broken kernel", "how do `airframe_observe`
  facts flow into the DB", "how do I certify". The data + methodology layer that
  sits underneath the tool invocations.
