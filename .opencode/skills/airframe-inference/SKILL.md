---
name: airframe-inference
description: "Use BEFORE ANY inference / model-output / decode / gibberish / probe / localize / certify / onboarding work on Airframe/Shimmy. Covers the PRESCRIBED Layer-1 tools (layer_dump_gpu, invariant_probe, quant_verify, shimmy generate) and the family factory (certify_family.sh). Math authority: quant_formula (GGUF spec) — never hand-rolled. FORBIDS custom probe binaries and hand-parsing GGUF. New quant = registry path (quant_formula -> FormulaSlot -> shader dispatch); new family = factory first. Triggers: 'qwen', 'debug model', 'gibberish', 'decode collapse', 'localize', 'probe', 'verify', 'certify', 'shimmy generate', 'forward pass', 'kv cache', 'rope', 'new model', 'new family', 'new quant', 'onboard', 'factory'."
---

# Airframe Inference Pathway — MATH-FIRST REGIME

> **Canonical copy for this sandbox:**
> `airframe-workspace/.opencode/skills/airframe-inference/SKILL.md`
> Nested copies under `airframe/.opencode/skills/` or `target/package/` may be **stale**.
> If they disagree with this file or workspace `AGENTS.md`, **this file + AGENTS win**.

Load **`airframe-discipline` first**, then this skill, before any inference work.

## THE GOLDEN RULE — DO NOT HAND-ROLL

**Never** hand-parse GGUF bytes, guess tensor offsets, or read shader/Rust source line-by-line
to "find" a bug. That is the thrash pattern. The toolchain exists precisely so you never have to.

The correct process is **layer-by-layer mathematical comparison**:

1. Build the **PLAN** from the GGUF by family (`scripts/cert_plan.py`) — reads the GGUF
   header/config and prints the exact math facts needed (dim_q, dim_kv, ffn_dim,
   norm_slots 4/6, stage counts, rope_base, head_dim, norm_kind). See
   `airframe/docs/CERT_REGIMEN.md` (origin commit `11de0cf`). **This is the onboarding
   process — not llama.cpp, not candle, not a golden trace.**
2. Compute the **expected** value independently from the GGUF using `quant_formula` (the law).
3. Capture the **actual** value from the GPU using the prescribed binary (`layer_dump_gpu`,
   `invariant_probe`, `quant_verify`, `shimmy generate`) — this is the **PEEL**.
4. Diff **PLAN vs PEEL** (`scripts/cert_reds.py`). RED = PLAN != PEEL. The first
   layer/operation that diverges is your bug.
5. Only if the math itself is elusive: read the pinned llama.cpp **source** for that model's
   family as a structural flashlight (never a pass bar).

**For bring-up of a brand-new family (e.g. gemma-4):** there is no golden trace to compare
against — that is NORMAL and by design. The vault is break-fix only, NEVER a bring-up
authority. You are the first person to do this. Build the PLAN from the GGUF, validate every
operator element-by-element against `quant_formula` (the law), and diff PLAN vs PEEL into a
RED ledger. That is the job. It is math, not archaeology, and it needs no other engine.

**llama.cpp / candle are OPTIONAL backups only** — used at the END to solidify a result, never
baked into the process. The predecessors' failure mode was becoming a llama.cpp/candle
wrapper. Airframe derives the math from the GGUF spec itself; that independence is the point.

---

## AUTHORITY HIERARCHY (never inverted)

| Priority | What | Role |
|----------|------|------|
| 1 | `airframe_observe::quant_formula` | **The law** — GGUF/GGML spec math. All dequant/RoPE/norm math is validated against it element-by-element. |
| 2 | `scripts/cert_plan.py` (CERT_REGIMEN, `airframe/docs/CERT_REGIMEN.md`) | **PLAN builder** — reads the GGUF by family, prints the math facts + stage counts + invariants. The onboarding process. |
| 3 | Prescribed Layer-1 binaries (`layer_dump_gpu`, `invariant_probe`, `quant_verify`, `shimmy generate`) | **PEEL capture** — actual GPU values per layer; diffed against PLAN. |
| 4 | `airframe/vault/vault.REAL.duckdb` + `seeds/` | **Break-fix ONLY** — regression oracle for models that have traces. **NOT populated for gemma-4.** NEVER a bring-up authority. |
| 5 | `candle_probe` (Layer 2) | Optional end-of-process second opinion. Not the authority, never baked in. |
| 6 | External engine / llama.cpp source (Layer 3) | Optional flashlight only — a structural hint, never a pass bar. **Prefer CERT_REGIMEN PLAN + quant_formula over reading it.** |

---

## LAYER 1 — INTERNAL (authoritative for development)

GPU-side tooling + spec math. This is the primary bar. **Run in this order.**

1. **`quant_verify -- --model-path <gguf>`** — dequant gate. Run FIRST if output is all-NaN
   (usually = wrong byte offsets, not a compute bug). Validates every weight tensor's dequant
   against `quant_formula`. ALL PASS required before compute work.

2. **`layer_dump_gpu <gguf> "<multi-token prompt>" out.json`** — Full per-layer hidden-state
   dump (RMS + FinalLogits). **Use a multi-token prompt** so cross-attention is exercised; a
   single token hides prefill bugs. RMS must grow monotonically embedding→output. The first
   layer where RMS spikes / goes non-monotonic is your divergence point. This is your primary
   localization tool — the JSON is diffed against expected values.

3. **`invariant_probe` with `AIRFRAME_CAPTURE_INVARIANT=1`** — RMS-ratio cage
   (`max(gpu,reference)/min ≤ 2.0` layers, `≤ 4.0` final_logits) against the probe's own
   captured reference.

4. **`AIRFRAME_TRACE_PREFILL_LAYERS=1`** on the server — per-layer NaN/values during a real
   prefill.

5. **`shimmy generate <gguf> --prompt "..." --max-tokens N`** — product end-to-end path
   (prefill + decode). The final correctness gate for a closed-domain prompt.

6. **Reference math = `airframe_observe::quant_formula`** — validate any shader-side math
   against it element-by-element in a test. **Do not reimplement it.**

### Layer 1 workflow for a RED gate (gibberish / NaN / wrong argmax)

```
quant_verify → ALL PASS?
  NO  → fix dequant/offset. STOP. do not touch compute.
  YES → layer_dump_gpu multi-token → first non-monotonic / NaN layer = N
         → invariant_probe → confirm cage
         → diff layer N against cert_plan (PLAN) + quant_formula → first diverge = RED
         → fix the operator, rebuild, re-run from top.
```

**At no point do you read shader source or hand-parse the GGUF.** The divergence is found by
comparing numbers the tools emit, then the fix is validated against the spec math.

---

## LAYER 2 — OPTIONAL BACKUP (independent CPU reference: `candle_probe`)

Optional end-of-process corroboration, never a gate:
`cargo run -p candle_probe -- <gguf>`.
It is a second opinion (not the authority) that cross-checks per-layer activations / logits
against the GPU path **after** the CERT_REGIMEN PLAN/PEEL ledger is green. **Keep it
independent of ALL oracles** (duckdb vault and invariant capture alike) — it must not share
their data path. Recreated 2026-07-26 after the earlier CPU-golden purge; do not delete it
again. Currently TinyLlama-only — SKIP with reason for unsupported arches.

---

## LAYER 3 — OPTIONAL FLASHLIGHT (external engine / llama.cpp source)

llama.cpp / another external engine is the LAST resort — only when the PLAN/PEEL math itself
stays elusive after quant_formula element checks. On this Linux machine llama.cpp is **not
installed** (the old `C:\llama.cpp` tree was deleted in the 2026-08-11 migration). If you must
look, **read the pinned source from GitHub** rather than install — most questions are "how does
the reference graph this operator," which the source answers without a build. A llama/candle
match is NEVER a GREEN source on its own; it is a hint at most.

---

## BREAK-FIX ORACLE (regression only): `airframe-vault`

`airframe/vault/vault.REAL.duckdb` (+ `seeds/*.json`) holds known-good per-layer traces for
models that were previously certified. **Use it ONLY to localize regressions** ("this model
worked last week, now it's broken — diff the live dump against the stored trace").

- **Populated for:** Llama-3.2 (1B/3B), Qwen3 (1.7B/8B), qwen2 (0.5B/1.5B/7B), TinyLlama.
- **NOT populated for gemma-4** (gemma-2 entries exist but have zero oracles).
- **NEVER** a bring-up authority. If the model isn't in the vault, you do independent math.

See the `airframe-vault` skill for the schema and populated list.

---

## llama.cpp SOURCE — OPTIONAL FLASHLIGHT (not a pass bar)

Airframe derives the math from the GGUF spec itself (`quant_formula` + CERT_REGIMEN PLAN);
llama.cpp source is never required. If the PLAN/PEEL diff stays elusive, the upstream graph is
a HINT for the operator structure:

- **gemma-4:** `llama.cpp src/models/gemma4.cpp` (pinned: `49f35421`). Read from GitHub via
  `webfetch`. Check: PLE (per-layer embedding) graph, proportional RoPE
  (`rope_freqs` as `theta / ff`, see `ggml_rope_cache_init`), dual RoPE base per layer,
  `f_attention_scale`, V plain-RMS-norm, FFN `LLM_FFN_GELU+PAR`.
- **General RoPE:** `ggml_rope_cache_init` in `ggml-cpu/ops.cpp` applies freq factors as
  `theta / ff` (DIVIDE), not multiply. If your table multiplies, that's the bug.
- Fetch with `webfetch`, never hand-transcribe. Pin the commit, name it in the test.
- **Remember:** matching llama.cpp/candle is NEVER a GREEN source. CERT_REGIMEN PLAN/PEEL +
  quant_formula is the bar.

---

## THE FULL CERTIFICATION REGIMEN — RUN THE FACTORY FIRST, EVERY MODEL, EVERY TIME

> **This is the ONLY way a model is certified.** There is no shortcut, no single-token
> argmax check, no "Paris says X so it's fine", no calling a model GREEN on one word.
> A model is certified **only** when `certify_family.sh` produces a cert package whose
> `STATUS.md` shows **every primary gate G0–G6 green**, PLUS the MATH box (0 reds in
> `reds.json`) **AND** the CHAT box (multi-prompt coherent output — not one word).
> If you are about to certify with anything less, STOP and run the factory.

### The machine (one picture)

```text
            certify_family.sh <family-id> <gguf> "<multi-token prompt>"
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                 ▼
 Layer 1 INTERNAL  Layer 2 SECONDARY  Layer 3 TERTIARY
 quant_verify       candle_probe       llama_ref_dump.py
 layer_dump_gpu     (independent CPU)  (llama.cpp — not on Linux yet)
 invariant_probe
 shimmy generate
     └─────────────────┼─────────────────┘
                       ▼
     cert/packages/<family-id>/  →  STATUS.md  (the ONLY green-light)
```

Authority: `airframe_observe::quant_formula` (GGUF/GGML spec math) is the law.
Candle/llama.cpp are **columns, not the law**. The vault is **break-fix only**.

### The runner (Linux)

```bash
cd /home/michael/repos/airframe-workspace
bash scripts/certify_family.sh <family-id> <path-to.gguf> "The capital of France is"
# env: SHIMMY_MAX_CTX=8192 (default), AIRFRAME_TRACE_PREFILL_LAYERS=1, FAMILY_MAX_TOKENS=24
```

**Required:** native Linux GPU. **llvmpipe / WSL is NOT a valid factory runner**
(FAMILY_FACTORY.md §5) — runs there are PRE-SCREENS ONLY, never a certification.

### Package layout (every model, always the same)

```text
cert/packages/<family-id>/
  00_meta.txt              # model path, arch, dims, head_dim, rope, qk_norm, date, git SHA
  01_quant_verify.log      # G1 dequant gate
  02_layer_dump.log        # G2 multi-token residual dump log
  02_layers.json           # layer_dump_gpu output
  03_invariant.log         # G3 production-path capture
  04_generate_trace.log    # G4 shimmy generate --raw (AIRFRAME_TRACE_PREFILL_LAYERS=1)
  04_generate_chat.log     # G4b shimmy generate (chat template / shimmyjinja)
  05_candle.json           # G5 L2 if candle_probe succeeds (else SKIP reason)
  06_llama_ref.log         # G6 L3 known-good continuation
  07_decode_chain.log      # G7 optional decode chain (ONLY after prefill GREEN)
  STATUS.md                # gate table + first red gate — the session handoff
```

### Gates (finite, ordered, stop on class, fix ONLY the first red)

| Gate | Tool | GREEN means | RED means | Next |
|------|------|-------------|-----------|------|
| **G0** | rebuild + env | airframe lib + bins + shimmy build; native GPU selected | build fail / llvmpipe | fix build/env; do not diagnose model |
| **G1** | `quant_verify` | ALL PASS | dequant/offset | fix metadata/quant; not "decode" |
| **G2** | `layer_dump_gpu` multi-token | logits finite, RMS monotonic, no FIRST_NAN | `FIRST_NAN_LAYER=N` | TRACE stage N only |
| **G2b** | stack dump vs llama reference | `stack_compare.py` compare written | divergence at a stage | localize stage (not decode) |
| **G3** | `invariant_probe` | capture sane (RMS changes across layers) | identical RMS all layers / all-NaN | fix capture sink or prefill |
| **G4** | `shimmy generate --raw` | coherent or finite non-gibberish | all-NaN / gibberish / crash | G2/G3 localization |
| **G4b** | `shimmy generate` chat | instruct-shaped; compare to G4 | only G4b broken → template | jinja/stop tokens |
| **G5** | `candle_probe` | optional L2; SKIP if arch unsupported | mismatch vs GPU residual | only after G2 green |
| **G6** | `llama_ref_dump` | known-good coherent continuation | llama fail | model file / CLI; not Airframe |
| **G7** | decode chain | teacher-forced multi-step matches ref | diverge at step k | orchestration after prefill green |

**Factory rule:** do not open G7 (decode) while G2/G4 are red for NaN prefill.

### The two checkboxes (BOTH required for "certified")

| Box | Meaning | Evidence |
|-----|---------|----------|
| **MATH** | Red ledger empty (or only waived with written reason) | `reds.json` + `REPORT.md` |
| **CHAT** | Multi-prompt generate coherent (NOT one-word only) | `chat_smoke.log` / `04_generate_chat.log` |

Neither substitutes for the other. **A single-token "Paris" argmax is NOT certification.**

### MATH box — PLAN vs PEEL (the mathematical levels, ALL of them)

1. **PLAN** from the GGUF via `cert_plan.py`:
   ```bash
   python3 scripts/cert_plan.py -o cert/packages/<id>/math/plan.json cert/packages/<id>/math/peel.json
   ```
   Plan rules (fastidious — this is where 4 vs 6 lives): `dim_q = n_head*head_dim`;
   `dim_kv = n_kv_head*head_dim`; `ffn_dim` from config; `norm_slots` = **6 if qk_norm else 4**;
   stage counts q/q_norm=dim_q, k/v/k_norm=dim_kv, attn_ctx=dim_q, ffn_gate/ffn_up=ffn_dim,
   residuals=n_embd; rope_base recorded (qwen* warn if ≈1e4); head_dim must be consistent —
   never force `head_dim = n_embd/n_head` if dump says otherwise.
2. **PEEL** from product `stack_dump_gpu` (multi-token prompt only).
3. **Diff** PLAN vs PEEL via `cert_reds.py`:
   ```bash
   python3 scripts/cert_reds.py --reds-out cert/packages/<id>/math/reds.json --report-out cert/packages/<id>/math/REPORT.md cert/packages/<id>/math/peel.json
   ```
   For EVERY layer: required stage present with `sampled == "real"`, `count` matches PLAN,
   `nan_count == 0`, residual rms finite. Final logits: `nan_count == 0`, `len > 0`.
   Dequant: every present type must PASS vs `quant_formula`; FAIL → `RED DEQUANT.<type>` with max_err.
4. **RED** = PLAN ≠ PEEL, or PLAN incomplete, or dequant fails vs `quant_formula`.
5. Exit `0` only if MATH box green (zero unwaived REDs).

### CHAT box — full inference battery per model (NOT one word)

After MATH is green, certify the model's actual generation — this is the "extensive
inference testing" that must run on EVERY model, individually:

1. `04_generate_trace.log` — `shimmy generate --raw` with `AIRFRAME_TRACE_PREFILL_LAYERS=1`,
   check: no `logits_nans`, no `hidden_rms=NaN`, output finite and non-gibberish.
2. `04_generate_chat.log` — `shimmy generate` with the model's chat template (shimmyjinja
   when the GGUF has one). Instruct-shaped output. If only this gate is broken → it's a
   **template/jinja** problem, not a compute problem.
3. Multi-prompt coherent output across several prompts — **never** a single fixed prompt
   like "The capital of France is" used as the whole battery. Run a real prompt set per model.
4. G7 decode chain (after prefill green): teacher-forced multi-step matches reference; a
   divergence at step k localizes decode orchestration.

**There is no "certified" from a one-word answer.** If you only ran "The capital of France
is" and got "Paris", you have done NOTHING toward certification — you must still run the
full battery above.

### Fix discipline

- **First red gate is the only problem.** Do not invent decode stories while G2/G4 are red.
- Multi-token prompts only for dumps.
- No custom probe binaries as substitute for G1–G6.
- `quant_formula` is the certification authority; vault is break-fix only; llama.cpp source
  is a structural flashlight, never the pass bar.
- **Write STATUS.md; read it next session** — do not re-export 20 chats to rediscover the red gate.
- The package `STATUS.md` IS the session handoff.

### Known factory bugs (verified 2026-08-13; beads filed — fix before trusting a package)

1. `certify_family.sh`: `write_status` defined **after** first use → RED paths exit with
   "command not found" and **STATUS.md is never written**. Check ordering first.
2. Same script: G3/G4 NaN greps hardcode Qwen3's vocab `logits_nans=151936` — other vocab
   sizes (e.g. gemma-4 262144) never match → all-NaN misreported as clean.
3. Non-isf `formula_index_for_ggml` mirror lacks IQ4_XS (type 30) → F32 slot in non-isf builds.

## ONBOARDING A NEW QUANT TYPE (registry path)

The quant authority is `airframe_observe::quant_formula` (GGUF/GGML spec math).
Adding a new GGML type (e.g. Q3_K, TQ1_0, IQ4_NL) is a **6-step registration**,
never a shader hack:

1. **`quant_formula.rs`** — write `dequant_<name>` from the GGML block layout (cite
   it in the doc comment), add a `QuantFormula` entry (type_id, name, block_elems,
   block_bytes, dequant), and add a **hand-computed block test** (pattern:
   `q6_k_hand_computed_block` / `q4_0_known_value` — degenerate block, exact values).
2. **`FormulaSlot` enum** + `slot_for_type` arm (same file). Slot ∈ 0..8 = the WGSL
   dispatch ladder; the registry owns the mapping, never the raw GGML type id.
3. **`formula_index_for_ggml`** (`pipeline/mod.rs`): the `isf` build delegates to the
   registry automatically; keep the **non-isf mirror** in sync
   (⚠ 2026-08-13: mirror lacks type 30 / IQ4_XS → F32 fallback in non-isf builds;
   bead filed).
4. **WGSL dispatch ladders**: `sh_dequant_any.wgsl` (switch on `params.formula_index`)
   and `sh_layer_v1.wgsl` dequant paths. One branch per slot; slot 8 (IQ4_XS) is the
   current max. Validate the WGSL branch against `quant_formula` element-by-element
   in a test — do not reimplement the math.
5. **Block geometry** (`block_elems`/`block_bytes`) feeds `dequant_window`
   (`dequant.rs`) and `BlobWindow` planning — windows are sized from block geometry,
   not element counts. Wrong geometry = the window-planning failure class
   (`airframe-eyn`, `airframe-f41.4`).
6. **Gate = G1**: `quant_verify -- --model-path <gguf>` ALL PASS. It is
   registry-driven, so a registered quant is covered automatically; the crate unit
   tests are the second gate.

## MACHINE STATUS — known factory bugs (verified 2026-08-13, beads filed)

1. `scripts/certify_family.sh`: `write_status` is defined **after** its first use →
   every RED path exits with "command not found" and **STATUS.md is never written**.
   Check ordering before trusting a RED package.
2. Same script: G3/G4 NaN greps hardcode Qwen3's vocab `logits_nans=151936` —
   gemma-4 (262144) and other families never match → all-NaN misreported as clean.
3. Non-isf `formula_index_for_ggml` mirror lacks IQ4_XS (type 30) → F32 slot in
   non-isf builds (latent today: PLE runs under isf only).

## ANTI-THRASH RULES (enshrined)

1. **NaN or gibberish → reach for the tools** (`layer_dump_gpu` multi-token +
   `invariant_probe` + `quant_verify`) to **localize** before reading a line of WGSL.
2. **Reading 1500 lines of Rust/shader by hand to "find" a bug is the thrash pattern this
   process exists to prevent.** The bug is found by comparing numbers, then validated against
   spec math.
3. **Custom `kv_*` probes are not a substitute for Layer 1.** Use the prescribed binaries.
4. **Hand-parsing GGUF bytes / guessing tensor offsets is FORBIDDEN.** The tools read the file
   correctly; if a value is wrong, compare the tool's output against `quant_formula`, don't
   re-derive the file layout.
5. **Prefer a layer table (known-good vs ours) once L1 dumps exist.** First divergence = bug.
6. **New family with no oracle?** You are the first — that is NORMAL. Build the PLAN
   (`scripts/cert_plan.py`), validate element-by-element against `quant_formula`, diff PLAN
   vs PEEL. Do NOT reach for llama.cpp or candle as the authority; they are optional hints.

---

## FAMILY DIVERGENCE CHECKLIST (suspects to check via tooling, not by reading source)

| Divergence | Where to check | Known values |
|---|---|---|
| QK-norm (per-head RMS **before** RoPE) | GGUF qk_norm flag | Qwen3/gemma-2/gemma-4: yes; Llama: no |
| RoPE `freq_base` | GGUF rope_freq_base | Qwen2/3: 1e6; Llama/gemma: 1e4 |
| Proportional RoPE factor | `theta / ff` — **DIVIDE** (`ggml_rope_cache_init`) | gemma-4: rope_freqs |
| `rms_norm_eps` | GGUF | Qwen3: 1e-6; Llama: 1e-5 |
| `head_dim` | from attn_q shape / n_head — **never** `n_embd/n_head` alone | Qwen3-4B: 128; gemma-4: 512 |
| FFN gate function | GGUF ffn_gate type | gemma-4: GELU+PAR; Llama: SiLU |
| Attention scale | 1/sqrt(head_dim) vs override | gemma-4: 1.0 |
| KV heads / KV head_dim | attn_k shape | GQA models |
| PLE (per-layer embedding) | spec.ple_enabled, latent dim | gemma-4-E4B: latent 256 |
| MoE routing | n_experts / router tensors | Mixtral, qwen3-coder-30b-A3B |
| SSM/hybrid routing + norm banks | arch profile | qwen3.5-9b (`airframe-8v8`) |
| Chat template | GGUF tokenizer.chat_template | shimmyjinja when present |

### Qwen3 gibberish checklist (worked example)

1. Per-head Q/K RMSNorm **before** RoPE (QK-norm).
2. RoPE `freq_base` (Qwen2 = 1e6, Llama = 1e4).
3. `rms_norm_eps` (Qwen3 = 1e-6).
4. `head_dim` from attn_q shape / n_head (**Qwen3-4B = 128**, not `n_embd/n_head`=80).
5. Chat-template vs raw-prompt mismatch.
6. Prefill finite + decode garbage → product orchestration / feedback — still instrument first.

---

## TOOL INVOCATION (Linux, native)

**Always the repo binary by absolute path** — there is no global `shimmy` on Linux:

```
/home/michael/repos/airframe-workspace/shimmy/target/debug/shimmy ...
```

Verify before trusting output: must report `shimmy 2.5.0`. If `target/debug/shimmy` is missing,
rebuild first (`cd shimmy && cargo build`), then run the rebuilt binary.

Airframe bins (from `airframe/`):
```
cargo run --bin layer_dump_gpu --features isf -- <gguf> "<prompt>" out.json
cargo run --bin quant_verify --features isf -- --model-path <gguf>
cargo run --bin invariant_probe --features isf -- <gguf> "<prompt>"
cargo run -p candle_probe -- <gguf>
```

Self-terminating one-liner (no server/port), from `shimmy/`:
```
SHIMMY_MAX_CTX=8192 target/debug/shimmy generate <gguf> --prompt "..." --max-tokens 256
```

Base directory for this skill: `/home/michael/repos/airframe-workspace/.opencode/skills/airframe-inference`
Relative paths in this skill (e.g., scripts/, reference/) are relative to this base directory.
Note: file list is sampled.
