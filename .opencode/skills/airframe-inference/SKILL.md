---
name: airframe-inference
description: Use BEFORE ANY inference / model-output / decode / gibberish / probe / localize / certify work on Airframe/Shimmy. Covers the PRESCRIBED Layer-1 tools (layer_dump_gpu, invariant_probe, quant_verify, shimmy generate). Layer-2 candle_probe and Layer-3 llama.cpp are secondary only. Math authority: quant_formula (GGUF spec). FORBIDS custom probe binaries as a substitute for Layer-1 tools. Triggers: "qwen", "debug model", "gibberish", "decode collapse", "localize", "probe", "verify", "certify", "shimmy generate", "forward pass", "kv cache".
---

# Airframe Inference Pathway

> **Canonical copy for this sandbox:**  
> `airframe-workspace/.opencode/skills/airframe-inference/SKILL.md`  
> Nested copies under `airframe/.opencode/skills/` or `target/package/` may be **stale**.
> If they disagree with this file or workspace `AGENTS.md`, **this file + AGENTS win**.

Load **`airframe-discipline` first**, then this skill, before any inference work.

> **🔴 SHIMMY = THE REPO BINARY, NEVER THE GLOBAL ONE.** A bare `shimmy generate`
> / `shimmy list` via `cmd /c` resolves to the stale global `~/.cargo/bin/shimmy.exe`
> (v2.2.0) which links crates.io airframe — NOT this workspace's patched code.
> Always invoke the workspace binary explicitly from `shimmy/`:
> `cmd /c "target\debug\shimmy.exe ..."` (or `target\release\shimmy.exe`).
> Verify `--version` = `2.4.2` before trusting any shimmy output. If the repo
> binary is missing, rebuild it (`cd shimmy && cargo build`); never fall back to
> the global install. Full rule: workspace `AGENTS.md` (SHIMMY INVOCATION block).

Also read (when debugging the multi-session Qwen3 thrash):  
`SESSION_RESEARCH_SYNTHESIS.md` at the workspace root — what is proven / invalid / next.

**Family factory (onboarding machine):** `FAMILY_FACTORY.md` +  
`scripts/certify_family.bat <family-id> <gguf> "multi-token prompt"`.  
Produces `cert/packages/<family-id>/` with gates G0–G6 and `STATUS.md`.  
**Run the factory before freestyle debugging.** First red gate only.

---

## Three-layer certification (order is mandatory)

| Layer | Role | Tools |
|-------|------|--------|
| **1 — INTERNAL (authority for development)** | GPU tools + **spec math** | `quant_verify`, `layer_dump_gpu` (**multi-token**), `invariant_probe`, `AIRFRAME_TRACE_PREFILL_LAYERS=1`, `shimmy generate` |
| **2 — SECONDARY** | Independent CPU forward | `candle_probe` (`cargo run -p candle_probe`) — must stay independent of retired vault oracle |
| **3 — TERTIARY** | External engine if L1+L2 disagree | `llama.cpp` (e.g. `C:\llama.cpp`) |

**Math authority** = `airframe_observe::quant_formula` (GGUF/GGML registry).  
Never hand-roll WGSL/reference math. candle / llama.cpp are **diagnostics**, not the spec.

**Custom probes** (`kv_*`, one-off binaries): allowed **only** as a narrow delta after Layer-1
has localized a step/layer and only to bisect one variable. They are **not** a substitute
for the prescribed tools. Sessions that skipped Layer-1 burned weeks.

---

## Quick reference — Layer 1 commands

| Goal | Tool | Command (write output to a **file**, read verbatim) |
|------|------|------------------------------------------------------|
| Dequant gate (run first if all-NaN) | `quant_verify` | `cargo run --bin quant_verify --features isf -- --model-path <gguf> > QV.log 2>&1` |
| Multi-token layer dump | `layer_dump_gpu` | `cargo run --bin layer_dump_gpu --features isf -- <gguf> "<MULTI-TOKEN PROMPT>" out.json > LD.log 2>&1` |
| PPT invariant cage | `invariant_probe` | `AIRFRAME_CAPTURE_INVARIANT=1 cargo run --bin invariant_probe --features isf -- <gguf> <name> > INV.log 2>&1` |
| Production end-to-end | `shimmy generate` | from `shimmy/`: `set SHIMMY_MAX_CTX=8192` then `cmd /c "target\debug\shimmy.exe generate <model> --prompt \"...\" --max-tokens N > GEN.log 2>&1"`. **Never bare `shimmy` — repo binary only (see banner above).** |
| Prefill NaN trace | env | `AIRFRAME_TRACE_PREFILL_LAYERS=1` on the server / generate process |

**Never use a single-token prompt** (`"Hello"`) for localization — softmax of 1 token
masks cross-attention bugs. Use a multi-token prompt (e.g. `"The capital of France is"`).

---

## Correct investigation sequence

```
0. taskkill //f //im shimmy_server_gpu.exe   # MSYS: double-slash
   SHIMMY_MAX_CTX=8192
   cargo build --features isf   (airframe) + cargo build (shimmy)
1. quant_verify          → ALL PASS or fix byte offsets / quant type
2. layer_dump_gpu MULTI  → logits_nans=0; RMS monotonic embedding→output
3. invariant_probe       → cage / first non-monotonic layer
4. shimmy generate (repo binary) → product symptom (finite gibberish vs all-NaN vs coherent)
5. Only if L1 is green but product is wrong:
   candle_probe (L2) and/or llama.cpp (L3) for first-token / per-layer table
6. Optional narrow custom delta probe AFTER a layer/step is named by L1
```

### Layer table method (when comparing engines)

Build a table: one row per layer (0..N-1 + FinalLogits).

| Layer | Must-happen (algebra) | Known-good (candle/llama) RMS / top-1 | Ours (layer_dump_gpu) RMS / top-1 | Δ / notes |
|-------|----------------------|----------------------------------------|-------------------------------------|-----------|
| 0 | RMSNorm → QKV → … | … | … | first diverge = hunt target |

Cancel algebraically only against **spec** (`quant_formula`) or a **file-backed** known-good dump — never against improvised mental math.

---

## Layer 2 — candle_probe

```bash
cd airframe
cargo run -p candle_probe -- <model.gguf> [out.json]
```

- Independent CPU path (candle-transformers).  
- **Not** the vault oracle; do not delete it again.  
- May be LLaMA-family limited; extend carefully for Qwen3 if needed.  
- Use for corroboration after Layer 1, or when product is gibberish but Layer 1 looks “healthy” (self-consistent wrong).

## Layer 3 — llama.cpp

Local install (this machine): `C:\llama.cpp` (build under `build/`).  
Use only if Layer 1 + Layer 2 disagree or external top-1 is needed.  
Do not modify the llama.cpp tree; run read-only / generate dumps into **this workspace**.

---

## airframe_observe facts (instrumentation)

| Fact | Key | Data |
|------|-----|------|
| `LayerOutput` | 1 | `layer_idx`, `position`, `rms_bits`, `checksum` |
| `FinalLogits` | 2 | `position`, `rms_bits`, `checksum` |
| `OutputToken` | 3 | `step`, `token_id` |
| `PerTensorOutput` | 9 | Q/K/V/post/ffn/output RMS+checksum |
| `DispatchTiming` | 10 | layer, kernel, elapsed_ms |
| `PromptToken` | 12 | position, token_id |
| `DecodeStep` | 16 | step, token_id (and position when present) |
| `EmbeddingRequest` | 19 | token_id |

Env: `AIRFRAME_CAPTURE_INVARIANT=1`, `AIRFRAME_TRACE_PREFILL_LAYERS=1`, `SHIMMY_MAX_CTX=8192`.

---

## Common model issues

### All-NaN on GPU
1. Rebuild before diagnosing (stale binary phantom).  
2. `quant_verify` (byte offsets).  
3. Adapter = DiscreteGpu.  
4. `batch_count` ≥ 1; WGSL `Params` layout.

### Gibberish (finite wrong numbers) — Qwen3 checklist
1. Per-head Q/K RMSNorm **before** RoPE (QK-norm).  
2. RoPE `freq_base` (Qwen2/3 = **1e6**, Llama = 1e4).  
3. `rms_norm_eps` (Qwen3 = **1e-6**).  
4. `head_dim` from **attn_q shape / n_head**, not `n_embd/n_head` (Qwen3-4B = **128**, not 80).  
5. Chat-template vs raw-prompt mismatch.  
6. Prefill OK + decode garbage → orchestration / KV carry / feedback loop — still **instrument first**, do not start by reading 1500 lines of WGSL.

---

## Gotchas (non-negotiable)

- Kill stale GPU: `taskkill //f //im shimmy_server_gpu.exe` (MSYS double-slash).  
- **Write tool output to a file and read verbatim** — never `grep|head|tail` as the only check.  
- Multi-token prompts for `layer_dump_gpu`.  
- GPU capture needs `isf` feature + env vars.  
- Do **not** load retired `airframe-vault` as certification authority.  
- Do **not** claim “kernel proven” from self-matching probes that still emit garbage tokens.

## Anti-patterns (from SESSION_RESEARCH_SYNTHESIS — do not repeat)

- Single-token layer dump → false “forward correct”.  
- Hand-rolled `f16_to_f32` in WGSL instead of `quant_formula`.  
- Stale `shimmy.exe` diagnosis.  
- Custom `kv_*` probes instead of Layer-1 tools.  
- Argmax on wrong tuple field (`(_, logits, _)` vs `(_, _, logits)`).  
- Multi-step compare against a single last-position reference.  
- Probe `head_dim = n_embd/n_head` when padded heads need 128.
