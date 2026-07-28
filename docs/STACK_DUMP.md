# Stack Dump Contract (OBS-0)

**Schema:** `docs/stack.schema.v1.json` (`schema` field must be `"airframe.stack.v1"`)  
**Architecture:** workspace root `INFERENCE_OBSERVATORY.md`  
**Beads:** epic `airframe-ate` — implement dumps in OBS-1+, do not invent alternate field names without amending OBS-0 / this file.

## Package layout

```text
cert/packages/<family-id>/obs/<run-id>/
  manifest.json           # run meta: engines, prompt, paths, git SHA, timestamp
  airframe.stack.json     # airframe.stack.v1 from product path
  llama.stack.json        # or SKIP.json with reason
  candle.stack.json       # optional or SKIP.json
  compare.md              # human first-diverge (OBS-4)
  compare.json            # machine first-diverge (OBS-4)
```

`<run-id>` example: `20260728T153000Z` or UTC iso compact.

## Env / flags (frozen defaults)

| Variable | Default | Meaning |
|----------|---------|---------|
| `AIRFRAME_STACK_DUMP` | off | `1` / `true` → write stack JSON during product generate (OBS-1b) |
| `AIRFRAME_STACK_DUMP_PATH` | *(derived)* | Explicit output path; if unset, under `cert/packages/.../obs/` or CWD `stack_dump.json` |
| `AIRFRAME_STACK_DUMP_TOP_K` | `20` | Size of `final.logits.top_k` |
| `AIRFRAME_STACK_DUMP_STAGES` | `layer0` | `layer0` \| `all` \| `none` |
| `SHIMMY_MAX_CTX` | *(must set 8192 on 12GB)* | Cap context / KV VRAM |

## Level 4 honesty (`sampled`)

Every stage entry **must** set `sampled` when present:

| Value | Meaning |
|-------|---------|
| `real` | Read from the true stage tensor |
| `temp_last` / `activation_last` | Known buffer convention (document which) |
| `proxy` / `proxy_residual` | Not the named op’s buffer — comparison must treat as non-authoritative |
| `unavailable` | Not captured this run |
| `wrong_buffer` | Known bug; do not trust for first-diverge |

Never emit a stage named `qkv`/`qk_norm` with identical stats to `attn_norm` without an honesty tag.

## Missing levels

Use:

```json
{ "status": "unsupported", "reason": "..." }
```

or

```json
{ "status": "skipped", "reason": "..." }
```

## Validate fixture offline

```bat
cd airframe
python docs/validate_stack_fixture.py
```

Or:

```bat
pip install check-jsonschema
check-jsonschema --schemafile docs/stack.schema.v1.json docs/fixtures/stack_minimal.json
```

## Product dump command (OBS-1)

```bat
set SHIMMY_MAX_CTX=8192
cd airframe
cargo run --features isf --bin stack_dump_gpu -- ^
  "D:\path\model.gguf" "The capital of France is" out\airframe.stack.json --top-k 20
```

`AIRFRAME_STACK_DUMP=1` (OBS-1b): operators use `scripts\stack_dump.bat` which is the
flag-driven entrypoint (sets TOP_K / MAX_CTX and writes `obs/run_latest/`).

## InferenceFact → stack level (OBS-9)

| InferenceFact / capture | Stack level |
|-------------------------|-------------|
| PromptToken / encode | L1 tokens |
| Embedding dequant | L2 embedding |
| LayerOutput / StackLayerSnap | L3 residual |
| Stage TRACE (honest sampled) | L4 stages |
| PerTensorOutput / CapturedPerTensor | L5 tensors |
| FinalLogits + top_k | L6 final.logits |
| DecodeStep / OutputToken | L7 decode |
| prompt_render source | L8 product_shell |

Materialize path: product prefill asserts/captures → `stack_dump_gpu` writes JSON (not a third forward for L3 when stack sink installed during the same `run_full_model_*` call).

## Cross-links

- Workspace: `INFERENCE_OBSERVATORY.md`, `FAMILY_FACTORY.md`
- Airframe beads: `airframe-ate` (OBS-0…OBS-12)
- One command: `scripts\stack_dump.bat`

## Deep peel structure

See `docs/PEEL_STRUCTURE.md` — required intercept registry for full stage/tensor peel.

