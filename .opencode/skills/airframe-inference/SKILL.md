---
name: airframe-inference
description: Use when running any inference pathway on Airframe/Shimmy — GPU probing, CPU reference generation, per-tensor comparison, invariant certification. Covers candle_probe, vault_seed, layer_dump_gpu, invariant_probe, frontier_compare, quant_verify, and the airframe_observe observability stack.
---

# Airframe Inference Pathway

## Quick Reference — Which Tool For What

| Goal | Tool | Command |
|------|------|---------|
| CPU golden reference (all layers) | `vault_seed` | `cargo run --bin vault_seed -- <model.gguf> [out.json]` |
| CPU reference (final logits only, LLaMA only) | `candle_probe` | `cargo run -p candle_probe -- <model.gguf> [out.json]` |
| Full GPU layer activation dump | `layer_dump_gpu` | `cargo run --bin layer_dump_gpu -- <model.gguf> "Hello" <out.json>` |
| PPT invariant RMS/checksum | `invariant_probe` | `cargo run --bin invariant_probe -- <model.gguf> <name>` |
| Per-tensor Q/K/V/post/ffn comparison | `frontier_compare` | `cargo run --bin frontier_compare -- --model <gguf> --prompt "Hello" --output <out.json>` |
| GPU dequant type validation | `quant_verify` | `cargo run --bin quant_verify -- --model-path <gguf>` |
| Full invariant cage (CI gate) | `test_invariants` | `cargo test -p airframe --test test_invariants -- --test-threads=1` |
| 22-layer oracle CSV generation | `generate_oracle_for_gguf` | `cargo test --test generate_oracle_for_gguf -- --nocapture` (set `SHIMMY_BASE_GGUF`) |

## The Inference Pathway (Step-by-Step)

### Step 0: Inspect Model Metadata

```bash
cargo run --bin vault_seed -- <model.gguf> /dev/null 2>&1 | head -30
```

Supported architectures: **Llama, Mistral, Phi, Gemma, Qwen2, Qwen3**.
StarCoder2 loads but panics on missing `ffn_gate.weight` (non-gated FFN).
DeepSeek/Command-R may need metadata.rs updates.

Check model size vs VRAM (RTX 3060 = 12 GB). Models > 2 GB need the multi-buffer
path (automatic). Models > 6 GB fail at the 3-way blob cap.

### Step 1: CPU Reference (Known Good)

```bash
cargo run --bin vault_seed -- <model.gguf> golden_output.json
```

Output: per-layer `{layer_idx, rms, checksum, first20}` where layer_idx=-1 is
final logits. This is the ground truth.

### Step 2: GPU Layer Activation Dump

```bash
cargo run --bin layer_dump_gpu -- <model.gguf> "Hello" gpu_layers.json
```

Captures full hidden state after every layer. Compare RMS with CPU reference.

### Step 3: Invariant Probe (PPT Cage)

```bash
set AIRFRAME_CAPTURE_INVARIANT=1
cargo run --bin invariant_probe -- <model.gguf> <model_name>
```

Uses production capture hook. RMS ratio test: `max(gpu, vault) / min(gpu, vault)
≤ 2.0` for layers, `≤ 4.0` for final_logits.

### Step 4: Frontier Compare (Deep Per-Tensor)

```bash
cargo run --bin frontier_compare -- --model <model.gguf> --prompt "Hello" --output compare.json
```

Compares Q, K, V, post_attn, ffn_out, output CPU vs GPU at every layer.
Use when you need to find which specific kernel produces wrong results.

### Step 5: Quant Verify

```bash
cargo run --bin quant_verify -- --model-path <model.gguf>
```

Run FIRST if you suspect dequant issues (all-NaN often = wrong byte offsets).

## airframe_observe Facts

All facts in `InferenceFact` enum with alpha keys:

| Fact | Key | Data |
|------|-----|------|
| `LayerOutput` | 1 | `layer_idx`, `position`, `rms_bits`, `checksum` |
| `FinalLogits` | 2 | `position`, `rms_bits`, `checksum` |
| `OutputToken` | 3 | `step`, `token_id` |
| `PerTensorOutput` | 9 | `layer_idx`, `position`, 6×`rms_bits`+`checksum` |
| `DispatchTiming` | 10 | `layer`, `kernel` enum, `elapsed_ms` |
| `PromptToken` | 12 | `position`, `token_id` |
| `EmbeddingRequest` | 19 | `token_id` |
| `DecodeStep` | 16 | `step`, `token_id` |

## Observers

| Observer | On Key | Captures |
|----------|--------|----------|
| `VaultOracleObserver` | KEY_LAYER_OUTPUT | `OracleCapture {layer_idx, position, rms, checksum}` |
| `CandleCompareObserver` | KEY_FINAL_LOGITS | `LogitCapture {position, rms, checksum}` |
| `LayerStabilityObserver` | KEY_LAYER_OUTPUT | Derives `LayerOutputStable` if RMS sane |

## Emitting Facts in Code

```rust
session.emit_layer_output(layer_idx, position, &hidden_state_f32_slice);
session.emit_final_logits(position, &logits_f32_slice);
session.emit_per_tensor_output(layer_idx, position, &q, &k, &v, &post, &ffn, &output);
session.saturate();
session.vault_oracle().drain();  // get captures
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `AIRFRAME_CAPTURE_INVARIANT=1` | Enable per-layer capture | off |
| `AIRFRAME_TRACE_PREFILL_LAYERS=1` | Per-layer NaN trace during prefill | off |
| `SHIMMY_MAX_CTX=8192` | Context window (RTX 3060) | 32768 |
| `VAULT_DB=<path>` | Path to golden vault DuckDB | vault/vault.duckdb |
| `SHIMMY_BASE_GGUF=<path>` | Test model path for oracle tests | none |

## Common Model Issues

### All-NaN on GPU

1. Check adapter selection (prefer DiscreteGpu):
   `cargo run --bin <anything> -- <model> 2>&1 | grep adapter`
2. Check `batch_count` param in inference.rs (must be ≥ 1, fixed v0.2.9)
3. Check struct layout aligns with WGSL `struct Params`
4. Run `quant_verify` for dequant correctness

### Gibberish Output

1. Qwen3 needs per-head Q/K RMSNorm before RoPE (QK-norm)
2. RoPE freq_base differs: Qwen2=1e6, Llama=1e4
3. rms_norm_eps differs: Qwen3=1e-6, Llama=1e-5
4. Chat template vs raw prompt mismatch

## End-to-End: Certify a New Model

```
1. vault_seed <model.gguf> cpu_ref.json
2. layer_dump_gpu <model.gguf> "Hello" gpu.json
3. Compare RMS per layer (should match within 2×)
4. If step 3 fails → frontier_compare to find which tensor
5. If step 4 passes → invariant_probe for CI gate
6. cargo test -p airframe --test test_invariants -- --test-threads=1
7. shimmy generate <name> --prompt "Hello" --max-tokens 32
```

## Gotchas

- **Kill stale GPU procs first (MSYS/Git-Bash):** `taskkill //f //im shimmy_server_gpu.exe` (double-slash — MSYS mangles a single `/` into a path). Or `cmd /c "taskkill /f /im shimmy_server_gpu.exe"`.
- **Set `SHIMMY_MAX_CTX=8192`** on RTX 3060
- **Write test output to file and read verbatim** — never pipe through grep|head|tail
- **Single-threaded invariant tests:** `--test-threads=1`
- **candle_probe builds separately:** use `-p candle_probe`
- **GPU capture gated by `isf` feature + env var** — compiles to nothing in release
- **Qwen3 8B all-zeros:** may exceed 3-blob cap (>6 GB) or need unsupported quant types
