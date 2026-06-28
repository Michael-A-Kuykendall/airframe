# Session Handoff — NaN Fixed, StarCoder2 Non-Gated FFN Remains

## Current State
- **Branch**: `feat/phase4-pingpong-activation` (HEAD `ac961a3`)
- **Tests**: 369/369 passing (`cargo test --lib`)
- **GPU**: RTX 3060 (12GB VRAM, Vulkan, driver 596.49)
- **GGUF models**: All 5 on `D:/shimmy-test-models/gguf_collection/`

## Critical Findings This Session
1. **GPU pipeline NaN FIXED**. Root cause: `sh_layer_v1.wgsl:985` used `params.quant_attn_out` for both FFN gate and up projections instead of per-tensor `quant_ffn_gate`/`quant_ffn_up`. Also fixed `layer_dump_gpu.rs` which hardcoded all quant types to 0 (F32) and used Q4_0-only dequant. Verified: TinyLlama Q4_0 produces valid layer outputs.
2. **StarCoder2 panics at `metadata.rs:410`**: Missing `ffn_gate.weight` (non-gated FFN arch). NOT fused QKV — has separate Q/K/V.
3. **gguf_inspector rewritten**: Now detects arch, ffn_gate presence, fused QKV via raw byte scan (8MB window).
4. **Pool encoder/timestamp**: Fixes tracked in `airframe-dg3` and `airframe-f35`. Not wired into `mod.rs`.

## Quick Commands
```bash
# GGUF inspector (detects arch, ffn_gate, fused QKV)
cargo run --bin gguf_inspector -- "D:/shimmy-test-models/gguf_collection/<model-path>"

# Layer dump (runs all layers on GPU)
cargo run --bin layer_dump_gpu -- "<gguf_path>" "Hello" output.json

# TDR calibration (vault DB only, no GPU)
cargo run --bin tdr_calibrate -- qwen3-0.6b-q4_k_m
cargo run --bin model_calibrator -- qwen3-0.6b-q4_k_m
```

## Beads Overview
- `bd prime` — Shows BD workflow commands
- `bd ready` — Ready-to-work issues
- See `.beads/PRIME.md` for full context

## IMPORTANT: DO NOT USE THESE (don't exist)
- `cargo run --bin inference` — binary doesn't exist
- `cargo test --features tdr_test` — feature doesn't exist
