# Shimmy Vision — Session Brief
_Generated: 2026-06-01. Purpose: complete handoff for a new session starting vision work._

---

## One-Line Summary

The goal is to make MiniCPM-V-2.6 run natively through Airframe's wgpu/WGSL pipeline — **zero llama.cpp, zero cmake, zero sidecar binary**. A working llama.cpp-based implementation already exists but is parked on an archived branch. The forward path is pure Rust.

---

## What Exists Right Now

### Repo: `airframe` (this repo, `master` branch)

| File | What It Is | Status |
|------|-----------|--------|
| `VISION_OPTION_A_SPIKE.md` | Full engineering spike — model selection rationale (why MiniCPM-V-2.6, why Qwen2-VL-7B as v2), gap analysis, full implementation checklist Phases 0–7 | ✅ Complete, authoritative |
| `docs/vision-task-breakdown.md` | 28 atomic tasks across 8 phases. Each task is self-contained — an agent can pick one up cold. Full Rust code stubs included. | ✅ Complete, authoritative |
| `docs/architecture-map.md` | Full Airframe architecture reference | Existing |

**There is no `feat/vision-multimodal` branch yet in airframe.** The planning docs are on `master`. Implementation has NOT started.

### Repo: `shimmy-private` — archived branch `archive/public/2026-05-24/feature/shimmy-vision-phase1`

This is the OLD implementation (llama.cpp wrapper). It is complete and working end-to-end but is **not the forward path**. It exists as:
1. A live parity baseline (the subprocess inference path to test against)
2. The source of truth for the product UX: 6 modes, JSON schema, license enforcement, Stripe/Keygen wiring

| File (on archive branch) | What It Is |
|--------------------------|-----------|
| `src/vision.rs` | 1,399 lines — MiniCPM-V inference via llama.cpp subprocess, HTTP `/api/vision`, CLI `shimmy vision`, 6 response modes, full JSON schema, prompt templates |
| `src/vision_license.rs` | 674 lines — Keygen license enforcement, Ed25519 verification, offline grace period, usage metering, hardware fingerprinting |
| `docs/SHIMMY_VISION_SPEC.md` | Product spec — system requirements, model files (with SHA256 checksums), Stripe/Keygen provisioning, 6 operating modes |
| `docs/VISION_PRODUCTION_CHECKLIST.md` | 95% production ready as of last audit (2025-12-15). Critical remaining: Step 1 of private crate migration (copy vision.rs + vision_license.rs to shimmy-vision-private) was NOT STARTED. |

**Key model details (from SHIMMY_VISION_SPEC):**
- LLM: `ggml-model-Q4_K_M.gguf` (~5.0 GB) — SHA256: `3a4078d53b46f22989adbf998ce5a3fd090b6541f112d7e936eb4204a04100b1`
- ViT+Resampler: `mmproj-model-f16.gguf` (~600 MB) — SHA256: `4485f68a0f1aa404c391e788ea88ea653c100d8e98fe572698f701e5809711fd`
- HF repo: `openbmb/MiniCPM-V-2_6-gguf`

---

## The Architecture Decision (Why Option A)

The llama.cpp path (`shimmy-vision-phase1`) died at packaging. `llama-mtmd-cli` requires cmake, LLVM, CUDA SDK, Vulkan SDK — cross-compiling for 5 platforms in GitHub Actions CI is unsolvable. Option B (vendored sidecar binary) just delays the same problem.

**Option A**: vision encoder lives inside Airframe as pure Rust + WGSL. Release = `cargo build --release --features vision`. Standard hosted CI runners, zero native SDK requirements.

**MiniCPM-V-2.6 was chosen because:**
- Only model in its class trained specifically for OCR + document analysis + UI spatial grounding
- `dom_map` feature (normalized bounding boxes for interactive UI elements) requires the Resampler's position-aware cross-attention — no 2B model can do this
- SigLIP-So400M encoder + Qwen2-7B LLM backbone — Airframe already has 80% of Qwen2-7B via Qwen3 support
- v2 upgrade (Qwen2-VL-7B) shares the same LLM backbone, so delta is only in the vision encoder

---

## What Airframe Already Has (Don't Rebuild)

From `src/ops/dispatch.rs`:
- ✅ `matmul`, `matvec`, `rmsnorm`, `rope`, `softmax`, `attention` (with and without KV cache), `ffn_swiglu`, `silu`, `multiply`, `add`
- ✅ Full `LlamaModel` forward pass (text-only) — `src/family/llama.rs`
- ✅ GGUF weight loader + `WeightId` enum — `src/core/weight_id.rs`, `src/core/model.rs`
- ✅ KV cache, Engine prefill + decode — `src/runtime/`
- ✅ GPU bindless backend (wgpu/WebGPU) — `src/backend/bindless/`
- ✅ HTTP server binary — `src/bin/shimmy_server_gpu.rs`

---

## What's Missing (The 28-Task Build Plan)

Full details with Rust code stubs are in `docs/vision-task-breakdown.md`. Summary:

### Phase 0 — Research & Oracle (~3–5 days, NO Rust code)
- **T-0.1** Download MiniCPM-V-2.6 GGUFs to `~/models/minicpm-v-2.6/`
- **T-0.2** Map all mmproj tensor names → `artifacts/mmproj_tensor_map.txt`
- **T-0.3** Find `<image>` token ID in tokenizer → `artifacts/image_token_id.txt`
- **T-0.4** Python oracle: extract SigLIP ViT features → `artifacts/oracle_vit_features.npy` (shape `[1, 1025, 1152]`)
- **T-0.5** Python oracle: extract Resampler output → `artifacts/oracle_resampler_output.npy` (shape `[1, 64, 3584]`)
- **T-0.6** Document multi-scale tiling behavior → `artifacts/tiling_behavior.md`

### Phase 1 — Missing Ops (~1 week, all tasks independent/parallelizable)
- **T-1.1** Add `layernorm` to `OpDispatcher` — ViT uses LayerNorm, NOT RMSNorm
- **T-1.2** Add `gelu` to `OpDispatcher` — ViT FFN uses GELU, NOT SwiGLU
- **T-1.3** Add `patch_embed` to `OpDispatcher` — 14×14 patch conv as reshape+matmul
- **T-1.4** Verify bidirectional attention (`causal_mask=false`) — write test, fix if broken
- **T-1.5** Add `add_broadcast` if existing `add` requires matching shapes

### Phase 2 — ViT Encoder + Resampler (~2 weeks, both tasks independent)
- **T-2.1** Create `src/family/vit.rs` — SigLIP-So400M encoder (27 layers, 1152 hidden, 16 heads, GELU FFN, LayerNorm)
  - Acceptance: `rms_diff(output, oracle_vit_features.npy) < 0.01`
- **T-2.2** Create `src/family/resampler.rs` — Perceiver Resampler (64 learned queries, 1 cross-attn layer, output dim 3584)
  - Acceptance: `rms_diff(output, oracle_resampler_output.npy) < 0.01`

### Phase 3 — Weight Loading (~3–5 days)
- **T-3.1** Add `WeightId` variants for all mmproj tensors (`vision_model.encoder.layers.N.*`, `resampler.*`)
- **T-3.2** Add tensor name mapping in `src/core/model.rs` for mmproj GGUF
- **T-3.3** Create `src/core/image_preprocess.rs` — resize + normalize to SigLIP format (mean=0.5, std=0.5)

### Phase 4 — Token Injection (~3–5 days)
- **T-4.1** Add `prefill_multimodal()` on Engine — injects 64 visual tokens at `<image>` position
- **T-4.2** Add `forward_with_image_embeds()` on LlamaModel — replaces `<image>` token IDs with Resampler output vectors

### Phase 5 — Image Preprocessing + Tiling (~2–3 days)
- **T-5.1** Create `src/family/image_tiler.rs` — multi-scale tiler: 448×448 base + N crop tiles depending on aspect ratio (max 6 tiles)

### Phase 6 — Smoke Binary (~2 days)
- **T-6.1** Create `src/bin/vision_smoke.rs` — standalone binary: load mmproj + LLM GGUF, run one test image, print text output. Used for local verification before wiring into the HTTP server.

### Phase 7 — Wire into Shimmy (~3–5 days)
- **T-7.1** Add `src/runtime/vision_engine.rs` — orchestrates tile → encode → resample → inject → generate
- **T-7.2** Add Cargo feature `vision` to airframe/shimmy `Cargo.toml`
- **T-7.3** Port HTTP endpoint `/api/vision` from `vision.rs` (archive branch) to use `VisionEngine` instead of `run_mtmd_cli_minicpm_v()`
- **T-7.4** Port 6 operating modes + JSON schema from `vision.rs` — these are already correct, just swap the inference call
- **T-7.5** Wire Keygen license enforcement from `vision_license.rs` — already production-ready, just call it

---

## New Files to Create in Airframe

| File | Phase | Purpose |
|------|-------|---------|
| `src/family/vit.rs` | P2 | SigLIP-So400M ViT encoder |
| `src/family/resampler.rs` | P2 | Perceiver Resampler projector |
| `src/family/image_tiler.rs` | P5 | Multi-scale tiling |
| `src/core/image_preprocess.rs` | P3 | Resize + SigLIP normalization |
| `src/runtime/vision_engine.rs` | P7 | Orchestration |
| `src/bin/vision_smoke.rs` | P6 | Standalone smoke test binary |

---

## Existing Files to Modify in Airframe

| File | Change |
|------|--------|
| `src/ops/reference/activations.rs` | Add `layernorm_f32`, `gelu_f32`, `add_broadcast_f32`, `patch_embed_f32` |
| `src/ops/dispatch.rs` | Expose `layernorm`, `gelu`, `add_broadcast`, `patch_embed` methods |
| `src/core/weight_id.rs` | Add `WeightId` variants for all mmproj tensors |
| `src/core/model.rs` | Add tensor name mapping for mmproj GGUF |
| `src/family/llama.rs` | Add `forward_with_image_embeds()` |
| `src/runtime/engine.rs` | Add `prefill_multimodal()` |
| `src/family/mod.rs` | Register `vit`, `resampler`, `image_tiler` modules |
| `Cargo.toml` | Add `image` crate (resize), `vision` feature flag |

---

## The 6 Vision Modes (from archive branch `vision.rs`)

These are already tested and working. Port these prompt templates and output parsers wholesale:

| Mode | What it produces |
|------|-----------------|
| `full` | Complete analysis — layout + OCR + UI elements + metadata JSON |
| `ocr` | Text extraction only — ordered by reading flow |
| `layout` | Structural regions — header, nav, sidebar, content, footer |
| `brief` | 1–3 sentence summary, no structured JSON |
| `web` | Interactive element detection — buttons, links, inputs |
| `dom_map` | Normalized bounding boxes for interactive elements (spatial grounding) |

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `master` | Current stable, no vision code |
| `feat/vision-multimodal` | **Create this** — all vision implementation work |
| `archive/public/2026-05-24/feature/shimmy-vision-phase1` (shimmy-private) | Old llama.cpp impl — reference only, DO NOT copy |

Create the branch before starting Phase 1:
```bash
cd /c/Users/micha/repos/airframe
git checkout -b feat/vision-multimodal
```

---

## Total Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 0 (oracle/research) | 3–5 days |
| Phase 1 (ops) | ~1 week |
| Phase 2 (ViT + Resampler) | ~2 weeks |
| Phase 3 (weight loading) | 3–5 days |
| Phase 4 (token injection) | 3–5 days |
| Phase 5 (tiling) | 2–3 days |
| Phase 6 (smoke binary) | 2 days |
| Phase 7 (shimmy wiring) | 3–5 days |
| **Total** | **~6–8 weeks** |

New Rust code: ~900–1,200 lines across 6 new files + ~150 lines of changes to existing files. License enforcement + HTTP endpoints are already written and just need to be rewired to the new inference call.

---

## Recommended First Action for New Session

1. Read `VISION_OPTION_A_SPIKE.md` — model selection rationale, gap analysis
2. Read `docs/vision-task-breakdown.md` — the full 28-task build plan
3. Create branch `feat/vision-multimodal`
4. Start with **T-1.1** (`layernorm`) or **T-0.1** (download GGUFs) — both have zero blockers

Do NOT look at `archive/public/2026-05-24/feature/shimmy-vision-phase1` in shimmy-private unless you need the prompt templates, JSON schema, or license enforcement code from `vision.rs` / `vision_license.rs`. Those are the only parts worth porting. Everything else in that branch is the llama.cpp path being discarded.
