# Airframe Multimodal (Vision) — Option A Spike
_Engineering spike: May 27, 2026. Goal: define every step to make Airframe a first-class multimodal inference engine and ship Shimmy Vision with zero llama.cpp, zero cmake, zero sidecar binaries._
_Updated after model selection spike: MiniCPM-V-2.6 confirmed as primary. Qwen2-VL-7B queued as v2. Moondream2 dropped._

---

## The Thesis

jiangwei-chen and the original Shimmy Vision both died at the same place: packaging `llama-mtmd-cli` for 5 platforms. That binary requires cmake, LLVM, CUDA SDK, Vulkan SDK — and cross-compiling it in GitHub Actions CI is the problem that killed both efforts. There is no way to win that fight. Option B (vendored sidecar) just delays it.

Option A eliminates it permanently. When the vision encoder lives inside Airframe, a release is just `cargo build --release --features vision` — pure Rust, standard hosted CI runners, zero native SDK requirements.

---

## Model Selection: Decision Made

**Primary: MiniCPM-V-2.6. Secondary (v2 upgrade): Qwen2-VL-7B. All other candidates dropped.**

MiniCPM-V-2.6 was not chosen for quality-per-size on constrained hardware. It was chosen because it was **purpose-built and specifically trained** for the exact task Shimmy Vision performs: document OCR, UI screenshot analysis, structured JSON output, and spatial grounding with bounding box prediction. The `dom_map` feature — normalized bounding boxes for UI elements — requires a model trained for spatial grounding. No 2B model does this reliably.

### Why the Other Candidates Were Rejected

| Model | Rejection Reason |
|---|---|
| Moondream2 | No spatial grounding; `dom_map` would fail; trained for captioning not UI analysis |
| LLaVA-1.5-Phi-3 | Not trained on OCR/documents; CLIP encoder loses text fidelity |
| InternVL2-2B | Better than Moondream but still 2B — insufficient OCR quality at the detail level Shimmy needs |
| SmolVLM | Same problem as Moondream — general VQA, not document/UI focus |

### Primary + v2 Architecture

| Model | Role | Size | Vision Encoder | Projector | LLM Backbone | GGUF |
|---|---|---|---|---|---|---|
| **MiniCPM-V-2.6** | **v1 (ship first)** | 8B | SigLIP-So400M (400M ViT) | Resampler (cross-attn) | **Qwen2-7B** | ✅ |
| **Qwen2-VL-7B** | **v2 (add after)** | 7B | SigLIP + 2D RoPE | MRoPE projector | Qwen2-7B | ✅ |

Both models share the **Qwen2-7B LLM backbone** — which Airframe already has 80% complete via Qwen3 support. The implementation delta between v1 and v2 is primarily in the vision encoder (MRoPE vs Resampler), not the LLM.

### Why MiniCPM-V-2.6 Is Uniquely Good at This

- **OCRBench**: #1 in its size class at time of release. Trained on TextVQA, DocVQA, InfoVQA, ChartQA.
- **Multi-scale tiling**: Slices high-res screenshots into N overlapping 448×448 tiles. Each tile is independently encoded, then all tile features are fed through the Resampler. This is why the codebase has `preprocess_config_for_mode` — web mode uses smaller images to control tile count and avoid Resampler memory exhaustion.
- **Spatial grounding**: Resampler output tokens are position-aware (cross-attention over patch positions). Model can predict WHERE things are, not just WHAT they are. This is the `dom_map` feature.
- **Structured JSON output**: Was specifically prompted and fine-tuned for structured output. The existing 6-mode prompt templates in `vision.rs` already produce reliable JSON with this model.

### What Makes an Encoder Easy vs Hard in Airframe

**Easy (standard ViT):**
- Standard LayerNorm (NOT RMSNorm) — need to add this op, ~20 lines
- GELU activation (NOT SwiGLU) — need to add this op, ~5 lines
- Standard multi-head self-attention (NO KV cache, NO RoPE in the vision encoder)
- Fixed-size patch embedding: image → non-overlapping 14×14 or 16×16 patches → flatten → linear project
- 2-layer MLP projector: `linear → GELU → linear` maps vision hidden dim to LLM hidden dim
- No causal masking (bidirectional attention in the encoder)

---

## What Airframe Needs (Gap Analysis)

### Current ops inventory (`src/ops/dispatch.rs`):
✅ matmul, matvec, rmsnorm, rope, softmax, attention, attention_with_cache, ffn_swiglu, silu, multiply, add

### Gaps for ViT encoder:
| Op | Needed For | Lines of Work | Notes |
|---|---|---|---|
| `layernorm` | ViT LayerNorm (pre/post attention) | ~30 | Standard LN with gamma/beta, not RMS |
| `gelu` | ViT FFN and MLP projector | ~10 | Standard GELU (tanh approx ok) |
| `patch_embed` | Image → patch token conversion | ~50 | Reshape + linear project; no conv2d needed |
| `add_2d_pos_embed` | Add ViT position embeddings | ~20 | Simple tensor addition |
| `bidirectional_attention` | ViT self-attention (no causal mask) | ~20 | Reuse existing attention, just pass `causal_mask=false` |

### Gaps for token injection in LlamaModel.forward():
| Change | Location | Work |
|---|---|---|
| `prefill_multimodal()` on Engine | `src/runtime/engine.rs` | ~40 lines |
| `forward_with_image_embeds()` on LlamaModel | `src/family/llama.rs` | ~20 lines |
| New `WeightId` variants for vision tensors | `src/core/weight_id.rs` | ~30 lines |
| Vision GGUF tensor name mapping | `src/core/model.rs` | ~50 lines |

### New files needed:
| File | Purpose |
|---|---|
| `src/family/vit.rs` | SigLIP-So400M ViT encoder (27 layers, 1152 hidden, 16 heads) |
| `src/family/resampler.rs` | Resampler projector — cross-attn with 64 learned queries |
| `src/family/image_tiler.rs` | Multi-scale tiler (448×448 base + N crop tiles) |
| `src/runtime/vision_engine.rs` | Orchestrates tile → encode → resample → inject → generate |
| `src/core/image_preprocess.rs` | Resize/normalize to SigLIP format (mean=0.5, std=0.5) |

### Total new Rust code estimate: ~900–1200 lines across 5 new files + ~150 lines of changes to existing files.

---

## Implementation Checklist

### PHASE 0 — Research & Reference (3–5 days)

- [ ] **Download MiniCPM-V-2.6 GGUFs**: `huggingface-cli download openbmb/MiniCPM-V-2_6-gguf --include "*.gguf"`. Two files: `ggml-model-Q4_K_M.gguf` (~5GB, LLM) + `mmproj-model-f16.gguf` (~600MB, ViT+Resampler). Inspect tensor names: `python -c "import gguf; r=gguf.GGUFReader('mmproj-model-f16.gguf'); [print(t.name, t.shape) for t in r.tensors]"`
- [ ] **Map mmproj tensor naming**: Document every tensor in `mmproj-model-f16.gguf`. Expected groups: `resampler.*` (Resampler weights), `vision_model.*` (SigLIP ViT). These names become the new `WeightId` variants.
- [ ] **Confirm SigLIP-So400M config**: From HuggingFace `config.json` — expected: patch_size=14, image_size=448, hidden_dim=1152, n_layers=27, n_heads=16, mlp_dim=4304. Verify against mmproj tensor shapes.
- [ ] **Identify image token ID**: MiniCPM-V uses `<image>` token. Find its token ID in the tokenizer vocab. This is what gets replaced by visual tokens in the LLM sequence.
- [ ] **Run reference inference with llama.cpp mtmd-cli**: The existing `run_mtmd_cli_minicpm_v()` in shimmy-workspace already does this. Run a known test image through it and save: raw output text, token count, timing. This is the parity baseline.
- [ ] **Python oracle for ViT+Resampler**: Write a script using `transformers` + `torch` to extract: (a) SigLIP patch features after 27 ViT layers, (b) Resampler output (64 visual tokens in LLM embedding space). Save both as numpy arrays. These are the ground truth for unit testing Airframe's implementation.
- [ ] **Understand multi-scale tiling**: Run `transformers` MiniCPM-V preprocessing on a 1280×720 web screenshot. Document: how many tiles are generated, what sizes, how tile features are concatenated before the Resampler.

---

### PHASE 1 — Add Missing Ops to Airframe (~1 week)

- [ ] **Add `layernorm` to `OpDispatcher`** (`src/ops/dispatch.rs`):
  - Signature: `pub fn layernorm(&self, input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor>`
  - Standard: `(x - mean) / sqrt(var + eps) * weight + bias`
  - Add unit tests: zero-mean unit-var input → output equals weight. Compare against PyTorch reference.
- [ ] **Add `gelu` to `OpDispatcher`** (`src/ops/dispatch.rs`):
  - Signature: `pub fn gelu(&self, input: &Tensor) -> Result<Tensor>`
  - Use tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
  - Add unit test: gelu(0.0) == 0.0, gelu(1.0) ≈ 0.841.
- [ ] **Add `patch_embed` to `OpDispatcher`** (`src/ops/dispatch.rs`):
  - Signature: `pub fn patch_embed(&self, image: &Tensor, patch_proj: &Tensor, patch_size: usize) -> Result<Tensor>`
  - Takes image `[H, W, 3]`, slices into `[n_patches, patch_size*patch_size*3]`, runs matmul with projection weight → `[n_patches, hidden_dim]`
  - No actual Conv2D needed — patch extraction is a reshape + matmul.
- [ ] **Verify `attention` with `causal_mask=false` works correctly**: The existing `attention()` op takes a `causal_mask: bool`. Test it with `false` for bidirectional (encoder) attention. Write a unit test confirming all tokens attend to all tokens.
- [ ] **Add `add` broadcast variant**: Position embeddings add a `[1, n_patches+1, hidden_dim]` learned embedding to `[n_patches+1, hidden_dim]`. Confirm existing `add` handles this or add a broadcast version.

---

### PHASE 2 — ViT Encoder Implementation (~1 week)

- [ ] **Create `src/family/vit.rs`** — SigLIP-So400M ViT encoder:
  ```rust
  pub struct ViTConfig {
      pub patch_size: usize,   // 14 for SigLIP-So400M
      pub image_size: usize,   // 448 for MiniCPM-V tiles
      pub hidden_dim: usize,   // 1152 for So400M
      pub n_layers: usize,     // 27 for So400M
      pub n_heads: usize,      // 16 for So400M
      pub mlp_dim: usize,      // 4304 for So400M
      pub eps: f32,            // 1e-6
  }
  // ViT layer: LayerNorm → bidirectional MHSA → LayerNorm → GELU FFN
  // Full encoder: patch_embed [H,W,3→1024 patches of dim 1152] → add pos_embed → 27×ViTBlock
  // Returns: [n_patches, 1152] — patch features BEFORE Resampler
  ```
- [ ] **Create `src/family/resampler.rs`** — Resampler projector (the unique piece):
  ```rust
  pub struct ResamplerConfig {
      pub n_queries: usize,    // 64 learned query vectors
      pub hidden_dim: usize,   // 1152 (matches ViT output)
      pub out_dim: usize,      // 3584 (Qwen2-7B n_embd)
      pub n_heads: usize,      // 16
  }
  // Cross-attention: learned queries (Q) attend to patch features (K, V)
  // Uses existing `attention` op with causal_mask=false
  // Returns: [64, 3584] — the visual tokens injected into the LLM
  ```
- [ ] **Unit test ViT encoder**: Load Python oracle patch features (Phase 0), run Airframe ViT on the same image, assert RMS diff < 0.01. Gate on this before continuing.
- [ ] **Unit test Resampler**: Load Python oracle Resampler output (64 visual tokens), run Airframe Resampler, assert RMS diff < 0.01. This is the hardest parity gate.

---

### PHASE 3 — Weight Loading (~3 days)

- [ ] **Extend `WeightId`** (`src/core/weight_id.rs`) with vision tensor variants:
  ```rust
  // ViT encoder weights
  VitPatchProj,           // mm.image_encoder.encoder.patch_embedding.linear.weight
  VitPosEmbed,            // mm.image_encoder.encoder.pos_embed.weight (or pos_embedding)
  VitBlock { layer: usize, part: VitWeightPart },  // per-layer Q/K/V/O/LN1/LN2/MLP
  VitLayerNormPre,        // pre-encoder LN (if any)
  VitLayerNormPost,       // post-encoder LN
  
  // MLP projector weights
  ProjFc1,               // mm.image_projection.fc1.weight
  ProjFc2,               // mm.image_projection.fc2.weight
  ProjFc1Bias,
  ProjFc2Bias,
  ```
- [ ] **Add vision tensor name → WeightId mapping** in `src/core/model.rs`: Parse the mmproj GGUF and map each tensor name string to the appropriate `WeightId` variant. This is the same pattern as the existing LLM tensor name parsing but for a different naming scheme.
- [ ] **Add `Model::from_mmproj_gguf()`** — loads the vision encoder and projector weights from the mmproj GGUF file, returning a separate `VisionModel` struct (or adding to an existing `Model` with a namespace prefix to avoid key collisions with the LLM weights).
- [ ] **Validate all expected vision tensors are loaded**: Add a `validate_vision_weights()` function that checks all required WeightIds are present and have the expected shapes. Fail loudly (not silently) if shapes don't match the config.

---

### PHASE 4 — Token Injection in the LLM Forward Pass (~2 days)

- [ ] **Add `forward_with_image_embeds()` to `LlamaModel`** (`src/family/llama.rs`):
  ```rust
  pub fn forward_with_image_embeds(
      &self,
      input_ids: &[usize],
      image_embeds: Option<&Tensor>,   // [n_img_tokens, n_embd] from projector
      image_token_id: usize,           // the special <image> token ID
      weights: &HashMap<WeightId, Tensor>,
      kv_cache: &mut KvCache,
      ops: &OpDispatcher,
  ) -> Result<Tensor>
  ```
  Logic:
  1. Call `embed_tokens(input_ids)` as normal
  2. Find positions where `input_ids[i] == image_token_id`
  3. Splice `image_embeds` rows into `hidden_states` at those positions (replace the single image placeholder token with N image patch tokens)
  4. Continue the existing forward pass unchanged
- [ ] **Update `Engine::prefill()`** to have a multimodal variant:
  ```rust
  pub fn prefill_multimodal(
      &mut self,
      input_ids: &[usize],
      image_bytes: &[u8],
      vision_model: &VisionEncoder,
      weights: &HashMap<WeightId, Tensor>,
  ) -> Result<Tensor>
  ```
  Orchestrates: preprocess image → ViT encode → MLP project → inject into prefill.
- [ ] **Handle variable sequence length expansion**: MiniCPM-V replaces 1 `<image>` token with 64 visual tokens per tile (Resampler output). A web screenshot with 3 tiles produces 192 visual tokens. Update `kv_cache.complete_prefill()` to use the expanded sequence length, not the original `input_ids.len()`.

---

### PHASE 5 — Image Preprocessing (~2 days)

- [ ] **Create `src/core/image_preprocess.rs`**:
  - `pub fn preprocess_for_siglip(image_bytes: &[u8], target_size: usize) -> Result<Tensor>`
  - Decode → resize tile to 448×448 → normalize: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] → `Tensor [448, 448, 3]` f32.
  - `pub fn tile_image(image_bytes: &[u8], max_slices: usize) -> Result<Vec<Tensor>>` — generates base tile + crop tiles per MiniCPM-V tiling strategy. Web mode: max 2 tiles (512px source). Full mode: up to 6 tiles.
  - Reuse existing `image` crate (already a dep in shimmy-workspace).
- [ ] **Handle RGBA → RGB conversion**: Screenshots often have an alpha channel. Strip alpha before normalizing.
- [ ] **Handle URL → image bytes**: For "web mode" requests, fetch the URL and screenshot it (or fetch the raw image). The existing `vision.rs` already has this logic — reuse it.
- [ ] **Unit test preprocessing**: Run the same image through the Python reference preprocessing (`torchvision.transforms`) and the Airframe preprocessing, assert pixel values match to 4 decimal places.

---

### PHASE 6 — End-to-End Inference Test (~2 days)

- [ ] **Write a `vision_smoke` binary** (`src/bin/vision_smoke.rs`):
  - Usage: `vision_smoke --mmproj mmproj-model-f16.gguf --model ggml-model-Q4_K_M.gguf --image test.png --prompt "Describe this image"`
  - Run full pipeline: load weights → preprocess → ViT encode → MLP project → inject → prefill → decode → print output
  - This is the standalone test harness before wiring into the server.
- [ ] **Parity test vs llama.cpp llava-cli**: Run the same image+prompt through both `vision_smoke` and `llava-cli --mmproj ... --model ...`. Assert the first 50 tokens of output are identical or nearly identical (token-level parity isn't required; semantic parity is).
- [ ] **Benchmark**: Time the full pipeline (preprocess + encode + prefill + 100 decode steps) on the local machine. Compare to the documented MiniCPM-V timings (GPU: 5–15s, CPU: 60–120s). Target: better or equal.

---

### PHASE 7 — Wire into shimmy-workspace Vision API (~3 days)

- [ ] **Update `src/vision.rs`** — replace the `// TODO: Implement actual vision processing` stub:
  ```rust
  // Replace Err("Vision processing not yet implemented".into()) with:
  let airframe_result = airframe::vision::run_vision_inference(
      &preprocessed.bytes,
      &prompt,
      &state.vision_engine,  // new field on AppState
  ).await?;
  ```
- [ ] **Add `vision_engine: Option<VisionEngine>` to `AppState`** in `src/lib.rs` or `src/server.rs`. Load on startup if `SHIMMY_VISION_MODEL_DIR` or auto-download path has both GGUF files.
- [ ] **Add `POST /v1/chat/completions` multimodal support**: Accept `content: [{type: "image_url", ...}, {type: "text", ...}]` message format. This is the standard OpenAI vision API shape — all vision-capable clients already use it.
- [ ] **Wire license check**: Ensure `VisionLicenseManager::validate_license()` is called before any vision inference (existing logic, just verify it still gates the new code path).
- [ ] **Remove all llama.cpp feature flags**: `llama`, `llama-cuda`, `llama-vulkan` feature flags in `Cargo.toml` are no longer needed for vision. The `vision` feature now only requires Airframe's internal vision module.
- [ ] **Test the full API round-trip**: `curl -X POST localhost:11435/api/vision -d '{"image": "base64...", "mode": "image"}' -H "SHIMMY_LICENSE_KEY: test-key"` → expect `VisionResponse` JSON.

---

### PHASE 8 — Qwen2-VL-7B (v2 Upgrade, after MiniCPM-V-2.6 ships)

_Qwen2-VL uses the same Qwen2-7B LLM backbone as MiniCPM-V-2.6. The LLM forward pass is already done at this point. The delta is in the vision encoder._

- [ ] **Research Qwen2-VL vision encoder**: Uses SigLIP encoder + MRoPE (Multimodal Rotary Position Embedding — 2D RoPE applied in the vision encoder itself, not just the LLM). Document: image token resolution, MRoPE position ID computation.
- [ ] **Implement `mrope_2d` position embedding**: Qwen2-VL computes 2D RoPE position IDs for image patches (row × col instead of sequential 1D). This is a new variant of the existing `rope` op. ~60 lines.
- [ ] **Qwen2-VL weight loader**: Map Qwen2-VL mmproj tensor names to WeightId variants. Tensor naming differs from MiniCPM-V mmproj.
- [ ] **Benchmarks vs MiniCPM-V-2.6**: Run both models on the Shimmy Vision test suite (OCR, dom_map, layout, full modes). Qwen2-VL-7B is reported 2–5% better on OCRBench and DocVQA. Verify this holds for web screenshots specifically.
- [ ] **Add `--model qwen2-vl` flag** to `shimmy-vision serve`. Both models load from separate GGUF files; wire to the same tier license check.
- [ ] **Update sales site**: Position Qwen2-VL as the premium model (available on Professional tier and above).

---

### PHASE 9 — CI/CD (Zero cmake, hosted runners only)

- [ ] **New `release.yml` for vision build** — single job matrix, no cmake, no CUDA SDK:
  ```yaml
  strategy:
    matrix:
      include:
        - os: ubuntu-latest   target: x86_64-unknown-linux-gnu
        - os: ubuntu-latest   target: aarch64-unknown-linux-gnu    cross: true
        - os: windows-latest  target: x86_64-pc-windows-msvc
        - os: macos-13        target: x86_64-apple-darwin
        - os: macos-latest    target: aarch64-apple-darwin
  steps:
    - uses: actions/checkout@v4
    - run: cargo build --release --features "vision"    # pure Rust, just works
    - run: cargo test --features "vision"
  ```
- [ ] **Remove Tesseract bundling complexity**: Tesseract OCR runs as a separate process (`ocr.rs`). Keep the existing bundled binary approach from shimmy-workspace but confirm it still builds cleanly without llama.cpp in the deps.
- [ ] **Vision gate in CI**: Add `cargo test --test vision_smoke -- --ignored` as a CI step that downloads MiniCPM-V-2.6 GGUFs (~5.6GB total) and runs a real inference test on the CI runner. Gate the release on this passing. Cache the model files in CI to avoid re-downloading.
- [ ] **Artifact size check**: The model files (~5.6GB) are never bundled in the binary. Auto-download on first use via the existing `ensure_minicpm_v_files()` logic in shimmy-workspace. Binary itself is <5MB.

---

## Summary: Why This Leapfrogs jiangwei-chen

| Problem | jiangwei's situation | Airframe Option A |
|---|---|---|
| Cross-compile llama.cpp | Still fighting cmake/LLVM/CUDA | Eliminated entirely |
| Ship GPU binaries | Needs CUDA SDK in CI | Pure Rust + wgpu, works on any runner |
| Model coupling | Hard-coded to mtmd-cli subprocess | Model is a first-class Airframe crate |
| Precision | Q4 quantized (MiniCPM-V-2.6) | f32 (MiniCPM-V-2.6) — same model, better precision |
| API shape | Custom `/api/vision` | `/v1/chat/completions` multimodal — standard OpenAI shape |
| Packaging | 5-platform sidecar bundle | `cargo install shimmy-vision` — one command, BSL-1.1 |

The moment Airframe ships multimodal, the entire packaging problem that killed this product twice goes away. Anyone can install it with `cargo install shimmy --features vision`. No DLLs, no cmake, no separate model CLI.

---

## Rough Timeline

| Phase | Days |
|---|---|
| Phase 0 — Research & reference oracle | 3–5 |
| Phase 1 — Add ops (layernorm, gelu, patch_embed) | 3–5 |
| Phase 2 — ViT encoder (SigLIP-So400M) | 5–7 |
| Phase 3 — Resampler projector | 3–4 |
| Phase 4 — Multi-scale tiler | 2 |
| Phase 5 — Weight loading (mmproj GGUF) | 2–3 |
| Phase 6 — Token injection in LLM forward pass | 2 |
| Phase 7 — Image preprocessing | 1–2 |
| Phase 8 — E2E smoke test | 2 |
| Phase 9 — Wire into shimmy-workspace API + BSL license | 2–3 |
| Phase 10 — CI/CD (no cmake, hosted runners) | 1–2 |
| **Total (MiniCPM-V-2.6 v1)** | **~4–5 weeks** |
| Phase 11 — Qwen2-VL-7B upgrade | +1.5 weeks |

## Branch Locations

- Airframe work: `feat/vision-multimodal` branch in `C:/Users/micha/repos/airframe`
- shimmy-workspace work: `feat/airframe-vision` branch in `C:/Users/micha/repos/shimmy-workspace`
- Business architecture: see `VISION_BUSINESS_ARCHITECTURE.md` in shimmy-workspace
- License: shimmy-workspace will be updated to BSL-1.1 when vision feature ships
