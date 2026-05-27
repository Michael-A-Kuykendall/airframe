# Airframe Vision — Execution Task Breakdown

**Purpose**: Atomic, self-contained tasks for MiniCPM-V-2.6 multimodal support in Airframe.  
Any AI agent can pick up any task and execute it cold. Each task has zero hidden prereqs beyond what's stated.

---

## What's Already Done (Don't Re-do)

### Airframe (this repo — `feat/vision-multimodal` branch)
- ✅ All LLM ops: `matmul`, `matvec`, `rmsnorm`, `rope`, `softmax`, `attention`, `attention_with_cache`, `ffn_swiglu`, `silu`, `multiply`, `add` — all in `src/ops/dispatch.rs`
- ✅ `LlamaModel` full forward pass (text-only) — `src/family/llama.rs`
- ✅ GGUF weight loader + `WeightId` enum — `src/core/weight_id.rs`, `src/core/model.rs`
- ✅ KV cache, Engine prefill + decode — `src/runtime/`
- ✅ GPU bindless backend (wgpu/WebGPU) — `src/backend/bindless/`
- ✅ HTTP server binary — `src/bin/shimmy_server_gpu.rs`

### shimmy-workspace (separate repo — `feat/airframe-vision` branch)
- ✅ `src/vision.rs` — full response schema + working mtmd-cli subprocess inference (llama.cpp)
- ✅ `src/vision_license.rs` — complete Keygen integration, Ed25519 verification, offline cache, usage metering
- ✅ Stripe → Keygen provisioning pipeline (webhook + Cloudflare worker)
- ✅ `run_mtmd_cli_minicpm_v()` — today's working inference path (subprocess into llama.cpp binary)

**The current shimmy vision path works end-to-end using llama.cpp as the subprocess.**  
The goal of Phase 0–7 is to replace that subprocess with native Airframe inference.

---

## What Needs to Be Built

Below are 28 tasks across 8 phases. The unblocking dependency chain is:
```
Phase 0 (research + oracle) → Phase 1 (ops) → Phase 2 (ViT + Resampler) →
Phase 3 (weight loading) → Phase 4 (token injection) →
Phase 5 (image preprocess) → Phase 6 (smoke binary) → Phase 7 (wire shimmy)
```

Phase 1 tasks are independent of each other (parallel OK).  
Phase 2 tasks are independent of each other (parallel OK).  
All Phase 3 tasks are independent.

---

## Phase 0 — Research & Oracle Setup (No Rust code, ~3–5 days)
*Purpose: build the ground-truth reference data that phases 1–7 test against.*

---

### T-0.1: Download MiniCPM-V-2.6 GGUFs

**Where**: Local machine, any directory (recommend `~/.cache/huggingface/hub/`)  
**What to do**:
```bash
pip install huggingface_hub
huggingface-cli download openbmb/MiniCPM-V-2_6-gguf \
  --include "ggml-model-Q4_K_M.gguf" "mmproj-model-f16.gguf" \
  --local-dir ~/models/minicpm-v-2.6/
```
Expected: two files:
- `ggml-model-Q4_K_M.gguf` — ~5.0 GB (LLM backbone, Qwen2-7B)
- `mmproj-model-f16.gguf` — ~600 MB (SigLIP ViT + Resampler)

**Acceptance**: both files exist, `sha256sum` stored in `artifacts/model_checksums.txt`

---

### T-0.2: Map mmproj Tensor Names

**Where**: Run against the downloaded `mmproj-model-f16.gguf`  
**What to do**:
```python
import gguf
r = gguf.GGUFReader("mmproj-model-f16.gguf")
for t in r.tensors:
    print(t.name, t.shape, t.tensor_type)
```
**Document output in**: `artifacts/mmproj_tensor_map.txt`  
Expected groups: `vision_model.encoder.layers.N.*`, `resampler.query.*`, `resampler.attn.*`, `resampler.ln.*`  
Count all tensors. There should be ~27 ViT layers × ~6 tensors each + resampler weights.

**Acceptance**: `artifacts/mmproj_tensor_map.txt` committed, every tensor named and shaped.

---

### T-0.3: Identify the `<image>` Token ID

**Where**: Run against the LLM GGUF tokenizer  
**What to do**:
```python
import gguf
r = gguf.GGUFReader("ggml-model-Q4_K_M.gguf")
# Look for token pieces table
kv = {k.name: k for k in r.fields}
tokens = kv["tokenizer.ggml.tokens"].parts  # list of token strings
for i, tok in enumerate(tokens):
    s = bytes(tok).decode("utf-8", errors="replace")
    if "<image>" in s or "image" in s.lower():
        print(i, repr(s))
```
**Document in**: `artifacts/image_token_id.txt` (just the integer ID)

**Acceptance**: integer image token ID confirmed.

---

### T-0.4: Python Oracle — ViT Patch Features

**Where**: New script `scripts/vit_oracle.py`  
**What to do**: Using `transformers` + `torch`, load MiniCPM-V-2.6 from HuggingFace (full precision, CPU is fine), run a single 448×448 test image through the SigLIP ViT encoder. Save the output of every ViT layer + the final patch feature tensor as numpy arrays.
```python
from transformers import AutoModel, AutoProcessor
import numpy as np, torch

model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)

# Use test image: fixtures/oracle_22layer_hello.csv already in repo OR
# any 448x448 PNG works for a first pass
img = ...  # load image
inputs = processor(images=img, return_tensors="pt")

# Extract ViT features (before Resampler)
with torch.no_grad():
    vit_out = model.vpm(inputs["pixel_values"])  # [1, n_patches+1, 1152]

np.save("artifacts/oracle_vit_features.npy", vit_out.last_hidden_state.numpy())
```
**Acceptance**: `artifacts/oracle_vit_features.npy` saved, shape `[1, 1025, 1152]` (1024 patches + 1 CLS).

---

### T-0.5: Python Oracle — Resampler Output (64 Visual Tokens)

**Continuation of T-0.4 — same script or new `scripts/resampler_oracle.py`**  
Run the Resampler projector on top of the ViT output. This is the critical oracle — these 64 vectors are what Airframe must reproduce exactly.
```python
with torch.no_grad():
    resampler_out = model.resampler(vit_out.last_hidden_state)  # [1, 64, 3584]

np.save("artifacts/oracle_resampler_output.npy", resampler_out.numpy())
```
**Acceptance**: `artifacts/oracle_resampler_output.npy` saved, shape `[1, 64, 3584]`.

---

### T-0.6: Document Multi-Scale Tiling Behavior

**Where**: `scripts/tiling_probe.py`  
MiniCPM-V-2.6 slices images into 448×448 tiles before encoding. The number of tiles depends on input image aspect ratio.  
**What to do**: Run 4 test images through the MiniCPM-V processor and log how many tiles each produces + tile sizes:
- 448×448 (1 tile)
- 896×448 (2 tiles horizontal)
- 1280×720 (web screenshot — ~3 tiles)
- 1920×1080 (large screenshot — up to 6 tiles)

```python
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
for img in test_images:
    out = p(images=img, return_tensors="pt")
    print(out["pixel_values"].shape)  # [1, n_tiles, 3, 448, 448]
```
**Document in**: `artifacts/tiling_behavior.md` — table of input size → tile count.  
**Acceptance**: table committed; confirms max tiles is ≤ 6 for standard inputs.

---

## Phase 1 — Add Missing Ops to Airframe (~1 week, all tasks independent)
*Prereqs: Phase 0 not required — these are pure Rust implementations.*

---

### T-1.1: Add `layernorm` to OpDispatcher

**Files to touch**:
- `src/ops/reference/activations.rs` — add the function
- `src/ops/dispatch.rs` — add the method

**What to write**:
```rust
// In src/ops/reference/activations.rs
/// Layer Normalization: normalize input, apply scale (weight) + bias (optional)
/// input: [N, D] or [D] — normalized over last dimension
/// weight: [D], bias: Option<[D]>, eps: f32 (typically 1e-6)
pub fn layernorm_f32(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let d = input.shape()[input.shape().len() - 1];
    // For each row: mean, variance, normalize, scale, shift
    // See: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // Must work on 1D [D] and 2D [N, D] shapes
    todo!()
}

// In src/ops/dispatch.rs
pub fn layernorm(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    activations::layernorm_f32(input, weight, bias, eps)
}
```
**Acceptance**: unit test passes comparing against expected values from `((x - mean) / sqrt(var + eps)) * weight + bias` for a known input.

---

### T-1.2: Add `gelu` to OpDispatcher

**Files to touch**:
- `src/ops/reference/activations.rs`
- `src/ops/dispatch.rs`

**What to write**:
```rust
// Gaussian Error Linear Unit — used in ViT FFN blocks (not SwiGLU)
// Standard formula: x * 0.5 * (1 + erf(x / sqrt(2)))
// OR the fast tanh approximation used by most transformers:
// x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Use the tanh approximation — it matches PyTorch's default `gelu` behavior.
pub fn gelu_f32(input: &Tensor) -> Result<Tensor> { todo!() }
```
In `dispatch.rs`:
```rust
pub fn gelu(&self, input: &Tensor) -> Result<Tensor> {
    activations::gelu_f32(input)
}
```
**Acceptance**: for input `[0.0, 1.0, -1.0, 2.0]` output matches PyTorch `torch.nn.functional.gelu(t)` to 4 decimal places.

---

### T-1.3: Add `patch_embed` to OpDispatcher

**Files to touch**:
- `src/ops/reference/mod.rs` — new function or new submodule `vision.rs`
- `src/ops/dispatch.rs`

**What it does**: Converts a raw image tile (float32 pixel values, shape `[3, H, W]`) into patch tokens via a 2D Conv with kernel_size=14, stride=14. Each 14×14 patch maps to a 1152-d vector. Output shape: `[n_patches, 1152]` where `n_patches = (H/14) * (W/14)`.

**What to write**:
```rust
// patch_embed: 2D Conv equivalent for non-overlapping patches
// image: [3, H, W] (C, H, W layout, float32, already normalized to mean/std)
// weight: [1152, 3, 14, 14] (out_channels, in_channels, kH, kW)
// bias: [1152]
// Returns: [n_patches, 1152]
//
// Implementation: for each patch at (ph, pw):
//   Extract sub-array [3, 14, 14], flatten to [3*14*14=588]
//   Matmul with weight reshaped to [1152, 588], add bias
//   Result: [1152]
// Collect all patches -> [n_patches, 1152]
pub fn patch_embed_f32(image: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> { todo!() }
```
**Acceptance**: for a zero-filled 448×448 image with zero weight, output is all zeros with shape `[1024, 1152]`.

---

### T-1.4: Verify Bidirectional Attention

**Files to touch**: `src/ops/reference/attention.rs` — add a test only, no code change expected.

**What to do**: Write a unit test that calls `attention_f32` with `causal_mask = false` and verifies all attention scores are valid (no -inf masking applied). The ViT encoder uses bidirectional attention — every patch must attend to every other patch.

```rust
#[test]
fn test_bidirectional_attention() {
    // 4-token sequence, 2 heads, head_dim=4
    // With causal_mask=false, position [0] should attend to positions [1,2,3]
    // Verify by checking that the output at position 0 is influenced by all tokens
    // (output should differ from causal output at positions that would be masked)
}
```
**Acceptance**: test passes; documents any bug found. If `causal_mask=false` is broken, fix it.

---

### T-1.5: Add `add_broadcast` to OpDispatcher

**Files to touch**: `src/ops/reference/activations.rs`, `src/ops/dispatch.rs`

**What it does**: Standard `add` already exists. ViT adds a learned positional embedding `[1, n_patches+1, D]` to patch features `[n_patches+1, D]`. Check if existing `add` handles this shape pair. If it requires exactly matching shapes, add a broadcast version that follows NumPy broadcasting rules.

**What to do**:
1. Check `src/ops/reference/activations.rs` `add_f32` — read the shape handling
2. If shapes must match exactly, add:
```rust
pub fn add_broadcast_f32(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // follows numpy broadcasting: [1, N, D] + [N, D] -> [N, D]
    // For the specific ViT case: squeeze leading 1 dimension if needed
    todo!()
}
```
**Acceptance**: `[1, 1025, 1152] + [1025, 1152]` returns `[1025, 1152]` with correct elementwise sum.

---

## Phase 2 — ViT Encoder + Resampler (~2 weeks total, both tasks independent)
*Prereqs: T-1.1, T-1.2, T-1.3, T-1.4, T-1.5 all done. T-0.4 + T-0.5 needed for acceptance tests.*

---

### T-2.1: Create `src/family/vit.rs` — SigLIP-So400M Encoder

**File to create**: `src/family/vit.rs`  
**Register in**: `src/family/mod.rs` — add `pub mod vit;`

**Architecture** (from `config.json` on HuggingFace):
- `hidden_dim = 1152`
- `n_layers = 27`
- `n_heads = 16`
- `head_dim = 72` (= 1152 / 16)
- `mlp_dim = 4304` (= intermediate_size in config)
- `patch_size = 14`
- `image_size = 448`
- `n_patches = (448/14)^2 = 1024`
- Positional embedding: `[1, 1025, 1152]` (1 CLS token + 1024 patches)
- Attention: **bidirectional** (`causal_mask = false`)
- FFN activation: **GELU** (not SwiGLU — different from the LLM backbone)
- Normalization: **LayerNorm** (not RMSNorm — different from LLM backbone)

**Struct to write**:
```rust
pub struct SigLipBlock {
    // per-layer weights
    attn_q_weight: Tensor,  attn_k_weight: Tensor,
    attn_v_weight: Tensor,  attn_o_weight: Tensor,
    attn_q_bias: Tensor,    attn_k_bias: Tensor,
    attn_v_bias: Tensor,    attn_o_bias: Tensor,
    ln1_weight: Tensor,     ln1_bias: Tensor,
    ln2_weight: Tensor,     ln2_bias: Tensor,
    mlp_fc1_weight: Tensor, mlp_fc1_bias: Tensor,
    mlp_fc2_weight: Tensor, mlp_fc2_bias: Tensor,
}

pub struct SigLipEncoder {
    patch_weight: Tensor,  // [1152, 3, 14, 14]
    patch_bias: Tensor,    // [1152]
    pos_embedding: Tensor, // [1, 1025, 1152]
    pre_norm_weight: Tensor, pre_norm_bias: Tensor,  // LayerNorm before attention layers
    post_norm_weight: Tensor, post_norm_bias: Tensor, // LayerNorm after all layers
    layers: Vec<SigLipBlock>,  // 27 blocks
}

impl SigLipEncoder {
    pub fn forward(&self, image_pixels: &Tensor, ops: &OpDispatcher) -> Result<Tensor> {
        // 1. patch_embed(image_pixels) -> [1024, 1152]
        // 2. prepend CLS token (learned, from pos_embedding[0]) -> [1025, 1152]
        // 3. add pos_embedding -> [1025, 1152]
        // 4. pre_norm
        // 5. for each layer: pre_layernorm, bidirectional attention, residual
        //    then: post_layernorm, GELU FFN (fc1 -> gelu -> fc2), residual
        // 6. post_norm
        // 7. return [1025, 1152] — ALL tokens including CLS
        todo!()
    }
}
```

**Attention note**: SigLIP attention has Q/K/V bias tensors. The existing `attention_f32` may not accept bias — check signature. If not, either add bias support to `attention_f32` or do the QKV projection manually (matmul + add bias) before calling the core attention kernel.

**Acceptance**: Load oracle from `artifacts/oracle_vit_features.npy`. Run same image through `SigLipEncoder::forward()`. Assert `rms_diff(output, oracle) < 0.01`.

---

### T-2.2: Create `src/family/resampler.rs` — MiniCPM-V Resampler Projector

**File to create**: `src/family/resampler.rs`  
**Register in**: `src/family/mod.rs` — add `pub mod resampler;`

**Architecture** (Perceiver Resampler):
- `n_queries = 64` — learned query vectors `[64, 3584]`
- `hidden_dim = 3584` — Qwen2-7B embedding space
- `n_heads = 16` (typically)
- `n_layers = 1` (single cross-attention layer in MiniCPM-V-2.6)
- Input: ViT output `[1025, 1152]`; Output: `[64, 3584]`

**How it works**: Cross-attention where queries (learned) attend to ViT key/values.
```
Q = learned_query @ W_Q  # [64, 3584] @ [3584, head_dim*n_heads]
K = vit_output @ W_K     # [1025, 1152] @ [1152, head_dim*n_heads]
V = vit_output @ W_V     # same
output = cross_attn(Q, K, V)  # [64, head_dim*n_heads]
output = output @ W_O + bias  # [64, 3584]
output = layernorm(output)
```

**Struct to write**:
```rust
pub struct Resampler {
    query_embeds: Tensor,       // [64, 3584] — the learned queries
    ln_q_weight: Tensor,        // layernorm on queries
    ln_kv_weight: Tensor,       // layernorm on ViT features before KV projection
    attn_q_weight: Tensor,
    attn_k_weight: Tensor,
    attn_v_weight: Tensor,
    attn_out_weight: Tensor,
    attn_out_bias: Tensor,
    ln_post_weight: Tensor,
    ln_post_bias: Tensor,
    proj_weight: Tensor,  // optional final linear [3584, 3584]
}

impl Resampler {
    pub fn forward(&self, vit_features: &Tensor, ops: &OpDispatcher) -> Result<Tensor> {
        // vit_features: [1025, 1152]
        // Returns: [64, 3584]
        todo!()
    }
}
```

**Key note**: The actual tensor names in `mmproj-model-f16.gguf` come from T-0.2. Use that map to confirm the exact WeightId names before writing this.

**Acceptance**: Load oracle from `artifacts/oracle_resampler_output.npy`. Run `Resampler::forward(vit_oracle_features)`. Assert `rms_diff(output, oracle) < 0.01`.

---

## Phase 3 — Weight Loading (~3 days, all tasks independent)
*Prereqs: T-0.2 (tensor map). These tasks can be done before Phase 2 code is finished — just needs T-0.2 done first.*

---

### T-3.1: Add Vision WeightId Variants

**File to touch**: `src/core/weight_id.rs`

**What to do**: Open the file and look at the existing `WeightId` enum. Add new variants for ViT and Resampler tensors. Use the exact names from `artifacts/mmproj_tensor_map.txt` (T-0.2).

Expected variants to add (exact names TBD from T-0.2 output, but typically):
```rust
// SigLIP patch embedding
VisionPatchWeight,
VisionPatchBias,
VisionPosEmbed,
VisionPreNormWeight,
VisionPreNormBias,
VisionPostNormWeight,
VisionPostNormBias,

// Per SigLIP layer (parameterized by layer index)
VisionLayerAttnQWeight(usize),
VisionLayerAttnQBias(usize),
VisionLayerAttnKWeight(usize),
VisionLayerAttnKBias(usize),
VisionLayerAttnVWeight(usize),
VisionLayerAttnVBias(usize),
VisionLayerAttnOWeight(usize),
VisionLayerAttnOBias(usize),
VisionLayerLn1Weight(usize),
VisionLayerLn1Bias(usize),
VisionLayerLn2Weight(usize),
VisionLayerLn2Bias(usize),
VisionLayerMlpFc1Weight(usize),
VisionLayerMlpFc1Bias(usize),
VisionLayerMlpFc2Weight(usize),
VisionLayerMlpFc2Bias(usize),

// Resampler
ResamplerQueryEmbeds,
ResamplerLnQWeight, ResamplerLnQBias,
ResamplerLnKvWeight, ResamplerLnKvBias,
ResamplerAttnQWeight, ResamplerAttnKWeight,
ResamplerAttnVWeight, ResamplerAttnOutWeight, ResamplerAttnOutBias,
ResamplerLnPostWeight, ResamplerLnPostBias,
ResamplerProjWeight,
```

**Acceptance**: `cargo check` passes with new variants; no existing code broken.

---

### T-3.2: Add mmproj Tensor Name → WeightId Mapping

**File to touch**: `src/core/model.rs`

**What to do**: Find where LLM tensor names are parsed (the string → `WeightId` match block). Add a parallel section for vision tensor names. Use the tensor name strings from `artifacts/mmproj_tensor_map.txt`.

Typical pattern in the file:
```rust
// Existing LLM parsing (example)
"token_embd.weight" => WeightId::TokenEmbed,
"blk.0.attn_q.weight" => WeightId::AttnQ(0),
// ...

// New: vision parsing
"vision_model.encoder.layers.0.self_attn.q_proj.weight" => WeightId::VisionLayerAttnQWeight(0),
// etc. for all 27 layers
```

Add a `parse_vision_tensor_name(name: &str) -> Option<WeightId>` function that handles the `vision_model.*` and `resampler.*` prefixes.

**Acceptance**: Can parse all tensor names from `artifacts/mmproj_tensor_map.txt` without returning `None`.

---

### T-3.3: Add `Model::from_mmproj_gguf()`

**File to touch**: `src/core/model.rs`

**What to do**: Add a new constructor that loads the vision encoder GGUF (the `mmproj-model-f16.gguf` file, ~600MB). Returns either a new `VisionModel` struct or extends the existing `Model` with a vision namespace.

```rust
pub struct VisionModel {
    pub tensors: HashMap<WeightId, Tensor>,
}

impl VisionModel {
    pub fn from_mmproj_gguf(path: &Path) -> Result<Self> {
        // Read GGUF file (reuse existing GGUF parsing infrastructure)
        // For each tensor: parse name -> WeightId, dequant if needed, store
        // Call validate_vision_weights() before returning
    }
    
    fn validate_vision_weights(&self) -> Result<()> {
        // Check all required WeightIds are present
        // Check shapes match expected config (e.g., VisionPatchWeight == [1152, 3, 14, 14])
        // Fail loudly if anything is wrong — do NOT silently skip
    }
    
    pub fn get(&self, id: WeightId) -> Result<&Tensor> { ... }
}
```

**Acceptance**: Load `mmproj-model-f16.gguf` from disk. `validate_vision_weights()` passes. Spot-check: `model.get(WeightId::VisionPatchWeight)?.shape() == [1152, 3, 14, 14]`.

---

## Phase 4 — Token Injection in LLM Forward Pass (~2 days)
*Prereqs: Phase 2 AND Phase 3 both done. This is the seam between the vision encoder and the text decoder.*

---

### T-4.1: Add `forward_with_image_embeds()` to `LlamaModel`

**File to touch**: `src/family/llama.rs`

**What it does**: The LLM forward pass currently takes `input_ids: &[u32]` and looks up embeddings. Vision mode needs to splice `[64, 3584]` visual token embeddings into the embedding sequence at the position of the `<image>` token.

**What to add**:
```rust
impl LlamaModel {
    /// Multimodal forward pass — image embeddings replace the <image> token
    /// input_ids: full token sequence including the <image> placeholder
    /// image_embeds: [n_visual_tokens, hidden_dim] — from Resampler output
    ///   For MiniCPM-V-2.6: n_visual_tokens = 64 * n_tiles
    /// image_token_id: the integer ID of the <image> placeholder token (from T-0.3)
    pub fn forward_with_image_embeds(
        &self,
        input_ids: &[u32],
        image_embeds: &Tensor,  // [n_visual_tokens, hidden_dim]
        image_token_id: u32,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        // 1. Look up text token embeddings as usual (n_tokens, hidden_dim)
        // 2. Find the position(s) of image_token_id in input_ids
        // 3. Replace that row with the n_visual_tokens rows from image_embeds
        //    (expand: splice 64 rows in place of 1 row)
        // 4. Run the resulting embedding sequence through all transformer layers
        //    (same as normal forward pass from here on)
        // Returns logits for the last token position
        todo!()
    }
}
```

**Acceptance**: Unit test — create a 5-token sequence with one `<image>` at position 2. After splice, the embedding matrix has shape `[5 + 63, hidden_dim]` (1 replaced by 64). Verify the shape is correct before the first transformer layer.

---

### T-4.2: Update `Engine::prefill()` for Multimodal Input

**File to touch**: `src/runtime/engine.rs`

**What to add**:
```rust
pub struct MultimodalInput {
    pub input_ids: Vec<u32>,
    pub image_embeds: Tensor,    // [n_visual_tokens, hidden_dim]
    pub image_token_id: u32,
}

impl Engine {
    pub fn prefill_multimodal(&mut self, input: MultimodalInput) -> Result<()> {
        // Call model.forward_with_image_embeds(...)
        // Update KV cache with the expanded sequence length (input_ids.len() - 1 + n_visual_tokens)
        // Store the next-token logits for decode to pick up
    }
}
```

**KV cache note**: After multimodal prefill, `kv_cache.complete_prefill()` must record the *expanded* sequence length (e.g., for a 100-token prompt with 1 `<image>` token replaced by 64 visual tokens, `kv_cache_len = 100 + 63 = 163`). Check what `complete_prefill()` currently accepts and update accordingly.

**Acceptance**: `prefill_multimodal` runs without panic on a fake `[64, 3584]` image embed tensor. KV cache length is `input_ids.len() - 1 + 64` after completion.

---

## Phase 5 — Image Preprocessing (~2 days)
*Prereqs: T-0.6 (tiling behavior). No dependency on Phase 2/3/4 — can be done in parallel with them.*

---

### T-5.1: Create `src/core/image_preprocess.rs`

**File to create**: `src/core/image_preprocess.rs`  
**Register in**: `src/core/mod.rs` — add `pub mod image_preprocess;`  
**Add to `Cargo.toml`**: `image = "0.25"` (if not already present)

**What to write**:
```rust
/// MiniCPM-V-2.6 image preprocessing pipeline
/// 
/// 1. Resize to fit within max long edge (preserving aspect ratio)
/// 2. Determine tile layout based on aspect ratio (see T-0.6 output)
/// 3. For each tile: resize to 448x448, normalize with SigLIP mean/std
/// 4. Return Vec<Tensor> of tiles, each [3, 448, 448] (CHW, float32)
///
/// SigLIP normalization constants:
///   mean = [0.5, 0.5, 0.5]
///   std  = [0.5, 0.5, 0.5]
///
/// RGBA images: strip alpha channel before normalizing.

pub struct ImagePreprocessor {
    pub max_tiles: usize,  // default 6
    pub tile_size: usize,  // always 448
}

pub struct PreprocessedImage {
    pub tiles: Vec<Tensor>,  // each [3, 448, 448], n_tiles total
    pub original_size: (u32, u32),
    pub tile_grid: (usize, usize),  // (rows, cols)
}

impl ImagePreprocessor {
    pub fn from_bytes(&self, image_bytes: &[u8]) -> Result<PreprocessedImage> { todo!() }
    pub fn from_path(&self, path: &Path) -> Result<PreprocessedImage> { todo!() }
}
```

**Acceptance**: Run the same JPEG through this and through the Python reference in T-0.6. Pixel values in each tile must match to 4 decimal places.

---

## Phase 6 — End-to-End Smoke Test (~2 days)
*Prereqs: ALL of Phase 1–5 done. This is the integration gate before wiring shimmy.*

---

### T-6.1: Create `src/bin/vision_smoke.rs`

**File to create**: `src/bin/vision_smoke.rs`

**What to write**: A binary that takes `--image path/to/image.png --prompt "describe this image"` and:
1. Loads `mmproj-model-f16.gguf` → `VisionModel`
2. Loads `ggml-model-Q4_K_M.gguf` → LLM `Model`
3. Preprocesses the image → `PreprocessedImage`
4. Runs `SigLipEncoder::forward()` on each tile
5. Concatenates ViT outputs across tiles, runs `Resampler::forward()`
6. Tokenizes prompt text + builds input_ids with `<image>` placeholder
7. Calls `engine.prefill_multimodal()`
8. Decodes 200 tokens with greedy sampling
9. Prints the result

**Acceptance**: On a test image (e.g., `fixtures/test_image.png`), produces coherent text describing the image. Does not require token-exact parity with llama.cpp — semantic parity is enough. Time the full pipeline and report it.

---

### T-6.2: Parity Check vs llama.cpp mtmd-cli

**No new files.** Run the same image through:
1. `vision_smoke` (Airframe)
2. shimmy-workspace's existing `run_mtmd_cli_minicpm_v()` (llama.cpp)

Compare output texts. Both should describe the same image content. Document the comparison in `artifacts/vit_parity_check.md`.

**Acceptance**: Both outputs identify the same major objects/content. Timing comparison documented.

---

## Phase 7 — Wire into shimmy-workspace (~3 days)
*Prereqs: Phase 6 done + T-6.2 parity check passed. These tasks are in `shimmy-workspace`, not `airframe`.*

---

### T-7.1: Replace `run_mtmd_cli_minicpm_v()` with Airframe VisionEngine

**File to touch**: `shimmy-workspace/src/vision.rs`

The current `process_image()` function calls out to `run_mtmd_cli_minicpm_v()` which shells out to a llama.cpp binary. Replace the internals with an `airframe` library call.

```rust
// In Cargo.toml of shimmy-workspace, add:
// airframe = { path = "../airframe", features = ["vision"] }

// In vision.rs:
use airframe::vision::{VisionEngine, VisionEngineConfig};

async fn process_image_airframe(
    request: &VisionRequest,
    engine: &VisionEngine,
) -> Result<VisionResponse> {
    let image_bytes = load_image_bytes(request)?;
    let prompt = build_vision_prompt(request);
    let raw_output = engine.run(image_bytes, &prompt).await?;
    parse_vision_response(raw_output, request)
}
```

**Note**: `VisionEngine` is a new public struct to be added to `airframe/src/lib.rs` — a thin wrapper over `ImagePreprocessor` + `SigLipEncoder` + `Resampler` + `Engine` that exposes a simple `run(image_bytes, prompt) -> String` API.

**Acceptance**: `shimmy-workspace` builds with `--features vision`. `POST /api/vision` responds with a `VisionResponse` without calling any external binary.

---

### T-7.2: Add `VisionEngine` to `AppState`

**File to touch**: `shimmy-workspace/src/server.rs` or `src/lib.rs` (wherever `AppState` is defined)

```rust
pub struct AppState {
    // ... existing fields ...
    pub vision_engine: Option<VisionEngine>,  // None if model files not found
}
```

Load on startup:
```rust
let vision_engine = VisionEngine::from_paths(
    &config.vision_mmproj_path,   // mmproj-model-f16.gguf
    &config.vision_llm_path,      // ggml-model-Q4_K_M.gguf
).ok();  // None is fine — vision feature just won't be available
```

**Acceptance**: Server starts cleanly with `vision_engine = None` when model files are absent. Does not panic, returns `503 Vision model not loaded` gracefully.

---

### T-7.3: Verify License Gate Still Works

**File to touch**: `shimmy-workspace/src/vision.rs` — read and trace the `VisionLicenseManager::validate_license()` call.

The existing license check in `vision.rs` must still fire before any `VisionEngine::run()` call. Verify the call order hasn't changed. If refactoring vision.rs breaks the gate, fix it.

**Acceptance**: Without a valid license key, `process_image_airframe()` returns `401 Unauthorized` before calling `engine.run()`. Confirm with a unit test using a mock `VisionLicenseManager`.

---

### T-7.4: Remove llama.cpp Feature Flags from shimmy-workspace

**File to touch**: `shimmy-workspace/Cargo.toml`

Remove or obsolete:
- `llama`, `llama-cuda`, `llama-vulkan` feature flags (only needed for the mtmd-cli subprocess path)
- Any `llama-cpp-sys` or similar C bindings dep

**Acceptance**: `cargo build --features vision` completes without compiling any C/C++ code. `cargo build` (no vision feature) also still works. Check `cargo tree | grep llama` returns nothing.

---

## Phase 8 — Qwen2-VL-7B Upgrade (After MiniCPM-V-2.6 ships)
*Don't start until Phase 7 is fully complete and deployed.*

| Task | Description |
|------|-------------|
| T-8.1 | Research Qwen2-VL vision encoder: MRoPE (2D RoPE for patches) vs MiniCPM-V's standard RoPE |
| T-8.2 | Implement `mrope_2d` op: 2D rotary position embedding for image patch coordinates |
| T-8.3 | New `src/family/qwen2vl_vit.rs` using MRoPE instead of position embedding additive |
| T-8.4 | Qwen2-VL mmproj weight loader (different tensor names from MiniCPM-V) |
| T-8.5 | `vision_smoke --model qwen2-vl` benchmark vs MiniCPM-V on web screenshots |
| T-8.6 | Add `--model qwen2-vl` flag to shimmy-workspace vision path |
| T-8.7 | Update Keygen: Qwen2-VL gated to Professional tier and above |

---

## Phase 9 — CI/CD (After Phase 7, before public launch)

| Task | Description |
|------|-------------|
| T-9.1 | Add `cargo test --features vision` to CI (mocked, no real model files) |
| T-9.2 | Add `cargo test --test vision_smoke -- --ignored` step that downloads GGUFs and runs real inference (gated, cache model files in CI) |
| T-9.3 | Confirm `cargo build --release` produces a single binary with no cmake/CUDA SDK dependency |
| T-9.4 | Artifact size check: binary < 5MB; model files auto-downloaded on first use (not bundled) |
| T-9.5 | Remove Tesseract bundling complexity if Airframe covers the OCR path natively |

---

## Task Dependency Summary

```
Phase 0 (research):
  T-0.1 → T-0.2 → T-3.1, T-3.2, T-3.3, T-2.1, T-2.2
  T-0.4 → T-0.5 (oracle for parity tests)
  T-0.3 (image token ID, needed for T-4.1)
  T-0.6 (tiling, needed for T-5.1)

Phase 1 (ops) — all independent, no prereqs:
  T-1.1, T-1.2, T-1.3, T-1.4, T-1.5

Phase 2 (encoder) — needs Phase 1 + T-0.2:
  T-2.1 (ViT) — independent of T-2.2
  T-2.2 (Resampler) — independent of T-2.1

Phase 3 (weight loading) — needs T-0.2:
  T-3.1, T-3.2, T-3.3 — all independent of each other

Phase 4 (token injection) — needs Phase 2 + Phase 3 complete:
  T-4.1 → T-4.2

Phase 5 (image preprocess) — needs T-0.6 only (parallel with 2/3/4):
  T-5.1

Phase 6 (smoke test) — needs all of 1-5:
  T-6.1 → T-6.2

Phase 7 (wire shimmy) — needs Phase 6:
  T-7.1, T-7.2 (independent) → T-7.3 → T-7.4
```

---

## Rough Timeline

| Phase | Parallelizable? | Days |
|-------|----------------|------|
| 0 — Research + oracle | Some tasks parallel | 3–5 |
| 1 — Add ops | All 5 tasks in parallel | 3–5 |
| 2 — ViT + Resampler | 2 tasks in parallel | 5–7 |
| 3 — Weight loading | 3 tasks in parallel | 2–3 |
| 4 — Token injection | Sequential | 2 |
| 5 — Image preprocess | Parallel with 2/3/4 | 2 |
| 6 — Smoke test + parity | Sequential | 2 |
| 7 — Wire shimmy | Mostly parallel | 3 |
| **Total to v1** | | **~4.5 weeks** |
| 8 — Qwen2-VL | After v1 ships | +1.5 weeks |

---

*Document current as of May 2026. Branch: `feat/vision-multimodal` (airframe) + `feat/airframe-vision` (shimmy-workspace).*
