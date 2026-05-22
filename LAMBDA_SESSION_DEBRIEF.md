# Lambda A100 — Shimmy v2.0 Validation Session Brief
**Date:** 2026-05-21 / 2026-05-22  
**Instance:** 150.136.92.11 — Lambda Labs A100-SXM4-40GB (40 960 MiB VRAM)  
**Session goal:** Bring `shimmy_server_gpu` from first-boot-panic to multi-model inference on Vulkan/Linux

---

## FINAL DEBRIEF — May 22, 2026 (End of Session — Safe to Terminate Instance)

### Gemma-2-2B-IT Final Smoke Results (`airframe-v2-gpu` @ `f23d42e`)

Run on A100-SXM4-40GB, model: `gemma-2-2b-it-Q4_K_M.gguf`, temperature 0.

| Prompt | Response | Stop Reason | Pass? |
|--------|----------|-------------|-------|
| `"What is 2+2? Reply with just the number."` | `"2 "` | `end_of_turn` | ✅ |
| `"The capital of France is"` | `"The capital of France is **Paris**."` | `max_tokens` | ✅ |
| `"What is 3+3? Answer with one word."` | `"Six "` | `end_of_turn` | ✅ |
| `"What is 2+2?"` | `"2 + 2 = **4** "` | `end_of_turn` | ✅ |

TinyLlama regression: `"What is 2+2?"` → `"2 + 2 = 4"` ✅

### Commits Merged into `airframe-v2-gpu`

```
f23d42e  docs: update RELEASE_STATUS.md — all v2.0 gates green, 7-model coverage, full commit list
f4ca4c1  fix(pipeline): split compute passes for memory barriers; add PostAttnNorm/PostFfwNorm dispatch
d33048f  fix: Gemma-2 inference - GELU activation, embedding scale, logit softcap, parse_special tokenization
32455a1  feat(gemma2): post_attention_norm, post_ffw_norm, final_logit_softcap
8b5f9ef  feat(gemma2): attn_logit_softcap in LayerParams; use_cpu_head binding-limit guard
4dabeb9  fix(gemma2): correct head_dim to 256 via attention.key_length; remove formula dump diagnostics
```

### Root Cause Summary — 6 Gemma-2 Bugs Fixed

**Bug 1 — Wrong gate activation (SiLU → GeGLU)** — `d33048f`, `sh_layer_v1.wgsl`  
Gemma-2 uses Gated GELU (GeGLU), not SiLU. Fixed by branching on `attn_logit_softcap > 0.0`.

**Bug 2 — Missing embedding scale** — `d33048f`, `server_inference.rs`  
Gemma-2 requires all input embeddings multiplied by `sqrt(n_embd)` = 48.0. Applied at all 4 embedding lookup sites.

**Bug 3 — Missing final logit softcap** — `d33048f`, `server_inference.rs`  
Gemma-2 applies `tanh(logit / 30.0) * 30.0` to all output logits before sampling.

**Bug 4 — parse_special tokenization** — `d33048f`, `server_inference.rs`  
Gemma-2's `<start_of_turn>` (token 106) and `<end_of_turn>` (token 107) are `Control` type — not in the trie. Without `parse_special: true`, template strings encode as individual bytes causing garbage output. Fix: `encode_with_options(&prompt, &EncodeOptions::with_parse_special(true, true))`.

**Bug 5 — No stop on end_of_turn** — `d33048f`, `server_inference.rs`  
Decode loop had no mechanism to stop on token 107. Fixed by encoding the stop token once at startup.

**Bug 6 — No GPU memory barriers between dependent kernels** — `f4ca4c1`, `pipeline/inference.rs`  
All kernels per layer were in a single `ComputePass`. Fixed by giving each kernel its own pass scope.

### Validated Model Coverage (v2.0)

| Model | Arch | Quant | Status |
|-------|------|-------|--------|
| TinyLlama-1.1B-Chat-v1.0 | llama | Q4_0 | ✅ |
| Llama-3.2-1B-Instruct | llama | Q4_K_M | ✅ |
| Llama-3.2-3B-Instruct | llama | Q4_K_M | ✅ |
| Gemma-2-2B-IT | gemma2 | Q4_K_M | ✅ |
| phi-2 | phi2 | Q4_K_M | ✅ |
| starcoder2-3b | starcoder2 | Q4_K_M | ✅ |
| gpt2 | gpt2 | Q4_K_M | ✅ |

WGSL quant formats: F32, F16, Q4_0, Q8_0, Q4_K(M/S), Q5_K(M/S), Q6_K ≈ **85–90%** of GGUF downloads.  
Not implemented: Q5_0, IQ2/IQ3/IQ4 imatrix — deferred to v2.1.

### Known Limitations Deferred to v2.1

- Models with single tensors >2 GB (wgpu buffer cap) — Llama-3.1-8B, Gemma-2-27B
- Gemma-2 sliding window attention (alternates local 4096 / global per layer)
- IQ quant formats (IQ2/IQ3/IQ4)
- Lazy KV allocation for Llama 128K on consumer GPU

### Instance Shutdown Checklist

- [x] All 6 Gemma-2 bugs fixed and committed
- [x] `feat/gemma2-post-norms` merged into `airframe-v2-gpu`
- [x] TinyLlama regression passes
- [x] Gemma-2 final smoke passes (4/4 prompts)
- [x] RELEASE_STATUS.md updated
- [x] `git push origin airframe-v2-gpu` — pushed ✅ (done post-session by local agent)
- [x] Terminate Lambda instance

---

## 1. What was proven

### TinyLlama 1.1B Q4_0 — FULL PASS ✅
- Server starts, all pipelines compile, inference completes
- `"What is 2+2?" → "2 + 2 = 4"`
- `prompt_tokens: 35`, `tokens_generated: 8`, `stop_reason: max_tokens`

### Gemma-2-2b-it Q4_K_M — FULL PASS ✅ (all 6 bugs fixed during session)
- Server starts without crash
- Dynamic ModelSpec read from GGUF metadata: `arch=Gemma 2 2b It, n_layer=26, n_embd=2304, n_vocab=256000`
- Q4K pipelines (7 entry points) all compiled on A100 Vulkan
- Output head correctly routed to Q6K GPU path (2250 MB F32 > 2 GB wgpu limit)
- Inference completes: 13 tokens, `stop_reason: eos`
- **Bug:** output is wrong — ChatML template (`<|im_start|>`) sent to a model trained on `<start_of_turn>user` format. Model echoes question instead of answering.
- Root cause is a 3-line code fix; GPU path itself is sound.

---

## 2. Root causes fixed during this session

| # | Bug | Fix |
|---|-----|-----|
| 1 | `create_buffer_init` triggered `StagingBuffer::new` → `handle_hal_error` → permanent device loss on Vulkan; every subsequent buffer returned `Fallible::Invalid`, causing panic on first inference | All 29 inference-path `create_buffer_init` calls in `pipeline.rs`, `pipeline_shift.rs`, `preflight.rs` replaced with `create_buffer(mapped_at_creation: false)` + `queue.write_buffer` |
| 2 | 608 MB GGUF and 250 MB output head used `queue.write_buffer`, which internally also uses the staging belt → same device-loss path | Large buffers switched to `mapped_at_creation: true` + `copy_buffer_to_buffer` in 8 MB chunks |
| 3 | Heap fragmentation: pipeline compilation after a 250 MB output-head staging upload exhausted HOST_VISIBLE heap | Startup reordered: GGUF load → output head upload → pipeline compilation |
| 4 | `main_attn_out` WGSL entry used `var scores: array<f32, 8192>` — 32 KB per thread × 256 threads = 8 MB thread-private memory per workgroup. NVIDIA Vulkan compiler returned `VK_ERROR_DEVICE_LOST` during `create_compute_pipeline`, silently killing the device | Replaced with online Flash Attention (O(1) scalars: `running_max`, `running_sum`, `running_out`) |
| 5 | `/v1/chat/completions` endpoint ignored the `messages[]` array and always ran a hard-coded "story" prompt | Added `ChatMessage` struct + `messages: Vec<ChatMessage>` to `InferenceRequest`; builds ChatML template before dispatch |
| 6 | `<|im_end|>` stop token leaked into TinyLlama output | `EncodeOptions::with_parse_special(true)` so the tokenizer recognises it as a single special ID |
| 7 | `prompt_tokens` / `completion_tokens` absent from response | Added to `InferenceResponse` struct and wired through the token loop |
| 8 | Gemma-2's 256K-vocab output head is 2.25 GB — exceeds wgpu 2 GB buffer limit | Added `GPU_MAX_BUFFER_BYTES` guard: skip F32 pre-dequant for large-vocab models; use Q6K GPU output projection path instead |
| 9 | All models previously assumed TinyLlama's hardcoded spec | `ModelSpec::from_gguf_metadata()` now reads arch, layer count, dims, vocab from the GGUF file; any model auto-configures |

---

## 3. Branch state at session close

### Local (Windows, RTX 3060) — post-merge state

| Branch | Tip | Notes |
|--------|-----|-------|
| `feat/streaming-and-discovery` | `144246d` | HEAD — pushed to origin ✅ |
| `airframe-v2-gpu` | `8a1b1c3` | Lambda merged + Q4K shader fix — pushed to origin ✅ |
| `master` | `419c0d8` | Not yet fast-forwarded |
| `shimmy_integration/main` | `961cbf8` | Pushed to `private` ✅ |

**Dirty working tree (not committed):**
- `chat-lambda.md` — session log (append)
- `chat.md` — untracked session log
- `shimmy_integration` pointer — already reflects `beadc6d`

### Lambda (A100)

| Branch | Tip | Notes |
|--------|-----|-------|
| `lambda-vulkan-gpu-fix` | `fff71ab` | 3 commits on top of `origin/master` |

All 3 Lambda commits are fetched into local repo under `lambda/lambda-vulkan-gpu-fix`.  
Lambda's origin key is read-only so it cannot push to GitHub directly; fetch via `lambda` remote works.

**Lambda commits:**

| Hash | Summary |
|------|---------|
| `fff71ab` | feat(gpu): Q4K/Q6K shader support, multi-model pipeline expansion |
| `cbfb9bd` | fix(gpu): online softmax, chat messages support, prompt_tokens |
| `5a73474` | fix(vulkan): bypass staging belt for large buffer uploads |

---

## 4. What the Lambda branch has that local does NOT yet integrate

The 3 new WGSL shaders have been ported to local `airframe-v2-gpu` (`419c0d8`). The following **Rust-side** changes from `lambda-vulkan-gpu-fix` still need to be ported into the locally-refactored `pipeline/` structure and `server_inference.rs`:

| File (Lambda) | Local equivalent | What needs porting |
|---|---|---|
| `pipeline.rs` (monolith) | `pipeline/mod.rs` + sub-files | Q4K/Q6K pipeline fields, `readback_a` buffer, Q4K struct fields (`post_attn_norm`, `post_ffn_norm`, `attn_softcap`, `v_is_q4k`, `ffn_down_is_q4k`) |
| `shimmy_server_gpu.rs` | `shimmy_server_gpu/server_inference.rs` | NaN sanitization in `sample_token`, `completion_tokens` alias, dynamic spec from GGUF, `use_q4k` flag, conditional output head upload, Q6K offset lookup, weight-tied fallback (`token_embd.weight`) |
| `spec.rs` | same | `GgufFileType::Q4_K` / `Q4_K_M` variants |
| `model.rs` / `metadata.rs` | same | `get_tensor_type`, `get_tensor_offset` APIs |
| `llama.rs`, `engine.rs`, `multi_token_engine.rs` | same | Minor Q4K-aware updates |

---

## 5. What still needs to be done for Gemma-2 to fully pass

1. **Chat template per arch** — detect `arch == "gemma2"` from spec, use:
   ```
   <start_of_turn>user
   {content}<end_of_turn>
   <start_of_turn>model
   ```
   instead of the ChatML `<|im_start|>` format.

2. **Stop token** — Gemma-2's EOS for the chat format is `<end_of_turn>` (token ID 107). The job completed at `id=107` in the log, which is correct — but the text should be stripped before that token leaks.

3. **Post-attention / post-FFN norms** — Gemma-2 has extra RMSNorm after attention output and after FFN output (unique to the Gemma-2 architecture). The Q4K layer shader has entry points (`main_post_attn_norm`, `main_post_ffn_norm`) already written; the dispatch loop in `server_inference.rs` needs to call them conditionally when `use_post_norms == true`.

4. **Attention logit soft-cap** — Gemma-2 applies `tanh(logit / 50.0) * 50.0` before softmax. The Q4K attn shader already has the `attn_softcap` field in the params struct; the server just needs to populate it (`spec.attn_softcap = 50.0` for Gemma-2).

---

## 6. What was NOT done (next Lambda session or large-VRAM environment)

The A100 had 37.9 GB free at the end of this session. These items were not done because only 2 models were on the instance:

| Item | Why it matters |
|------|---------------|
| Download + test Llama-3.2-1B/3B Q4_K_M | First Q4K cross-validation against a non-Gemma model on Vulkan |
| Download + test Phi-2 Q4_K_M (phi2 arch) | Different architecture; validates arch-detection path |
| Download + test StarCoder2-3B Q4_K_M | Code-generation model; validates Q4K on starcoder2 arch |
| `quant_verify` on TinyLlama and Llama-3.2-1B | CPU vs GPU dequant agreement on A100 Vulkan (≤2 GB models) |
| Needle bench at 8K–16K ctx | Llama-3.2 supports 131 072 ctx; A100 VRAM can hold much larger KV cache than RTX 3060 |
| Download + test DeepSeek-Coder-6.7B / Qwen2-7B | 4+ GB models — impossible on RTX 3060 12GB; needs A100 or equivalent |
| Shimmy provider smoke against Lambda server | End-to-end shimmy_integration → Airframe → Vulkan stack validation |

---

## 7. Next session action items (ordered)

1. **`git branch -f master airframe-v2-gpu`** — advance local master pointer (no checkout needed)
2. **Delete `merge-vet-preview` worktree** — `git worktree remove --force c:/Users/micha/repos/airframe-merge-vet`
3. **Port Lambda Rust-side Q4K/Q6K changes** to the locally-refactored pipeline/ and server_inference.rs (see §4 above)
4. **Fix Gemma-2 chat template** in `build_templated_prompt` — arch-conditional dispatch
5. **Wire `post_attn_norm`, `post_ffn_norm`, `attn_softcap`** into the Q4K dispatch loop
6. **Run local build + tests** — confirm pipeline compilation and all 346 shimmy_integration tests still pass
7. **Push `airframe-v2-gpu` to origin**
8. **Run local model smoke test** with TinyLlama to verify nothing regressed
9. **On next GPU instance:** download all verified models, run full smoke test, validate Gemma-2 with corrected template

---

## 8. GPU memory note (A100 at shutdown)

```
NVIDIA A100-SXM4-40GB: 40 960 MiB total | 2 511 MiB used | 37 931 MiB free
```

Gemma-2-2b-it Q4_K_M uses ~2.5 GB VRAM including KV cache — leaves 37+ GB free. A 7B model at Q4_K_M (~4 GB) would run comfortably. A 13B model (~8 GB) would also fit. This instance is well-provisioned for multi-model concurrent testing.

---

*End of debrief. Safe to shut down the Lambda instance.*
