# Shimmy v2.0 — Issue Triage, PR Guidance & Pre-Release Checklist

**Generated:** May 20, 2026  
**Repo state:** `private/main` @ `efe7c17`  
**Engine:** Airframe (wgpu/WebGPU) — llama.cpp fully removed from history

> **How to use this doc:** Click the linked issue number → read the thread → paste the response below it → close.

---

## What Changed in v2.0 (Context for All Responses Below)

| Item | Status |
|------|--------|
| llama.cpp removed | ✅ Deleted from source AND scrubbed from all git history |
| Airframe GPU engine (WebGPU/wgpu) | ✅ Default engine for all GGUF inference |
| `src/engine/llama.rs` | ✅ Deleted — does not exist in v2.0 |
| `src/engine/universal.rs` | ✅ Updated — LlamaGGUF arm returns helpful error pointing to Airframe |
| `--legacy` flag | ✅ Routes to CPU adapter (no llama.cpp) |
| `cargo install shimmy` | ✅ Works, uses HuggingFace engine |
| GPU detection | ✅ Automatic via wgpu — no CUDA/Vulkan SDK required |
| Ollama-compat API `/api/generate`, `/api/tags` | ✅ Supported |
| Multi-part content array (issue #191) | ✅ Fixed, regression test added |
| Chinese user manuals | ✅ `docs/USER_MANUAL.zh-CN.md` + `docs/USER_MANUAL.zh-TW.md` |
| Sensitive file history scrub | ✅ 59+ paths removed from all 1540+ commits |

---

## Open PRs — Contributor Guidance

### [PR #198](https://github.com/Michael-A-Kuykendall/shimmy/pull/198) — LopezNuance — `feat/ollama-stop-tokens`
**Branch:** `feat/ollama-stop-tokens`  
**What it does:** Reads per-model stop tokens from `ollama show <model>` at load time and applies them during generation. Adds `src/model_registry.rs`, `src/model_manager.rs`, `src/main_integration.rs`, `src/preloading.rs`.

**The problem with merging as-is:**
- Modifies `src/engine/llama.rs` — **this file no longer exists in v2.0**
- Requires Ollama CLI to be installed and running — we're moving to direct GGUF metadata reads, not Ollama as an operational dependency
- `src/model_registry.rs` conflicts with our rewritten version
- Needs rebase — `universal.rs`, `engine/mod.rs`, and `main.rs` all changed significantly

**The diagnosis is 100% correct.** Models like `exaone-deep:2.4b` and `cogito:3b` do generate past EOS because their GGUF doesn't advertise stop tokens the way our sampler expects. This is a real bug that needs fixing.

**What to tell LopezNuance — post this as a PR comment:**

> Hey, thank you for digging into this — the stop token gap you identified is real and affects a meaningful set of models. We've done a major architecture change for v2.0: llama.cpp has been fully removed and replaced with Airframe, our pure-Rust WebGPU inference engine. `src/engine/llama.rs` no longer exists.
>
> The right fix for v2.0 is reading stop tokens directly from GGUF metadata fields (`tokenizer.ggml.eos_token_id`, `tokenizer.ggml.eot_token_id`, `tokenizer.ggml.padding_token_id`) rather than from `ollama show`, which introduces an Ollama runtime dependency we're trying to eliminate.
>
> Your `src/model_manager.rs` and `src/preloading.rs` infrastructure is solid and we'd love to carry it forward. Would you be willing to revise this PR to:
> 1. Drop the `src/engine/llama.rs` changes (file deleted)
> 2. Replace the `ollama show` call with GGUF metadata reads from the already-loaded `GgufFile` struct  
> 3. Rebase on current `main`
>
> We're happy to review quickly. The core work here is good and solves a real user-visible issue.

---

### [PR #196](https://github.com/Michael-A-Kuykendall/shimmy/pull/196) — LopezNuance — `fix/sampler-chain-and-kv-eviction`
**Branch:** `fix/sampler-chain-and-kv-eviction`  
**What it does:** Fixes degenerate repetitive output on long generations. Diagnosed 5 root causes vs Ollama: penalty params swapped (repeat penalty was amplifying repetition), sampler chain inverted (penalties applied after top-k instead of before), greedy not probabilistic, KV eviction destroying the system prompt, no DRY sampler.

**The problem with merging as-is:**
- Modifies `src/engine/llama.rs` — **deleted in v2.0**
- Needs rebase — `universal.rs` and `engine/mod.rs` both changed significantly
- The sampler/penalty fixes in `huggingface.rs` and `openai_compat.rs` are still valid and still needed

**The diagnosis is solid.** Side-by-side against Ollama with identical params proved the degeneration is Shimmy-specific. The penalty param swap and sampler chain order issues are real bugs that apply to the HuggingFace engine path too.

**Action items before responding:**
- [ ] Verify current `src/engine/huggingface.rs` sampling chain order — does it still have the chain-inverted bug?
- [ ] Verify `src/openai_compat.rs` penalty param mapping — are `frequency_penalty` and `presence_penalty` mapped correctly?
- [ ] Check KV eviction behavior — does v2.0's Airframe path preserve system prompt across context window?

**What to tell LopezNuance — post this as a PR comment:**

> This is excellent research — running identical prompts side-by-side against Ollama to isolate the degeneration to Shimmy-specific behavior is exactly the right methodology. The five root causes you identified are real.
>
> We've shipped a major architecture change for v2.0: llama.cpp is fully removed, replaced by Airframe (pure-Rust WebGPU). The changes to `src/engine/llama.rs` won't apply, but the fixes to `src/engine/huggingface.rs` and `src/openai_compat.rs` still do.
>
> To get this merged we need:
> 1. Drop all changes to `src/engine/llama.rs` (file deleted)
> 2. Rebase on current `main` — `universal.rs` and `engine/mod.rs` changed
> 3. The `huggingface.rs` sampler chain + penalty fixes: keep and rebase
> 4. The `openai_compat.rs` stop token and penalty mapping: keep and rebase
> 5. The test files (`tests/api_error_handling_test.rs`, `tests/openai_api_real_tests.rs`): keep and rebase
>
> We'll also apply the equivalent sampler chain and penalty logic to the Airframe engine path in a companion commit. Would you like to do the rebase or should we take it over?

---

### [PR #175](https://github.com/Michael-A-Kuykendall/shimmy/pull/175) — gyc567 (eric) — `docs: add Chinese tutorial`
**Branch:** `main` (contributor's main)  
**What it does:** Adds `Shimmy中文教程.md` (root-level Chinese tutorial) and a README link.

**The situation:** We have now shipped comprehensive Simplified and Traditional Chinese user manuals (`docs/USER_MANUAL.zh-CN.md` and `docs/USER_MANUAL.zh-TW.md`), both ~700 lines covering 19 sections. The contributor's tutorial is a good community signal but is now superseded and would create a conflicting README.md edit.

**What to tell gyc567 — post this as a PR comment:**

> 谢谢你的贡献！Thank you for writing this — it's exactly the kind of community contribution that means the most to us.
>
> We've just merged comprehensive Simplified Chinese and Traditional Chinese user manuals (`docs/USER_MANUAL.zh-CN.md` and `docs/USER_MANUAL.zh-TW.md`) for v2.0, which cover the full v2.0 feature set including the new Airframe GPU engine. They were partly inspired by your PR showing the demand for Chinese documentation.
>
> We're going to close this PR as superseded by those manuals, but your work directly influenced our decision to prioritize Chinese-language docs for launch. If you'd like to contribute improvements, translations, or corrections to `docs/USER_MANUAL.zh-CN.md`, we'd be glad to have you.
>
> 非常感谢你的支持！

---

## Issues — Linked, Full Response Ready to Paste

---

### [#200](https://github.com/Michael-A-Kuykendall/shimmy/issues/200) — `gguf_init_from_file: failed to open GGUF file`
**Opened by:** @tiansiyuan

> Hey @tiansiyuan — this error comes from llama.cpp's file loader, which no longer exists in Shimmy v2.0. We've replaced the entire llama.cpp backend with Airframe, our own pure-Rust WebGPU inference engine that reads GGUF files natively.
>
> A couple things to check with the new version:
> 1. Make sure you're pointing at the right path — `SHIMMY_BASE_GGUF=/absolute/path/to/phi3-mini.gguf` or pass `--model-path` directly
> 2. Run `shimmy list` to confirm the model was found before making requests
> 3. If you're on Intel Mac, grab the new `shimmy-macos-intel` binary from the v2.0 release — it includes the Airframe engine built in
>
> Closing this out since the underlying error source (llama.cpp) is gone in v2.0. Drop a comment if it's still giving you trouble with the new binary and I'll dig in.

**Action:** Close ✅

---

### [#191](https://github.com/Michael-A-Kuykendall/shimmy/issues/191) — 422 on multi-part content array
**Opened by:** @jeffbski

> @jeffbski — this one's fixed. You were totally right, the OpenAI spec allows `content` to be either a string or an array of content parts, and we were only accepting string. Zed's file attachment behavior was perfectly valid and we were the ones out of spec.
>
> The fix is in v2.0 — `src/openai_compat.rs` now deserializes both formats correctly and we've added a regression test specifically for this so it can't quietly break again. If you're using Zed with file attachments, grab the v2.0 binary and it should work without any workaround.
>
> Thanks for the detailed report — the exact error message made this a fast fix.

**Action:** Close ✅

---

### [#190](https://github.com/Michael-A-Kuykendall/shimmy/issues/190) — Cannot use GPU on Windows
**Opened by:** @sunisstar

> Hey @sunisstar — in v1.9.0 you had to build with explicit CUDA/Vulkan flags to get GPU support, which was a real pain and caught a lot of people off guard.
>
> v2.0 changes this completely. We've replaced the llama.cpp GPU stack with Airframe, which uses WebGPU (wgpu) and **automatically detects your GPU at runtime** — no special build flags, no CUDA toolkit, no Vulkan SDK. On Windows 11 it picks up your GPU through Direct3D 12 or Vulkan, whichever your driver supports.
>
> Grab the v2.0 binary, run `shimmy gpu-info` and you should see your GPU listed. If it still shows CPU-only, drop the output of `shimmy gpu-info` here and I'll figure out what's going on with your driver config.

**Action:** Close ✅

---

### [#189](https://github.com/Michael-A-Kuykendall/shimmy/issues/189) — Crashes in llama
**Opened by:** @iilyak

> @iilyak — the crashes you're hitting are in the llama.cpp backend, which we've fully removed in v2.0. The Airframe engine (pure Rust, WebGPU) takes over all GGUF inference now and doesn't have the same crash surface.
>
> For your setup — VSCode with Local Model Provider pointing at `http://127.0.0.1:11435/v1` — that should work out of the box with v2.0. The models you mentioned (`typst-coder-9b` and `qwen3-coder-30b`) should load fine if they're standard GGUF format.
>
> Give v2.0 a shot and let me know if you hit anything. If you see a crash, an error, or silence from the model, I want to know about it.

**Action:** Close ✅

---

### [#184](https://github.com/Michael-A-Kuykendall/shimmy/issues/184) — Add `~/.cache/lm-studio/models` to auto-discovery
**Opened by:** @vsenn

> @vsenn — 43 models and 1.36TB, that's a real collection. You absolutely shouldn't have to re-download anything.
>
> Adding `~/.cache/lm-studio/models` to the auto-discovery path list is a quick win and I agree it should be there. I'm adding it to the v2.0 release. In the meantime, you can use:
> ```
> SHIMMY_MODEL_PATHS="$HOME/.cache/lm-studio/models" shimmy serve
> ```
> and `shimmy discover` will show you everything it's finding.
>
> I'll close this once the path is in the auto-discovery list in the release. Thanks for the detailed use case — the specific path and the `lms ls` context made it easy to action.

**Action:** Add path to `auto_discovery.rs`, then close ⚙️

---

### [#183](https://github.com/Michael-A-Kuykendall/shimmy/issues/183) — Ollama-compatible API
**Opened by:** @longzou

> @longzou — yes, v2.0 includes Ollama-compatible endpoints. You get `/api/generate` and `/api/tags` in addition to the OpenAI-compatible `/v1/` endpoints. If you have something pointed at Ollama's address, change the base URL to `http://127.0.0.1:11435` and the Ollama-format requests should work as-is.
>
> Closing this out — let me know if there's a specific endpoint or parameter you're hitting that doesn't behave the same as Ollama and I'll look at it.

**Action:** Close ✅

---

### [#182](https://github.com/Michael-A-Kuykendall/shimmy/issues/182) — `unknown model architecture: 'qwen35'`
**Opened by:** @eonun | **Also commented:** @getpool, @fceex49

> @eonun @getpool @fceex49 — the `qwen35` architecture (Qwen3.5) is a newer GGUF architecture identifier that wasn't recognized in v1.9.0 because the llama.cpp version we were pinned to predated it.
>
> In v2.0 we've moved to Airframe, our own inference engine, which reads architecture from GGUF metadata directly. Qwen3.5 support in Airframe is on the near-term roadmap — I want to give you an honest answer rather than "coming soon" with no timeline. I'll update this thread when it's in.
>
> If you need Qwen3.5 working today, the workaround is to build from source with the `--features huggingface` flag which uses the HuggingFace candle backend and has broader architecture coverage. I know that's not ideal.

**Action:** Keep open, update when Qwen3.5 lands in Airframe ⚙️

---

### [#181](https://github.com/Michael-A-Kuykendall/shimmy/issues/181) — 503 Service Unavailable
**Opened by:** @dzhl

> @dzhl — a 503 from Shimmy usually means one of two things: the model is still loading when the first request hits, or the model path wasn't found and the server came up with nothing loaded.
>
> Try this sequence:
> 1. Start shimmy and wait for the `✅ Ready to serve requests` line before sending anything
> 2. Hit `GET http://127.0.0.1:11435/api/health` — it'll tell you if models are loaded
> 3. Run `shimmy list` to confirm the model is actually registered
>
> If you're still getting 503 after the server says ready, restart with `SHIMMY_LOG_LEVEL=debug` and paste the output here. That'll tell us exactly what's happening at the request level.

**Action:** Keep open pending response 🔍

---

### [#180](https://github.com/Michael-A-Kuykendall/shimmy/issues/180) — Discord server
**Opened by:** @awdemos

> @awdemos — noted and agreed. A Discord would give people a place to share model configs, integration tips, and get faster help than GitHub issues allow. This is on my list alongside the v2.0 launch. I'll drop the link in the README and pin it here when it's live.

**Action:** Close when Discord is created ⚙️

---

### [#177](https://github.com/Michael-A-Kuykendall/shimmy/issues/177) — `cargo install shimmy` installs 1.8.1 instead of 1.9.0
**Opened by:** @Slach

> @Slach — the crates.io version was behind during that window because the publish pipeline had a path dependency issue that blocked automated publishing. That's been sorted out and v2.0 will publish cleanly to crates.io as part of the release.
>
> Once v2.0 drops, `cargo install shimmy` will get you the current version. If you want GPU acceleration, the v2.0 binary includes Airframe (WebGPU) by default — no extra feature flags needed for the precompiled release.
>
> Closing this out since it's a timing artifact that the v2.0 release resolves.

**Action:** Close on v2.0 crates.io publish ✅

---

### [#174](https://github.com/Michael-A-Kuykendall/shimmy/issues/174) — POST failed — EOF while parsing JSON
**Opened by:** @windows10do

> @windows10do — that `EOF while parsing a value` error means the request body arrived empty. The `/api/generate` endpoint expects a JSON body with at least a `prompt` field. A few things that cause this:
>
> 1. The Content-Type header is missing — add `-H "Content-Type: application/json"`
> 2. The body flag is wrong — use `-d '{"prompt":"your text here"}'` with curl
> 3. Some HTTP clients send the request before the body is attached
>
> Here's the minimum working curl:
> ```bash
> curl -X POST http://127.0.0.1:11435/api/generate \
>   -H "Content-Type: application/json" \
>   -d '{"prompt": "Hello", "max_tokens": 50}'
> ```
>
> Also — Shimmy doesn't use API keys, so you don't need one. If you paste the exact command you're running I can tell you exactly what's off.

**Action:** Close ✅

---

### [#173](https://github.com/Michael-A-Kuykendall/shimmy/issues/173) — Cannot run on Debian 11 (glibc too old)
**Opened by:** @higkoo | **Also commented:** @nwtgck, @LGinC, @sonderlau

> @higkoo @nwtgck @LGinC @sonderlau — Debian 11 ships glibc 2.31 and our Linux binary currently requires 2.32+. This is a real gap and it affects anyone on Debian 11, Ubuntu 20.04, or other distros that haven't moved to a newer glibc.
>
> Two options right now:
> 1. **Build from source** — `cargo install shimmy` on Debian 11 will compile against your local glibc and work fine
> 2. **Use the musl binary** — we're adding a `shimmy-linux-x86_64-musl` build to v2.0 releases that has zero glibc dependency and will run on Debian 11 without issue
>
> The musl build is coming in v2.0. I'll close this and link the release when it's out. Thanks to everyone who piled on with confirmation — it helped establish that this is a build target gap, not a one-off environment issue.

**Action:** Close when musl binary lands in v2.0 release ⚙️

---

### [#172](https://github.com/Michael-A-Kuykendall/shimmy/issues/172) — Not detecting GPU / Metal on M4
**Opened by:** @zz85

> @zz85 — in v1.9.0 you had to build with `--features mlx,apple` specifically to get Metal acceleration, and even then it was routing through MLX rather than direct Metal. v2.0 changes the whole model.
>
> The Airframe engine uses WebGPU (wgpu) and on Apple Silicon it goes through Metal natively — no `--features apple` flag, no MLX dependency, just `shimmy serve` and it finds your M4 GPU automatically. Run `shimmy gpu-info` and you should see `Apple M4 (Metal)` listed as the selected adapter.
>
> If you grab the v2.0 macOS ARM64 binary and it still shows CPU, let me know and I'll look at what wgpu is seeing on your machine.

**Action:** Close ✅

---

### [#171](https://github.com/Michael-A-Kuykendall/shimmy/issues/171) — Fail on Ubuntu 24 with CUDA
**Opened by:** @maded2

> @maded2 — the CUDA dependency is gone in v2.0. We've replaced llama.cpp with Airframe, which uses WebGPU (wgpu) for GPU acceleration. On Ubuntu 24, wgpu uses Vulkan, which your NVIDIA driver ships with.
>
> The `llama_context: constructing llama_context` log lines you saw are from the old llama.cpp path. You won't see those anymore. Grab the v2.0 `shimmy-linux-x86_64` binary — no CUDA toolkit needed, your existing NVIDIA driver is sufficient.

**Action:** Close ✅

---

### [#170](https://github.com/Michael-A-Kuykendall/shimmy/issues/170) — Split GGUF files not loading correctly
**Opened by:** @tom-nom-nom

> @tom-nom-nom — split GGUF (the `model-00001-of-00002.gguf` format) is a real gap. Right now Shimmy discovers GGUF files individually and can pick up the wrong shard when it finds `-00002-of-00002.gguf` before `-00001-of-00002.gguf` depending on filesystem order.
>
> This is on the roadmap for a proper fix — discovery needs to recognize the shard naming pattern and only register the first file as the model entry point.
>
> Workaround until then: merge the shards before loading with `llama-gguf-split --merge model-00001-of-00002.gguf merged.gguf` (part of llama.cpp utilities). I know that's annoying but it's the cleanest path right now.
>
> Keeping this open as a tracked issue. GLM-4.5 is a good test case for when we get to it.

**Action:** Keep open, roadmap ⚙️

---

### [#169](https://github.com/Michael-A-Kuykendall/shimmy/issues/169) — SSD offloading
**Opened by:** @moyanj

> @moyanj — SSD offloading is a good long-term capability for running models too large for RAM+VRAM. It's on the roadmap but not in scope for v2.0 — we need to stabilize the Airframe GPU path first before adding a third tier to the memory hierarchy.
>
> Keeping this open as a tracked feature request. The reference you posted (ollm) is useful context for the implementation approach.

**Action:** Keep open, roadmap ⚙️

---

### [#168](https://github.com/Michael-A-Kuykendall/shimmy/issues/168) — GPU support missing from precompiled Linux binary
**Opened by:** @Dominiquini

> @Dominiquini — you were right. The v1.9.0 precompiled Linux binary shipped CPU-only because building with CUDA/Vulkan/OpenCL in CI required the full driver stack, and we weren't doing that.
>
> v2.0 fixes this properly. The Airframe engine uses WebGPU (wgpu), which doesn't require any special build-time SDK — it links against your runtime Vulkan driver instead. The precompiled v2.0 `shimmy-linux-x86_64` binary has full GPU support out of the box. Run `shimmy gpu-info` after installing and you'll see your GPU listed.

**Action:** Close ✅

---

### [#166](https://github.com/Michael-A-Kuykendall/shimmy/issues/166) — Model 'default' not found in registry
**Opened by:** @vsenn

> @vsenn — the VSCode extension is sending `"model": "default"` and Shimmy is looking for a model registered under that exact ID, which doesn't exist. The extension needs to know your model's actual name.
>
> Quick fix: run `shimmy list` to see the exact model IDs Shimmy registered, then configure the extension to use one of those names. The ID is usually derived from the filename — e.g., `phi3-mini-4k-instruct`.
>
> In v2.0 there's also clearer error messaging in this case that points directly to `shimmy list` instead of just logging the missing ID. Closing this — if the extension itself needs a setting for which model to request, that's a config question on the extension side.

**Action:** Close ✅

---

### [#165](https://github.com/Michael-A-Kuykendall/shimmy/issues/165) — `--features moe` doesn't exist
**Opened by:** @jpicht | **Also commented:** @vsenn

> @jpicht @vsenn — the `moe` feature flag was in older documentation that got out of sync with the actual Cargo features. It never shipped as a standalone feature — MoE CPU offloading (`--cpu-moe`) was a runtime flag on top of the llama.cpp backend, not a compile-time feature.
>
> In v2.0 we've removed llama.cpp entirely. MoE architecture support in the Airframe engine is on the roadmap, but `--features moe` has never been a valid cargo feature and that doc reference was wrong. Fixing the docs as part of this close.
>
> Sorry for the confusion — bad documentation is a real bug.

**Action:** Close ✅

---

### [#163](https://github.com/Michael-A-Kuykendall/shimmy/issues/163) — Can't load models on NVIDIA DGX Spark (GB10)
**Opened by:** @Slach | **Also back-and-forth with:** @Slach, @Michael-A-Kuykendall

> @Slach — the DGX Spark has an NVIDIA GB10 (Blackwell architecture, compute 12.1), which is brand new hardware. The llama.cpp version we were on in v1.9.0 had limited Blackwell support, and the `null result from llama cpp` you were seeing was it failing to initialize the CUDA context properly.
>
> In v2.0 we've replaced llama.cpp with Airframe (WebGPU/wgpu). On the DGX Spark, wgpu should be able to use Vulkan against the GB10 driver. The DGX Spark runs ARM64 + Linux, and our v2.0 ARM64 binary currently uses the HuggingFace candle backend (Airframe ARM64 cross-compile is in progress). That should still load your models.
>
> Can you try `shimmy-linux-aarch64` from the v2.0 release and run `shimmy gpu-info`? I want to see what adapters wgpu enumerates on the GB10. This hardware is rare enough that I'd like to get it properly verified.

**Action:** Keep open pending v2.0 ARM64 test on DGX Spark 🔍

---

### [#162](https://github.com/Michael-A-Kuykendall/shimmy/issues/162) — Clarify auto-discovery ("automatically finds models")
**Opened by:** @Eason0729 | **Also commented:** @Michael-A-Kuykendall

> @Eason0729 — to clarify what "automatically finds models" means: Shimmy scans local directories for GGUF files at startup. It does **not** pull models from Ollama hub or any remote source — it only looks at what's already on your disk.
>
> The directories it searches include `~/models`, `~/.cache/huggingface/hub`, and the Ollama model cache if it exists locally. Run `shimmy discover` to see every path it's checking and every model it found.
>
> If you want to use a model that's in Ollama's local cache (already pulled via `ollama pull`), it should be discoverable automatically since Shimmy checks the Ollama model directory. But it won't reach out to pull anything that isn't already local.

**Action:** Close ✅

---

### [#161](https://github.com/Michael-A-Kuykendall/shimmy/issues/161) — Environment variables don't affect Docker config
**Opened by:** @Tenount

> @Tenount — environment variables in Docker Compose were being ignored in v1.8.2 because of how the binary was reading its config at startup. That's fixed in v2.0 — `SHIMMY_BASE_GGUF`, `SHIMMY_PORT`, `SHIMMY_BIND_ADDRESS`, and `SHIMMY_MAX_CTX` all work correctly when passed via Docker Compose environment.
>
> One note on your compose config: `SHIMMY_HOST=0.0.0.0` isn't the right variable. Use `--bind 0.0.0.0:11435` in the command, or set `SHIMMY_BIND_ADDRESS=0.0.0.0:11435`. The v2.0 docs have the full environment variable reference.
>
> Closing this — let me know if anything doesn't behave as expected in the new version.

**Action:** Close ✅

---

### [#160](https://github.com/Michael-A-Kuykendall/shimmy/issues/160) — Fix broken README examples (stream: false)
**Opened by:** @Michael-A-Kuykendall

> Fixed as part of the v2.0 doc pass. README curl and Python examples now include `"stream": false` explicitly, and the streaming behavior is documented separately. Closing.

**Action:** Close ✅

---

### [#153](https://github.com/Michael-A-Kuykendall/shimmy/issues/153) — Swagger / OpenAPI UI
**Opened by:** @7dir | **Also commented:** @7dir

> @7dir — Swagger/OpenAPI spec is a good ask, especially as more integrations appear that do spec-based client generation. It's not in v2.0 scope but it's a realistic post-launch addition — the API surface is stable enough now to generate a spec from.
>
> If you're interested in contributing a PR for this, the `/v1/chat/completions`, `/v1/models`, and `/api/generate` endpoints would be the core of it. I'd merge it. Keeping this open as a tracked enhancement.

**Action:** Keep open, roadmap ⚙️

---

### [#151](https://github.com/Michael-A-Kuykendall/shimmy/issues/151) — How does shimmy work?
**Opened by:** @windows10do

> @windows10do — Shimmy is a local inference server. You point it at a GGUF model file, it starts an HTTP server, and you send it OpenAI-format requests. It handles the inference on your hardware (GPU if available, CPU otherwise) and streams or returns the response.
>
> The v2.0 README has a 30-second quickstart, a full architecture section explaining the Airframe engine, and examples in curl, Python, Node.js, and Go. That should answer most "how does it work" questions — if something specific is still unclear after reading it, let me know and I'll add it to the docs.

**Action:** Close ✅

---

### [#150](https://github.com/Michael-A-Kuykendall/shimmy/issues/150) — Server reloads model on every request
**Opened by:** @confidentmeerkat | **Also commented:** @dbrucknr

> @confidentmeerkat @dbrucknr — models should load once at startup and stay resident. If you're seeing per-request reloading, that's either a v1.x bug that's been fixed, or something in your setup is killing and restarting the process between requests.
>
> In v2.0 with Airframe, the model is loaded into GPU memory at startup and stays there. The startup log says `✅ Ready to serve requests` when it's done loading — requests before that point get queued or rejected, not a fresh load.
>
> If you're on v2.0 and still seeing per-request load times, run with `SHIMMY_LOG_LEVEL=debug` and paste the first 20 lines of server output. I want to confirm the model is actually staying loaded between requests.

**Action:** Close ✅

---

### [#146](https://github.com/Michael-A-Kuykendall/shimmy/issues/146) — Docker image not published
**Opened by:** @prabirshrestha | **Also commented:** @sucream, @Michael-A-Kuykendall

> @prabirshrestha @sucream — the `ghcr.io/michael-a-kuykendall/shimmy:latest` image was never published, which means the `docker-compose.yml` in the repo was pointing at something that didn't exist. That's on us.
>
> v2.0 release includes a Docker image publish step in the release workflow. Once that's live you'll be able to `docker pull ghcr.io/michael-a-kuykendall/shimmy:2.0.0` and the `docker-compose.yml` will work as documented.
>
> Sorry this sat broken for as long as it did. Closing when the v2.0 image is published.

**Action:** Close on v2.0 Docker publish ⚙️

---

### [#145](https://github.com/Michael-A-Kuykendall/shimmy/issues/145) — TTS / STT / image / video / embeddings
**Opened by:** @prabirshrestha | **Also commented:** @Michael-A-Kuykendall

> @prabirshrestha — TTS, STT, image, and video generation are all architecturally different inference workloads from text completion and each would need its own engine path. They're not in v2.0 scope.
>
> Embeddings (`/v1/embeddings`) is the most likely near-term addition since the plumbing is closest to what we already have. I'll note that as a tracked post-2.0 item.
>
> For the others — if there's a specific use case driving this (e.g., local Whisper for STT, local SDXL for image) I'd be interested to hear more about what you're building toward. That shapes the priority.

**Action:** Keep open, roadmap ⚙️

---

### [#144](https://github.com/Michael-A-Kuykendall/shimmy/issues/144) — `cargo install` should auto-enable MLX on Apple Silicon
**Opened by:** @prabirshrestha | **Also commented:** @Michael-A-Kuykendall

> @prabirshrestha — the challenge here is that crates.io packages can't have platform-conditional default features — `Cargo.toml` default features are static. The only way to do this properly is either a build script that detects the target at compile time or a separate `shimmy-macos` crate.
>
> In practice, the v2.0 precompiled macOS ARM64 binary has Airframe (WebGPU/Metal) built in and doesn't need MLX at all for basic GPU acceleration — Metal is handled through wgpu directly. MLX becomes relevant for Apple-native model formats which is a narrower use case.
>
> Keeping this open as a tracked issue for the Homebrew formula work, which is a cleaner place to handle Mac-specific defaults.

**Action:** Keep open, roadmap ⚙️

---

### [#143](https://github.com/Michael-A-Kuykendall/shimmy/issues/143) — Add uvx support
**Opened by:** @prabirshrestha | **Also commented:** @Michael-A-Kuykendall

> @prabirshrestha — uvx support would mean publishing Shimmy as a Python package that wraps the Rust binary, which is possible but not something we have bandwidth for in v2.0. The awkwardness is that Shimmy is a Rust binary, not Python, so the "package" would just be a downloader/installer shim.
>
> The Homebrew formula and direct binary downloads cover the "install without cargo" use case for now. uvx is a good long-term option if the Python ecosystem is where most users are coming from — I'll keep it tracked.

**Action:** Keep open, roadmap ⚙️

---

### [#141](https://github.com/Michael-A-Kuykendall/shimmy/issues/141) — Does it support `response.create` (OpenAI Realtime API)?
**Opened by:** @hyunW3 | **Also commented:** @Michael-A-Kuykendall

> @hyunW3 — `response.create` is part of the OpenAI Realtime API which is a separate WebSocket-based protocol, different from the Chat Completions API. We don't support it in v2.0.
>
> What we do support: streaming chat completions via SSE (`"stream": true` in your request) and WebSocket generation via `/ws/generate`. If your use case is streaming text output, those paths work. If you specifically need the Realtime API schema for tool use and audio, that's not there yet.

**Action:** Close ✅

---

### [#137](https://github.com/Michael-A-Kuykendall/shimmy/issues/137) — Better quickstart + demo video/GIF
**Opened by:** @Michael-A-Kuykendall

> v2.0 ships with an updated quickstart in README and two new comprehensive user manuals (Simplified Chinese and Traditional Chinese). A demo GIF is still on the content list — recording a 30-second terminal session showing startup → first curl → streamed response is the target. Closing this as a tracked content item to be done before the public launch post.

**Action:** Close when GIF is added to README ⚙️

---

### [#127](https://github.com/Michael-A-Kuykendall/shimmy/issues/127) — Smoke test broken on MLX
**Opened by:** @iamkroot | **Also back-and-forth with:** @rpattcorner, @alistairheath, @Michael-A-Kuykendall

> @iamkroot @rpattcorner @alistairheath — the MLX response wrapper was including the `data: ` SSE prefix in the non-streaming response body, which is wrong. That was a v1.x regression.
>
> The MLX code path hasn't changed in v2.0 (Airframe handles GGUF, MLX handles native Apple format), so I need to verify whether this is still present before closing. Running the MLX smoke test against v2.0 is on my pre-release list. I'll update this thread with the result and close it once confirmed fixed or with a fix in hand.

**Action:** Verify in v2.0, close when confirmed 🔍

---

### [#114](https://github.com/Michael-A-Kuykendall/shimmy/issues/114) — MLX missing from crates.io and Homebrew
**Opened by:** @Michael-A-Kuykendall | **Also back-and-forth with:** @newnight, @petri, @Michael-A-Kuykendall

> @newnight @petri — the crates.io publish is now unblocked for v2.0. MLX remains a source-build feature (`--features mlx`) because mlx-swift is macOS-only and crates.io packages must build on all platforms. That constraint doesn't go away.
>
> What does change: the precompiled macOS ARM64 binary in v2.0 has Airframe with Metal acceleration built in, which covers the GPU use case for most users without needing MLX at all. MLX is now specifically for users who want Apple-native model format support.
>
> The Homebrew formula needs to be updated to point at v2.0 and the Homebrew tap with a macOS-ARM64 precompiled bottle is the right long-term path for Mac users who want `brew install`. That's post-2.0 work but I'll get to it quickly after launch.

**Action:** Close on v2.0 crates.io publish ⚙️

---

### [#113](https://github.com/Michael-A-Kuykendall/shimmy/issues/113) — Open WebUI / AnythingLLM compatibility
**Opened by:** @Michael-A-Kuykendall | **Extended thread with:** @barseghyanartur, @DecayingSec, @Michael-A-Kuykendall

> @barseghyanartur @DecayingSec — the `/v1/models` response format and `/api/tags` response format that Open WebUI and AnythingLLM require are both correct in v2.0. The specific issues you hit (malformed model list, missing fields in the response structure) have regression tests in place now so they can't silently break again.
>
> If you were using Zed, Continue.dev, Open WebUI, or SillyTavern with Shimmy and hitting integration issues: update to v2.0, point your client at `http://127.0.0.1:11435` and it should work. If you find a specific frontend that's still broken, open a new issue with the exact request/response and I'll fix it fast.

**Action:** Close ✅

---

## Pre-Release Checklist — v2.0

### Must-Do Before Going Live

- [ ] **Bump version** — `Cargo.toml` version to `2.0.0`
- [ ] **CHANGELOG.md** — Finalize v2.0.0 entry (Airframe announcement, llama.cpp removal, migration notes)
- [ ] **`cargo fmt -- --check`** — Must pass clean
- [ ] **`cargo clippy --no-default-features --features huggingface`** — Zero warnings
- [ ] **`cargo test --no-default-features --features huggingface`** — All gates green
- [ ] **`cargo check --features airframe`** — Clean compile
- [ ] **`cargo publish --dry-run --features huggingface`** — crates.io package validates
- [ ] **Binary size gate** — `shimmy` binary ≤ 20MB
- [ ] **Docker image** — Ensure `release.yml` publishes `ghcr.io/michael-a-kuykendall/shimmy:2.0.0`
- [ ] **Git tag** — Create `v2.0.0` tag

### Nice-to-Have Before Launch

- [ ] **#184** — Add LM Studio path to auto-discovery (30-min task)
- [ ] **#137** — Record demo GIF for README
- [ ] **PR #196 response** — Post comment to LopezNuance with rebase guidance
- [ ] **PR #198 response** — Post comment to LopezNuance with GGUF metadata approach
- [ ] **PR #175 close** — Post thank-you comment to gyc567, close as superseded

### Post-Launch (Do Not Block Release)

- [ ] **#182** — Qwen3 (`qwen35`) architecture support in Airframe
- [ ] **#170** — Split GGUF support
- [ ] **#163** — NVIDIA DGX Spark ARM64 investigation
- [ ] **#173** — Debian 11 glibc compatibility
- [ ] **#165** — MoE support roadmap in Airframe
- [ ] **#114** — MLX Homebrew formula update
- [ ] **#144** — Conditional MLX default feature on Apple Silicon

---

## What We Did This Session (Audit Trail)

| Action | Commit / Status |
|--------|----------------|
| Deleted `src/engine/llama.rs` + removed all llama.cpp bindings | `5898e6a` |
| Removed llama features from `Cargo.toml` | `5898e6a` |
| Removed CUDA CI gate, added wgpu-compat gate | `933eda0` |
| git filter-repo pass 1: scrubbed 59 sensitive paths from all history | SHA rewrite |
| Updated `.gitignore` | `2637b0a` |
| Created `docs/USER_MANUAL.zh-CN.md` (19-section Simplified Chinese) | `efe7c17` |
| Created `docs/USER_MANUAL.zh-TW.md` (19-section Traditional Chinese) | `efe7c17` |
| Added language links to `README.md` | `efe7c17` |
| git filter-repo pass 2: scrubbed SALES_PIPELINE.md, workspace file, 4 root scripts | SHA rewrite |
| Force-pushed clean history to `private/main` | `efe7c17` via gh CLI |

**Repo current HEAD:** `efe7c17`  
**Remote:** `private` = `github.com/Michael-A-Kuykendall/shimmy-private.git` ✅  
**`origin` remote:** Removed (expected — filter-repo removes remotes)
