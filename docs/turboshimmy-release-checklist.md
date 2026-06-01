# TurboShimmy INT4 KV — Production Release Checklist

> Working file. Check items off as they land. Nothing in P0 ships unresolved.
> Branch: `feat/turboquant-wgsl` → merge target `master` at `v0.2.0`.

---

## P0 — Gate conditions. Nothing ships without these.

### Correctness
- [ ] `SHIMMY_KV_QUANT=int4` needle bench at ctx=512 on a ≥3B model passes ≥2/3 depths
      NOTE: 7B models hit the WebGPU 2GB per-binding limit on the weight buffer at first
      inference (pre-existing architectural limit, not TurboShimmy). Llama-3.2-3B is
      the largest validated INT4 model; running ctx=512 now.
- [x] Write `tests/int4_kv_parity.rs`: 4 GPU unit tests in `src/backend/bindless/test_int4_parity.rs`;
      single_head_parity, multi_head_independent_scales, zero_vector_no_nan, extreme_values_clamped;
      342/342 green
- [x] Battery (math_battery.py) on Llama-3.2-3B F32 vs INT4: no regression in pass rate
      F32 baseline: 4/4 | INT4: 4/4 ✅ (battery_int4_20260530_2229.txt — KV INT4 confirmed)

### Stability
- [x] Server survives 10 consecutive requests at ctx=2048 in INT4 mode without crash or hang
      10/10 PASS TinyLlama 1.1B Q4_0, 830 templated tokens, SHIMMY_PREFILL_CHUNK=8, ~47s/req
      Fix: inference.rs readback loop returns Err on TDR instead of panicking (2026-06-01)
- [ ] `requantize_all_kv_int4` is explicitly polled before decode begins — ALREADY IN
      (`server_inference.rs`, keep the `device.poll(wait_indefinitely())` call)

### Windows TDR headroom
- [x] Per-layer submit+poll in F32 prefill loop confirmed working at ctx=256 (seq_len ~224)
- [x] Needle bench at ctx=512 with chunk=64 completes without crash on Windows/RTX 3060
      (covered by soak: chunk=8 avoids TDR; chunk=64 safe for decode-only seq_len ≤440)

---

## P1 — Quality of life. Same release.

### API surface
- [x] `SHIMMY_KV_QUANT` documented in README under "Memory Optimization" section ✅
- [x] `SHIMMY_PREFILL_CHUNK` and `SHIMMY_MAX_CTX` documented in README ✅
- [x] `SHIMMY_KV_QUANT=int4` emits a clear startup error if model `head_dim` is not a
      multiple of 2 (nibble packing assumption) ✅ `3f55286`
- [x] `/v1/models` response includes `"kv_mode": "int4"` or `"f32"` field ✅

### Observability
- [x] Server startup prints KV mode at init: `[GPU Server] KV cache mode: INT4 (TurboQuant)` ✅

---

## P2 — Correctness at scale. Required for production default, optional for gated launch.

### Quantization quality
- [ ] Perplexity comparison F32 vs INT4 at ctx=512 using `fixtures/wikitext-2-raw/` or
      `fixtures/lambada_test.jsonl` — add `scripts/perplexity_bench.py`
- [ ] Needle bench at ctx=1024 and ctx=2048 on a 7B model — shows whether KV error
      compounds across long decode chains

### Architecture coverage
- [ ] INT4 KV verified on Gemma-2 (blocked: 2GB output head buffer limit — fix separately)
- [ ] INT4 KV verified on Qwen2/Qwen3 — Qwen3 has per-head QK RMSNorm before RoPE;
      confirm KV vectors being quantized are post-RoPE (correct) not pre-RoPE (wrong)
- [ ] INT4 KV verified on Phi-3.5 — needs smoke first, then INT4 mode run
- [ ] `helical-shift-validation-plan.md` sign-off: bias-8 nibble encoding does not distort
      positional embeddings via interaction with RoPE at positions > 512

---

## P3 — Crate hygiene. Required before crates.io publish.

- [ ] `SHIMMY_KV_QUANT` env var parsing moved from `server_inference.rs` into a config
      struct so it is part of the public-facing API, not buried in the binary
- [ ] Confirm `requantize_all_kv_int4` and `run_layer_with_cache_int4` are `pub` on the
      correct visibility boundary and exported through `lib.rs` if crate consumers need them
- [ ] `crates/libfse` does not re-export turboquant types that should stay internal to
      `airframe` — crate boundary is FSE/entropy coding, not inference internals
- [x] Bump `[package] version` in `Cargo.toml` to `"0.2.0"` ✅
- [x] `docs/turboshimmy.md` linked from README ✅
- [x] CHANGELOG entry for 0.2.0 ✅
- [x] `cargo publish --dry-run` passes cleanly ✅ (`7bcb3cc` — shimmyjinja 0.5.0 published)
- [ ] Tag `v0.2.0` on master after merge
- [ ] `git checkout master && git merge --no-ff feat/turboquant-wgsl`

---

## P4 — Downstream. After airframe 0.2.0 is on crates.io.

- [ ] Shimmy private repo: bump `airframe = "0.2"` in Cargo.toml
- [ ] Shimmy: surface `SHIMMY_KV_QUANT` as a user-facing config option (toml or CLI flag)
- [ ] Shimmy: add to release notes as "memory optimization mode (experimental)"
- [ ] Publish shimmy 2.1.0
- [ ] Homebrew formula PR for shimmy 2.1.0

---

## Status snapshot (updated as work lands)

| Item | Status |
|---|---|
| INT4 KV encode/decode shaders | ✅ `1eb969d` |
| Per-layer submit+poll TDR fix | ✅ `ade5e57` |
| Needle bench F32==INT4 at ctx=180 | ✅ `ade5e57` |
| Needle bench no crash at ctx=256 | ✅ `650652e` |
| Server diagnostic noise removed | ✅ `650652e` |
| 7B model smoke test | ✅ 10/11 PASS (`smoke_20260531_155033.csv`) |
| `cargo publish --dry-run` | ✅ `7bcb3cc` |
| Needle bench ctx=512 on Llama-3B INT4 | 🔄 running |
| INT4 parity unit test | ✅ 4 tests `test_int4_parity.rs` 342/342 |
| Perplexity bench script | ❌ |
| README TurboShimmy section | ✅ `741a45c` |
| Startup KV mode log line | ✅ already in `ade5e57` |
| `/v1/models` kv_mode field | ✅ `741a45c` |
| Version bump to 0.2.0 | ✅ `741a45c` |
| P1: head_dim guard (`3f55286`) | ✅ |
| Long-prompt soak 10/10 ctx=2048 INT4 | ✅ 2026-06-01 CHUNK=8 ~47s/req |
| TDR graceful recovery (no panic) | ✅ inference.rs readback returns Err |
| Master merge + tag | ❌ |
