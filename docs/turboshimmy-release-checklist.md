# TurboShimmy INT4 KV — Production Release Checklist

> Working file. Check items off as they land. Nothing in P0 ships unresolved.
> Branch: `feat/turboquant-wgsl` → merge target `master` at `v0.2.0`.

---

## P0 — Gate conditions. Nothing ships without these.

### Correctness
- [ ] `SHIMMY_KV_QUANT=int4` needle bench at ctx=512 on a ≥6B model passes ≥2/3 depths
      — confirms INT4 KV fidelity holds where model capability is not the bottleneck
- [ ] Write `tests/int4_kv_parity.rs`: F32 and INT4 outputs are token-identical for a
      short (≤32 token) prompt at decode step 0 — fast CI regression check with no server
- [ ] Battery (math_battery.py) on Llama-3.2-3B F32 vs INT4: no regression in pass rate
      (current F32 baseline: 4/4)

### Stability
- [ ] Server survives 10 consecutive requests at ctx=2048 in INT4 mode without crash or hang
      (currently only tested at ctx≤256 with short outputs)
- [ ] `requantize_all_kv_int4` is explicitly polled before decode begins — ALREADY IN
      (`server_inference.rs`, keep the `device.poll(wait_indefinitely())` call)

### Windows TDR headroom
- [x] Per-layer submit+poll in F32 prefill loop confirmed working at ctx=256 (seq_len ~224)
- [ ] Needle bench at ctx=512 with chunk=64 completes without crash on Windows/RTX 3060
      (seq_len ~440 during decode — well above old 136 limit, but verify)

---

## P1 — Quality of life. Same release.

### API surface
- [ ] `SHIMMY_KV_QUANT` documented in README under a "Memory Optimization" section
      (currently only in `docs/turboshimmy.md`)
- [ ] `SHIMMY_PREFILL_CHUNK` and `SHIMMY_MAX_CTX` documented in README alongside
      `SHIMMY_PORT`
- [ ] `SHIMMY_KV_QUANT=int4` emits a clear startup error if model `head_dim` is not a
      multiple of 2 (nibble packing assumption) — currently silent wrong behavior
- [ ] `/v1/models` response includes `"kv_mode": "int4"` or `"f32"` field when active

### Observability
- [ ] Server startup prints one log line for KV mode:
      `[GPU Server] KV cache: INT4 (SHIMMY_KV_QUANT=int4)` or `F32`
      — currently silent

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
- [ ] Bump `[package] version` in `Cargo.toml` to `"0.2.0"`
- [ ] `docs/turboshimmy.md` linked from README
- [ ] CHANGELOG entry for 0.2.0
- [ ] `cargo publish --dry-run` passes cleanly
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
| 7B model smoke test | 🔄 running |
| Needle bench ctx=512 on 7B INT4 | ⏳ |
| INT4 parity unit test | ❌ |
| Perplexity bench script | ❌ |
| README TurboShimmy section | ❌ |
| Startup KV mode log line | ❌ |
| `/v1/models` kv_mode field | ❌ |
| Version bump to 0.2.0 | ❌ |
| Master merge + tag | ❌ |
