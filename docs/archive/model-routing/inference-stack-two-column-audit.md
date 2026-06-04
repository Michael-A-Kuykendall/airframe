# Inference Stack Two-Column Audit (Golden Trace Style)

Scope:
- Left column = canonical llama.cpp behavior (golden trace intent)
- Right column = current Airframe behavior on this branch (with comparison)
- Primary target context = weak-model triage (StarCoder2, Qwen3, phi-2)

Notes:
- This is intentionally architecture-aware. "Mismatch" means Airframe differs from the expected behavior for that model family, not from every llama.cpp model.
- Canonical references come from llama.cpp graph/model paths (e.g. `src/llama-graph.cpp` norm dispatch and model attention/FFN blocks).

| llama.cpp Golden (top-to-bottom) | Airframe Current (top-to-bottom + compare) |
|---|---|
| 1) Request shaping and prompt contract are model-family specific before graph execution. Completion families are plain completion prompts; chat families use their chat template path. | Prompt renderer has explicit family/Jinja routing and trace fields (`prompt_renderer_mode`, `prompt_renderer_family`, `prompt_template_source`). StarCoder2 routes through `Completion` family. Compare: MATCH for routing framework; content quality still unresolved. |
| 2) Tokenization and context budgeting are enforced before decode. | Server enforces `prompt_tokens + max_tokens <= n_ctx` and rejects overrun. Compare: MATCH. |
| 3) Embedding lookup creates initial residual stream; some families apply embedding scale (e.g., Gemma variants). | Airframe applies Gemma embed scale and keeps scale = 1.0 for others. Compare: MATCH. |
| 4) Per-layer pre-attention normalization uses architecture-appropriate norm operator (`ggml_norm` for LayerNorm families, `ggml_rms_norm` for RMS families). | Airframe now routes norm mode via `ModelSpec::uses_layer_norm()` and logs `layer_norm_enabled` in trace/log. Compare: MATCH after `fb56570` for StarCoder2/GPT-like `Other` architectures. |
| 5) Q/K/V projection reads architecture-correct tensor layout (separate Q,K,V or fused QKV split) and quant dequant path. | Metadata compiler handles separate or fused QKV offsets and quant packing per layer. StarCoder2 logs show separate Q/K/V tensors found. Compare: MATCH for tensor layout selection. |
| 6) RoPE is applied to Q/K where architecture requires it, with configured rope parameters; optional per-head Q/K norm where model defines it. | Airframe applies RoPE via shader path and uses `qk_norm_enabled` flag (Qwen3 true, StarCoder2 false). Compare: MATCH for flag-level routing; deeper numeric parity still under investigation. |
| 7) KV cache write/read path must preserve attention semantics (F32 or quantized variant) without changing math intent. | Airframe supports F32 and INT4 KV paths (`main_attn_out` / `main_attn_out_int4`, `quantize_kv` pass). Compare: PARTIAL; behavior is stable but weak-model quality indicates likely deeper numeric/path issue remains. |
| 8) Attention scores: `softmax((QK^T)/sqrt(d_k))` (plus model-specific modifiers) then weighted V aggregation. | Airframe performs QK dot, scaling, softmax, and V accumulation in layer shader. Formula diffs show major divergence loci often in decode-layer `output_energy` and `ffn_gain`, not solely prompt text. Compare: PARTIAL; operator order appears correct but output parity is still off for weak models. |
| 9) Attention output projection returns to residual dimension and residual update occurs. | `main_attn_proj` writes projected attention output into residual stream. Compare: MATCH at pipeline-order level. |
| 10) Optional post-attention norm executes only for architectures that require it. | `post_norm_enabled` is architecture gated (Gemma family), warnings in preflight now narrowed to Gemma-only expectation. Compare: MATCH after warning-hygiene fix. |
| 11) FFN pre-norm/operator choice is architecture-specific; non-gated and gated families differ in exact flow. | Airframe has explicit `main_ffn_norm`, `main_ffn_proj`, `main_ffn_down` and non-gated/gated handling in shader. Recent fixes removed an F32 branch inconsistency and cleaned staged-flow ambiguity. Compare: IMPROVED; still a high-probability area for remaining weak-model drift. |
| 12) FFN activation path (SiLU/GELU variants) follows model family. | Airframe selects activation behavior in shader (including Gemma-related behavior via softcap checks). Compare: PARTIAL; likely correct for baseline pass models, but weak models still suggest potential family-specific nuance remains. |
| 13) FFN down projection and residual update close the layer block; optional post-FFN norm if architecture needs it. | `main_ffn_down` + optional `main_post_ffw_norm` dispatch sequence is present and architecture-gated. Compare: MATCH for sequencing; numeric parity still pending for weak models. |
| 14) Final output norm uses the same norm family contract as model architecture before lm_head. | Airframe final norm now reuses the same layer-norm mode routing and logs mode/eps in runtime output. Compare: MATCH after norm-mode fix. |
| 15) lm_head matmul + optional logit softcap + sampling produce next token. | Airframe uses blob lm_head path with final softcap support; sampling runs with configured temperature/top-p/repetition settings. Compare: MATCH for pipeline shape. |
| 16) Trace/invariant capture should make mismatches obvious without guesswork. | Airframe trace now includes norm eps, norm-mode flags, qk/post-norm flags, quant pack, prompt routing metadata, plus formula diff tooling. Compare: STRONG MATCH to desired debugging workflow. |

## Current Audit Verdict

- High-level equation routing now largely matches canonical intent for StarCoder2-family inference.
- Remaining failures are likely below "which step exists" and inside "how step math is executed" (kernel-level numeric behavior, per-architecture tensor interpretation, or logits-path specifics).
- This means the correct next phase is not broad rollback; it is layer-local numeric parity checks at the top divergent decode loci already surfaced by formula diff.

## Next Probe Focus (single-model, single-surface)

1. Keep StarCoder2 as active model.
2. Use this row order as golden trace checklist.
3. Instrument only one additional invariant at a time around rows 8-12 (attention output energy and FFN gain regions).
4. Re-run probe + side-by-side report + formula diff after each tiny change.
