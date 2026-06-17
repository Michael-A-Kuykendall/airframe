# Current Status & Next Action (2026-06-15)

**Goal**: Get DeepSeek Coder producing coherent output on the current branch with zero forced polls.

## Status

**Working**:
- TinyLlama 1.1B Q4_0 works correctly (uses GGUF template, 0 polls on decode).
- All forced QKV polls eliminated (Phase 4a Step 5).
- KV cache write positions fixed (`params.batch_offset` added in both shaders).
- Embedding dequant fixed (Phase 0).
- TDR facts + TdrScheduler in place.
- Diagnostics show correct `current_pos` and `kv_inc` count after prefill.
- Template application improved on shimmy side.

**Broken**:
- DeepSeek Coder 6.7B Q4_K_M still produces garbage (repeating weird tokens).
- Engine metrics are healthy (`hidden_rms` good, no NaNs).
- The remaining issue is in the **Q4K attention math** (`main_attn_out` and related kernels), not control flow or positions.

## Next Action for Local Model

**Do this in order**:

1. **Confirm current state**
   - Run: `git status` in both repos.
   - Report the current branch and whether there are uncommitted changes.

2. **Test with a known-good model (fastest path to usable aider model)**
   - Try `Llama-3.2-3B-Instruct.Q4_K_M.gguf` (or any small Llama Q4_K_M you have).
   - Use the same test command you use for DeepSeek.
   - If this model produces coherent output → we have a working aider-capable model today. Report results.

3. **If you want to continue debugging DeepSeek instead**
   - Add a temporary diagnostic in `main_attn_out` (or just before/after it) that logs the first 4 attention scores for the first token of a small prompt.
   - Compare the pattern to what you expect from a working model.
   - Stop and say **"RAISE HAND – needs shader comparison to llama.cpp reference"** if this feels too open-ended.

**Rule**: If any step requires changes to more than 2 files or needs deep reasoning about the attention formula, stop immediately and say:

> "RAISE HAND – needs more context from Grok"

Only apply small, obvious patches. Do not refactor.

---

**Verification after any change**:
- Rebuild airframe + shimmy
- Run a short prompt on the model you are testing
- Check `/tmp/shimmy_isf_run.log` for `hidden_rms`, `logits_nans`, and first token
- Report those three values + whether output is coherent

Start with step 1 and report back.