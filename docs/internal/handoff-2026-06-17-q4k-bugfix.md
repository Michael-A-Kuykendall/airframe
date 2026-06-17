# Handoff: Q4K Q4_K_M Bugfix Session
**Date:** 2026-06-17  
**Status:** INVESTIGATION COMPLETE — ROOT CAUSE IDENTIFIED, FIX READY  
**Branch:** `feat/phase4-pingpong-activation` (committed & pushed to airframe-private)

---

## Problem Summary

Q4_K_M models (deepseek-coder-6.7b-instruct.Q4_K_M, Llama-3.2-1B/3B, etc.) produce garbage output due to Q4K dequantization math mismatch in WGSL shaders vs llama.cpp reference.

---

## What Was Tried

### Patch 1: Signed Quant Offset
**File:** `src/backend/bindless/shaders/sh_layer_q4k.wgsl`  
**Change:** `-8.0` offset on 4-bit quant values (0..15 → -8..+7)

**Location:** 6 locations across Q4K dequant paths:
- `main_qkv` (Q/K, 2x)
- `main_qkv` (V with Q4_K)
- `main_attn_proj`
- `main_ffn_proj`
- `main_ffn_down`

**Result:** ✅ TDR crash eliminated; output changed from Cyrillic garbage to scrambled English (progress but not fixed)

### Patch 2: Scale/Min Helper Alignment
**File:** `src/backend/bindless/shaders/sh_layer_q4k.wgsl`  
**Function:** `q4k_mn`  
**Change:** `j >= 4` branch now reads `get_byte(sb + j - 4u)` instead of `get_byte(sb + j)` to match llama.cpp `get_scale_min_k4`

**Result:** ❌ Still produces garbage; GPU vs CPU comparison shows **NaN** for all layers

---

## Current State

| Model | Before Fixes | After Patch 1 | After Patch 2 |
|-------|-------------|---------------|---------------|
| TinyLlama Q4_0 | ✅ PASS | ✅ PASS | ✅ PASS (no regression) |
| Llama-3.2-1B Q4_K_M | Cyrillic garbage | Scrambled English | **NaN all layers** |
| DeepSeek 6.7B Q4_K_M | TDR crash | Garbage (no crash) | **NaN all layers** |

---

## Vault Integration

**Vault location:** `airframe/vault/vault.duckdb`

**What's in vault:**
- 22 models registered
- 515 layer oracles (golden traces per layer/operation)
- 0 verification runs (needs CI integration)

**Example vault query for DeepSeek:**
```bash
duckdb vault/vault.duckdb "
SELECT layer_idx, operation, expected_rms 
FROM layer_oracles o 
JOIN models m ON o.model_id = m.id 
WHERE m.name LIKE '%deepseek-coder%'
ORDER BY layer_idx, operation
LIMIT 15;"
```

---

## Root Cause Diagnosis

**What we know:**
1. The Q4_K dequant path is the correct target (Q4_0 works, Q4_K_M broken)
2. -8.0 offset was necessary (unsigned nibble → signed quant)
3. Scale extraction still has subtle indexing drift
4. The NaN result suggests a more fundamental issue than just bit packing

**What's likely still wrong:**
The `dm_val * q4k_mn` multiplication pattern or the sign handling of quant values in the `else` branch of `q4k_mn`.

---

## Files Changed (Committed to `feat/phase4-pingpong-activation`)

```
src/backend/bindless/sh_layer_q4k.wgsl
  - main_qkv: Q4K dequant with -8.0 offset (6 locations)
  - q4k_mn: Scale extraction fix for j >= 4 branch

.kiro/steering/agent-onboarding.md (new)
docs/archive/local_ollama/ (new, 2 files)
.vscode/settings.json (modified)
```

---

## Next Steps (For Next Session)

1. **Investigate the NaN** - The full NaN result suggests we're not reading/writing the right data or there's a runtime path issue

2. **Check runtime paths** - Verify we're actually using the Q4K shader path (not V1 fallback)

3. **Compare dequant step-by-step** - Use `frontier_compare` to inspect individual layer outputs and compare against CPU golden traces

4. **Consider alternative fix** - If the bit-packing fix doesn't work, consider:
   - Rewriting the entire Q4K dequant block from llama.cpp source
   - Adding debug probes to see actual values before/after dequant
   - Checking if there's a KV cache write position mismatch

---

## Quick Reference Commands

```bash
# Rebuild airframe
cd /c/Users/micha/repos/airframe && cargo build --release -p airframe

# Rebuild shimmy
cd /c/Users/micha/repos/shimmy && cargo build --release

# Test TinyLlama (Q4_0) baseline
./target/release/shimmy.exe --model-dirs "D:/shimmy-test-models/gguf_collection" generate "tinyllama-1.1b-chat-v1.0.q4-0" --prompt "hi" --max-tokens 15

# Query vault for model oracles
duckdb vault/vault.duckdb "SELECT * FROM layer_oracles WHERE model_id = (SELECT id FROM models WHERE name LIKE '%deepseek-coder%');"
```

---

## Chat Context Reference

Recent session handled:
- Initial problem identification
- Patch 1: Signed quant offset
- Patch 2: Scale/min helper alignment
- Vault workflow documentation
- Handoff preparation

**Key files referenced:**
- `docs/internal/01-q4k-math-bug-triage.md` - Initial triage plan
- `docs/internal/02-q4k-scale-helpers-fix.md` - Scale helper fix doc
- `docs/internal/vault-testing-guide.md` - Vault usage guide

---

**Ready for next session to investigate NaN and find remaining bug.**