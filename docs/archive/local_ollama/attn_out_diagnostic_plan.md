# attn_out Diagnostic Plan (Only if Needed)

**Only follow this if the local model raised its hand on the main status document or if you specifically want to debug the Q4K attention math instead of testing Llama-3.2 first.**

## Goal
Find the numerical difference between `sh_layer_q4k.wgsl` `main_attn_out` and the canonical llama.cpp Q4_K attention implementation.

## Steps (Local Model)

1. Locate the `main_attn_out` function in `src/backend/bindless/sh_layer_q4k.wgsl`.

2. Extract the full function (from `fn main_attn_out` to the closing `}`).

3. Also extract the relevant parts of `main_qkv` that write to the KV cache (the parts that compute `cache_idx`).

4. Paste those two functions into a new file called `q4k_attn_current.wgsl` in the artifacts folder.

5. Then say:  
   **"RAISE HAND – ready for llama.cpp reference comparison"**

Do **not** try to fix anything yet. Just extract the current shader code cleanly.

**Rule**: Stop at step 5. Do not start comparing implementations yourself.