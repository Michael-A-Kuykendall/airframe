---
name: shimmy-generate
description: Using the shimmy generate one-liner for end-to-end inference testing with chat template rendering.
---

# Shimmy Generate Skill

## One-Liner (run after building shimmy)

```powershell
cd C:\Users\micha\repos\shimmy
.\target\release\shimmy.exe generate "model-name" --prompt "hello" --max-tokens 20
```

Model names available (auto-discovered from `D:\shimmy-test-models\gguf_collection`):
- `Phi-3.5-mini-instruct` — ChatML template, good baseline
- `Llama-3.2-1B-Instruct` — Llama3 template
- `Llama-3.2-3B-Instruct` — Llama3 template
- `tinyllama-1.1b` — TinyLlama template
- `Qwen3-0.6B` — Qwen3 (ChatML)
- `Gemma-2-2B` — Gemma-2
- `deepseek-coder-6.7b` — DeepSeek Coder template

## Full Build + Test

```powershell
# Build shimmy (after airframe changes)
cd C:\Users\micha\repos\shimmy
cargo build --release

# Quick smoke test
.\target\release\shimmy.exe generate "Phi-3.5-mini-instruct" --prompt "hi" --max-tokens 20
```

## What Good Looks Like

| Check | Expected |
|-------|----------|
| Output is coherent | Real words/sentences, not garbage |
| No panic/crash | Exit code 0 |
| Chat template applied | Output formatted per model's template (e.g., Llama3 has `<|start_header_id|>`) |
| No `[DIAG]`/`[ISF-TDR` noise on stderr | Filtered or absent |

## Current Bugs Affecting This Command

- **airframe-e0b** [P1]: `shimmy generate` passes raw prompt without template wrapping. Fix pending — see AGENTS.md.
- **airframe-0h5** [P1]: Uncommitted `classify_template()` fix in `shimmy_server_gpu.rs` needs commit+push.

Until airframe-e0b is fixed, template wrapping must be done manually or testing is limited to models whose `spec.template` matches the hardcoded fallback.
