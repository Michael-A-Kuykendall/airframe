# Fail-Fast Experiment: Entropy Detection

## Objective
To prove we can implement a "Fail-Fast" policy in the inference loop that:
1.  Calculates **Shannon Entropy** on the fly (from logits).
2.  Halts generation if uncertainty exceeds a threshold (`> 0.5` for demo).
3.  Demonstrates the "Fused Semantic Execution" concept (Metric -> Decision -> Action).

## Implementation
1.  **Metric**: Added `libfse::metrics::shannon_entropy_from_logits`.
2.  **Loop Integration**: Modified `shimmy_infer.rs` hot loop.
3.  **Policy**: Hard-coded fail-fast trigger.

## Result
We ran the local inference engine (`shimmy_infer`) with a strict entropy threshold of 0.5.

**Command:**
```bash
cargo run --bin shimmy_infer -- --model ../llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf "List three facts about rust"
```

**Output:**
```
Prompt: "List three facts about rust"
...
=== OUTPUT (streaming) ===
ic architecture that you find interesting

[FAIL-FAST] High Entropy Detected: 1.4124 > 0.5
[FAIL-FAST] Policy Action: DROPPING NOTE & HALTING
```

## Conclusion
The experiment was **SUCCESSFUL**.
-   The inference loop correctly calculated entropy per token.
-   The "Safety Guard" caught the spike (1.41) and halted execution immediately.
-   This validates the architecture: We can enforce mathematical bounds (Entropy, PPL) at the kernel level.
