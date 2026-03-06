# Fail-Fast Experiment: "Vague Entanglement" (High Risk)

## Objective
Test if a deliberately ambiguous prompt ("Describe quantum entanglement vaguely") triggers the entropy guard more aggressively than a simple prompt.

## Results
**Command:**
```bash
cargo run --bin shimmy_infer -- --model ../llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf "Describe quantum entanglement vaguely"
```

**Output:**
```
Prompt: "Describe quantum entanglement vaguely"
...
=== OUTPUT (streaming) ===
.

[FAIL-FAST] High Entropy Detected: 3.4605 > 0.5
[FAIL-FAST] Policy Action: DROPPING NOTE & HALTING
```

## Analysis
1.  **Entropy Spike**: The entropy jumped to **3.4605**.
    *   Previous experiment ("Facts about Rust"): ~1.41.
    *   This confirms the hypothesis: Ambiguous/Complex queries cause higher model uncertainty (flat logit distributions).
2.  **Early Termination**: Halting occurred almost immediately (after 6 tokens, with the first few being whitespace/punctuation).
3.  **Effectiveness**: The threshold of `0.5` is extremely sensitive, effectively demonstrating the mechanism. A production threshold would likely be around `2.5 - 3.0` to allow for normal English complexity while catching pure "babble" or "hallucination loops".

## Conclusion
The **Interior Chat** fail-fast mechanism is validated. We have:
1.  A way to harness "Inference Loop Data" (logits).
2.  A way to "Compute Immediately" (metrics.rs).
3.  A way to "Decide and Halt" (Fail-Closed).
