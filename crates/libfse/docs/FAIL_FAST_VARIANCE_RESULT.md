# Experiment 3.2: Logit Variance Collapse

## Objective
To detect "Model Loops" or "Repetitive Ruts" where the model gets stuck repeating the same token (e.g., "the the the the").
**Metric**: Logit Variance ($\sigma^2$).
**Hypothesis**: Stuck models exhibit extremely low logit variance (< 0.1) because the probability mass collapses onto a single token.

## Implementation
- **Server**: Updated `shimmy_server.rs` to Halt if $\sigma^2 < 0.1$.
- **Harness**: `tools/experiment_3_2.py` with prompts designed to trigger loops ("Repeat 'hello world' 50 times").

## Results

| Prompt Type | Prompt | Result | Stop Reason | Trigger | Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | "List facts about Rust" | **Pass** | EOS | N/A | Healthy inference. |
| **Repetitive** | "Repeat 'hello world'..." | **FAIL-FAST** | Fail Fast | **PPL > 1325** | Model became **Confused**, not Stuck. |
| **Repetitive** | "Say 'repeat' 100 times" | **FAIL-FAST** | Fail Fast | **PPL > 800** | Model became **Confused**. |
| **Repetitive** | "Loop 'Kansas' forever" | **FAIL-FAST** | Fail Fast | **PPL > 522** | Model became **Confused**. |
| **Counting** | "Count to 1000..." | **Pass** | Max Tokens | N/A | Model output valid numbers (healthy variance). |

## Key Insight
**For TinyLlama Q4_0, "Repetitive" instructions (bad prompts) trigger Confusion (Perplexity Explosion) before they trigger Variance Collapse.**

The model does not simply "get stuck" in a low-entropy loop; it breaks down semantically.
This means the **Perplexity Guard (Experiment 3.1)** is the dominant defensive layer for this model class. The Variance guard acts as a secondary safety net for purely mechanical loops (which were not observed here).

## Next Steps
- **Retain both guards**: PPL for confusion, Variance for loops.
- **Proceed to Experiment 3.3**: Max Probability Overconfidence (Is the model "sure" about its garbage?).
