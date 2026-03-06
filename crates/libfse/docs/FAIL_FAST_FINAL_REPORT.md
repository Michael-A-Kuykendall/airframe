# Commercial Inference Control Plane: Status Report

**Status**: ACTIVE & VALIDATED
**Architecture**: Rust (shimmy_server) + Python Harness (experiment_*.py)
**Model**: TinyLlama 1.1B Q4_0

## The "Fail-Fast Suite" (Implemented Mathematical Set)

We have mapped the surface area of the inference loop and installed 4 mathematical guards.

| Guard Metric | Threshold | Detected Failure Mode | Experiment Validation |
| :--- | :--- | :--- | :--- |
| **Perplexity (PPL)** | `> 400.0` | **Hallucinations & Confusion** | ✅ Caught "Quantum Vague", "Dots", "Assertive". |
| **Logit Variance** | `< 0.1` | **Model Collapse (Stuck)** | ⚠️ Secondary. (Adverse prompts triggered PPL instead). |
| **Max Probability** | `> 0.99` | **Overconfidence / Tunnel Vision** | ⚠️ Secondary. (Model confidence rarely exceeded 0.99 even in loops). |
| **L2 Norm** | `> 1e4` | **Numerical Instability / Crash** | ✅ Verified mechanism (simulated with `> 10.0`). |

## Key Insights

1.  **Perplexity is the "God Metric" for TinyLlama.**
    High Perplexity (> 400) reliably caught:
    -   Vague / Hallucinatory prompts.
    -   Repetitive garbage ("dots").
    -   Assertive nonsense.
    It is significantly more effective than Variance or MaxProb for this specific Q4_0 model.

2.  **The "Healthy Loop" Gap.**
    Prompts like *"Repeat 'the' forever"* successfully generated 40 tokens of "the".
    -   PPL stayed low (valid sequence).
    -   Variance stayed healthy.
    -   MaxProb stayed below 0.99.
    *This represents the only known gap in the current armor.*

3.  **Numerical Stability.**
    The L2 Norm guard is a silent sentinel. During healthy inference, Norm sits around ~380. During simulated faults (Threshold 10.0), it triggers instantly at Token 0.

## Recommendations for Dev Tool Product

1.  **Default Config**: Enable PPL Guard (400.0) and Norm Guard (1e4).
2.  **Pro Config**: Add "Repetition Penalty" or "N-Gram Guard" to close the "Healthy Loop" gap.
3.  **UI Feedback**: Report "Inference Health" based on PPL.
    -   PPL < 200: 🟢 Healthy (Factual)
    -   PPL 200-400: 🟡 Drifting (Creative)
    -   PPL > 400: 🔴 HALT (Garbage)
