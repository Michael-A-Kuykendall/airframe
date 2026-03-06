# Experiment 4: Self-Healing Redo Switch

## Objective
To test if a **Recoverable Violation** (Perplexity Spike) can be "healed" by discarding the creative token and forcing a deterministic (Greedy) token from the same logit state.

## Implementation (Shimmy Loop v2)
- **Phase 1 (Propose)**: Calculate Metrics (PPL/Entropy) on the current logit distribution.
- **Phase 2 (Evaluate)**:
  - If PPL < 400: Use standard sampling (Implied Greedy/Top-P).
  - If PPL > 400: **INTERCEPT** the generation.
    - Log `[REDO]` event.
    - Force selection of `argmax(logits)` (Best Token).
    - Continue generation.
- **Phase 3 (Commit)**: Only commit the chosen token (Standard or Healed) to the engine state.

## Results

### 1. Factual Baseline
- **Prompt**: "List three facts about Rust"
- **Status**: **Pass (Clean)**
- **Behavior**: Metrics stayed green (PPL ~200). Standard path taken.

### 2. High Risk ("Quantum Vague")
- **Prompt**: "Describe quantum entanglement vaguely"
- **Original Behavior (Exp 3.1)**: **HALT** at Token 2. (PPL 1045). Output: `[Nothing]`
- **Healing Behavior (Exp 4.0)**: **HEALED**.
  - **Trigger**: Detected PPL 1045.51 at Token 2.
  - **Action**: Intercepted. Switched to Greedy Strategy.
  - **Result**: The model recovered and output a coherent list: 
    `". 2. Quantum entanglement is a fundamental aspect of quantum mechanics..."`
  - **Survival**: Generated full 30 tokens instead of crashing.

## Key Insight
**The Redo Switch works.**
We successfully turned a "Fatal Hallucination" (which would have stopped generation or output garbage) into a "Recoverable Stutter" that produced a factual (if boring) output.

## Conclusion
We have moved from **Passive Monitoring** to **Active Control**.
The inference engine now has a "Lane Assist" capability that automatically corrects course when the model tries to veer off the road into hallucination.
