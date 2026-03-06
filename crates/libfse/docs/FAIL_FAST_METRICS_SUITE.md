# LibFSE V4 Inference Control Plane: Metric Suite

**Atomic Step 2 Complete**: Expansion from single-metric entropy checks to a multi-dimensional "Math Set" for robust inference guarding.

## The Mathematical Set {}

We define the set of control signals $S = \{ H, PPL, \sigma^2, ||L||_2, P_{max} \}$ computed strictly in the hot path.

### 1. Shannon Entropy ($H$)
**Formula**: $H(x) = -\sum P(x_i) \log_2 P(x_i)$
**Role**: Measures uncertainty.
**Failure Mode**: High values ($>0.5$ in greedy contexts) indicate "Vague" or "Confused" states. High risk of hallucination.
**Threshold**: `> 0.5` (Strict), `> 3.0` (Creative)

### 2. Rolling Perplexity ($PPL$)
**Formula**: $PPL = \exp\left(\frac{1}{N} \sum_{i=1}^N H_i\right)$ (over window $N=10$)
**Role**: Measures how "surprised" the model is by its own output over time.
**Failure Mode**: Sudden explosions ($PPL > 100$) indicate a total semantic break or "schizophrenic" output.
**Threshold**: `> 100.0`

### 3. Logit Variance ($\sigma^2$)
**Formula**: $\sigma^2 = \frac{1}{K}\sum(l_i - \mu)^2$
**Role**: Measures the spread of raw logits before Softmax.
**Failure Mode**: 
- **Collapse** ($\sigma^2 \to 0$): The model sees everything as equally probable (Uniform distribution).
- **Repetition Loop**: Often precedes infinite loops.
**Threshold**: `< 0.1`

### 4. L2 Norm ($||L||_2$)
**Formula**: $||L||_2 = \sqrt{\sum l_i^2}$
**Role**: Measures the magnitude of the logit vector.
**Failure Mode**: 
- **Explosion** ($> 1e4$): Numerical instability, likely `NaN` imminent.
- **Vanishing**: Model weights have died.
**Threshold**: `> 10,000.0`

### 5. Max Probability ($P_{max}$)
**Formula**: $P_{max} = \max(Softmax(L))$
**Role**: Confidence check.
**Failure Mode**: 
- **Overconfidence** ($> 0.99$): Model is "stuck" on a token (e.g., repeating "the the the").
- **Underconfidence** ($< 0.1$): Model is guessing.
**Threshold**: `> 0.99` (Warning), `< 0.05` (Halt)

## Verification Experiment

**Input**: `"Describe quantum entanglement vaguely"`
**Goal**: Trigger the fail-fast mechanics.

### Results
```text
[FAIL-FAST] Metric Violation Detected at Token 2
  Entropy:    8.0027
  Perplexity: 1045.5143
  Max Prob:   0.0365
  Variance:   2.6925
  Norm:       387.0066
[FAIL-FAST] Action: CRITICAL: Perplexity Explosion (Hallucination)
```

### Analysis
The system correctly identified a catastrophic state:
1.  **Entropy (8.0)**: Extreme confusion.
2.  **Perplexity (1045)**: The model's sequence has lost all coherence immediately.
3.  **Max Prob (0.03)**: The "best" guess had only 3% confidence.

**Result**: FAIL-FAST Triggered. Inference Halted. GPU/Compute Saved.

## Commercial Value
By monitoring $S$, we can:
1.  **Stop Hallucinations Early**: Don't generate 500 tokens of garbage. Stop at token 2.
2.  **Detect Loops**: Stop infinite repetitions before they waste 30 seconds of inference.
3.  **Prevent Crashes**: Catch numerical explosions before they segfault the serving infrastructure.
