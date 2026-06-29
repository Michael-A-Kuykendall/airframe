//! Mathematical metrics for AI safety auditing.
//!
//! Provides hot-loop implementations of Shannon Entropy, Perplexity, etc.

#[inline]
pub fn shannon_entropy(probs: &[f32]) -> f32 {
    let mut entropy = 0.0;
    for &p in probs {
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

#[inline]
pub fn shannon_entropy_from_logits(logits: &[f32]) -> f32 {
    // 1. Find max for numerical stability
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // 2. Compute sum(exp(x - max))
    let mut sum_exp = 0.0;
    for &logit in logits {
        sum_exp += (logit - max_logit).exp();
    }

    // 3. Compute log_sum_exp
    // log(sum(exp(x_i))) = max + log(sum(exp(x_i - max)))
    let log_z = max_logit + sum_exp.ln();

    // 4. Compute entropy: - sum(p * log(p))
    // p = exp(x - log_z)
    // log(p) = x - log_z
    // entropy = - sum(exp(x - log_z) * (x - log_z))

    let mut entropy = 0.0;
    for &logit in logits {
        let log_p = logit - log_z;
        let p = log_p.exp();
        entropy -= p * log_p;
    }

    entropy
}

/// Computes the variance of the logits (measure of "flatness" vs "spikiness").
/// Low variance (< 0.1) often indicates an overconfident or "stuck" model.
#[inline]
pub fn logit_variance(logits: &[f32]) -> f32 {
    let n = logits.len() as f32;
    if n == 0.0 {
        return 0.0;
    }

    let mean = logits.iter().sum::<f32>() / n;
    let sum_sq_diff: f32 = logits.iter().map(|&x| (x - mean).powi(2)).sum();

    sum_sq_diff / n
}

/// Computes the maximum probability in the distribution (Confidence).
/// Extremely high values (> 0.99) can indicate "Collapse" or loops.
#[inline]
pub fn max_probability_from_logits(logits: &[f32]) -> f32 {
    // Softmax max only
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp = 0.0;
    for &logit in logits {
        sum_exp += (logit - max_logit).exp();
    }
    // The max logit corresponds to term exp(0)=1.
    // p_max = 1.0 / sum_exp
    1.0 / sum_exp
}

/// Computes the L2 Norm (Euclidean Norm) of the logits.
/// Sudden spikes (> 1e5) indicate numerical instability/overflow.
#[inline]
pub fn logit_l2_norm(logits: &[f32]) -> f32 {
    let sum_sq: f32 = logits.iter().map(|&x| x * x).sum();
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution: entropy = ln(N)
        // For N=2, entropy = ln(2) = 0.6931...
        // logits [0.0, 0.0] -> probs [0.5, 0.5]
        let logits = vec![0.0, 0.0];
        let h = shannon_entropy_from_logits(&logits);
        assert!((h - 0.693147).abs() < 1e-4);
    }

    #[test]
    fn test_variance() {
        let logits = vec![1.0, 2.0, 3.0];
        // Mean = 2.0
        // Variance = (1+0+1)/3 = 0.666...
        let v = logit_variance(&logits);
        assert!((v - 0.666666).abs() < 1e-4);
    }

    #[test]
    fn test_max_prob() {
        // Logits [0.0, 100.0] -> p[1] approx 1.0
        let logits = vec![0.0, 100.0];
        let p = max_probability_from_logits(&logits);
        assert!(p > 0.9999);
        assert!(p <= 1.0);
    }

    #[test]
    fn test_l2_norm() {
        let logits = vec![3.0, 4.0];
        // Sqrt(9+16) = 5
        let n = logit_l2_norm(&logits);
        assert!((n - 5.0).abs() < 1e-4);
    }
}
