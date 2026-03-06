//! Token sampling strategies for text generation.
//!
//! Currently implements greedy (argmax) sampling only.

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

/// Sampling strategies for token generation
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Greedy: always select token with highest logit
    Greedy,
}

/// Sampler for token generation
pub struct Sampler {
    pub strategy: SamplingStrategy,
}

impl Sampler {
    /// Create new greedy sampler
    pub fn greedy() -> Self {
        Self {
            strategy: SamplingStrategy::Greedy,
        }
    }

    /// Sample next token from logits
    pub fn sample(&self, logits: &Tensor) -> Result<usize> {
        if logits.ndim() != 1 {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "sample_logits".to_string(),
                expected: vec![1],
                got: vec![logits.ndim()],
            });
        }

        match &self.strategy {
            SamplingStrategy::Greedy => greedy_sample(logits),
        }
    }
}

/// Greedy sampling: select token with highest logit.
///
/// On ties, returns the first (lowest index) maximum.
#[must_use = "sampling result should be used"]
pub fn greedy_sample(logits: &Tensor) -> Result<usize> {
    logits
        .data
        .iter()
        .enumerate()
        .reduce(|(max_i, max_v), (i, v)| if v > max_v { (i, v) } else { (max_i, max_v) })
        .map(|(idx, _)| idx)
        .ok_or_else(|| LibshimmyError::Unsupported("cannot sample from empty logits".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampling() {
        let logits = Tensor::new(vec![0.1, 0.8, 0.3, 0.2], vec![4]).unwrap();
        let selected = greedy_sample(&logits).unwrap();
        assert_eq!(selected, 1);
    }

    #[test]
    fn test_greedy_sampling_ties() {
        let logits = Tensor::new(vec![0.5, 0.8, 0.8, 0.2], vec![4]).unwrap();
        let selected = greedy_sample(&logits).unwrap();
        // First occurrence of max wins
        assert_eq!(selected, 1);
    }

    #[test]
    fn test_sampler_interface() {
        let logits = Tensor::new(vec![0.1, 0.8, 0.3, 0.2], vec![4]).unwrap();
        let sampler = Sampler::greedy();
        assert_eq!(sampler.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn test_sampling_edge_cases() {
        // Empty logits
        let empty_logits = Tensor::new(vec![], vec![0]).unwrap();
        assert!(greedy_sample(&empty_logits).is_err());

        // Single token
        let single_logits = Tensor::new(vec![0.5], vec![1]).unwrap();
        assert_eq!(greedy_sample(&single_logits).unwrap(), 0);
    }

    #[test]
    fn test_deterministic_behavior() {
        let logits = Tensor::new(vec![0.1, 0.8, 0.3, 0.2], vec![4]).unwrap();
        let result1 = greedy_sample(&logits).unwrap();
        let result2 = greedy_sample(&logits).unwrap();
        assert_eq!(result1, result2);
    }
}
