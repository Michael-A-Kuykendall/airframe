use serde::{Deserialize, Serialize};
use shimmytok::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub abs_max: f32,
    pub first8: Vec<f32>,
}

impl TraceStats {
    pub fn from_slice(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std_dev: 0.0,
                abs_max: 0.0,
                first8: Vec::new(),
            };
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut abs_max: f32 = 0.0;
        let mut sum: f32 = 0.0;
        for &value in values {
            min = min.min(value);
            max = max.max(value);
            sum += value;
            abs_max = abs_max.max(value.abs());
        }
        let mean = sum / values.len() as f32;
        let variance = values
            .iter()
            .map(|&value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>()
            / values.len() as f32;

        Self {
            min,
            max,
            mean,
            std_dev: variance.sqrt(),
            abs_max,
            first8: values.iter().take(8).copied().collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorTrace {
    pub stats: TraceStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f32>>,
}

impl TensorTrace {
    pub fn from_slice(values: &[f32], include_values: bool) -> Self {
        Self {
            stats: TraceStats::from_slice(values),
            values: include_values.then(|| values.to_vec()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTrace {
    pub layer_idx: usize,
    pub current_pos: u32,
    pub seq_len: u32,
    pub logical_pos_base: u32,
    pub q: TensorTrace,
    pub k: TensorTrace,
    pub v: TensorTrace,
    pub post_attn: TensorTrace,
    pub ffn_out: TensorTrace,
    pub output: TensorTrace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitTopK {
    pub token_id: u32,
    pub logit: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_text: Option<String>,
}

pub fn topk_from_logits(logits: &[f32], k: usize, tokenizer: Option<&Tokenizer>) -> Vec<LogitTopK> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    indexed
        .into_iter()
        .take(k)
        .map(|(token_id, logit)| LogitTopK {
            token_id: token_id as u32,
            logit,
            token_text: tokenizer.and_then(|tok| tok.decode_single(token_id as u32, true).ok()),
        })
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTrace {
    pub phase: String,
    pub step_index: usize,
    pub token_id: u32,
    pub token_text: String,
    pub cache_len_before: u32,
    pub cache_len_after: u32,
    pub window_base_before: u32,
    pub window_base_after: u32,
    pub logits_topk: Vec<LogitTopK>,
    pub layers: Vec<LayerTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTracePackage {
    pub schema_version: u32,
    pub model_arch: String,
    pub prompt_mode: String,
    pub seed: u64,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub prompt_token_count: usize,
    pub templated_prompt: String,
    pub prefill_steps: Vec<TokenTrace>,
    pub decode_steps: Vec<TokenTrace>,
    pub final_stop_reason: String,
    pub final_tokens_generated: usize,
    pub final_text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // ── TraceStats::from_slice ────────────────────────────────────────────────

    #[test]
    fn test_trace_stats_empty() {
        let s = TraceStats::from_slice(&[]);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.std_dev, 0.0);
        assert_eq!(s.abs_max, 0.0);
        assert!(s.first8.is_empty());
    }

    #[test]
    fn test_trace_stats_single_element() {
        let s = TraceStats::from_slice(&[3.0]);
        assert!(approx_eq(s.min, 3.0));
        assert!(approx_eq(s.max, 3.0));
        assert!(approx_eq(s.mean, 3.0));
        assert!(approx_eq(s.std_dev, 0.0));
        assert!(approx_eq(s.abs_max, 3.0));
        assert_eq!(s.first8, vec![3.0]);
    }

    #[test]
    fn test_trace_stats_basic_values() {
        // [1, 2, 3]: mean=2, variance=2/3, std_dev=sqrt(2/3)
        let s = TraceStats::from_slice(&[1.0, 2.0, 3.0]);
        assert!(approx_eq(s.min, 1.0));
        assert!(approx_eq(s.max, 3.0));
        assert!(approx_eq(s.mean, 2.0));
        let expected_std = (2.0f32 / 3.0).sqrt();
        assert!((s.std_dev - expected_std).abs() < 1e-4, "std_dev={}", s.std_dev);
        assert!(approx_eq(s.abs_max, 3.0));
        assert_eq!(s.first8, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_trace_stats_negative_values() {
        let s = TraceStats::from_slice(&[-5.0, -1.0, 2.0]);
        assert!(approx_eq(s.min, -5.0));
        assert!(approx_eq(s.max, 2.0));
        assert!(approx_eq(s.abs_max, 5.0), "abs_max should be 5.0, got {}", s.abs_max);
    }

    #[test]
    fn test_trace_stats_all_same() {
        let v: Vec<f32> = vec![7.0; 10];
        let s = TraceStats::from_slice(&v);
        assert!(approx_eq(s.min, 7.0));
        assert!(approx_eq(s.max, 7.0));
        assert!(approx_eq(s.mean, 7.0));
        assert!(approx_eq(s.std_dev, 0.0));
    }

    #[test]
    fn test_trace_stats_first8_cap() {
        // 12 values — only first 8 should appear
        let v: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let s = TraceStats::from_slice(&v);
        assert_eq!(s.first8.len(), 8);
        assert_eq!(s.first8, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_trace_stats_large_positive_and_negative() {
        let s = TraceStats::from_slice(&[1e30, -1e30]);
        assert!(approx_eq(s.mean, 0.0));
        assert!(s.abs_max > 1e29);
    }

    // ── Property: min <= mean <= max for any non-empty slice ──────────────────

    #[test]
    fn test_trace_stats_mean_in_range_property() {
        let cases: &[&[f32]] = &[
            &[1.0, 2.0, 3.0],
            &[-5.0, -1.0, 0.0, 4.0],
            &[42.0],
            &[0.0, 0.0, 0.0],
            &[-1e10, 1e10],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            &[-100.0, -50.0, -1.0],
        ];
        for slice in cases {
            let s = TraceStats::from_slice(slice);
            assert!(
                s.min <= s.mean + 1e-4 && s.mean <= s.max + 1e-4,
                "slice {:?}: min={} mean={} max={} — invariant min<=mean<=max violated",
                slice, s.min, s.mean, s.max
            );
            assert!(s.max >= s.min, "max < min for slice {:?}", slice);
            assert!(s.mean.is_finite(), "mean not finite for slice {:?}", slice);
        }
    }

    #[test]
    fn test_trace_stats_first8_never_exceeds_8_elements() {
        let cases: &[&[f32]] = &[
            &[1.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[0.0; 100],
        ];
        for slice in cases {
            let s = TraceStats::from_slice(slice);
            assert!(
                s.first8.len() <= 8,
                "first8 has {} elements for slice of len {}",
                s.first8.len(), slice.len()
            );
        }
    }

    // ── TensorTrace::from_slice ───────────────────────────────────────────────

    #[test]
    fn test_tensor_trace_without_values() {
        let v = vec![1.0, 2.0, 3.0];
        let tt = TensorTrace::from_slice(&v, false);
        assert!(tt.values.is_none());
        assert!(approx_eq(tt.stats.mean, 2.0));
    }

    #[test]
    fn test_tensor_trace_with_values() {
        let v = vec![1.0, 2.0, 3.0];
        let tt = TensorTrace::from_slice(&v, true);
        assert_eq!(tt.values.as_ref().unwrap(), &v);
    }

    #[test]
    fn test_tensor_trace_empty() {
        let tt = TensorTrace::from_slice(&[], true);
        assert!(tt.values.as_ref().unwrap().is_empty());
        assert_eq!(tt.stats.min, 0.0);
    }

    // ── topk_from_logits ──────────────────────────────────────────────────────

    #[test]
    fn test_topk_basic_ordering() {
        // Token 1 has highest logit (5.0), token 2 second (3.0)
        let logits = vec![0.1, 5.0, 3.0, 1.0];
        let top = topk_from_logits(&logits, 2, None);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].token_id, 1);
        assert!(approx_eq(top[0].logit, 5.0));
        assert_eq!(top[1].token_id, 2);
        assert!(approx_eq(top[1].logit, 3.0));
    }

    #[test]
    fn test_topk_k_larger_than_vocab_returns_all() {
        let logits = vec![1.0, 2.0, 3.0];
        let top = topk_from_logits(&logits, 100, None);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_topk_k_zero_returns_empty() {
        let logits = vec![1.0, 2.0, 3.0];
        let top = topk_from_logits(&logits, 0, None);
        assert!(top.is_empty());
    }

    #[test]
    fn test_topk_empty_logits() {
        let top = topk_from_logits(&[], 5, None);
        assert!(top.is_empty());
    }

    #[test]
    fn test_topk_no_tokenizer_token_text_is_none() {
        let logits = vec![1.0, 2.0];
        let top = topk_from_logits(&logits, 2, None);
        for t in &top {
            assert!(t.token_text.is_none());
        }
    }

    #[test]
    fn test_topk_negative_logits() {
        let logits = vec![-10.0, -1.0, -5.0];
        let top = topk_from_logits(&logits, 3, None);
        // Should still be sorted descending: -1.0, -5.0, -10.0
        assert_eq!(top[0].token_id, 1);
        assert_eq!(top[1].token_id, 2);
        assert_eq!(top[2].token_id, 0);
    }

    #[test]
    fn test_topk_single_element() {
        let top = topk_from_logits(&[42.0], 5, None);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].token_id, 0);
        assert!(approx_eq(top[0].logit, 42.0));
    }
}
