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
pub struct HelicalShiftTrace {
    pub phase: String,
    pub step_index: usize,
    pub keep_sink: u32,
    pub shift_amt: u32,
    pub seq_len_before: u32,
    pub seq_len_after: u32,
    pub window_base_before: u32,
    pub window_base_after: u32,
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
    pub helical_shifts: Vec<HelicalShiftTrace>,
    pub final_stop_reason: String,
    pub final_tokens_generated: usize,
    pub final_text: String,
}
