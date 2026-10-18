//! Exact generation accounting from the tokenizer and sampled token IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GenerationOutput {
    pub text: String,
    /// Tokenized, rendered prompt including tokenizer-owned special tokens.
    pub prompt_tokens: usize,
    /// Sampled output IDs, including a terminal EOS/stop ID when sampled.
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}
