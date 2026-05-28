use crate::runtime::kvcache::KvSnapshot;

/// Read-only snapshot of inference state passed to a control hook.
///
/// This is deliberately metadata-only: no KV tensor copies, no logits buffers.
#[derive(Debug, Clone)]
pub struct InferenceEvent<'a> {
    /// Sequence so far (prompt + generated), excluding `candidate_token`.
    pub tokens: &'a [usize],

    /// Next token selected by the sampler (candidate to be appended).
    pub candidate_token: usize,

    /// Generation step index (0-based within this generation loop).
    pub step: usize,

    /// KV snapshot metadata (cheap): length + monotonic version.
    pub kv: KvSnapshot,

    /// Decoded text buffer (incremental).
    ///
    /// If a `TokenDecoder` was provided to the engine, this contains the
    /// accumulated text of the sequence so far. If not, it is empty.
    /// This allows policy to regex/match on text content.
    pub text: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlDecision {
    /// Continue normally.
    Allow,

    /// Stop generation cleanly (no error).
    EarlyExit,

    /// Fail closed with a reason.
    BlockAndTerminate(String),

    /// Discard the sampler's candidate and emit this token ID instead.
    ForceToken(usize),
}

/// Interface for converting token IDs to text for the control loop.
pub trait TokenDecoder: Send + Sync {
    /// Decode a single token ID to a string fragment.
    fn decode_single(&self, token: usize) -> String;
}

/// Extension point for policy/gating/audit without touching math kernels.
pub trait InferenceControl: Send + Sync {
    fn intervene(&self, event: &InferenceEvent<'_>) -> ControlDecision;
}

#[derive(Debug, Default, Clone)]
pub struct NoopControl;

impl InferenceControl for NoopControl {
    fn intervene(&self, _event: &InferenceEvent<'_>) -> ControlDecision {
        ControlDecision::Allow
    }
}
