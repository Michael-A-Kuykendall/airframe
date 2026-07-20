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

/// Run multiple controls in order; the first non-`Allow` decision wins.
///
/// This is the engine-level primitive for composing independent gates
/// (e.g. FSE reject + grammar enforce) without re-implementing chaining
/// at every call site.
pub struct ChainControl(pub Vec<Box<dyn InferenceControl + Send + Sync>>);

impl InferenceControl for ChainControl {
    fn intervene(&self, event: &InferenceEvent<'_>) -> ControlDecision {
        for c in &self.0 {
            match c.intervene(event) {
                ControlDecision::Allow => continue,
                other => return other,
            }
        }
        ControlDecision::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kvcache::KvSnapshot;

    fn ev(token: usize) -> InferenceEvent<'static> {
        InferenceEvent {
            tokens: &[],
            candidate_token: token,
            step: 0,
            kv: KvSnapshot { len: 0, version: 0 },
            text: "",
        }
    }

    struct Allow;
    impl InferenceControl for Allow {
        fn intervene(&self, _: &InferenceEvent<'_>) -> ControlDecision {
            ControlDecision::Allow
        }
    }
    struct Block;
    impl InferenceControl for Block {
        fn intervene(&self, _: &InferenceEvent<'_>) -> ControlDecision {
            ControlDecision::BlockAndTerminate("nope".into())
        }
    }
    struct Early;
    impl InferenceControl for Early {
        fn intervene(&self, _: &InferenceEvent<'_>) -> ControlDecision {
            ControlDecision::EarlyExit
        }
    }

    #[test]
    fn chain_returns_allow_when_all_allow() {
        let c = ChainControl(vec![Box::new(Allow), Box::new(Allow)]);
        assert_eq!(c.intervene(&ev(0)), ControlDecision::Allow);
    }

    #[test]
    fn chain_returns_first_non_allow() {
        let c = ChainControl(vec![Box::new(Allow), Box::new(Block), Box::new(Early)]);
        assert!(matches!(
            c.intervene(&ev(0)),
            ControlDecision::BlockAndTerminate(_)
        ));
    }

    #[test]
    fn chain_early_exit_beats_later_block() {
        let c = ChainControl(vec![Box::new(Allow), Box::new(Early), Box::new(Block)]);
        assert_eq!(c.intervene(&ev(0)), ControlDecision::EarlyExit);
    }
}
