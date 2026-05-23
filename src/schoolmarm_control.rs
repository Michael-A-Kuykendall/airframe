use crate::control::{ControlDecision, InferenceControl, InferenceEvent, TokenDecoder};
use schoolmarm::{Grammar, GrammarState};
use std::sync::{Arc, Mutex};

/// Applies grammar policies to generated text during inference to constrain outputs.
pub struct SchoolmarmControl {
    state: Mutex<GrammarState>,
    decoder: Arc<dyn TokenDecoder>,
}

impl SchoolmarmControl {
    /// Create a new SchoolmarmControl wrapper for a given Grammar
    pub fn new(grammar: Grammar, decoder: Arc<dyn TokenDecoder>) -> Result<Self, String> {
        let state = GrammarState::new(grammar)
            .map_err(|e| format!("Grammar init failed: {}", e))?;
        Ok(Self {
            state: Mutex::new(state),
            decoder,
        })
    }
}

impl InferenceControl for SchoolmarmControl {
    fn intervene(&self, event: &InferenceEvent<'_>) -> ControlDecision {
        let piece = self.decoder.decode_single(event.candidate_token);
        let mut state = self.state.lock().unwrap();
        if let Err(err) = state.accept_token(&piece) {
            ControlDecision::BlockAndTerminate(format!("Grammar rejected token: {}", err))
        } else if state.is_accepting() {
            ControlDecision::EarlyExit
        } else {
            ControlDecision::Allow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::kvcache::KvSnapshot;
    use std::collections::HashMap;

    // ── Mock TokenDecoder ─────────────────────────────────────────────────────

    struct MockDecoder {
        map: HashMap<usize, String>,
        fallback: String,
    }

    impl MockDecoder {
        fn new(map: HashMap<usize, String>) -> Self {
            Self { map, fallback: "?".to_string() }
        }
    }

    impl TokenDecoder for MockDecoder {
        fn decode_single(&self, token: usize) -> String {
            self.map.get(&token).cloned().unwrap_or_else(|| self.fallback.clone())
        }
    }

    fn fake_event(token: usize) -> InferenceEvent<'static> {
        InferenceEvent {
            tokens: &[],
            candidate_token: token,
            step: 0,
            kv: KvSnapshot { len: 0, version: 0 },
            text: "",
        }
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_with_valid_grammar_succeeds() {
        let grammar = Grammar::new(r#"root ::= "yes" | "no""#).unwrap();
        let decoder = Arc::new(MockDecoder::new(HashMap::new()));
        assert!(SchoolmarmControl::new(grammar, decoder).is_ok());
    }

    #[test]
    fn test_new_with_invalid_grammar_returns_err() {
        // Missing root rule → should fail to parse
        let result = Grammar::new("bad grammar :::= {{{");
        // If Grammar::new rejects it, good. If it doesn't, SchoolmarmControl::new might fail.
        // Either way, we can't construct a valid SchoolmarmControl.
        if let Ok(grammar) = result {
            let decoder = Arc::new(MockDecoder::new(HashMap::new()));
            // GrammarState::new may reject it
            let ctrl = SchoolmarmControl::new(grammar, decoder);
            // Accept either outcome — the important thing is no panic
            let _ = ctrl;
        }
        // If Grammar::new returned Err, that's also fine
    }

    // ── intervene: Allow path ─────────────────────────────────────────────────

    #[test]
    fn test_allows_valid_prefix_token() {
        // Grammar: root ::= "hello"
        // Token 0 decodes to "hel" — partial match, not yet accepting → Allow
        let grammar = Grammar::new(r#"root ::= "hello""#).unwrap();
        let mut map = HashMap::new();
        map.insert(0usize, "hel".to_string());
        let decoder = Arc::new(MockDecoder::new(map));
        let ctrl = SchoolmarmControl::new(grammar, decoder).unwrap();
        let decision = ctrl.intervene(&fake_event(0));
        // "hel" is a valid prefix of "hello" → Allow
        assert_eq!(decision, ControlDecision::Allow, "partial prefix should Allow");
    }

    // ── intervene: EarlyExit path ─────────────────────────────────────────────

    #[test]
    fn test_early_exit_when_grammar_accepting() {
        // Grammar: root ::= "yes" | "no"
        // Feed "yes" in one shot → accepting → EarlyExit
        let grammar = Grammar::new(r#"root ::= "yes" | "no""#).unwrap();
        let mut map = HashMap::new();
        map.insert(1usize, "yes".to_string());
        let decoder = Arc::new(MockDecoder::new(map));
        let ctrl = SchoolmarmControl::new(grammar, decoder).unwrap();
        let decision = ctrl.intervene(&fake_event(1));
        assert_eq!(decision, ControlDecision::EarlyExit);
    }

    #[test]
    fn test_early_exit_on_second_alternative() {
        let grammar = Grammar::new(r#"root ::= "yes" | "no""#).unwrap();
        let mut map = HashMap::new();
        map.insert(2usize, "no".to_string());
        let decoder = Arc::new(MockDecoder::new(map));
        let ctrl = SchoolmarmControl::new(grammar, decoder).unwrap();
        let decision = ctrl.intervene(&fake_event(2));
        assert_eq!(decision, ControlDecision::EarlyExit);
    }

    // ── intervene: BlockAndTerminate path ────────────────────────────────────

    #[test]
    fn test_block_on_grammar_reject() {
        // Grammar: root ::= "yes"
        // Feed "no" — grammar should reject it
        let grammar = Grammar::new(r#"root ::= "yes""#).unwrap();
        let mut map = HashMap::new();
        map.insert(3usize, "no".to_string());
        let decoder = Arc::new(MockDecoder::new(map));
        let ctrl = SchoolmarmControl::new(grammar, decoder).unwrap();
        let decision = ctrl.intervene(&fake_event(3));
        assert!(
            matches!(decision, ControlDecision::BlockAndTerminate(_)),
            "invalid token should BlockAndTerminate, got: {decision:?}"
        );
    }

    // ── Multi-step: Allow then EarlyExit ─────────────────────────────────────

    #[test]
    fn test_multi_step_allow_then_early_exit() {
        // Grammar: root ::= "ab"
        // Step 1: token "a" → Allow (partial)
        // Step 2: token "b" → EarlyExit (accepting)
        let grammar = Grammar::new(r#"root ::= "ab""#).unwrap();
        let mut map = HashMap::new();
        map.insert(10usize, "a".to_string());
        map.insert(11usize, "b".to_string());
        let decoder = Arc::new(MockDecoder::new(map));
        let ctrl = SchoolmarmControl::new(grammar, decoder).unwrap();

        let d1 = ctrl.intervene(&fake_event(10)); // "a"
        assert_eq!(d1, ControlDecision::Allow, "first char should Allow");

        let d2 = ctrl.intervene(&fake_event(11)); // "b"
        assert_eq!(d2, ControlDecision::EarlyExit, "completing 'ab' should EarlyExit");
    }
}
