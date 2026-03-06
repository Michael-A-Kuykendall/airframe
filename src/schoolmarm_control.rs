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
