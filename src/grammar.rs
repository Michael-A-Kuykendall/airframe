//! Grammar-constrained decoding helpers (engine level).
//!
//! This is the single source of truth for grammar enforcement. Every caller
//! (the `shimmy generate`/`serve` adapter, the `shimmy_server_gpu` dev server)
//! builds its hooks from here so the behavior cannot diverge between paths.
//!
//! The pre-sample mask and the post-sample control share ONE
//! `Arc<Mutex<GrammarState>>`: the mask reads the state, the control advances
//! it — exactly as the old inline decode loop did with a single variable.

use crate::control::{InferenceControl, TokenDecoder};
use schoolmarm::{Grammar, GrammarState};
use shimmytok::Tokenizer;
use std::sync::{Arc, Mutex};

/// `TokenDecoder` backed by the airframe tokenizer.
pub struct TokenizerDecoder(pub Arc<Tokenizer>);

impl TokenDecoder for TokenizerDecoder {
    fn decode_single(&self, token: usize) -> String {
        self.0.decode_single(token as u32, true).unwrap_or_default()
    }
}

/// Canonical "developer" grammar. Single definition for the whole engine.
pub fn developer_mode_grammar() -> &'static str {
    r#"
root ::= start body end
start ::= "fn " | "use " | "struct " | "enum " | "impl "
body ::= [\x09\x0A\x0D\x20-\x7E]*
end ::= "// END_RUST_FILE"
"#
}

/// Build the pre-sample mask closure AND the post-sample `SchoolmarmControl`
/// from a single shared `GrammarState`.
///
/// Returns `None` when `mode` is not grammar-driven (e.g. `"none"`/`"creative"`),
/// so callers can use the result directly as an `Option` of each hook.
pub fn grammar_hooks(
    mode: &str,
    tokenizer: Arc<Tokenizer>,
    n_vocab: usize,
    eos_token: u32,
    im_end_token: Option<u32>,
) -> Option<(
    Box<dyn Fn(&mut [f32]) + Send + Sync>,
    Box<dyn InferenceControl + Send + Sync>,
)> {
    if mode != "developer" {
        return None;
    }

    let grammar = Grammar::new(developer_mode_grammar()).ok()?;
    let state: Arc<Mutex<GrammarState>> = Arc::new(Mutex::new(GrammarState::new(grammar).ok()?));

    // Precompute the full vocab once; the mask closure re-borrows it per step.
    let vocab: Vec<String> = (0..n_vocab)
        .map(|tid| {
            tokenizer
                .decode_single(tid as u32, true)
                .unwrap_or_default()
        })
        .collect();

    let mask_state = state.clone();
    let mask = Box::new(move |logits: &mut [f32]| {
        let gs = match mask_state.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let vocab_refs: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();
        let allowed = gs.allowed_tokens(&vocab_refs);
        for (idx, l) in logits.iter_mut().enumerate() {
            if idx >= allowed.len() || !allowed[idx] {
                *l = f32::NEG_INFINITY;
            }
        }
        if gs.is_accepting() {
            if (eos_token as usize) < logits.len() {
                logits[eos_token as usize] = 0.0;
            }
            if let Some(im) = im_end_token {
                if (im as usize) < logits.len() {
                    logits[im as usize] = 0.0;
                }
            }
        }
    });

    let control = crate::schoolmarm_control::SchoolmarmControl::new_shared(
        state,
        Arc::new(TokenizerDecoder(tokenizer)),
    );

    Some((mask, Box::new(control)))
}
