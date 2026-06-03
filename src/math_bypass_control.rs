//! `MathBypassControl` — arithmetic interception for the inference control loop.
//!
//! When a prompt contains a simple arithmetic expression ("What is 37 times 4?"),
//! this control computes the answer deterministically, encodes it to token IDs
//! via the model's own tokenizer, and feeds those tokens directly into the stream
//! — bypassing the sampler entirely for those positions.
//!
//! The model's KV cache still processes each forced token normally, so the
//! sequence is coherent from the model's perspective.  It just never gets the
//! chance to make a carry-propagation error.
//!
//! # Usage
//!
//! ```rust,ignore
//! let bypass = MathBypassControl::new(&rendered_prompt, &tokenizer);
//! if bypass.is_active() {
//!     log::info!("[MathBypass] overriding arithmetic answer");
//! }
//! engine.generate(prompt_ids, max_tokens, weights, Some(&bypass), None)?;
//! ```

use crate::control::{ControlDecision, InferenceControl, InferenceEvent};
use std::collections::VecDeque;
use std::sync::Mutex;

// ─── Public type ─────────────────────────────────────────────────────────────

pub struct MathBypassControl {
    state: Mutex<BypassState>,
}

struct BypassState {
    /// Token IDs to force, in order.  Empty means "passthrough".
    queue: VecDeque<usize>,
    /// Set to `true` when a math expression was detected.
    /// Once the queue drains we EarlyExit instead of continuing.
    armed: bool,
}

impl MathBypassControl {
    /// Construct from the already-rendered prompt string and the model's
    /// tokenizer.
    ///
    /// If the prompt contains a detectable arithmetic expression the answer is
    /// pre-encoded and stored as a forced-token queue.  Otherwise this is a
    /// zero-overhead passthrough.
    pub fn new(prompt: &str, tokenizer: &shimmytok::Tokenizer) -> Self {
        let queue = detect_and_compute(prompt)
            .and_then(|answer| {
                tokenizer
                    .encode(&answer.to_string(), /*add_special_tokens=*/ false)
                    .ok()
                    .map(|ids| ids.into_iter().map(|t| t as usize).collect::<VecDeque<_>>())
            })
            .unwrap_or_default();

        let armed = !queue.is_empty();
        Self {
            state: Mutex::new(BypassState { queue, armed }),
        }
    }

    /// `true` if a math expression was detected and the answer will be forced.
    pub fn is_active(&self) -> bool {
        self.state.lock().unwrap().armed
    }
}

impl InferenceControl for MathBypassControl {
    fn intervene(&self, _event: &InferenceEvent<'_>) -> ControlDecision {
        let mut s = self.state.lock().unwrap();
        if let Some(token) = s.queue.pop_front() {
            // Baby-bird: discard sampler's candidate, emit our pre-computed token.
            ControlDecision::ForceToken(token)
        } else if s.armed {
            // Queue drained — full answer written, stop cleanly.
            ControlDecision::EarlyExit
        } else {
            ControlDecision::Allow
        }
    }
}

// ─── Arithmetic detection + evaluation ───────────────────────────────────────

/// Try to find a simple "X op Y" pattern in the prompt and evaluate it.
/// Returns `None` if nothing arithmetic is detected.
pub fn detect_and_compute(prompt: &str) -> Option<i64> {
    let lower = prompt.to_lowercase();

    // (operator text, evaluator) — ordered so longer phrases match before substrings
    let ops: &[(&str, fn(i64, i64) -> Option<i64>)] = &[
        ("multiplied by", |a, b| a.checked_mul(b)),
        ("divided by",    |a, b| if b == 0 { None } else { Some(a / b) }),
        ("added to",      |a, b| a.checked_add(b)),
        ("subtracted",    |a, b| a.checked_sub(b)),
        ("times",         |a, b| a.checked_mul(b)),
        ("plus",          |a, b| a.checked_add(b)),
        ("minus",         |a, b| a.checked_sub(b)),
        ("÷",             |a, b| if b == 0 { None } else { Some(a / b) }),
        ("×",             |a, b| a.checked_mul(b)),
    ];

    for (op_str, compute) in ops {
        if let Some(op_pos) = lower.find(op_str) {
            let before = &lower[..op_pos];
            let after  = &lower[op_pos + op_str.len()..];
            if let (Some(a), Some(b)) = (last_integer(before), first_integer(after)) {
                if let Some(result) = compute(a, b) {
                    return Some(result);
                }
            }
        }
    }
    None
}

/// Parse the last integer in a string slice (scans right-to-left).
fn last_integer(s: &str) -> Option<i64> {
    let digits: String = s
        .chars()
        .rev()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse().ok()
}

/// Parse the first integer in a string slice.
fn first_integer(s: &str) -> Option<i64> {
    let start = s.find(|c: char| c.is_ascii_digit())?;
    let digits: String = s[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Convenience helper for callers that manage their own generation loop.
///
/// Detects arithmetic in `prompt`, evaluates it, and encodes the result into
/// token IDs using `tokenizer`.  Returns an empty `Vec` if no math is found.
pub fn compute_bypass_tokens(prompt: &str, tokenizer: &shimmytok::Tokenizer) -> Vec<u32> {
    detect_and_compute(prompt)
        .and_then(|answer| tokenizer.encode(&answer.to_string(), false).ok())
        .unwrap_or_default()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(prompt: &str) -> Option<i64> {
        detect_and_compute(prompt)
    }

    // ── Multiplication ────────────────────────────────────────────────────────

    #[test]
    fn times_word() {
        assert_eq!(ev("What is 37 times 4? Reply with only the number."), Some(148));
    }

    #[test]
    fn times_word_large_carry() {
        assert_eq!(ev("What is 77 times 77? Reply with only the number."), Some(5929));
    }

    #[test]
    fn times_word_carry_medium() {
        assert_eq!(ev("What is 48 times 52? Reply with only the number."), Some(2496));
    }

    #[test]
    fn multiplied_by_phrase() {
        assert_eq!(ev("What is 6 multiplied by 6?"), Some(36));
    }

    #[test]
    fn unicode_times() {
        assert_eq!(ev("What is 6 × 6? Reply with only the number."), Some(36));
    }

    // ── Addition ─────────────────────────────────────────────────────────────

    #[test]
    fn plus_word() {
        assert_eq!(ev("What is 127 plus 456? Reply with only the number."), Some(583));
    }

    #[test]
    fn plus_carry() {
        assert_eq!(ev("What is 999 plus 1? Reply with only the number."), Some(1000));
    }

    // ── Subtraction ──────────────────────────────────────────────────────────

    #[test]
    fn minus_word() {
        assert_eq!(ev("What is 100 minus 37? Reply with only the number."), Some(63));
    }

    #[test]
    fn minus_large() {
        assert_eq!(ev("What is 1000 minus 1? Reply with only the number."), Some(999));
    }

    // ── Division ─────────────────────────────────────────────────────────────

    #[test]
    fn divided_by_phrase() {
        assert_eq!(ev("What is 144 divided by 12? Reply with only the number."), Some(12));
    }

    #[test]
    fn unicode_division() {
        assert_eq!(ev("What is 81 ÷ 9? Reply with only the number."), Some(9));
    }

    #[test]
    fn division_by_zero_is_none() {
        assert_eq!(ev("What is 5 divided by 0?"), None);
    }

    // ── No-op cases ───────────────────────────────────────────────────────────

    #[test]
    fn non_math_prompt_is_none() {
        assert_eq!(ev("Tell me something interesting."), None);
    }

    #[test]
    fn geography_system_prompt_is_none() {
        assert_eq!(ev("You are a geography expert. Always mention the word GEOGRAPHY."), None);
    }
}
