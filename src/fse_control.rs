//! FSE Hit #3 — Aho-Corasick multi-policy text scan wired into InferenceControl.
//!
//! Instead of N independent `InferenceControl` implementors each doing their own
//! `contains()` / regex scan on accumulated text, `FseControl` compiles ALL rules
//! into one `FseMap` (Aho-Corasick DFA) and maintains a persistent `ScanCursor`
//! across `intervene()` calls.
//!
//! Per-token cost: one `scan_with_cursor()` call over the *delta* bytes only
//! (bytes added since the last intervene).  All rules fire simultaneously in
//! that single O(delta) pass — zero per-rule allocation, zero repeated scans.
//!
//! # Example
//!
//! ```rust
//! use libfse::{FseMap, Rule, FseOpcode};
//! use airframe::fse_control::FseControl;
//!
//! let rules = vec![
//!     Rule { pattern: b"badword".to_vec(), opcode: FseOpcode::Reject(0) },
//!     Rule { pattern: b"secret".to_vec(),  opcode: FseOpcode::Reject(1) },
//! ];
//! let map = FseMap::compile(rules).unwrap();
//! let control = FseControl::new(map).unwrap();
//! ```

use crate::control::{ControlDecision, InferenceControl, InferenceEvent};
use libfse::{FseMap, ScanCursor};
use std::sync::{Arc, Mutex};

/// Multi-policy text scanner that implements [`InferenceControl`].
///
/// Maintains an incremental Aho-Corasick DFA cursor across `intervene()` calls.
/// Only the *new* bytes appended since the last call are scanned — O(delta)
/// per token regardless of how many rules are registered.
pub struct FseControl {
    map: Arc<FseMap>,
    /// Persistent DFA walker + rule-bit state.
    cursor: Mutex<ScanCursor>,
    /// How many bytes of `event.text` have already been fed to the scanner.
    bytes_consumed: Mutex<usize>,
}

impl FseControl {
    /// Compile `map` and return a ready-to-use `FseControl`.
    ///
    /// Returns an error string if the DFA start state cannot be initialised
    /// (indicates a corrupt or empty map — should not happen in practice).
    pub fn new(map: FseMap) -> Result<Self, String> {
        let arc = Arc::new(map);
        let cursor = arc
            .new_cursor()
            .map_err(|e| format!("FseControl: failed to init cursor: {e}"))?;
        Ok(Self {
            map: arc,
            cursor: Mutex::new(cursor),
            bytes_consumed: Mutex::new(0),
        })
    }

    /// Reset scanner state (DFA position + rule bits + byte offset).
    ///
    /// Call this between generations if the same `FseControl` is reused across
    /// multiple independent inference runs.
    pub fn reset(&self) -> Result<(), String> {
        let fresh = self
            .map
            .new_cursor()
            .map_err(|e| format!("FseControl::reset: {e}"))?;
        *self.cursor.lock().unwrap() = fresh;
        *self.bytes_consumed.lock().unwrap() = 0;
        Ok(())
    }
}

impl InferenceControl for FseControl {
    /// Called once per candidate token.
    ///
    /// `event.text` contains the *full* accumulated decoded text up to (but not
    /// including) the candidate token.  We scan only the bytes appended since
    /// the previous call (the delta), keeping the DFA state live across calls.
    fn intervene(&self, event: &InferenceEvent<'_>) -> ControlDecision {
        let text_bytes = event.text.as_bytes();
        let mut consumed = self.bytes_consumed.lock().unwrap();
        let delta = &text_bytes[*consumed..];

        if delta.is_empty() {
            return ControlDecision::Allow;
        }

        let mut cursor = self.cursor.lock().unwrap();
        match self.map.scan_with_cursor(&mut cursor, delta) {
            Ok(_) => {
                *consumed = text_bytes.len();
                ControlDecision::Allow
            }
            Err(v) => ControlDecision::BlockAndTerminate(format!("FSE policy violation: {v}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{ControlDecision, InferenceControl, InferenceEvent, NoopControl};
    use crate::runtime::kvcache::KvSnapshot;
    use libfse::{FseMap, FseOpcode, Rule};

    fn make_map(patterns: &[(&[u8], u32)]) -> FseMap {
        let rules = patterns
            .iter()
            .map(|(pat, id)| Rule {
                pattern: pat.to_vec(),
                opcode: FseOpcode::Reject(*id),
            })
            .collect();
        FseMap::compile(rules).expect("test map compile")
    }

    fn dummy_event<'a>(text: &'a str) -> InferenceEvent<'a> {
        InferenceEvent {
            tokens:          &[],
            candidate_token: 0,
            step:            0,
            kv:              KvSnapshot { len: 0, version: 0 },
            text,
        }
    }

    #[test]
    fn test_allow_on_clean_text() {
        let map = make_map(&[(b"badword", 0)]);
        let ctrl = FseControl::new(map).unwrap();

        let e = dummy_event("hello world");
        assert_eq!(ctrl.intervene(&e), ControlDecision::Allow);
    }

    #[test]
    fn test_block_on_reject_pattern() {
        let map = make_map(&[(b"badword", 0)]);
        let ctrl = FseControl::new(map).unwrap();

        let e = dummy_event("this contains badword in it");
        let decision = ctrl.intervene(&e);
        assert!(
            matches!(decision, ControlDecision::BlockAndTerminate(_)),
            "expected BlockAndTerminate, got {decision:?}"
        );
    }

    #[test]
    fn test_incremental_scan_accumulates_state() {
        // The pattern spans a word boundary — only fires once the full pattern
        // has been scanned across successive calls.
        let map = make_map(&[(b"secret", 0)]);
        let ctrl = FseControl::new(map).unwrap();

        // First call: partial match — "sec" doesn't trigger anything
        let e1 = dummy_event("sec");
        assert_eq!(ctrl.intervene(&e1), ControlDecision::Allow);

        // Second call: text now "secret" — the cursor should complete the match
        let e2 = dummy_event("secret");
        let decision = ctrl.intervene(&e2);
        assert!(
            matches!(decision, ControlDecision::BlockAndTerminate(_)),
            "expected block on completed cross-call pattern, got {decision:?}"
        );
    }

    #[test]
    fn test_multiple_rules_simultaneous_dispatch() {
        // Both rules compiled into one DFA. Only rule 1 fires here.
        let map = make_map(&[(b"alpha", 0), (b"beta", 1)]);
        let ctrl = FseControl::new(map).unwrap();

        let e = dummy_event("contains beta inside");
        let decision = ctrl.intervene(&e);
        assert!(matches!(decision, ControlDecision::BlockAndTerminate(_)));
    }

    #[test]
    fn test_reset_clears_state() {
        let map = make_map(&[(b"secret", 0)]);
        let ctrl = FseControl::new(map).unwrap();

        // Scan "sec" — partial match, DFA is mid-word
        let e1 = dummy_event("sec");
        assert_eq!(ctrl.intervene(&e1), ControlDecision::Allow);

        // Reset clears DFA state and byte offset
        ctrl.reset().unwrap();

        // After reset, "ret" alone should NOT trigger (no longer completing "secret")
        let e2 = dummy_event("ret");
        assert_eq!(ctrl.intervene(&e2), ControlDecision::Allow);
    }

    #[test]
    fn test_empty_delta_is_allow() {
        let map = make_map(&[(b"x", 0)]);
        let ctrl = FseControl::new(map).unwrap();

        // Call with empty text — no bytes to scan
        let e = dummy_event("");
        assert_eq!(ctrl.intervene(&e), ControlDecision::Allow);

        // Call again with same text (delta == 0) — should still Allow
        assert_eq!(ctrl.intervene(&e), ControlDecision::Allow);
    }
}
