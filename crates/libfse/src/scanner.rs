// src/scanner.rs

use crate::store::{FseMap, PackedAction};
use crate::RuleId;

use aho_corasick::automaton::Automaton;
use aho_corasick::Anchored;

// ──────────────────────────────────────────────────────────────────────────────
// ScanCursor: borrowless mutable walker state
//
// Separates the "what have I seen so far" state from the `&FseMap` borrow.
// This allows callers to hold cursor + `Arc<FseMap>` without lifetime friction,
// which is exactly what `FseControl` (an `InferenceControl` implementor) needs.
// ──────────────────────────────────────────────────────────────────────────────

/// Persistent DFA walker state, divorced from the `FseMap` borrow.
///
/// Construct via [`FseMap::new_cursor`], advance via [`FseMap::scan_with_cursor`].
/// Can be stored alongside an `Arc<FseMap>` without lifetime parameters.
#[derive(Debug, Clone)]
pub struct ScanCursor {
    pub(crate) sid: aho_corasick::automaton::StateID,
    pub(crate) rule_bits: Vec<u64>,
    pub(crate) rules_recorded: u32,
}

impl ScanCursor {
    /// Reset the rule-tracking bits (Record flags/counters). DFA state is kept.
    #[inline]
    pub fn reset_rule_state(&mut self) {
        for w in self.rule_bits.iter_mut() {
            *w = 0;
        }
        self.rules_recorded = 0;
    }
}

/// Fail-closed violation returned immediately on Reject.
#[derive(Debug, Clone)]
pub enum Violation {
    /// A policy rule explicitly rejected the input.
    PolicyReject {
        rule_id: RuleId,
        pattern_index: usize,
        span: core::ops::Range<usize>,
    },
    /// Internal integrity failure (e.g. strict mode saw a missing opcode).
    /// This ensures fail-closed behavior on corrupted state.
    IntegrityError {
        pattern_index: usize,
        details: &'static str,
    },
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Violation::PolicyReject { rule_id, .. } => {
                write!(f, "Policy Violation: Rule {}", rule_id)
            }
            Violation::IntegrityError { details, .. } => write!(f, "Integrity Error: {}", details),
        }
    }
}

impl std::error::Error for Violation {}

impl Violation {
    pub fn rejected_rule_id(&self) -> Option<RuleId> {
        match self {
            Violation::PolicyReject { rule_id, .. } => Some(*rule_id),
            _ => None,
        }
    }
}

/// Summary of a completed scan (no reject).
#[derive(Debug, Clone)]
pub struct ScanSummary {
    pub bytes_scanned: usize,
    pub match_states_seen: u64,
    pub pattern_hits: u64,
    pub rules_recorded: u32,
    pub rules_rejected: u32, // should be 0 on Ok
}

/// The hot-loop scanner.
///
/// Design goal:
/// - no per-match allocation
/// - execute opcode immediately
/// - fail closed on Reject
pub struct FseScanner<'m> {
    map: &'m FseMap,

    // DFA walker state
    sid: aho_corasick::automaton::StateID,

    // Dense rule bitset (Record tracking). This is policy-defined; keep it cheap.
    // 1 bit per RuleId.
    rule_bits: Vec<u64>,
    rules_recorded: u32,
}

impl<'m> FseScanner<'m> {
    pub fn new(map: &'m FseMap) -> Result<Self, ScanError> {
        let sid = map
            .dfa()
            .start_state(Anchored::No)
            .map_err(ScanError::StartState)?;

        let words = words_for_bits(map.rule_count());
        Ok(Self {
            map,
            sid,
            rule_bits: vec![0u64; words],
            rules_recorded: 0,
        })
    }

    /// Reset rule tracking (Record bits/counters). DFA state remains as-is.
    #[inline]
    pub fn reset_rule_state(&mut self) {
        for w in self.rule_bits.iter_mut() {
            *w = 0;
        }
        self.rules_recorded = 0;
    }

    /// Reset DFA walker to start state (unanchored).
    #[inline]
    pub fn reset_automaton_state(&mut self) -> Result<(), ScanError> {
        self.sid = self
            .map
            .dfa()
            .start_state(Anchored::No)
            .map_err(ScanError::StartState)?;
        Ok(())
    }

    /// Scan a byte slice.
    ///
    /// CRITICAL properties:
    /// - no heap allocation per match
    /// - no `Match` objects
    /// - immediate opcode execution
    pub fn scan(&mut self, input: &[u8]) -> Result<ScanSummary, Violation> {
        let aut = self.map.dfa();

        let mut match_states_seen: u64 = 0;
        let mut pattern_hits: u64 = 0;

        // We use a local copy of sid for speed; write back at end.
        let mut sid = self.sid;

        // Hot loop: byte-by-byte transition, then handle special states.
        for (at, &b) in input.iter().enumerate() {
            sid = aut.next_state(Anchored::No, sid, b);

            let acts = self.map.actions_for_state(sid);
            if acts.is_empty() {
                continue;
            }

            match_states_seen += 1;
            let end = at + 1;

            for act in acts {
                match *act {
                    PackedAction::Ignore => {}
                    PackedAction::Record { word_idx, bit_mask } => {
                        pattern_hits += 1;
                        if let Some(word) = self.rule_bits.get_mut(word_idx as usize) {
                            if (*word & bit_mask) == 0 {
                                *word |= bit_mask;
                                self.rules_recorded = self.rules_recorded.saturating_add(1);
                            }
                        } else {
                            // This branch should be practically unreachable given MAX_ALLOWED_RULE_ID,
                            // but fail-closed demands it just in case.
                            self.sid = sid;
                            return Err(Violation::IntegrityError {
                                pattern_index: 0,
                                details: "Precomputed word_idx out of bounds",
                            });
                        }
                    }
                    PackedAction::Reject {
                        rule_id,
                        pattern_index,
                        pattern_len,
                    } => {
                        let start = end.saturating_sub(pattern_len as usize);
                        self.sid = sid;
                        return Err(Violation::PolicyReject {
                            rule_id,
                            pattern_index: pattern_index as usize,
                            span: start..end,
                        });
                    }
                    PackedAction::ControlResetRuleState => {
                        pattern_hits += 1;
                        self.reset_rule_state()
                    }
                    PackedAction::IntegrityError { pattern_index } => {
                        self.sid = sid;
                        return Err(Violation::IntegrityError {
                            pattern_index: pattern_index as usize,
                            details: "Precomputed integrity error in compiled map",
                        });
                    }
                }
            }
        }

        self.sid = sid;
        Ok(ScanSummary {
            bytes_scanned: input.len(),
            match_states_seen,
            pattern_hits,
            rules_recorded: self.rules_recorded,
            rules_rejected: 0,
        })
    }
}

#[derive(Debug)]
pub enum ScanError {
    StartState(aho_corasick::MatchError),
}

impl core::fmt::Display for ScanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ScanError::StartState(e) => write!(f, "failed to get start state: {e}"),
        }
    }
}

impl std::error::Error for ScanError {}

#[inline]
fn words_for_bits(bit_count: usize) -> usize {
    bit_count.div_ceil(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify div_ceil replacement is behaviorally equivalent to old formula.
    /// Old: (bit_count + 63) / 64
    /// New: bit_count.div_ceil(64)
    #[test]
    fn words_for_bits_equivalence() {
        // Test edge cases and boundary values
        let test_cases = [
            (0, 0),
            (1, 1),
            (63, 1),
            (64, 1),
            (65, 2),
            (127, 2),
            (128, 2),
            (129, 3),
            (1000, 16),
            (10_000, 157),
        ];

        for (bit_count, expected) in test_cases {
            let actual = words_for_bits(bit_count);
            assert_eq!(
                actual, expected,
                "words_for_bits({}) = {}, expected {}",
                bit_count, actual, expected
            );
        }
    }
}
