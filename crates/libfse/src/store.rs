// src/store.rs

use crate::{ControlOp, FseOpcode, Rule};
use aho_corasick::dfa::DFA;
use aho_corasick::automaton::Automaton;
use aho_corasick::Anchored;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub struct ActionRange {
    pub start: u32,
    pub len: u16,
}

#[derive(Clone, Copy, Debug)]
pub enum PackedAction {
    Ignore,
    Record { word_idx: u16, bit_mask: u64 },
    Reject { rule_id: u32, pattern_index: u32, pattern_len: u16 },
    ControlResetRuleState,
    IntegrityError { pattern_index: u32 },
}

/// FseMap: compiled DFA + dense opcode table + fused action tables.
#[derive(Debug)]
pub struct FseMap {
    dfa: DFA,
    rule_count: usize,
    state_actions: Vec<ActionRange>,
    actions: Vec<PackedAction>,
}

// Hard cap on RuleId to prevent memory DoS (sparse bitsets)
// 65535 rules = ~8KB bitset per scanner. Safe for high-concurrency.
const MAX_ALLOWED_RULE_ID: u32 = 65535;

impl FseMap {
    pub fn compile(rules: Vec<Rule>) -> Result<Self, BuildError> {
        if rules.is_empty() {
            return Err(BuildError::EmptyRuleSet);
        }

        let mut patterns: Vec<Vec<u8>> = Vec::with_capacity(rules.len());
        let mut opcodes: Vec<FseOpcode> = Vec::with_capacity(rules.len());

        let mut max_rule_id: u32 = 0;
        for r in rules {
            if r.pattern.is_empty() {
                return Err(BuildError::EmptyPattern);
            }
            let rid = extract_rule_id(r.opcode);
            if rid > MAX_ALLOWED_RULE_ID {
                return Err(BuildError::RuleIdTooLarge(rid));
            }
            max_rule_id = max_rule_id.max(rid);
            patterns.push(r.pattern);
            opcodes.push(r.opcode);
        }

        let dfa = DFA::new(patterns).map_err(BuildError::AhoBuild)?;
        let rule_count = (max_rule_id as usize).saturating_add(1);

        let (state_actions, actions) = build_state_tables(&dfa, &opcodes)?;

        Ok(Self {
            dfa,
            rule_count,
            state_actions,
            actions,
        })
    }


    #[inline]
    pub fn dfa(&self) -> &DFA {
        &self.dfa
    }

    #[inline]
    pub fn actions_for_state(&self, sid: aho_corasick::automaton::StateID) -> &[PackedAction] {
        let idx = sid.as_usize();
        if idx >= self.state_actions.len() { return &[]; }
        let r = self.state_actions[idx];
        if r.len == 0 { return &[]; }
        let start = r.start as usize;
        let end = start + (r.len as usize);
        &self.actions[start..end]
    }

    #[inline]
    pub fn rule_count(&self) -> usize {
        self.rule_count
    }
}

fn build_state_tables(
    dfa: &DFA,
    opcodes: &[FseOpcode],
) -> Result<(Vec<ActionRange>, Vec<PackedAction>), BuildError> {
    let aut = dfa;

    let start = aut.start_state(Anchored::No)
        .map_err(BuildError::StartState)?;

    // BFS to discover reachable states
    let mut seen = vec![false; start.as_usize() + 1];
    let mut q = VecDeque::new();
    q.push_back(start);
    seen[start.as_usize()] = true;

    let mut max_sid = start.as_usize();

    while let Some(sid) = q.pop_front() {
        for b in 0u16..=255 {
            let ns = aut.next_state(Anchored::No, sid, b as u8);
            let u = ns.as_usize();
            if u >= seen.len() {
                seen.resize(u + 1, false);
            }
            if !seen[u] {
                seen[u] = true;
                q.push_back(ns);
                if u > max_sid { max_sid = u; }
            }
        }
    }

    let mut state_actions = vec![ActionRange { start: 0, len: 0 }; max_sid + 1];
    let mut actions: Vec<PackedAction> = Vec::new();

    // Fill actions for reachable states
    // Note: iterating 0..=max_sid assumes StateID::from_usize is valid/safe for these indices
    // because they were returned by next_state() previously.
    for sid_usize in 0..=max_sid {
        if sid_usize >= seen.len() || !seen[sid_usize] { continue; }
        
        // Safety: We only iterate indices we discovered from the DFA itself.
        // Aho-corasick StateID is generally just an index wrapper.
        // If StateID info is hidden we loop over q... but here we assume indexability.
        // We can't construct StateID from usize publicly in standard crate sometimes?
        // Let's rely on BFS queue if needed, but for now we iterate indices.
        // BUT wait: StateID constructor is not always public.
        // Better strategy: Collect (sid, actions) during BFS or a second pass over SIDs.
        // Let's refine: We iterate over all valid StateIDs. Since we cannot forge them easily,
        // we'll actually use the queue to build the map, or we accept that we need to store them.
    }
    
    // Correct Approach: Re-traverse or just collect unique StateIDs into a list during BFS
    // Re-run the BFS logic but simplified to just collecting the list of unique SIDs.
    // Actually, we already set `seen`. But we can't map `i -> StateID`.
    // So we change the valid-loop above.
    
    // REDO: Collect list of SIDs during BFS.
    let mut unique_sids = Vec::with_capacity(max_sid);
    
    // Reset BFS
    let mut seen_bfs = vec![false; start.as_usize() + 1];
    let mut q_bfs = VecDeque::new();
    q_bfs.push_back(start);
    seen_bfs[start.as_usize()] = true;
    unique_sids.push(start);

    while let Some(sid) = q_bfs.pop_front() {
        for b in 0u16..=255 {
            let ns = aut.next_state(Anchored::No, sid, b as u8);
            let u = ns.as_usize();
            if u >= seen_bfs.len() {
                seen_bfs.resize(u + 1, false);
            }
            if !seen_bfs[u] {
                seen_bfs[u] = true;
                q_bfs.push_back(ns);
                unique_sids.push(ns);
            }
        }
    }

    // Now populate tables
    for sid in unique_sids {
        if !aut.is_match(sid) { continue; }

        let base = actions.len() as u32;
        let mlen = aut.match_len(sid);
        if mlen == 0 { continue; }

        let mut rejects = Vec::new();
        let mut records = Vec::new();
        let mut rest = Vec::new();

        for i in 0..mlen {
            let pid = aut.match_pattern(sid, i);
            let pidx = pid.as_usize();

            let op = opcodes.get(pidx).copied()
                .ok_or(BuildError::MissingOpcode(pidx))?;

            match op {
                FseOpcode::Ignore => rest.push(PackedAction::Ignore),
                FseOpcode::Record(rule_id) => {
                     // Precompute bitmasks here
                    let word_idx = (rule_id >> 6) as u16;
                    let bit_mask = 1u64 << (rule_id & 63);
                    records.push(PackedAction::Record { word_idx, bit_mask });
                },
                FseOpcode::Reject(rule_id) => {
                    let plen = aut.pattern_len(pid);
                    rejects.push(PackedAction::Reject {
                        rule_id,
                        pattern_index: pidx as u32,
                        pattern_len: plen as u16,
                    });
                }
                FseOpcode::Control(ControlOp::ResetRuleState) => {
                    rest.push(PackedAction::ControlResetRuleState)
                }
                FseOpcode::Control(_) => {
                    // Future modes -> Ignore for now
                    rest.push(PackedAction::Ignore)
                }
            }
        }

        actions.extend(rejects);
        actions.extend(records);
        actions.extend(rest);

        let len = (actions.len() as u32 - base) as u16;
        let idx = sid.as_usize();
        if idx >= state_actions.len() {
            state_actions.resize(idx + 1, ActionRange { start: 0, len: 0 });
        }
        state_actions[idx] = ActionRange { start: base, len };
    }

    Ok((state_actions, actions))
}

/// Crate-local build errors.
#[derive(Debug)]
pub enum BuildError {
    EmptyRuleSet,
    EmptyPattern,
    AhoBuild(aho_corasick::BuildError),
    RuleIdTooLarge(u32),
    StartState(aho_corasick::MatchError),
    MissingOpcode(usize),
}

impl core::fmt::Display for BuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BuildError::EmptyRuleSet => write!(f, "cannot compile empty rule set"),
            BuildError::EmptyPattern => write!(f, "cannot compile empty pattern"),
            BuildError::AhoBuild(e) => write!(f, "aho-corasick build error: {e}"),
            BuildError::RuleIdTooLarge(id) => write!(f, "RuleId {id} exceeds maximum allowed (65535)"),
            BuildError::StartState(e) => write!(f, "failed to get start state: {e}"),
            BuildError::MissingOpcode(idx) => write!(f, "internal error: missing opcode for pattern {idx}"),
        }
    }
}

impl std::error::Error for BuildError {}

#[inline]
fn extract_rule_id(op: FseOpcode) -> u32 {
    match op {
        FseOpcode::Record(id) | FseOpcode::Reject(id) => id,
        _ => 0,
    }
}
