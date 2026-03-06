// Fused Semantic Execution (FSE)
// Copyright (c) 2026 AI State Pilot. All Rights Reserved.
//
// PROPRIETARY & CONFIDENTIAL.
// This library implements Patent Pending methods for Fail-Closed Semantic Enforcement.
//
// Portions of the graph compilation logic may utilize algorithms from `aho-corasick`
// (MIT License, Andrew Gallant). The *Execution Kernel* and *Policy Fusion* logic
// are original works.

pub mod scanner;
pub mod store;
pub mod metrics;

#[cfg(test)]
mod tests_integration;


pub use scanner::{FseScanner, ScanSummary, Violation};
pub use store::FseMap;
// pub mod automaton; // Deprecated by scanner/store
// pub mod compiler;  // Deprecated by scanner/store
// pub mod runtime;   // Deprecated by scanner/store

/// Dense RuleId (0..rule_count). Keep it small and indexable.
pub type RuleId = u32;

/// Optional: context-shift / mode control hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlOp {
    /// Reset scanner-local rule state (bitsets/counters) but keep automaton state.
    ResetRuleState,
    /// User-defined mode switch slot.
    Mode(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FseOpcode {
    Ignore,
    Record(RuleId),
    Reject(RuleId),
    Control(ControlOp),
}

/// Input rule definition for compilation.
#[derive(Debug, Clone)]
pub struct Rule {
    pub pattern: Vec<u8>,
    pub opcode: FseOpcode,
}

impl Rule {
    pub fn new<P: AsRef<[u8]>>(pattern: P, opcode: FseOpcode) -> Self {
        Self {
            pattern: pattern.as_ref().to_vec(),
            opcode,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanResult {
    /// Continue scanning.
    Continue,
    /// Stop immediately due to critical rule violation.
    Rejected(RuleId),
}
