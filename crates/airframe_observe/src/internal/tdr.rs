//! Private TDR / Saturation Fabric support.
//! Implements TDR navigation as part of the FSE/D0 reactive layer.
//! Uses d0-engine SaturationFabric internally for one-pass fact propagation,
//! chunk calibration, vault writes, and beads updates.
//!
//! All types and functions here are private to the crate.

use crate::facts::InferenceFact;
use d0_engine::{AlphaKey, ClosureProgram, RunBudget, SaturationFabric};
use std::sync::Arc;

/// TDR-specific facts (kept private).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TdrFact {
    DispatchStart {
        layer: u32,
        kernel: String,
        batch: u32,
        quant: u32,
    },
    DispatchTiming {
        layer: u32,
        kernel: String,
        ms: u32,
    },
    TdrRisk {
        level: u8,
    },
    SafeChunk {
        size: u32,
    },
    // Can be extended for beads task links, etc.
}

/// Key function for alpha dispatch (private).
pub fn tdr_alpha_key(f: &InferenceFact) -> Option<AlphaKey> {
    // Map heavy dispatch facts to a dedicated key.
    // Extend with real logic from InferenceFact variants.
    match f {
        InferenceFact::LayerOutput { .. } => Some(AlphaKey(10)), // example
        _ => None,
    }
}

/// Build the TDR rule program (private).
pub fn build_tdr_program() -> ClosureProgram<InferenceFact> {
    let mut prog = ClosureProgram::new();

    // Example rule: on dispatch timing, derive safe chunk.
    // Real version will use TIMESTAMP data + model context.
    prog.register(AlphaKey(10), |fact, _store| {
        // Placeholder: in real impl, compute from timing facts.
        if let InferenceFact::LayerOutput { layer_idx, .. } = fact {
            if *layer_idx == 0 {
                vec![InferenceFact::WriteOracleRow { layer_idx: 0 }] // example tie-in
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    });

    prog
}

/// Private TDR navigator session.
/// Wraps SaturationFabric for TDR + vault + beads in one pass.
pub struct PrivateTdrNavigator {
    fabric: SaturationFabric<InferenceFact>,
}

impl PrivateTdrNavigator {
    pub fn new() -> Self {
        let program = build_tdr_program();
        let key_fn = tdr_alpha_key;
        let handler = |c: d0_engine::Consequent<InferenceFact>,
                       store: &mut d0_engine::FactStore<InferenceFact>|
         -> Vec<InferenceFact> {
            // Handle consequents: drive chunking, vault write, beads update.
            // This is where the "saturation fabric" actions live.
            match c {
                d0_engine::Consequent::Custom(msg) if msg == "apply_chunk" => {
                    // In real: patch LayerParams, submit micro batch, etc.
                    vec![]
                }
                _ => vec![],
            }
        };

        let fabric = SaturationFabric::new(program, key_fn, handler);
        Self { fabric }
    }

    /// Emit a fact into the fabric (e.g. from GPU prefill).
    pub fn emit(&mut self, fact: InferenceFact) {
        self.fabric.assert(fact);
    }

    /// Run saturation. Returns whether we reached safe fixpoint for TDR.
    pub fn run(&mut self) -> bool {
        let result = self.fabric.run_to_fixpoint(RunBudget::default());
        result.saturated
    }

    /// Get current safe chunk recommendation (from derived facts).
    pub fn recommended_chunk(&self) -> u32 {
        // In real impl: inspect last SafeChunk fact.
        8 // placeholder, will come from fabric
    }
}
