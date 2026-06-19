//! ObservationPlan — the compiled FSE execution plan for inference observation.
//!
//! Analogous to `FseMap::compile()` in libfse:
//! - Rules (selectors + observers) are compiled once at startup
//! - The compiled plan is immutable and shareable across threads (Arc)
//! - Per-inference cost is O(unique_selectors), not O(N_observers)

use crate::observer::ObserverId;
use std::collections::HashMap;

/// A selector identifies a specific data point in the inference graph.
///
/// Selectors are deduplicated at compile time — if two observers want
/// `FinalLogits`, the logit tensor is extracted exactly once and broadcast.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Selector {
    /// Hidden state output after transformer layer N (0-indexed).
    LayerOutput(usize),

    /// Hidden state output for ALL layers (0..n_layers).
    /// Expands to N individual LayerOutput selectors at compile time.
    AllLayerOutputs,

    /// Final logits after output_norm + output_proj (vocab-size vector).
    /// This is what candle_probe captures for cross-validation.
    FinalLogits,

    /// Decoded output text (post-sampling, per-token).
    /// This is what FseControl uses for policy scanning.
    OutputText,

    /// Q projection output at layer N, position P.
    AttnQ { layer: usize },

    /// K projection output at layer N.
    AttnK { layer: usize },

    /// V projection output at layer N.
    AttnV { layer: usize },
}

/// A compiled observation entry: one selector → one or more observers.
#[derive(Debug, Clone)]
pub struct CompiledEntry {
    pub selector: Selector,
    /// All observer IDs that registered interest in this selector.
    /// Broadcast: when selector fires, all observers receive the data.
    pub observers: Vec<ObserverId>,
}

/// The compiled observation plan.
///
/// Built once from a list of (Selector, ObserverId) registrations.
/// Immutable after compilation — share via Arc.
#[derive(Debug)]
pub struct ObservationPlan {
    /// Deduplicated selector index: selector → list of observers.
    /// This is the trie/hashmap at the heart of FSE.
    pub entries: Vec<CompiledEntry>,
    /// Total unique selectors after deduplication.
    pub selector_count: usize,
    /// Total registered (selector, observer) pairs before deduplication.
    pub registration_count: usize,
}

/// Registration: an observer declares which selectors it cares about.
pub struct Registration {
    pub selector: Selector,
    pub observer: ObserverId,
}

impl ObservationPlan {
    /// Compile a list of registrations into a deduplicated execution plan.
    ///
    /// This is the FSE compile phase:
    /// - Parse registrations into (selector, observer) pairs
    /// - Deduplicate selectors — merge all observers sharing a selector
    /// - Output: CompiledEntries with broadcast lists
    ///
    /// Cost: O(N_registrations). Amortized at startup, not per-inference.
    pub fn compile(registrations: Vec<Registration>) -> Self {
        let registration_count = registrations.len();

        // Selector deduplication: build selector → Vec<ObserverId> map
        let mut selector_index: HashMap<SelectorKey, Vec<ObserverId>> = HashMap::new();

        for reg in registrations {
            // Expand AllLayerOutputs into individual selectors at compile time
            // so the runtime loop is uniform — no special cases during execution
            let selectors = match reg.selector {
                Selector::AllLayerOutputs => {
                    // Placeholder: actual n_layers comes from model spec at runtime
                    // For the skeleton, we record as-is and expand at bind time
                    vec![reg.selector]
                }
                s => vec![s],
            };

            for selector in selectors {
                let key = SelectorKey::from(&selector);
                selector_index
                    .entry(key)
                    .or_insert_with(|| vec![])
                    .push(reg.observer.clone());
            }
        }

        let selector_count = selector_index.len();

        // Build CompiledEntries — stable ordering for deterministic execution
        let mut entries: Vec<CompiledEntry> = selector_index
            .into_iter()
            .map(|(key, observers)| CompiledEntry {
                selector: key.into(),
                observers,
            })
            .collect();

        // Sort for deterministic order: layer outputs first, logits last, text last
        entries.sort_by_key(|e| selector_sort_key(&e.selector));

        ObservationPlan {
            entries,
            selector_count,
            registration_count,
        }
    }

    /// How many unique selectors are in this plan.
    /// This is the M in O(M) — the runtime cost invariant.
    pub fn unique_selector_count(&self) -> usize {
        self.selector_count
    }

    /// How many (selector, observer) pairs were registered.
    /// Useful for verifying deduplication efficiency.
    pub fn registration_count(&self) -> usize {
        self.registration_count
    }
}

// ─── Internal key type for HashMap ───────────────────────────────────────────

/// Hashable key derived from Selector (Selector itself may not be Hash in all variants).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SelectorKey {
    LayerOutput(usize),
    AllLayerOutputs,
    FinalLogits,
    OutputText,
    AttnQ(usize),
    AttnK(usize),
    AttnV(usize),
}

impl From<&Selector> for SelectorKey {
    fn from(s: &Selector) -> Self {
        match s {
            Selector::LayerOutput(n) => SelectorKey::LayerOutput(*n),
            Selector::AllLayerOutputs => SelectorKey::AllLayerOutputs,
            Selector::FinalLogits => SelectorKey::FinalLogits,
            Selector::OutputText => SelectorKey::OutputText,
            Selector::AttnQ { layer } => SelectorKey::AttnQ(*layer),
            Selector::AttnK { layer } => SelectorKey::AttnK(*layer),
            Selector::AttnV { layer } => SelectorKey::AttnV(*layer),
        }
    }
}

impl From<SelectorKey> for Selector {
    fn from(k: SelectorKey) -> Self {
        match k {
            SelectorKey::LayerOutput(n) => Selector::LayerOutput(n),
            SelectorKey::AllLayerOutputs => Selector::AllLayerOutputs,
            SelectorKey::FinalLogits => Selector::FinalLogits,
            SelectorKey::OutputText => Selector::OutputText,
            SelectorKey::AttnQ(n) => Selector::AttnQ { layer: n },
            SelectorKey::AttnK(n) => Selector::AttnK { layer: n },
            SelectorKey::AttnV(n) => Selector::AttnV { layer: n },
        }
    }
}

fn selector_sort_key(s: &Selector) -> (u32, usize) {
    match s {
        Selector::LayerOutput(n) => (0, *n),
        Selector::AllLayerOutputs => (0, usize::MAX),
        Selector::AttnQ { layer } => (1, *layer),
        Selector::AttnK { layer } => (2, *layer),
        Selector::AttnV { layer } => (3, *layer),
        Selector::FinalLogits => (4, 0),
        Selector::OutputText => (5, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::ObserverId;

    #[test]
    fn deduplicates_shared_selectors() {
        // Two observers both want FinalLogits — should produce ONE compiled entry
        // with both observers in the broadcast list.
        let regs = vec![
            Registration {
                selector: Selector::FinalLogits,
                observer: ObserverId("vault_oracle".into()),
            },
            Registration {
                selector: Selector::FinalLogits,
                observer: ObserverId("candle_compare".into()),
            },
        ];

        let plan = ObservationPlan::compile(regs);

        assert_eq!(
            plan.unique_selector_count(),
            1,
            "should deduplicate to 1 selector"
        );
        assert_eq!(
            plan.registration_count(),
            2,
            "should record 2 registrations"
        );

        let entry = &plan.entries[0];
        assert_eq!(
            entry.observers.len(),
            2,
            "both observers should be in broadcast list"
        );
    }

    #[test]
    fn independent_selectors_not_merged() {
        let regs = vec![
            Registration {
                selector: Selector::LayerOutput(0),
                observer: ObserverId("vault_oracle".into()),
            },
            Registration {
                selector: Selector::FinalLogits,
                observer: ObserverId("candle_compare".into()),
            },
        ];

        let plan = ObservationPlan::compile(regs);
        assert_eq!(plan.unique_selector_count(), 2);
    }

    #[test]
    fn layer_outputs_sort_before_logits() {
        let regs = vec![
            Registration {
                selector: Selector::FinalLogits,
                observer: ObserverId("a".into()),
            },
            Registration {
                selector: Selector::LayerOutput(5),
                observer: ObserverId("b".into()),
            },
            Registration {
                selector: Selector::LayerOutput(0),
                observer: ObserverId("c".into()),
            },
        ];

        let plan = ObservationPlan::compile(regs);
        // LayerOutput(0), LayerOutput(5), FinalLogits — in that order
        assert!(matches!(plan.entries[0].selector, Selector::LayerOutput(0)));
        assert!(matches!(plan.entries[1].selector, Selector::LayerOutput(5)));
        assert!(matches!(plan.entries[2].selector, Selector::FinalLogits));
    }
}
