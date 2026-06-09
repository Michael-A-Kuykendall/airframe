//! # airframe-observe
//!
//! FSE-based inference observation layer.
//!
//! ## The Core Inversion
//!
//! Conventional inference capture is **observer-first**:
//! ```text
//! vault_seed:   forward_pass() → capture hidden_states
//! candle_probe: forward_pass() → capture logits
//! fse_control:  forward_pass() → scan output_text
//! ```
//! Each observer triggers its own extraction. Cost: O(N_observers × M_selectors).
//!
//! This crate is **selector-first** (FSE architecture):
//! ```text
//! ObservationPlan::compile([
//!     Selector::LayerOutput(0..N),   // → broadcast to: vault_oracle, checksum
//!     Selector::FinalLogits,          // → broadcast to: vault_oracle, candle_compare
//!     Selector::OutputText,           // → broadcast to: fse_policy_scan
//! ])
//! ```
//! One forward pass. All observers receive their data simultaneously.
//! Adding a new observer with a shared selector costs zero additional extraction.
//!
//! ∂runtime / ∂observers ≈ 0 (for shared selectors)
//!
//! ## Patent Notice
//!
//! This selector-first, compile-once, single-pass observation architecture
//! applied to AI inference is covered by a pending US patent by
//! Michael A. Kuykendall. All rights reserved.
//! Contact: michaelallenkuykendall@gmail.com
//!
//! ## Status: Skeleton
//!
//! This is the foundational skeleton. The observation plan compiles,
//! selectors are defined, observers register interest.
//! The execution module wires into the Airframe forward pass in Phase 2.

pub mod plan;
pub mod observer;
pub mod output;
