//! # airframe-observe
//!
//! FSE-based inference observation layer, powered by d0-engine.
//!
//! ## The Core Inversion
//!
//! Conventional inference capture is observer-first (broken pattern):
//! ```text
//! vault_seed:   forward_pass() → capture hidden_states
//! candle_probe: forward_pass() → capture logits
//! fse_policy:   forward_pass() → scan output_text
//! Cost: O(N_observers × M_selectors)
//! ```
//!
//! This crate is selector-first (FSE/d0-engine architecture):
//! ```text
//! ObservationSession::new(plan)  ← compile once
//! forward_pass emits facts        ← single pass
//! All observers receive data simultaneously via d0-engine broadcast
//! Cost: O(M_selectors), independent of observer count
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use airframe_observe::{ObservationSession, InferenceFact};
//!
//! // 1. Build a session with registered observers
//! let mut session = ObservationSession::new();
//! session.register_vault_oracle();
//! session.register_candle_compare();
//!
//! // 2. During forward pass — emit facts via convenience helpers
//! let hidden: Vec<f32> = vec![0.1; 2048];
//! session.emit_layer_output(0, 1, &hidden);
//!
//! let logits: Vec<f32> = vec![0.01; 32000];
//! session.emit_final_logits(1, &logits);
//!
//! // 3. Run to fixpoint — all observers receive their data
//! let result = session.saturate();
//! assert!(result.saturated);
//! ```
//!
//! ## Patent Notice
//!
//! This selector-first, compile-once, single-pass observation architecture
//! applied to AI inference is covered by a pending US patent by
//! Michael A. Kuykendall. All rights reserved.
//! Contact: michaelallenkuykendall@gmail.com

pub mod facts;
pub mod observers;
pub mod session;

pub use facts::{alpha_key_of, InferenceFact};
pub use session::ObservationSession;
