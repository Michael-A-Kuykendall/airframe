//! # airframe-observe 0.1.4
//!
//! Inference facts, observers and host-side coordination using Dzero.
//!
//! ObservationSession registers observers, receives emitted facts and runs
//! saturation. ISF uses Dzero's fact store and caller-supplied inference
//! callbacks. Work includes fact processing and registered observer callbacks;
//! no observer-count-independent performance guarantee is implied.
//!
//! These contracts neither implement GPU arithmetic nor certify inference
//! correctness, driver stability or performance. See the crate README for
//! release scope, dependency and licensing details.
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
pub mod isf;
pub mod observer;
pub mod observers;
pub mod output;
pub mod plan;
pub mod profiling;
pub mod quant_formula;
pub mod session;

// Private experimentation / tuning surface (Saturation Fabric, TDR, beads integration).
// Not re-exported publicly. Use only from within airframe crate.
mod internal;

pub use facts::{alpha_key_of, HaltReason, InferenceFact};
pub use isf::{GenerateOutput, ISFState, InferenceSaturationFabric};
pub use observer::{CandleCompareObserver, VaultOracleObserver};
pub use output::ObservationData;
pub use session::ObservationSession;
