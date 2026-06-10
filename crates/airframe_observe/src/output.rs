//! ObservationData — the value broadcast to observers when a selector fires.
//!
//! This is the "value" side of FSE's (selector, value) dispatch.
//! When a selector fires during the single-pass inference run, the
//! execution module wraps the tensor/text data into ObservationData
//! and broadcasts it to all registered observers simultaneously.
//!
//! Zero-copy where possible: observers receive a reference during the
//! broadcast window. Clone only when the observer needs to retain the data.

/// Data payload broadcast to observers when a selector fires.
#[derive(Debug, Clone)]
pub enum ObservationData {
    /// Hidden state after transformer layer N.
    /// Shape: [n_embd]
    LayerOutput {
        layer_idx: usize,
        position: usize,
        values: Vec<f32>,
        rms: f32,
        checksum: i64,
    },

    /// Final logits after output_norm + output_proj.
    /// Shape: [n_vocab]
    FinalLogits {
        position: usize,
        values: Vec<f32>,
        rms: f32,
        checksum: i64,
    },

    /// Q projection output at layer N.
    AttnQ {
        layer_idx: usize,
        values: Vec<f32>,
    },

    /// K projection output at layer N.
    AttnK {
        layer_idx: usize,
        values: Vec<f32>,
    },

    /// V projection output at layer N.
    AttnV {
        layer_idx: usize,
        values: Vec<f32>,
    },

    /// Decoded output text (UTF-8, incremental).
    OutputText {
        step: usize,
        text: String,
    },
}

impl ObservationData {
    /// Compute RMS of a float slice — used for oracle fingerprinting.
    pub fn rms(v: &[f32]) -> f32 {
        let sq: f32 = v.iter().map(|x| x * x).sum();
        (sq / v.len() as f32).sqrt()
    }

    /// Compute row-wise checksum — deterministic, catches silent corruption.
    pub fn checksum(v: &[f32]) -> i64 {
        v.iter()
            .map(|x| x.to_bits() as i64)
            .fold(0i64, |a, b| a.wrapping_add(b))
    }
}
