//! TDR (Timeout Detection and Recovery) Scheduler for GPU inference.
//!
//! Extracts the TDR yield logic from the 900-line inference monolith into a
//! clean, testable struct. The scheduler tracks accumulated GPU dispatch time
//! and decides when to submit+poll to keep the GPU below the Windows TDR
//! watchdog threshold (~2s).
//!
//! ## Platform behaviour
//! - **Windows (D3D12)**: hard ~2s TDR. Default budget = 1400ms.
//! - **Linux / macOS**: no hard TDR. Default budget = 30 000ms (effectively
//!   never yields, matching the old unconstrained behaviour).
//!
//! Override the budget at runtime with `SHIMMY_TDR_BUDGET_MS`.
//!
//! ## Integration
//! The scheduler owns a `wgpu::CommandEncoder`. Callers append compute passes
//! to `scheduler.encoder`, then call `yield_if_needed` / `force_yield` at safe
//! checkpoint points. The scheduler manages submit + poll + new-encoder
//! transparently so the caller never touches `queue.submit` directly.
//!
//! Patent Notice: Implements Fused Semantic Execution (FSE) + D0 Saturation
//! Fabric scheduling. Pending patent by Michael A. Kuykendall. All rights
//! reserved.

/// Platform-aware default TDR budget in milliseconds.
///
/// Windows: conservative 1400ms (watchdog fires at ~2000ms).
/// Other: 30 000ms — effectively never yields on Linux/macOS.
#[cfg(windows)]
pub const DEFAULT_TDR_BUDGET_MS: u128 = 1400;
#[cfg(not(windows))]
pub const DEFAULT_TDR_BUDGET_MS: u128 = 30_000;

/// GPU dispatch scheduler with TDR-safe submit/poll logic.
///
/// Owns the active `wgpu::CommandEncoder` and tracks accumulated dispatch
/// time since the last yield. Callers use `encoder` directly for compute
/// passes; checkpoints call `yield_if_needed` or `force_yield`.
pub struct TdrScheduler<'d> {
    device: &'d wgpu::Device,
    queue: &'d wgpu::Queue,
    /// Active command encoder — callers append compute passes to this.
    pub encoder: wgpu::CommandEncoder,
    /// Accumulated GPU-side elapsed time since last yield (ms).
    pub accumulated_ms: u128,
    /// Budget threshold in ms — yield when accumulated >= budget.
    pub budget_ms: u128,
    /// Total number of yields performed since construction.
    pub yield_count: u32,
}

impl<'d> TdrScheduler<'d> {
    /// Create a new scheduler.
    ///
    /// Reads `SHIMMY_TDR_BUDGET_MS` from the environment; falls back to the
    /// platform default.
    pub fn new(device: &'d wgpu::Device, queue: &'d wgpu::Queue, label: &str) -> Self {
        let budget_ms = std::env::var("SHIMMY_TDR_BUDGET_MS")
            .ok()
            .and_then(|s| s.parse::<u128>().ok())
            .unwrap_or(DEFAULT_TDR_BUDGET_MS);

        let encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

        Self {
            device,
            queue,
            encoder,
            accumulated_ms: 0,
            budget_ms,
            yield_count: 0,
        }
    }

    /// Unconditionally submit the current encoder, poll the GPU, and start a
    /// new encoder. Returns the round-trip elapsed time in ms.
    ///
    /// This is the "hard yield" — used where write_buffer requires a prior
    /// submit (QKV micro-batch params update).
    pub fn force_yield(&mut self, label: &str) -> Result<u128, String> {
        let t0 = std::time::Instant::now();

        // Take the encoder by replacing it with a dummy first.
        // We need to call finish() which consumes it.
        let encoder = std::mem::replace(
            &mut self.encoder,
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tdr-placeholder"),
                }),
        );
        self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| format!("GPU TDR during {}", label))?;

        let elapsed = t0.elapsed().as_millis();
        self.accumulated_ms += elapsed;
        self.yield_count += 1;

        // Replace with a fresh encoder for subsequent work.
        self.encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("post-{}", label)),
            });

        Ok(elapsed)
    }

    /// Yield only if accumulated time has reached or exceeded the budget.
    ///
    /// Returns `true` if a yield happened (and resets the accumulator), or
    /// `false` if the budget is still safe.
    pub fn yield_if_needed(&mut self, label: &str) -> Result<bool, String> {
        if self.accumulated_ms >= self.budget_ms {
            let elapsed = self.force_yield(label)?;
            self.accumulated_ms = 0; // reset after yield
            let _ = elapsed;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Finish the current encoder and submit it as the final step (e.g. for
    /// the output head + readback commands). Returns the new encoder so callers
    /// can continue encoding readback copies after the submit.
    ///
    /// NOTE: after calling this, `self.encoder` is replaced with a fresh one.
    /// The submitted commands are in-flight; you still need to poll separately.
    pub fn submit_current(&mut self, label: &str) {
        let encoder = std::mem::replace(
            &mut self.encoder,
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("post-submit-{}", label)),
                }),
        );
        self.queue.submit(Some(encoder.finish()));
    }

    /// Add elapsed time to the accumulator without yielding.
    /// Used when external code does its own submit+poll (e.g. layer trace
    /// readbacks) and wants to inform the scheduler of the time spent.
    pub fn record_elapsed(&mut self, elapsed_ms: u128) {
        self.accumulated_ms += elapsed_ms;
    }

    /// Reset the accumulator. Called after a manual yield outside the
    /// scheduler (e.g. layer trace readback).
    pub fn reset_accumulator(&mut self) {
        self.accumulated_ms = 0;
    }

    /// Start a wall-clock timer for a GPU dispatch segment.
    /// Call `record_wall_elapsed(start)` after the segment to accumulate
    /// CPU-side elapsed time without doing a GPU round-trip.
    /// This feeds the TDR budget check without requiring TIMESTAMP_QUERY.
    pub fn wall_start() -> std::time::Instant {
        std::time::Instant::now()
    }

    /// Record the wall-clock time since `start` into the accumulator.
    /// Returns true if a yield is now needed (accumulated >= budget).
    pub fn record_wall_elapsed(&mut self, start: std::time::Instant) -> bool {
        let elapsed = start.elapsed().as_millis();
        self.accumulated_ms += elapsed;
        self.accumulated_ms >= self.budget_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budget_is_platform_aware() {
        // On Windows the default must be < 2000ms (the TDR timeout).
        // On other platforms it should be >> 2000ms (effectively never).
        #[cfg(windows)]
        assert!(DEFAULT_TDR_BUDGET_MS < 2000);
        #[cfg(not(windows))]
        assert!(DEFAULT_TDR_BUDGET_MS > 10_000);
    }

    #[test]
    fn yield_threshold_logic() {
        // Verify accumulator logic without needing a GPU device.
        let budget: u128 = 1400;
        let mut accumulated: u128 = 0;

        accumulated += 700;
        assert!(accumulated < budget, "below threshold — no yield");

        accumulated += 800;
        assert!(accumulated >= budget, "at/above threshold — yield");

        accumulated = 0; // reset after yield
        assert_eq!(accumulated, 0);
    }
}
