//! GPU timestamp query pool for accurate dispatch timing.
//!
//! Wraps `wgpu::QuerySet` with `TIMESTAMP_QUERY` to capture GPU-side
//! timestamps before/after compute dispatches. Provides a readback path
//! via resolve + staging buffer + `map_async`.
//!
//! Falls back to `None` (no timestamps) when
//! `TIMESTAMP_QUERY_INSIDE_PASSES` is unsupported.

use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, QuerySet, QueryType};

/// A pool of timestamp queries arranged in pairs (start, end).
///
/// Each pair captures GPU elapsed time for one dispatch.
pub struct TimestampPool {
    query_set: QuerySet,
    /// Resolve destination buffer (written by resolve_query_set).
    resolve_buf: Buffer,
    /// Staging readback buffer (mapped for CPU read).
    readback_buf: Buffer,
    /// Number of timestamp pairs allocated.
    pair_count: u32,
    /// GPU timestamp period (nanoseconds per tick, from device).
    period_ns: f64,
    /// Index of the next free pair.
    next_pair: u32,
    /// Whether timestamp queries are supported (checked at construction).
    supported: bool,
}

impl TimestampPool {
    pub fn for_compute_passes(
        device: &wgpu::Device,
        pair_count: u32,
        period_ns: f64,
    ) -> Option<Self> {
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if !supported
            || pair_count == 0
            || pair_count > wgpu::QUERY_SET_MAX_QUERIES / 2
            || !period_ns.is_finite()
            || period_ns <= 0.0
        {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamp_pool"),
            ty: QueryType::Timestamp,
            count: pair_count * 2,
        });
        let resolve_buf = device.create_buffer(&BufferDescriptor {
            label: Some("timestamp_resolve"),
            size: (pair_count as u64 * 2 * 8), // u64 per timestamp
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&BufferDescriptor {
            label: Some("timestamp_readback"),
            size: (pair_count as u64 * 2 * 8),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            query_set,
            resolve_buf,
            readback_buf,
            pair_count,
            period_ns,
            next_pair: 0,
            supported: true,
        })
    }
    pub fn reserve_pass(&mut self) -> Option<(u32, wgpu::ComputePassTimestampWrites<'_>)> {
        if self.next_pair >= self.pair_count {
            return None;
        }
        let index = self.next_pair;
        self.next_pair += 1;
        Some((
            index,
            wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(index * 2),
                end_of_pass_write_index: Some(index * 2 + 1),
            },
        ))
    }
    pub fn used_pairs(&self) -> u32 {
        self.next_pair
    }
    pub fn unmap(&self) {
        self.readback_buf.unmap();
    }

    /// Create a new timestamp pool.
    ///
    /// Returns `None` when `TIMESTAMP_QUERY_INSIDE_PASSES` is not supported
    /// (common on some OpenGL/Vulkan backends).
    pub fn new(device: &wgpu::Device, pair_count: u32, period_ns: f64) -> Option<Self> {
        if !device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
        {
            return None;
        }
        Self::for_compute_passes(device, pair_count, period_ns)
    }

    /// Write a start timestamp for the next pair.
    pub fn write_start(&mut self, encoder: &mut CommandEncoder) -> Option<u32> {
        if !self.supported || self.next_pair >= self.pair_count {
            return None;
        }
        let idx = self.next_pair;
        self.next_pair += 1;
        encoder.write_timestamp(&self.query_set, idx * 2);
        Some(idx)
    }

    /// Write an end timestamp for the given pair index (returned by `write_start`).
    pub fn write_end(&self, encoder: &mut CommandEncoder, pair_idx: u32) {
        if self.supported && pair_idx < self.next_pair {
            encoder.write_timestamp(&self.query_set, pair_idx * 2 + 1);
        }
    }

    /// Resolve all pending timestamp pairs into the readback buffer.
    pub fn resolve(&self, encoder: &mut CommandEncoder) {
        if !self.supported || self.next_pair == 0 {
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..self.next_pair * 2, &self.resolve_buf, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buf,
            0,
            &self.readback_buf,
            0,
            self.next_pair as u64 * 16,
        );
    }

    /// Read elapsed nanoseconds for a single pair.
    ///
    /// Call after the resolve buffer has been mapped and copied.
    /// Returns `None` if timestamps are unsupported or pair is out of range.
    pub fn elapsed_ns(&self, pair_idx: u32, data: &[u64]) -> Option<f64> {
        if !self.supported || pair_idx >= self.next_pair {
            return None;
        }
        let start = *data.get(pair_idx as usize * 2)?;
        let end = *data.get(pair_idx as usize * 2 + 1)?;
        if start == 0 || end <= start {
            return None;
        }
        let elapsed = (end - start) as f64 * self.period_ns;
        (elapsed.is_finite() && elapsed > 0.0).then_some(elapsed)
    }

    /// Map the readback buffer for CPU access.
    ///
    /// Returns a future-compatible handle. Typical usage:
    /// ```ignore
    /// let slice = pool.readback_slice();
    /// slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    /// device.poll(wgpu::Maintain::Wait);
    /// let data: &[u64] = bytemuck::cast_slice(slice.get_mapped_range());
    /// ```
    pub fn readback_slice(&self) -> wgpu::BufferSlice<'_> {
        self.readback_buf.slice(0..self.next_pair as u64 * 16)
    }

    /// Reset the pair index for a new frame.
    pub fn reset(&mut self) {
        self.next_pair = 0;
    }

    pub fn is_supported(&self) -> bool {
        self.supported
    }

    pub fn pair_count(&self) -> u32 {
        self.pair_count
    }
}
