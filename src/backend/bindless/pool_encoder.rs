//! Bounded pool of `CommandEncoder`s for non-blocking GPU work submission.
//!
//! Instead of a single encoder that blocks on submit+poll, the pool allows
//! dispatching one encoder's work while immediately encoding into the next.
//! This keeps the GPU fed without pipeline bubbles.
//!
//! ## Usage
//! ```ignore
//! let mut pool = EncoderPool::new(&device, &queue, 4);
//! let encoder = pool.acquire();
//! // ... append compute passes ...
//! pool.submit_and_recycle(encoder);
//! ```

use wgpu::CommandEncoder;

/// Bounded pool of `CommandEncoder`s.
///
/// Each slot holds an encoder in the recording state. Acquire returns the
/// next available encoder; submit_and_recycle submits it and replaces it
/// with a fresh encoder, never blocking on GPU completion.
pub struct EncoderPool {
    device: wgpu::Device,
    queue: wgpu::Queue,
    encoders: Vec<Option<CommandEncoder>>,
    next_idx: usize,
    max_pending: usize,
}

impl EncoderPool {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, max_pending: usize) -> Self {
        let mut encoders = Vec::with_capacity(max_pending);
        for _ in 0..max_pending {
            let desc = wgpu::CommandEncoderDescriptor {
                label: Some("pool_encoder"),
            };
            encoders.push(Some(device.create_command_encoder(&desc)));
        }
        Self {
            device: device.clone(),
            queue: queue.clone(),
            encoders,
            next_idx: 0,
            max_pending,
        }
    }

    /// Acquire the next available encoder for recording.
    pub fn acquire(&mut self) -> CommandEncoder {
        let idx = self.next_idx;
        self.next_idx = (self.next_idx + 1) % self.max_pending;
        self.encoders[idx].take().expect(
            "pool_encoder: slot was empty — call submit_and_recycle before acquiring all slots",
        )
    }

    /// Submit the given encoder and recycle it with a fresh encoder.
    pub fn submit_and_recycle(&mut self, encoder: CommandEncoder) {
        let idx = (self.next_idx + self.max_pending - 1) % self.max_pending;
        let submit_idx = idx;
        let cmd_buf = encoder.finish();
        self.queue.submit([cmd_buf]);
        let desc = wgpu::CommandEncoderDescriptor {
            label: Some("pool_encoder"),
        };
        self.encoders[submit_idx] = Some(self.device.create_command_encoder(&desc));
    }

    /// Submit all pending encoders and wait for GPU completion.
    pub fn drain(&mut self) {
        for slot in &mut self.encoders {
            if let Some(encoder) = slot.take() {
                let cmd_buf = encoder.finish();
                self.queue.submit([cmd_buf]);
            }
        }
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let desc = wgpu::CommandEncoderDescriptor {
            label: Some("pool_encoder"),
        };
        for slot in &mut self.encoders {
            *slot = Some(self.device.create_command_encoder(&desc));
        }
    }

    pub fn max_pending(&self) -> usize {
        self.max_pending
    }
}
