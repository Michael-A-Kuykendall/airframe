// GPU KV Cache Implementation for Transformer Inference
// Phase 4A: F32 Cache Architecture (704 MB VRAM)
// TurboQuant: INT4 packed+scale buffers (feat/turboquant-wgsl)

use wgpu;

/// GPU-resident KV cache for multi-token transformer inference
///
/// Stores Key and Value tensors across all layers to enable:
/// - Coherent text generation (context memory)
/// - Efficient autoregressive decoding (no recomputation)
/// - Causal attention masking
///
/// **Architecture Decision (Spike 2):** F32 cache required
/// - FP16 precision insufficient (9.65e-4 error, 960x threshold)
/// - Attention values need 6-7 decimal digits
/// - Cost: 704 MB VRAM (5.9% of RTX 3060)
pub struct KVCache {
    /// K buffers: One per layer [n_head_kv, max_seq, head_dim] in F32
    k_buffers: Vec<wgpu::Buffer>,

    /// V buffers: One per layer [n_head_kv, max_seq, head_dim] in F32
    v_buffers: Vec<wgpu::Buffer>,

    /// INT4 packed K nibbles: One per layer [max_seq, n_head_kv, head_dim/8] in U32
    /// Each U32 holds 8 nibbles (bias-8 encoded). Allocated only in INT4 mode.
    k_packed_buffers: Option<Vec<wgpu::Buffer>>,

    /// INT4 packed V nibbles: same layout as k_packed_buffers
    v_packed_buffers: Option<Vec<wgpu::Buffer>>,

    /// Per-head-vector K scale: One per layer [max_seq, n_head_kv] in F32
    /// scale = max_abs / 7.0 for the corresponding head-vector.
    k_scale_buffers: Option<Vec<wgpu::Buffer>>,

    /// Per-head-vector V scale: same layout as k_scale_buffers
    v_scale_buffers: Option<Vec<wgpu::Buffer>>,

    /// Current sequence length (number of tokens cached)
    seq_len: u32,

    /// Logical base position of the compacted sliding window.
    ///
    /// Sink tokens remain at absolute positions 0..keep_sink-1, but shifted
    /// window tokens conceptually start at `window_base + keep_sink`.
    window_base: u32,

    /// Maximum sequence length (context window size)
    max_seq_len: u32,

    /// Number of transformer layers
    // dead_code: n_layers stored for future multi-layer cache management (currently n_layer is implicit)
    #[allow(dead_code)]
    n_layers: usize,

    /// Number of KV heads (GQA: 4 for TinyLlama)
    n_head_kv: u32,

    /// Dimension per head (n_embd / n_head)
    head_dim: u32,
}

impl KVCache {
    /// Create new KV cache, allocating VRAM buffers
    ///
    /// # Memory Layout
    /// Per layer buffer: [max_seq_len, n_head_kv, head_dim] in F32
    /// TinyLlama: 2048 × 4 × 64 × 4 bytes = 2,097,152 bytes (2 MB)
    /// Total: 22 layers × 2 buffers × 2 MB = 88 MB
    ///
    /// # Arguments
    /// * `device` - WGPU device for buffer allocation
    /// * `n_layers` - Number of transformer layers (22 for TinyLlama)
    /// * `n_head_kv` - Number of KV heads for GQA (4 for TinyLlama)
    /// * `head_dim` - Dimension per head (64 for TinyLlama)
    /// * `max_seq_len` - Context window size (2048 for TinyLlama)
    pub fn new(
        device: &wgpu::Device,
        n_layers: usize,
        n_head_kv: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) -> Self {
        Self::new_inner(device, n_layers, n_head_kv, head_dim, max_seq_len, false)
    }

    /// Create KV cache with INT4 packed+scale buffers enabled.
    pub fn new_int4(
        device: &wgpu::Device,
        n_layers: usize,
        n_head_kv: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) -> Self {
        Self::new_inner(device, n_layers, n_head_kv, head_dim, max_seq_len, true)
    }

    fn new_inner(
        device: &wgpu::Device,
        n_layers: usize,
        n_head_kv: u32,
        head_dim: u32,
        max_seq_len: u32,
        enable_int4: bool,
    ) -> Self {
        // Buffer size per layer: [max_seq, n_head_kv, head_dim] in F32
        // Layout: position-major for cache append efficiency
        let buffer_size = (max_seq_len * n_head_kv * head_dim * 4) as u64;

        eprintln!("[KVCache] Allocating GPU buffers:");
        eprintln!("  Layers: {}", n_layers);
        eprintln!("  Max seq len: {}", max_seq_len);
        eprintln!("  KV heads: {}", n_head_kv);
        eprintln!("  Head dim: {}", head_dim);
        eprintln!(
            "  Buffer size per layer: {} bytes ({:.2} MB)",
            buffer_size,
            buffer_size as f64 / 1_048_576.0
        );

        let k_buffers: Vec<_> = (0..n_layers)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("KV_Cache_K_Layer_{}", i)),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let v_buffers: Vec<_> = (0..n_layers)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("KV_Cache_V_Layer_{}", i)),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // INT4 packed buffers: head_dim elements × 4 bits = head_dim/2 bytes per head-vector.
        // Stored as U32 arrays: head_dim/8 U32s per head-vector.
        // Layout per layer: [max_seq_len, n_head_kv, head_dim/8] U32.
        let (k_packed_buffers, v_packed_buffers, k_scale_buffers, v_scale_buffers) = if enable_int4
        {
            assert!(
                head_dim.is_multiple_of(8),
                "head_dim must be divisible by 8 for INT4 packing (got {})",
                head_dim
            );
            // packed: head_dim nibbles per head-vector = head_dim/8 U32s per head-vector
            let packed_size = (max_seq_len * n_head_kv * (head_dim / 8) * 4) as u64;
            // scale: one F32 per head-vector per position
            let scale_size = (max_seq_len * n_head_kv * 4) as u64;

            eprintln!(
                "  INT4 packed buf/layer: {} bytes ({:.2} MB)",
                packed_size,
                packed_size as f64 / 1_048_576.0
            );
            eprintln!(
                "  INT4 scale buf/layer:  {} bytes ({:.2} KB)",
                scale_size,
                scale_size as f64 / 1024.0
            );

            let mk_packed = |prefix: &str| -> Vec<wgpu::Buffer> {
                (0..n_layers)
                    .map(|i| {
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some(&format!("KV_Cache_{}_Packed_Layer_{}", prefix, i)),
                            size: packed_size,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        })
                    })
                    .collect()
            };
            let mk_scale = |prefix: &str| -> Vec<wgpu::Buffer> {
                (0..n_layers)
                    .map(|i| {
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some(&format!("KV_Cache_{}_Scale_Layer_{}", prefix, i)),
                            size: scale_size,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        })
                    })
                    .collect()
            };

            (
                Some(mk_packed("K")),
                Some(mk_packed("V")),
                Some(mk_scale("K")),
                Some(mk_scale("V")),
            )
        } else {
            (None, None, None, None)
        };

        let total_vram = buffer_size * (n_layers as u64) * 2;
        eprintln!(
            "  Total VRAM allocated: {} bytes ({:.2} MB)",
            total_vram,
            total_vram as f64 / 1_048_576.0
        );

        Self {
            k_buffers,
            v_buffers,
            k_packed_buffers,
            v_packed_buffers,
            k_scale_buffers,
            v_scale_buffers,
            seq_len: 0,
            window_base: 0,
            max_seq_len,
            n_layers,
            n_head_kv,
            head_dim,
        }
    }

    /// Get current sequence length (number of cached positions)
    pub fn get_seq_len(&self) -> u32 {
        self.seq_len
    }

    /// Get the logical base position for the shifted sliding window.
    pub fn get_window_base(&self) -> u32 {
        self.window_base
    }

    /// Get reference to all K buffers (for full pipeline execution)
    pub fn get_k_buffers(&self) -> &[wgpu::Buffer] {
        &self.k_buffers
    }

    /// Get reference to all V buffers (for full pipeline execution)
    pub fn get_v_buffers(&self) -> &[wgpu::Buffer] {
        &self.v_buffers
    }

    /// Reset cache to empty state (for new conversation/prompt)
    ///
    /// Buffers remain allocated, only position counter is reset.
    /// This is much faster than deallocating and reallocating.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.window_base = 0;
    }

    /// Explicitly override sequence length (used during helical cache compaction)
    pub fn set_seq_len(&mut self, new_len: u32) {
        if new_len > self.max_seq_len {
            // Invariant: helical compaction must never produce a length larger than the window.
            panic!(
                "KV cache override overflow: {} > {}",
                new_len, self.max_seq_len
            );
        }
        self.seq_len = new_len;
    }

    /// Advance the logical base after helical compaction.
    pub fn advance_window_base(&mut self, shift_amt: u32) {
        self.window_base = self.window_base.saturating_add(shift_amt);
    }

    /// Increment sequence length after appending new K/V pair
    ///
    /// Returns `Err` if the sequence exceeds max_seq_len (context overflow).
    pub fn increment(&mut self) -> Result<(), String> {
        self.seq_len += 1;
        if self.seq_len > self.max_seq_len {
            return Err(format!(
                "KV cache overflow: {} > {} (context window exceeded)",
                self.seq_len, self.max_seq_len
            ));
        }
        Ok(())
    }

    /// Get K buffer for a specific layer
    ///
    /// # Arguments
    /// * `layer` - Layer index (0..n_layers)
    pub fn get_k_buffer(&self, layer: usize) -> &wgpu::Buffer {
        &self.k_buffers[layer]
    }

    /// Get V buffer for a specific layer
    ///
    /// # Arguments
    /// * `layer` - Layer index (0..n_layers)
    pub fn get_v_buffer(&self, layer: usize) -> &wgpu::Buffer {
        &self.v_buffers[layer]
    }

    /// Get INT4 packed K buffer for a specific layer. Panics if INT4 mode not enabled.
    pub fn get_k_packed_buffer(&self, layer: usize) -> &wgpu::Buffer {
        self.k_packed_buffers
            .as_ref()
            .expect("KVCache: get_k_packed_buffer called but INT4 buffers not allocated")
            .get(layer)
            .expect("KVCache: layer index out of range for k_packed_buffers")
    }

    /// Get INT4 packed V buffer for a specific layer. Panics if INT4 mode not enabled.
    pub fn get_v_packed_buffer(&self, layer: usize) -> &wgpu::Buffer {
        self.v_packed_buffers
            .as_ref()
            .expect("KVCache: get_v_packed_buffer called but INT4 buffers not allocated")
            .get(layer)
            .expect("KVCache: layer index out of range for v_packed_buffers")
    }

    /// Get K scale buffer for a specific layer. Panics if INT4 mode not enabled.
    pub fn get_k_scale_buffer(&self, layer: usize) -> &wgpu::Buffer {
        self.k_scale_buffers
            .as_ref()
            .expect("KVCache: get_k_scale_buffer called but INT4 buffers not allocated")
            .get(layer)
            .expect("KVCache: layer index out of range for k_scale_buffers")
    }

    /// Get V scale buffer for a specific layer. Panics if INT4 mode not enabled.
    pub fn get_v_scale_buffer(&self, layer: usize) -> &wgpu::Buffer {
        self.v_scale_buffers
            .as_ref()
            .expect("KVCache: get_v_scale_buffer called but INT4 buffers not allocated")
            .get(layer)
            .expect("KVCache: layer index out of range for v_scale_buffers")
    }

    /// Returns true if this cache was allocated with INT4 packed+scale buffers.
    pub fn is_int4(&self) -> bool {
        self.k_packed_buffers.is_some()
    }

    /// Get maximum sequence length (context window size)
    pub fn max_len(&self) -> u32 {
        self.max_seq_len
    }

    /// Calculate byte offset for accessing cache at position/head/dimension
    ///
    /// # Arguments
    /// * `pos` - Sequence position (0..seq_len)
    /// * `head` - KV head index (0..n_head_kv)
    /// * `dim` - Dimension within head (0..head_dim)
    ///
    /// # Returns
    /// Byte offset into K or V buffer (multiply element offset by 4 for F32)
    pub fn calculate_offset(&self, pos: u32, head: u32, dim: u32) -> usize {
        let element_offset = (pos * self.n_head_kv * self.head_dim) + (head * self.head_dim) + dim;
        (element_offset * 4) as usize // F32 = 4 bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test KVCache allocation for TinyLlama configuration
    ///
    /// Validates:
    /// - Correct buffer count (22 K + 22 V = 44 buffers)
    /// - Correct VRAM usage (88 MB total for GQA with 4 KV heads)
    /// - Buffer indexing works
    #[test]
    fn test_kv_cache_allocation() {
        // TinyLlama 1.1B specs
        let n_layers = 22;
        let n_head_kv = 4; // GQA: 4 KV heads (32 Q heads share these)
        let head_dim = 64;
        let max_seq_len = 2048;

        // Create GPU instance for testing
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No GPU adapter found");

        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            }))
            .expect("Failed to create device");

        // Create cache
        let cache = KVCache::new(&device, n_layers, n_head_kv, head_dim, max_seq_len);

        // Validate allocation
        assert_eq!(
            cache.k_buffers.len(),
            22,
            "Should have 22 K buffers (one per layer)"
        );
        assert_eq!(
            cache.v_buffers.len(),
            22,
            "Should have 22 V buffers (one per layer)"
        );
        assert_eq!(cache.get_seq_len(), 0, "Initial seq_len should be 0");
        assert_eq!(cache.max_len(), 2048, "Max seq len should be 2048");

        // Calculate expected VRAM usage
        // Per buffer: 2048 × 4 × 64 × 4 bytes (F32) = 2,097,152 bytes
        let buffer_size = max_seq_len * n_head_kv * head_dim * 4;
        let total_vram = buffer_size * (n_layers as u32) * 2; // K + V

        println!(
            "Buffer size: {} bytes ({:.2} MB)",
            buffer_size,
            buffer_size as f64 / 1_048_576.0
        );
        println!(
            "Total VRAM: {} bytes ({:.2} MB)",
            total_vram,
            total_vram as f64 / 1_048_576.0
        );

        // Validate buffer sizes
        assert_eq!(cache.get_k_buffer(0).size(), buffer_size as u64);
        assert_eq!(cache.get_v_buffer(0).size(), buffer_size as u64);

        // Expected: ~88 MB for GQA (4 KV heads)
        // Note: Master plan may show 704 MB for full 32 heads, but GQA uses 4
        let expected_mb = 88.0;
        let actual_mb = total_vram as f64 / 1_048_576.0;
        assert!(
            (actual_mb - expected_mb).abs() < 1.0,
            "Expected ~{:.0} MB, got {:.2} MB",
            expected_mb,
            actual_mb
        );
    }

    /// Test sequence length management
    #[test]
    fn test_sequence_management() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("No GPU adapter");
        let (device, _queue) = pollster::block_on(adapter.request_device(&Default::default()))
            .expect("Failed to create device");

        let mut cache = KVCache::new(&device, 22, 4, 64, 2048);

        // Test increment
        assert_eq!(cache.get_seq_len(), 0);
        cache.increment();
        assert_eq!(cache.get_seq_len(), 1);
        cache.increment();
        assert_eq!(cache.get_seq_len(), 2);

        // Test reset
        cache.reset();
        assert_eq!(cache.get_seq_len(), 0);
    }

    /// Test offset calculation for cache indexing
    #[test]
    fn test_offset_calculation() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("No GPU adapter");
        let (device, _queue) = pollster::block_on(adapter.request_device(&Default::default()))
            .expect("Failed to create device");

        let cache = KVCache::new(&device, 22, 4, 64, 2048);

        // Test offset for position 0, head 0, dim 0
        let offset = cache.calculate_offset(0, 0, 0);
        assert_eq!(offset, 0, "First element should be at offset 0");

        // Test offset for position 0, head 1, dim 0 (should skip 64 dims)
        let offset = cache.calculate_offset(0, 1, 0);
        assert_eq!(offset, 64 * 4, "Second head should be 64 F32s after first");

        // Test offset for position 1, head 0, dim 0 (should skip all heads)
        let offset = cache.calculate_offset(1, 0, 0);
        assert_eq!(
            offset,
            4 * 64 * 4,
            "Second position should be 4 heads * 64 dims * 4 bytes"
        );
    }

    /// Test cache overflow detection
    #[test]
    #[should_panic(expected = "KV cache overflow")]
    fn test_overflow_panic() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("No GPU adapter");
        let (device, _queue) = pollster::block_on(adapter.request_device(&Default::default()))
            .expect("Failed to create device");

        let mut cache = KVCache::new(&device, 22, 4, 64, 8); // Small context for test

        // Increment beyond max_seq_len
        for _ in 0..9 {
            cache.increment().unwrap(); // Should panic at 9th call
        }
    }
}
