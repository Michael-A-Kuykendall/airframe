//! Key-value cache for transformer attention.
//!
//! Stores K/V tensors across sequence positions to avoid recomputation
//! during autoregressive generation.

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

/// Cheap immutable metadata view of KV state.
///
/// `len` must mirror `KvCache::current_len`.
/// `version` is monotonic and currently equals `len`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KvSnapshot {
    pub len: usize,
    pub version: usize,
}

impl KvSnapshot {
    pub fn empty() -> Self {
        Self { len: 0, version: 0 }
    }
}

/// KV cache for multi-layer attention.
///
/// Layout per layer: `[max_seq_len, n_head_kv, head_dim]`
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Key cache for each layer: Vec<Tensor> where each tensor is [max_seq_len, n_head_kv, head_dim]
    pub key_cache: Vec<Tensor>,
    /// Value cache for each layer: Vec<Tensor> where each tensor is [max_seq_len, n_head_kv, head_dim]
    pub value_cache: Vec<Tensor>,
    /// Current sequence length (number of positions filled)
    pub current_len: usize,
    /// Maximum sequence length (allocated capacity)
    pub max_seq_len: usize,
    /// Number of layers
    pub n_layer: usize,
    /// Number of key/value heads per layer
    pub n_head_kv: usize,
    /// Dimension of each head
    pub head_dim: usize,

    /// Cheap metadata snapshot of KV state.
    pub snapshot: KvSnapshot,
}

impl KvCache {
    /// Create new KV cache with specified dimensions
    pub fn new(max_seq_len: usize, n_layer: usize, n_head_kv: usize, head_dim: usize) -> Self {
        let mut key_cache = Vec::with_capacity(n_layer);
        let mut value_cache = Vec::with_capacity(n_layer);

        // Allocate cache for each layer
        for _ in 0..n_layer {
            let k_tensor = Tensor::zeros(vec![max_seq_len, n_head_kv, head_dim]);
            let v_tensor = Tensor::zeros(vec![max_seq_len, n_head_kv, head_dim]);
            key_cache.push(k_tensor);
            value_cache.push(v_tensor);
        }

        Self {
            key_cache,
            value_cache,
            current_len: 0,
            max_seq_len,
            n_layer,
            n_head_kv,
            head_dim,
            snapshot: KvSnapshot::empty(),
        }
    }

    /// Reset cache to empty state
    pub fn reset(&mut self) {
        self.current_len = 0;
        self.snapshot = KvSnapshot::empty();
        // Note: We don't need to zero the data, just reset the length
    }

    /// Get current snapshot metadata.
    pub fn snapshot(&self) -> KvSnapshot {
        self.snapshot
    }

    /// Get current sequence length
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.current_len >= self.max_seq_len
    }

    /// Get remaining capacity
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len.saturating_sub(self.current_len)
    }

    /// Prefill cache with multiple tokens for a specific layer
    /// k_states, v_states: [seq_len, n_head_kv, head_dim]
    pub fn prefill_layer(
        &mut self,
        layer: usize,
        k_states: &Tensor,
        v_states: &Tensor,
    ) -> Result<()> {
        self.validate_layer(layer)?;
        self.validate_kv_shapes(k_states, v_states)?;

        let seq_len = k_states.shape[0];

        // Check capacity
        if self.current_len + seq_len > self.max_seq_len {
            return Err(LibshimmyError::Unsupported(format!(
                "Cannot prefill {} tokens: would exceed max_seq_len {} (current: {})",
                seq_len, self.max_seq_len, self.current_len
            )));
        }

        // Copy data into cache
        self.copy_to_cache(layer, k_states, v_states, self.current_len, seq_len)?;

        Ok(())
    }

    /// Complete prefill operation (updates current_len after all layers processed)
    pub fn complete_prefill(&mut self, seq_len: usize) -> Result<()> {
        if self.current_len + seq_len > self.max_seq_len {
            return Err(LibshimmyError::Unsupported(format!(
                "Prefill would exceed capacity: {} + {} > {}",
                self.current_len, seq_len, self.max_seq_len
            )));
        }

        self.current_len += seq_len;
        self.snapshot = KvSnapshot {
            len: self.current_len,
            version: self.current_len,
        };
        Ok(())
    }

    /// Append single token for a specific layer (decode step)
    /// k_state, v_state: [1, n_head_kv, head_dim]
    pub fn append_layer(&mut self, layer: usize, k_state: &Tensor, v_state: &Tensor) -> Result<()> {
        self.validate_layer(layer)?;
        self.validate_kv_shapes(k_state, v_state)?;

        if k_state.shape[0] != 1 {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "decode_k_state".to_string(),
                expected: vec![1, self.n_head_kv, self.head_dim],
                got: k_state.shape.clone(),
            });
        }

        if self.is_full() {
            return Err(LibshimmyError::Unsupported(
                "KV cache is full, cannot append more tokens".to_string(),
            ));
        }

        // Copy single token
        self.copy_to_cache(layer, k_state, v_state, self.current_len, 1)?;

        Ok(())
    }

    /// Complete decode operation (updates current_len after all layers processed)
    pub fn complete_decode(&mut self) -> Result<()> {
        if self.is_full() {
            return Err(LibshimmyError::Unsupported(
                "Cannot complete decode: cache is full".to_string(),
            ));
        }

        self.current_len += 1;
        self.snapshot = KvSnapshot {
            len: self.current_len,
            version: self.current_len,
        };
        Ok(())
    }

    /// Get cached K, V for a layer up to current length
    /// Returns: (k_cache, v_cache) each [current_len, n_head_kv, head_dim]
    pub fn get_layer_cache(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        self.validate_layer(layer)?;

        if self.current_len == 0 {
            // Return empty tensors
            let empty_k = Tensor::zeros(vec![0, self.n_head_kv, self.head_dim]);
            let empty_v = Tensor::zeros(vec![0, self.n_head_kv, self.head_dim]);
            return Ok((empty_k, empty_v));
        }

        // Extract current portion of cache
        let k_data = self.extract_cache_data(&self.key_cache[layer], self.current_len)?;
        let v_data = self.extract_cache_data(&self.value_cache[layer], self.current_len)?;

        let k_tensor = Tensor::new(
            k_data,
            vec![self.current_len, self.n_head_kv, self.head_dim],
        )?;
        let v_tensor = Tensor::new(
            v_data,
            vec![self.current_len, self.n_head_kv, self.head_dim],
        )?;

        Ok((k_tensor, v_tensor))
    }

    /// Truncate cache to specified length (for testing/debugging)
    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.current_len {
            return Err(LibshimmyError::Unsupported(format!(
                "Cannot truncate to {} > current length {}",
                new_len, self.current_len
            )));
        }

        self.current_len = new_len;
        Ok(())
    }

    // Helper methods

    fn validate_layer(&self, layer: usize) -> Result<()> {
        if layer >= self.n_layer {
            return Err(LibshimmyError::Unsupported(format!(
                "Layer {} >= n_layer {}",
                layer, self.n_layer
            )));
        }
        Ok(())
    }

    fn validate_kv_shapes(&self, k_states: &Tensor, v_states: &Tensor) -> Result<()> {
        if k_states.shape != v_states.shape {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "kv_shapes".to_string(),
                expected: k_states.shape.clone(),
                got: v_states.shape.clone(),
            });
        }

        if k_states.ndim() != 3 {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "kv_ndim".to_string(),
                expected: vec![3],
                got: vec![k_states.ndim()],
            });
        }

        let expected_shape_suffix = vec![self.n_head_kv, self.head_dim];
        if k_states.shape[1..] != expected_shape_suffix {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "kv_head_dims".to_string(),
                expected: expected_shape_suffix,
                got: k_states.shape[1..].to_vec(),
            });
        }

        Ok(())
    }

    fn copy_to_cache(
        &mut self,
        layer: usize,
        k_states: &Tensor,
        v_states: &Tensor,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<()> {
        let head_size = self.n_head_kv * self.head_dim;

        for seq_idx in 0..seq_len {
            let cache_pos = start_pos + seq_idx;
            let src_offset = seq_idx * head_size;
            let dst_offset = cache_pos * head_size;

            // Copy K
            for i in 0..head_size {
                self.key_cache[layer].data[dst_offset + i] = k_states.data[src_offset + i];
            }

            // Copy V
            for i in 0..head_size {
                self.value_cache[layer].data[dst_offset + i] = v_states.data[src_offset + i];
            }
        }

        Ok(())
    }

    fn extract_cache_data(&self, cache_tensor: &Tensor, len: usize) -> Result<Vec<f32>> {
        let head_size = self.n_head_kv * self.head_dim;
        let total_size = len * head_size;

        if total_size > cache_tensor.data.len() {
            return Err(LibshimmyError::Unsupported(format!(
                "Extract size {} > cache size {}",
                total_size,
                cache_tensor.data.len()
            )));
        }

        Ok(cache_tensor.data[..total_size].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kvcache_creation() {
        let cache = KvCache::new(10, 2, 4, 64);

        assert_eq!(cache.max_seq_len, 10);
        assert_eq!(cache.n_layer, 2);
        assert_eq!(cache.n_head_kv, 4);
        assert_eq!(cache.head_dim, 64);
        assert_eq!(cache.current_len, 0);
        assert!(cache.is_empty());
        assert!(!cache.is_full());
        assert_eq!(cache.remaining_capacity(), 10);
    }

    #[test]
    fn test_kvcache_monotonic_length() {
        let mut cache = KvCache::new(5, 1, 2, 4);

        // Length should only increase
        assert_eq!(cache.len(), 0);

        let k_state = Tensor::zeros(vec![1, 2, 4]);
        let v_state = Tensor::zeros(vec![1, 2, 4]);

        cache.append_layer(0, &k_state, &v_state).unwrap();
        cache.complete_decode().unwrap();
        assert_eq!(cache.len(), 1);

        cache.append_layer(0, &k_state, &v_state).unwrap();
        cache.complete_decode().unwrap();
        assert_eq!(cache.len(), 2);

        // Reset should set to 0
        cache.reset();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kvcache_prefill_vs_decode_equivalence() {
        let max_len = 5;
        let n_layer = 1;
        let n_head_kv = 2;
        let head_dim = 4;

        // Create test data based on actual dimensions
        let seq_len = 2;
        let total_k_size = seq_len * n_head_kv * head_dim;
        let total_v_size = seq_len * n_head_kv * head_dim;

        let k_data: Vec<f32> = (0..total_k_size).map(|i| i as f32).collect();
        let v_data: Vec<f32> = (total_k_size..total_k_size + total_v_size)
            .map(|i| i as f32)
            .collect();

        let k_prefill = Tensor::new(k_data.clone(), vec![seq_len, n_head_kv, head_dim]).unwrap();
        let v_prefill = Tensor::new(v_data.clone(), vec![seq_len, n_head_kv, head_dim]).unwrap();

        let token_size = n_head_kv * head_dim;
        let k_token1 =
            Tensor::new(k_data[..token_size].to_vec(), vec![1, n_head_kv, head_dim]).unwrap();
        let v_token1 =
            Tensor::new(v_data[..token_size].to_vec(), vec![1, n_head_kv, head_dim]).unwrap();
        let k_token2 =
            Tensor::new(k_data[token_size..].to_vec(), vec![1, n_head_kv, head_dim]).unwrap();
        let v_token2 =
            Tensor::new(v_data[token_size..].to_vec(), vec![1, n_head_kv, head_dim]).unwrap();

        // Method 1: Prefill 2 tokens at once
        let mut cache1 = KvCache::new(max_len, n_layer, n_head_kv, head_dim);
        cache1.prefill_layer(0, &k_prefill, &v_prefill).unwrap();
        cache1.complete_prefill(2).unwrap();

        // Method 2: Decode token by token
        let mut cache2 = KvCache::new(max_len, n_layer, n_head_kv, head_dim);
        cache2.append_layer(0, &k_token1, &v_token1).unwrap();
        cache2.complete_decode().unwrap();
        cache2.append_layer(0, &k_token2, &v_token2).unwrap();
        cache2.complete_decode().unwrap();

        // Both should have same length
        assert_eq!(cache1.len(), cache2.len());
        assert_eq!(cache1.len(), 2);

        // Both should have same cached content
        let (k1, v1) = cache1.get_layer_cache(0).unwrap();
        let (k2, v2) = cache2.get_layer_cache(0).unwrap();

        assert_eq!(k1.data, k2.data);
        assert_eq!(v1.data, v2.data);
    }

    #[test]
    fn test_kvcache_capacity_limits() {
        let mut cache = KvCache::new(2, 1, 1, 2);

        let k_state = Tensor::zeros(vec![1, 1, 2]);
        let v_state = Tensor::zeros(vec![1, 1, 2]);

        // Fill to capacity
        cache.append_layer(0, &k_state, &v_state).unwrap();
        cache.complete_decode().unwrap();
        cache.append_layer(0, &k_state, &v_state).unwrap();
        cache.complete_decode().unwrap();

        assert!(cache.is_full());
        assert_eq!(cache.remaining_capacity(), 0);

        // Should fail to add more
        let result = cache.append_layer(0, &k_state, &v_state);
        assert!(result.is_err());
    }

    #[test]
    fn test_kvcache_validation() {
        let mut cache = KvCache::new(5, 2, 2, 4);

        // Wrong layer
        let k_state = Tensor::zeros(vec![1, 2, 4]);
        let v_state = Tensor::zeros(vec![1, 2, 4]);
        assert!(cache.append_layer(2, &k_state, &v_state).is_err());

        // Wrong shape
        let k_wrong = Tensor::zeros(vec![1, 3, 4]); // Wrong n_head_kv
        assert!(cache.append_layer(0, &k_wrong, &v_state).is_err());

        // Mismatched K/V shapes
        let v_wrong = Tensor::zeros(vec![1, 2, 3]); // Wrong head_dim
        assert!(cache.append_layer(0, &k_state, &v_wrong).is_err());
    }
}
