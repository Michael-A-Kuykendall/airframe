//! Dense tensor storage for FP32 inference data.
//!
//! Provides a minimal `Tensor` type used throughout libshimmy for
//! activations, weights, and intermediate computations.

use crate::core::error::{LibshimmyError, Result};

/// Dense FP32 tensor with shape metadata.
#[derive(Debug, Clone)]
#[must_use]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "new_tensor".to_string(),
                expected: vec![expected_len],
                got: vec![data.len()],
            });
        }
        Ok(Self { data, shape })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![1.0; len],
            shape,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get element at flat index
    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }

    /// Get element at multi-dimensional index
    pub fn get_nd(&self, indices: &[usize]) -> Result<f32> {
        if indices.len() != self.shape.len() {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "index".to_string(),
                expected: vec![self.shape.len()],
                got: vec![indices.len()],
            });
        }

        let flat_index = self.nd_to_flat_index(indices)?;
        self.data.get(flat_index).copied().ok_or_else(|| {
            LibshimmyError::Unsupported(format!("Index {} out of bounds", flat_index))
        })
    }

    /// Convert multi-dimensional index to flat index
    fn nd_to_flat_index(&self, indices: &[usize]) -> Result<usize> {
        let mut flat_index = 0;
        let mut stride = 1;

        for (i, &idx) in indices.iter().enumerate().rev() {
            if idx >= self.shape[i] {
                return Err(LibshimmyError::Unsupported(format!(
                    "Index {} >= shape {} at dimension {}",
                    idx, self.shape[i], i
                )));
            }
            flat_index += idx * stride;
            stride *= self.shape[i];
        }

        Ok(flat_index)
    }

    /// Validate shape compatibility for operations
    pub fn validate_shape_eq(&self, other: &Tensor) -> Result<()> {
        if self.shape != other.shape {
            return Err(LibshimmyError::ShapeMismatch {
                tensor: "shape_validation".to_string(),
                expected: self.shape.clone(),
                got: other.shape.clone(),
            });
        }
        Ok(())
    }
}

/// Read-only view into a tensor.
#[derive(Debug)]
#[must_use]
pub struct TensorView<'a> {
    pub data: &'a [f32],
    pub shape: &'a [usize],
}

impl<'a> TensorView<'a> {
    pub fn new(tensor: &'a Tensor) -> Self {
        Self {
            data: &tensor.data,
            shape: &tensor.shape,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get element at flat index
    pub fn get(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_valid() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, shape).unwrap();

        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_creation_invalid_shape() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![2, 2]; // Expects 4 elements, got 3

        let result = Tensor::new(data, shape);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_zeros_and_ones() {
        let zeros = Tensor::zeros(vec![2, 3]);
        assert_eq!(zeros.len(), 6);
        assert_eq!(zeros.data, vec![0.0; 6]);

        let ones = Tensor::ones(vec![2, 3]);
        assert_eq!(ones.len(), 6);
        assert_eq!(ones.data, vec![1.0; 6]);
    }

    #[test]
    fn test_tensor_indexing_flat() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        assert_eq!(tensor.get(0), Some(1.0));
        assert_eq!(tensor.get(1), Some(2.0));
        assert_eq!(tensor.get(3), Some(4.0));
        assert_eq!(tensor.get(4), None); // Out of bounds
    }

    #[test]
    fn test_tensor_indexing_nd() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        assert_eq!(tensor.get_nd(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get_nd(&[0, 1]).unwrap(), 2.0);
        assert_eq!(tensor.get_nd(&[1, 0]).unwrap(), 3.0);
        assert_eq!(tensor.get_nd(&[1, 1]).unwrap(), 4.0);

        // Invalid indices
        assert!(tensor.get_nd(&[2, 0]).is_err()); // Out of bounds
        assert!(tensor.get_nd(&[0]).is_err()); // Wrong number of dimensions
    }

    #[test]
    fn test_tensor_3d_indexing() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::new(data, vec![2, 3, 4]).unwrap();

        assert_eq!(tensor.get_nd(&[0, 0, 0]).unwrap(), 0.0);
        assert_eq!(tensor.get_nd(&[0, 1, 2]).unwrap(), 6.0); // 0*12 + 1*4 + 2 = 6
        assert_eq!(tensor.get_nd(&[1, 2, 3]).unwrap(), 23.0); // 1*12 + 2*4 + 3 = 23
    }

    #[test]
    fn test_shape_validation() {
        let tensor1 = Tensor::zeros(vec![2, 3]);
        let tensor2 = Tensor::zeros(vec![2, 3]);
        let tensor3 = Tensor::zeros(vec![3, 2]);

        assert!(tensor1.validate_shape_eq(&tensor2).is_ok());
        assert!(tensor1.validate_shape_eq(&tensor3).is_err());
    }

    #[test]
    fn test_tensor_view() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view = TensorView::new(&tensor);

        assert_eq!(view.len(), 4);
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.get(0), Some(1.0));
        assert_eq!(view.get(3), Some(4.0));
        assert_eq!(view.shape, &[2, 2]);
    }
}
