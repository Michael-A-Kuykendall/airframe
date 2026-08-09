//! Raw tensor byte views — bounded access to tensor data in a GGUF file.

use crate::raw_gguf::error::{RawGgufError, RawGgufResult};
use crate::raw_gguf::model_identity::TensorDescriptor;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// A bounded view into tensor data within a GGUF file.
///
/// This provides read-only access to a specific tensor's raw bytes
/// without loading the entire file into memory.
/// The view is bounded by the tensor's storage_upper_bound (not payload size).
#[derive(Debug, Clone)]
pub struct TensorView {
    descriptor: TensorDescriptor,
    file_path: std::path::PathBuf,
}

impl TensorView {
    /// Create a new tensor view.
    pub fn new(descriptor: TensorDescriptor, file_path: impl AsRef<Path>) -> Self {
        Self {
            descriptor,
            file_path: file_path.as_ref().to_path_buf(),
        }
    }

    /// Get the tensor descriptor.
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Get the tensor name.
    pub fn name(&self) -> &str {
        &self.descriptor.name
    }

    /// Get the tensor GGML type.
    pub fn ggml_type(&self) -> u32 {
        self.descriptor.ggml_type
    }

    /// Get the tensor shape.
    pub fn shape(&self) -> &[u64] {
        &self.descriptor.shape
    }

    /// Get the tensor storage upper bound in bytes.
    pub fn storage_upper_bound(&self) -> u64 {
        self.descriptor.storage_upper_bound
    }

    /// Get the tensor offset in the file.
    pub fn offset(&self) -> u64 {
        self.descriptor.offset
    }

    /// Get the available storage span in bytes (upper_bound - offset).
    pub fn storage_span(&self) -> u64 {
        self.descriptor
            .storage_upper_bound
            .saturating_sub(self.descriptor.offset)
    }

    /// Read the entire tensor storage span into a vector.
    ///
    /// This reads from offset to storage_upper_bound (not payload size).
    pub fn read_all(&self) -> RawGgufResult<Vec<u8>> {
        let span = self.storage_span() as usize;
        let mut file = File::open(&self.file_path)?;
        file.seek(SeekFrom::Start(self.descriptor.offset))?;
        let mut buffer = vec![0u8; span];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Read a portion of the tensor storage span.
    ///
    /// `offset` is relative to the tensor start (not the file).
    /// `length` is the number of bytes to read.
    pub fn read_range(&self, offset: u64, length: u64) -> RawGgufResult<Vec<u8>> {
        let span = self.storage_span();
        if offset + length > span {
            return Err(RawGgufError::TensorOffsetOutOfBounds(
                self.descriptor.offset + offset,
                length,
                self.descriptor.offset + span,
            ));
        }

        let mut file = File::open(&self.file_path)?;
        file.seek(SeekFrom::Start(self.descriptor.offset + offset))?;
        let mut buffer = vec![0u8; length as usize];
        file.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    /// Read the tensor storage span as a slice (requires the file to be mmapped externally).
    ///
    /// This is a zero-copy view that requires the caller to provide the
    /// memory-mapped file data.
    pub fn as_bytes<'a>(&self, mmap: &'a [u8]) -> RawGgufResult<&'a [u8]> {
        let start = self.descriptor.offset as usize;
        let end = self.descriptor.storage_upper_bound as usize;
        if end > mmap.len() {
            return Err(RawGgufError::TensorPastEndOfFile(
                self.descriptor.offset,
                self.storage_span(),
                mmap.len() as u64,
            ));
        }
        Ok(&mmap[start..end])
    }
}

/// A collection of tensor views for a model.
#[derive(Debug, Clone)]
pub struct TensorViewCollection {
    views: Vec<TensorView>,
    file_path: std::path::PathBuf,
}

impl TensorViewCollection {
    /// Create a new tensor view collection.
    pub fn new(
        descriptors: Vec<crate::raw_gguf::model_identity::TensorDescriptor>,
        file_path: impl AsRef<Path>,
    ) -> Self {
        let file_path = file_path.as_ref().to_path_buf();
        let views = descriptors
            .into_iter()
            .map(|d| TensorView::new(d, &file_path))
            .collect();
        Self { views, file_path }
    }

    /// Get a tensor view by name.
    pub fn get(&self, name: &str) -> Option<&TensorView> {
        self.views.iter().find(|v| v.name() == name)
    }

    /// Get all tensor views.
    pub fn all(&self) -> &[TensorView] {
        &self.views
    }

    /// Get tensor views matching a prefix.
    pub fn with_prefix(&self, prefix: &str) -> Vec<&TensorView> {
        self.views
            .iter()
            .filter(|v| v.name().starts_with(prefix))
            .collect()
    }

    /// Get tensor views for a specific layer (blk.N.*).
    pub fn for_layer(&self, layer: usize) -> Vec<&TensorView> {
        let prefix = format!("blk.{}.", layer);
        self.with_prefix(&prefix)
    }

    /// Get the file path.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }
}
