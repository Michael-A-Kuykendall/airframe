use super::metadata::BindlessMetadata;
use super::preflight::PreflightResources;
use crate::core::spec::ModelSpec;
use memmap2::Mmap;
use std::fs::File;
use std::num::NonZeroU64;
use std::path::Path;
use wgpu::util::DeviceExt;

/// Split point for the 3-way GGUF buffer binding.
/// 2,000,000,000 bytes = ~1.86 GiB — safely below the 2,147,483,647 binding limit,
/// and 256-byte aligned (required for sub-range offsets).
pub const BLOB_CHUNK_BYTES: u64 = 2_000_000_000;

/// A GPU-resident GGUF model.
/// The entire file content is loaded into a single read-only storage buffer.
///
/// For models > 2 GB the buffer is exposed to shaders through three sub-range
/// bindings (blob_0 / blob_1 / blob_2) so that each individual binding stays
/// within `max_storage_buffer_binding_size`.
pub struct BindlessModel {
    /// The massive buffer containing the raw GGUF file bytes.
    /// Usage: STORAGE | COPY_DST
    pub gpu_buffer: wgpu::Buffer,

    /// Size in bytes (for boundary checking)
    pub size: u64,

    /// A minimal 4-byte dummy STORAGE buffer used to fill blob_1 / blob_2
    /// bindings for models whose data fits within a single 2 GB chunk.
    pub dummy_buf: wgpu::Buffer,

    /// Parsed Metadata (tensor offsets)
    pub metadata: BindlessMetadata,

    /// Pre-fused resources (RoPE tables, Norm Banks)
    pub preflight: Option<PreflightResources>,
}

impl BindlessModel {
    // ------------------------------------------------------------------
    // Sub-range binding helpers
    // Each binding covers at most BLOB_CHUNK_BYTES bytes of gpu_buffer.
    // ------------------------------------------------------------------

    /// Binding resource for blob_0: bytes [0, min(CHUNK, size)).
    pub fn blob_binding_0(&self) -> wgpu::BindingResource<'_> {
        let sz = self.size.min(BLOB_CHUNK_BYTES);
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.gpu_buffer,
            offset: 0,
            size: Some(NonZeroU64::new(sz).unwrap()),
        })
    }

    /// Binding resource for blob_1: bytes [CHUNK, min(2·CHUNK, size)).
    /// Falls back to the 4-byte dummy if the model fits in blob_0.
    pub fn blob_binding_1(&self) -> wgpu::BindingResource<'_> {
        if self.size <= BLOB_CHUNK_BYTES {
            return self.dummy_buf.as_entire_binding();
        }
        let offset = BLOB_CHUNK_BYTES;
        let sz = (self.size - BLOB_CHUNK_BYTES).min(BLOB_CHUNK_BYTES);
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.gpu_buffer,
            offset,
            size: Some(NonZeroU64::new(sz).unwrap()),
        })
    }

    /// Binding resource for blob_2: bytes [2·CHUNK, size).
    /// Falls back to the 4-byte dummy for models < 4 GB.
    pub fn blob_binding_2(&self) -> wgpu::BindingResource<'_> {
        if self.size <= 2 * BLOB_CHUNK_BYTES {
            return self.dummy_buf.as_entire_binding();
        }
        let offset = 2 * BLOB_CHUNK_BYTES;
        let sz = self.size - 2 * BLOB_CHUNK_BYTES;
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.gpu_buffer,
            offset,
            size: Some(NonZeroU64::new(sz).unwrap()),
        })
    }
}

impl BindlessModel {
    /// Loads a GGUF file from disk and uploads it to VRAM.
    /// Also launches Preflight extraction (Norm fusion, RoPE tables).
    ///
    /// # Arguments
    /// * `device` - WGPU Device
    /// * `path` - Path to the .gguf file
    ///
    /// # Panics
    /// Panics if file IO fails or VRAM allocation fails.
    pub fn load_from_disk(device: &wgpu::Device, path: &Path, spec: Option<&ModelSpec>) -> Self {
        println!("[BindlessLoader] Opening GGUF: {:?}", path);

        let mut file = File::open(path).expect("Failed to open GGUF file");
        let metadata_fs = file.metadata().expect("Failed to read metadata");
        let size = metadata_fs.len();

        println!(
            "[BindlessLoader] File size: {} bytes ({:.2} MB)",
            size,
            size as f64 / 1024.0 / 1024.0
        );

        // Scan Metadata
        println!("[BindlessLoader] Scanning Metadata...");
        let metadata = BindlessMetadata::new(&mut file);
        println!(
            "[BindlessLoader] Found {} tensors. Data starts at {}.",
            metadata.tensor_count, metadata.data_start_offset
        );

        // Memory-map the file for zero-copy GPU upload
        // This avoids the ~60-240s blocking read_to_end() + RAM copy
        // OS pages data on-demand as GPU reads, no intermediate RAM copy
        println!("[BindlessLoader] Memory-mapping GGUF file...");
        let mmap = unsafe { Mmap::map(&file).expect("Failed to mmap GGUF file") };

        // Create buffer with mapped_at_creation, then copy from mmap
        // This is faster than read_to_end because mmap doesn't allocate
        // the full file in RAM - OS pages it on demand
        println!("[BindlessLoader] Creating GPU buffer from mmap...");
        let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GGUF Bindless Storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        // Copy from mmap to GPU buffer (fast, OS handles page faults)
        gpu_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(&mmap[..]);
        gpu_buffer.unmap();

        println!("[BindlessLoader] GPU buffer created from mmap (non-blocking)...");

        // JIT FUSION: Extract resources from mmap while GPU uploads
        // PreflightResources::new_from_ram accepts &[u8] so works with mmap
        let preflight = if let Some(spec) = spec {
            println!("[BindlessLoader] Launching Preflight Fusion (from mmap)...");
            Some(PreflightResources::new_from_ram(
                device,
                &mmap[..],
                &metadata,
                spec,
            ))
        } else {
            println!("[BindlessLoader] No Spec provided, skipping Preflight (Raw Mode).");
            None
        };

        // Explicitly drop mmap here to prove Preflight copied what it needed
        // In practice, Preflight completed while staging copy happened
        drop(mmap);

        let dummy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GGUF Dummy Blob"),
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::STORAGE,
        });

        println!("[BindlessLoader] Upload Complete.");

        Self {
            gpu_buffer,
            size,
            dummy_buf,
            metadata,
            preflight,
        }
    }
}
