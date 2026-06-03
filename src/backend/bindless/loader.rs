use super::metadata::BindlessMetadata;
use super::preflight::PreflightResources;
use crate::core::spec::ModelSpec;
use std::fs::File;
use std::io::{Read, Seek};
use std::num::NonZeroU64;
use std::path::Path;
use wgpu::util::DeviceExt; // Assuming ModelSpec is here or accessible

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

        // Reset for reading data
        file.seek(std::io::SeekFrom::Start(0)).unwrap();

        // Read into host memory first (Simplicity > Speed for V0.1)
        // TODO: Use memory mapping or streaming for >4GB models
        let mut raw_data = Vec::with_capacity(size as usize);
        file.read_to_end(&mut raw_data)
            .expect("Failed to read GGUF content");

        println!("[BindlessLoader] Uploading to VRAM...");
        let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GGUF Bindless Storage"),
            contents: &raw_data,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // JIT FUSION: Extract resources BEFORE we drop raw_data
        let preflight = if let Some(spec) = spec {
            println!("[BindlessLoader] Launching Preflight Fusion...");
            Some(PreflightResources::new_from_ram(
                device, &raw_data, &metadata, spec,
            ))
        } else {
            println!("[BindlessLoader] No Spec provided, skipping Preflight (Raw Mode).");
            None
        };

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
