use super::metadata::BindlessMetadata;
use super::preflight::PreflightResources;
use crate::core::spec::ModelSpec;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use wgpu::util::DeviceExt; // Assuming ModelSpec is here or accessible

/// A GPU-resident GGUF model.
/// The entire file content is loaded into a single read-only storage buffer (`ByteAddressBuffer` in HLSL terms).
///
/// This is the "Bindless" approach: instead of binding buffers for each tensor,
/// we bind the whole file once, and the shader reads by byte offset.
pub struct BindlessModel {
    /// The massive buffer containing the raw GGUF file bytes.
    /// Usage: STORAGE | COPY_DST
    pub gpu_buffer: wgpu::Buffer,

    /// Size in bytes (for boundary checking)
    pub size: u64,

    /// Parsed Metadata (tensor offsets)
    pub metadata: BindlessMetadata,

    /// Pre-fused resources (RoPE tables, Norm Banks)
    pub preflight: Option<PreflightResources>,
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

        println!("[BindlessLoader] Upload Complete.");

        Self {
            gpu_buffer,
            size,
            metadata,
            preflight,
        }
    }
}
