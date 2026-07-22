use super::metadata::BindlessMetadata;
use super::preflight::PreflightResources;
use crate::core::spec::ModelSpec;
use crate::invariant_ppt::airframe_invariants::*;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use wgpu::util::DeviceExt;

/// Hard ceiling for a single blob buffer. `effective_chunk` is always
/// `min(adapter.max_storage_buffer_binding_size, BLOB_CHUNK_BYTES)`, then
/// 256-byte aligned. 2,000,000,000 bytes = ~1.86 GiB — safely below the wgpu
/// 2 GB storage-buffer binding limit, and 256-byte aligned.
pub const BLOB_CHUNK_BYTES: u64 = 2_000_000_000;

/// The multi-buffer plan for a loaded model.
///
/// Produced by [`compute_chunk_plan`] (which embeds the PPT invariants) so a
/// malformed plan can never be constructed silently.
#[derive(Debug, Clone, Copy)]
pub struct ChunkPlan {
    /// Size of each blob buffer in bytes (256-aligned, ≤ binding limit).
    pub effective_chunk: u64,
    /// Number of independent blob buffers the model is split into.
    pub num_chunks: usize,
}

impl ChunkPlan {
    /// Resolves an absolute word index to `(buffer_index, word_offset_in_buffer)`
    /// under this plan. Embeds the word-range invariant so an out-of-range index
    /// trips the gate instead of reading garbage from the wrong buffer.
    pub fn buffer_for_word(&self, word_idx: u32) -> (usize, u32) {
        let chunk_words = (self.effective_chunk / 4) as u32;
        let total_words = (self.effective_chunk * self.num_chunks as u64 / 4) as u32;
        assert_word_index_in_range(word_idx, total_words, "loader::ChunkPlan::buffer_for_word");
        ((word_idx / chunk_words) as usize, word_idx % chunk_words)
    }
}

/// Computes the multi-buffer chunk plan for a file of `file_size` bytes given the
/// adapter's storage-buffer binding limit.
///
/// Embeds the PPT invariants directly:
/// - `effective_chunk` is capped at [`BLOB_CHUNK_BYTES`], floored to the adapter's
///   real limit, then 256-byte aligned (asserted).
/// - `effective_chunk` must not exceed the wgpu 2 GB binding limit (asserted).
/// - `num_chunks` is `ceil(file_size / effective_chunk)` and is capped at
///   `MAX_CHUNKS`; models needing more buffers are **rejected here**, not silently
///   split.
pub fn compute_chunk_plan(file_size: u64, adapter_limit: u64) -> ChunkPlan {
    let cap = BLOB_CHUNK_BYTES.min(adapter_limit);
    let effective_chunk = (cap / REQUIRED_ALIGNMENT) * REQUIRED_ALIGNMENT;
    assert_alignment(effective_chunk, "loader::compute_chunk_plan");
    assert_buffer_within_limit(effective_chunk, "loader::compute_chunk_plan");

    let num_chunks = ((file_size + effective_chunk - 1) / effective_chunk) as usize;
    assert_chunk_count_within_limit(num_chunks, "loader::compute_chunk_plan");

    ChunkPlan {
        effective_chunk,
        num_chunks,
    }
}

/// A GPU-resident GGUF model, stored as N independent read-only storage
/// buffers (one per 2 GB-ish chunk of the file).
///
/// Each `gpu_buffers[i]` holds `effective_chunk` bytes of the raw GGUF file
/// (the final buffer may be smaller). Shaders read tensor words through
/// `blob_binding_*` which maps directly onto `gpu_buffers`, so the WGSL
/// `read_blob` chunk-splitting logic is unchanged.
pub struct BindlessModel {
    /// The model split into N independent storage buffers.
    /// Usage: STORAGE | COPY_DST | COPY_SRC
    pub gpu_buffers: Vec<wgpu::Buffer>,

    /// Size in bytes (for boundary checking)
    pub size: u64,

    /// Size of each blob buffer in bytes (256-aligned, ≤ binding limit).
    pub effective_chunk: u64,

    /// A minimal 4-byte dummy STORAGE buffer used to pad unused blob bindings
    /// in the fixed bind-group layouts (e.g. a 2-chunk model still exposes a
    /// blob_2 slot so the layout shape is constant across models).
    pub dummy_buf: wgpu::Buffer,

    /// Parsed Metadata (tensor offsets)
    pub metadata: BindlessMetadata,

    /// Pre-fused resources (RoPE tables, Norm Banks)
    pub preflight: Option<PreflightResources>,
}

impl BindlessModel {
    // ------------------------------------------------------------------
    // Sub-range binding helpers
    // Each binding covers exactly one of the N blob buffers.
    // ------------------------------------------------------------------

    /// Binding resource for blob_0: bytes [0, min(effective_chunk, size)).
    pub fn blob_binding_0(&self) -> wgpu::BindingResource<'_> {
        if self.gpu_buffers.len() > 0 {
            self.gpu_buffers[0].as_entire_binding()
        } else {
            self.dummy_buf.as_entire_binding()
        }
    }

    /// Binding resource for blob_1: bytes [effective_chunk, min(2·effective_chunk, size)).
    /// Falls back to the 4-byte dummy if the model fits in a single chunk.
    pub fn blob_binding_1(&self) -> wgpu::BindingResource<'_> {
        if self.gpu_buffers.len() > 1 {
            self.gpu_buffers[1].as_entire_binding()
        } else {
            self.dummy_buf.as_entire_binding()
        }
    }

    /// Binding resource for blob_2: bytes [2·effective_chunk, size).
    /// Falls back to the 4-byte dummy for models < 2 chunks.
    pub fn blob_binding_2(&self) -> wgpu::BindingResource<'_> {
        if self.gpu_buffers.len() > 2 {
            self.gpu_buffers[2].as_entire_binding()
        } else {
            self.dummy_buf.as_entire_binding()
        }
    }

    /// Resolves an absolute word index to `(buffer_index, word_offset_in_buffer)`
    /// under the loaded multi-buffer plan.
    pub fn buffer_for_word(&self, word_idx: u32) -> (usize, u32) {
        let chunk_words = (self.effective_chunk / 4) as u32;
        let buffer_index = (word_idx / chunk_words) as usize;
        assert!(
            buffer_index < self.gpu_buffers.len(),
            "word index {} maps beyond available buffers ({})",
            word_idx,
            self.gpu_buffers.len()
        );
        (buffer_index, word_idx % chunk_words)
    }
}

impl BindlessModel {
    /// Loads a GGUF file from disk and uploads it to VRAM as N independent
    /// blob buffers.
    ///
    /// The chunk plan is computed from the device's real storage-buffer binding
    /// limit (via [`compute_chunk_plan`], which embeds the PPT invariants), so
    /// the loader is robust to adapters whose binding limit differs from 2 GB.
    /// Also launches Preflight extraction (Norm fusion, RoPE tables).
    ///
    /// # Arguments
    /// * `device` - WGPU Device
    /// * `path` - Path to the .gguf file
    /// * `spec` - Optional model spec (enables Preflight fusion)
    ///
    /// # Panics
    /// Panics if file IO fails, VRAM allocation fails, or the model needs more
    /// than `MAX_CHUNKS` buffers (rejected at load, not silently split).
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

        // Compute the multi-buffer plan from the device's REAL binding limit.
        // compute_chunk_plan embeds the alignment / 2 GB / chunk-count invariants.
        let chunk_plan =
            compute_chunk_plan(size, device.limits().max_storage_buffer_binding_size as u64);
        let effective_chunk = chunk_plan.effective_chunk;
        let num_chunks = chunk_plan.num_chunks;
        println!(
            "[BindlessLoader] Multi-buffer plan: {} chunks of {} bytes (256-aligned)",
            num_chunks, effective_chunk
        );

        // Scan Metadata
        println!("[BindlessLoader] Scanning Metadata...");
        let metadata = BindlessMetadata::new(&mut file);
        println!(
            "[BindlessLoader] Found {} tensors. Data starts at {}.",
            metadata.tensor_count, metadata.data_start_offset
        );

        // Memory-map the file for zero-copy GPU upload
        // OS pages data on-demand as GPU reads, no intermediate RAM copy
        println!("[BindlessLoader] Memory-mapping GGUF file...");
        let mmap = unsafe { Mmap::map(&file).expect("Failed to mmap GGUF file") };

        // Create N independent blob buffers, each ≤ effective_chunk bytes.
        let mut gpu_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let offset = (i as u64) * effective_chunk;
            let chunk_size = (size - offset).min(effective_chunk);
            // Each blob buffer must respect the binding limit and wgpu's 4-byte
            // size minimum. (256-byte alignment of `effective_chunk` itself is
            // asserted in compute_chunk_plan; the final partial chunk only needs
            // to satisfy wgpu's 4-byte alignment requirement.)
            assert_buffer_within_limit(chunk_size, "loader::load_from_disk");
            assert!(
                chunk_size.is_multiple_of(4),
                "blob buffer {} size {} must be 4-byte aligned for wgpu",
                i,
                chunk_size
            );

            println!(
                "[BindlessLoader] Creating blob buffer {}: bytes [{}, {})",
                i,
                offset,
                offset + chunk_size
            );
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("GGUF Bindless Blob {}", i)),
                contents: &mmap[offset as usize..(offset + chunk_size) as usize],
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
            gpu_buffers.push(buf);
        }

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
        drop(mmap);

        let dummy_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GGUF Dummy Blob"),
            contents: &vec![0u8; 1048576],
            usage: wgpu::BufferUsages::STORAGE,
        });

        println!(
            "[BindlessLoader] Upload Complete ({} buffers).",
            gpu_buffers.len()
        );

        Self {
            gpu_buffers,
            size,
            effective_chunk,
            dummy_buf,
            metadata,
            preflight,
        }
    }
}
