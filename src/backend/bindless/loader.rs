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

/// Fixed number of blob binding slots in the shader layout (bindings 0..7).
/// This is the shader-layout width and does NOT limit resident chunk count.
pub const BLOB_BINDING_SLOTS: usize = 8;

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
/// - `num_chunks` is `ceil(file_size / effective_chunk)` and represents the
///   total **resident** chunk count. There is no cap here — the loader allocates
///   all resident chunks. The shader binding limit (`BLOB_BINDING_SLOTS`) is
///   enforced at dispatch time via [`BlobWindow`], not at load time.
pub fn compute_chunk_plan(file_size: u64, adapter_limit: u64) -> ChunkPlan {
    let cap = BLOB_CHUNK_BYTES.min(adapter_limit);
    let effective_chunk = (cap / REQUIRED_ALIGNMENT) * REQUIRED_ALIGNMENT;
    assert_alignment(effective_chunk, "loader::compute_chunk_plan");
    assert_buffer_within_limit(effective_chunk, "loader::compute_chunk_plan");

    let num_chunks = file_size.div_ceil(effective_chunk) as usize;

    ChunkPlan {
        effective_chunk,
        num_chunks,
    }
}

/// Returns `(block_elems, block_bytes)` for a GGML quant type.
///
/// Under `isf` this defers to `airframe_observe::quant_formula`, the GGUF spec
/// registry and single source of truth; otherwise it mirrors the same values so
/// the always-built loader stays consistent. Mirrors the arrangement used by
/// `pipeline::formula_index_for_ggml`.
#[cfg(feature = "isf")]
fn quant_block_geometry(type_id: u32) -> Option<(usize, usize)> {
    let elems = airframe_observe::quant_formula::block_elems(type_id)?;
    let bytes = airframe_observe::quant_formula::block_bytes(type_id)?;
    Some((elems, bytes))
}

#[cfg(not(feature = "isf"))]
fn quant_block_geometry(type_id: u32) -> Option<(usize, usize)> {
    match type_id {
        0 => Some((1, 4)),      // F32
        1 => Some((1, 2)),      // F16
        2 => Some((32, 18)),    // Q4_0
        6 => Some((32, 22)),    // Q5_0
        8 => Some((32, 34)),    // Q8_0
        12 => Some((256, 144)), // Q4_K
        13 => Some((256, 176)), // Q5_K
        14 => Some((256, 210)), // Q6_K
        _ => None,
    }
}

/// A sliding window over resident blob chunks for a single GPU dispatch.
///
/// The shader layout has a fixed `BLOB_BINDING_SLOTS` (8) blob bindings.
/// A model may have more resident chunks than binding slots. `BlobWindow`
/// selects a consecutive range of `slot_count` resident chunks (default 8)
/// starting at `start_chunk` and maps absolute word indices to window-local
/// `(slot_index, word_offset)` pairs. Out-of-window accesses are rejected
/// before dispatch.
///
/// The window is defined by:
/// - `start_chunk`: index of the first resident chunk bound to slot 0
/// - `slot_count`: number of slots to bind (≤ `BLOB_BINDING_SLOTS`, default 8)
/// - `chunk_words`: words per chunk (`effective_chunk / 4`)
/// - `total_resident_chunks`: total number of resident chunks in the model
///
/// For a dispatch covering tensors in resident chunks `start_chunk..start_chunk+slot_count`,
/// the host subtracts `start_chunk * chunk_words` from all absolute word indices
/// passed to the shader (via `blob_base_words` and tensor offsets), so the shader
/// sees a contiguous window starting at word 0.
#[derive(Debug, Clone, Copy)]
pub struct BlobWindow {
    pub start_chunk: usize,
    pub slot_count: usize,
    pub chunk_words: u32,
    pub total_resident_chunks: usize,
}

impl BlobWindow {
    /// Creates a new window covering `slot_count` chunks starting at `start_chunk`.
    ///
    /// Fails if:
    /// - `start_chunk >= total_resident_chunks`
    /// - `slot_count == 0` or `slot_count > BLOB_BINDING_SLOTS`
    /// - `start_chunk + slot_count > total_resident_chunks` (window extends past resident chunks)
    pub fn new(
        start_chunk: usize,
        slot_count: usize,
        chunk_words: u32,
        total_resident_chunks: usize,
    ) -> Result<Self, String> {
        if start_chunk >= total_resident_chunks {
            return Err(format!(
                "window start_chunk {} >= total_resident_chunks {}",
                start_chunk, total_resident_chunks
            ));
        }
        if slot_count == 0 || slot_count > BLOB_BINDING_SLOTS {
            return Err(format!(
                "slot_count {} must be in [1, BLOB_BINDING_SLOTS={}]",
                slot_count, BLOB_BINDING_SLOTS
            ));
        }
        if start_chunk + slot_count > total_resident_chunks {
            return Err(format!(
                "window [{}, {}) exceeds total_resident_chunks {}",
                start_chunk,
                start_chunk + slot_count,
                total_resident_chunks
            ));
        }
        Ok(Self {
            start_chunk,
            slot_count,
            chunk_words,
            total_resident_chunks,
        })
    }

    /// Creates the narrowest window covering the absolute word range
    /// `[start_word, end_word]`.
    ///
    /// Fails if the range spans more than `BLOB_BINDING_SLOTS` chunks, or
    /// extends past the resident chunk count. Model-free so the algebra is
    /// exercised by the CPU-only PPT contract suite.
    pub fn for_range(
        start_word: u32,
        end_word: u32,
        chunk_words: u32,
        total_resident_chunks: usize,
    ) -> Result<Self, String> {
        let start_chunk = (start_word / chunk_words) as usize;
        let end_chunk = (end_word / chunk_words) as usize;
        Self::new(
            start_chunk,
            end_chunk - start_chunk + 1,
            chunk_words,
            total_resident_chunks,
        )
    }

    /// Creates a window covering exactly `BLOB_BINDING_SLOTS` chunks (8) starting at `start_chunk`.
    /// This is the normal dispatch window for layer shaders.
    pub fn full(
        start_chunk: usize,
        chunk_words: u32,
        total_resident_chunks: usize,
    ) -> Result<Self, String> {
        Self::new(
            start_chunk,
            BLOB_BINDING_SLOTS,
            chunk_words,
            total_resident_chunks,
        )
    }

    /// Returns the word offset of the window's start in the absolute GGUF space.
    pub fn window_base_words(&self) -> u32 {
        (self.start_chunk as u32) * self.chunk_words
    }

    /// Maps an absolute word index to `(slot_index, local_word_offset)` within this window.
    ///
    /// Returns an error if the word is not covered by this window (before start or at/after end).
    pub fn absolute_to_local(&self, absolute_word: u32) -> Result<(usize, u32), String> {
        let base = self.window_base_words();
        let end = base + (self.slot_count as u32) * self.chunk_words;

        if absolute_word < base {
            return Err(format!(
                "word {} before window base {}",
                absolute_word, base
            ));
        }
        if absolute_word >= end {
            return Err(format!(
                "word {} at/after window end {} (slot_count={}, chunk_words={})",
                absolute_word, end, self.slot_count, self.chunk_words
            ));
        }

        let rel = absolute_word - base;
        let slot = (rel / self.chunk_words) as usize;
        let offset = rel % self.chunk_words;

        // Exercise the word-index invariant for the local offset within the slot's chunk.
        assert_word_index_in_range(offset, self.chunk_words, "BlobWindow::absolute_to_local");
        Ok((slot, offset))
    }

    /// Reconstructs an absolute word index from a window-local `(slot_index, local_word)`.
    pub fn local_to_absolute(&self, slot_index: usize, local_word: u32) -> u32 {
        let base = self.window_base_words();
        base + (slot_index as u32) * self.chunk_words + local_word
    }

    /// Checks if an absolute word index is covered by this window.
    pub fn contains(&self, absolute_word: u32) -> bool {
        let base = self.window_base_words();
        let end = base + (self.slot_count as u32) * self.chunk_words;
        absolute_word >= base && absolute_word < end
    }

    /// Returns the binding resources for this window's slots.
    /// Slot `i` binds resident chunk `start_chunk + i` (or dummy_buf if missing).
    pub fn binding_resources<'a>(
        &self,
        model: &'a BindlessModel,
    ) -> [wgpu::BindingResource<'a>; BLOB_BINDING_SLOTS] {
        std::array::from_fn(|i| {
            if i < self.slot_count {
                let chunk_idx = self.start_chunk + i;
                if chunk_idx < model.gpu_buffers.len() {
                    return model.gpu_buffers[chunk_idx].as_entire_binding();
                }
            }
            model.dummy_buf.as_entire_binding()
        })
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

    /// Total number of resident chunks (may exceed BLOB_BINDING_SLOTS).
    /// Determined by file_size / effective_chunk at load time.
    pub total_resident_chunks: usize,

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
        if !self.gpu_buffers.is_empty() {
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

    /// Binding resource for blob_3: bytes [3·effective_chunk, size).
    /// Falls back to the 4-byte dummy for models < 4 chunks.
    pub fn blob_binding_3(&self) -> wgpu::BindingResource<'_> {
        if self.gpu_buffers.len() > 3 {
            self.gpu_buffers[3].as_entire_binding()
        } else {
            self.dummy_buf.as_entire_binding()
        }
    }

    /// Binding resource for blob_4: bytes [4·effective_chunk, size).
    /// Falls back to the 4-byte dummy for models < 5 chunks.
    pub fn blob_binding_4(&self) -> wgpu::BindingResource<'_> {
        if self.gpu_buffers.len() > 4 {
            self.gpu_buffers[4].as_entire_binding()
        } else {
            self.dummy_buf.as_entire_binding()
        }
    }

    /// Binding resource for blob_5: bytes [5·effective_chunk, size).
    /// Falls back to the 4-byte dummy for models < 6 chunks.
    pub fn blob_binding_5(&self) -> wgpu::BindingResource<'_> {
        if self.gpu_buffers.len() > 5 {
            self.gpu_buffers[5].as_entire_binding()
        } else {
            self.dummy_buf.as_entire_binding()
        }
    }

    /// Binding resource for blob_6: bytes [6·effective_chunk, size).
    /// Falls back to the 4-byte dummy for models < 7 chunks.
    pub fn blob_binding_6(&self) -> wgpu::BindingResource<'_> {
        if self.gpu_buffers.len() > 6 {
            self.gpu_buffers[6].as_entire_binding()
        } else {
            self.dummy_buf.as_entire_binding()
        }
    }

    /// Binding resource for blob_7: bytes [7·effective_chunk, size).
    /// Falls back to the 4-byte dummy for models < 8 chunks.
    pub fn blob_binding_7(&self) -> wgpu::BindingResource<'_> {
        if self.gpu_buffers.len() > 7 {
            self.gpu_buffers[7].as_entire_binding()
        } else {
            self.dummy_buf.as_entire_binding()
        }
    }

    /// Words per blob chunk (effective_chunk / 4). Shaders use this to resolve
    /// an absolute word index to `(chunk_index, offset_in_chunk)`.
    pub fn chunk_words(&self) -> u32 {
        (self.effective_chunk / 4) as u32
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

    /// Creates a full window (8 slots) starting at `start_chunk` for layer dispatches.
    /// Used by normal layer, prefill, and decode bind groups.
    pub fn layer_window(&self, start_chunk: usize) -> Result<BlobWindow, String> {
        BlobWindow::full(start_chunk, self.chunk_words(), self.total_resident_chunks)
    }

    /// Creates a window covering the exact chunks needed for a tensor range.
    /// Fails if the range spans more than BLOB_BINDING_SLOTS chunks.
    pub fn window_for_range(&self, start_word: u32, end_word: u32) -> Result<BlobWindow, String> {
        BlobWindow::for_range(
            start_word,
            end_word,
            self.chunk_words(),
            self.total_resident_chunks,
        )
    }

    /// Creates a window for dequant_any hot path (single tensor row).
    /// The window covers the chunk containing `offset_words`.
    pub fn dequant_window(&self, offset_words: u32, count: u32) -> Result<BlobWindow, String> {
        let chunk_words = self.chunk_words();
        let start_chunk = (offset_words / chunk_words) as usize;
        let end_chunk = ((offset_words + count) / chunk_words) as usize;
        let slot_count = end_chunk - start_chunk + 1;
        BlobWindow::new(
            start_chunk,
            slot_count,
            chunk_words,
            self.total_resident_chunks,
        )
    }

    /// Creates a window for RMSNorm covering the full weight (and bias) span.
    ///
    /// `count` is the element count of the norm tensor: the shader reads
    /// `weight_offset .. weight_offset + count`, so the window must cover the
    /// whole run, not just the word the tensor starts at.
    pub fn rmsnorm_window(
        &self,
        weight_offset: u32,
        bias_offset: Option<u32>,
        count: u32,
    ) -> Result<BlobWindow, String> {
        let last = count.saturating_sub(1);
        let start_word = weight_offset.min(bias_offset.unwrap_or(weight_offset));
        let end_word = weight_offset
            .saturating_add(last)
            .max(bias_offset.map_or(0, |b| b.saturating_add(last)));
        BlobWindow::for_range(
            start_word,
            end_word,
            self.chunk_words(),
            self.total_resident_chunks,
        )
    }

    /// Creates a window covering the LM-head weight rows `base_row..base_row+rows`.
    ///
    /// Row stride is derived from the head tensor's quant type via the GGUF
    /// spec registry (`block_bytes / block_elems`), NOT assumed to be 4 bytes
    /// per element — a Q4_K head is ~4.5 bits/element, so a f32 assumption
    /// overstates the span by ~7x and yields a window that cannot be bound.
    ///
    /// Falls back to `token_embd.weight` when the model ties its head weights,
    /// matching the tensor the dispatch actually reads.
    pub fn lm_head_window(&self, base_row: u32, rows: u32, dim: u32) -> Result<BlobWindow, String> {
        let name = if self.metadata.get_tensor_type("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let weight_bytes = self.metadata.get_tensor_offset(name).unwrap_or(0);
        let quant_type = self.metadata.get_tensor_type(name).unwrap_or(0);

        let (block_elems, block_bytes) = quant_block_geometry(quant_type)
            .ok_or_else(|| format!("unknown quant type {} for {}", quant_type, name))?;
        if !(dim as usize).is_multiple_of(block_elems) {
            return Err(format!(
                "{} row dim {} is not a multiple of block_elems {} for quant type {}",
                name, dim, block_elems, quant_type
            ));
        }
        let row_bytes = (dim as u64 / block_elems as u64) * block_bytes as u64;

        let start_byte = weight_bytes + (base_row as u64) * row_bytes;
        // Last byte actually read, not the exclusive end: an exclusive end that
        // lands on a chunk boundary would pull in a slot the dispatch never
        // touches, and can push the window past the resident chunk count.
        let end_byte = weight_bytes + ((base_row as u64) + (rows as u64)) * row_bytes
            - if rows == 0 { 0 } else { 1 };

        self.window_for_range((start_byte / 4) as u32, (end_byte / 4) as u32)
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
            total_resident_chunks: num_chunks,
            dummy_buf,
            metadata,
            preflight,
        }
    }
}
