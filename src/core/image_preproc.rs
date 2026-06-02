//! Image preprocessing for SigLIP ViT / MiniCPM-V-2.6 input.
//!
//! Converts a raw RGB image into one or more 448×448 tiles ready for
//! [`crate::core::vision_gpu::GpuVisionModel::encode_tiles`].
//!
//! # Pipeline
//!
//! ```text
//! raw HWC u8  ──normalize──►  CHW f32 [−1,1]
//!                                  │
//!                     ┌───────────────────────┐
//!                     │  pick_grid (aspect)   │
//!                     └───────────┬───────────┘
//!                   per tile      │
//!             ┌────────────────────────────────┐
//!             │  crop tile region (CHW f32)    │
//!             │  bilinear_resize → 448×448     │
//!             └────────────────────────────────┘
//!                     + thumbnail (whole image → 448×448)
//! ```
//!
//! # Normalisation
//! SigLIP uses mean = 0.5, std = 0.5 per channel, identical across R/G/B.
//!
//! ```text
//! pixel_norm = (pixel_u8 / 255.0 − 0.5) / 0.5
//!            = pixel_u8 / 127.5 − 1.0
//! ```

// ─── Constants ────────────────────────────────────────────────────────────────

/// Target tile size (pixels per side).
pub const IMAGE_SIZE: usize = 448;

/// Patch size used by SigLIP-So400M.
pub const PATCH_SIZE: usize = 14;

/// Number of patches per tile: (448/14)² = 1024.
pub const N_PATCHES: usize = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE);

/// ViT hidden dimension.
pub const HIDDEN_DIM: usize = 1152;

/// Maximum number of tiles (slices) when tiling a large image.
/// MiniCPM-V-2.6 default: 9 slices + 1 thumbnail.
pub const MAX_SLICES: usize = 9;

// ─── Normalisation ────────────────────────────────────────────────────────────

/// Convert a packed HWC `u8` RGB image to CHW `f32` in `[−1, 1]`.
///
/// # Arguments
/// * `pixels_hwc` – packed `[H × W × 3]` bytes, row-major, RGB order.
/// * `h`, `w`     – image height and width.
///
/// # Returns
/// `[3 × H × W]` f32 vector in channel-first (CHW) order.
pub fn normalize_hwc_u8_to_chw_f32(pixels_hwc: &[u8], h: usize, w: usize) -> Vec<f32> {
    assert_eq!(pixels_hwc.len(), h * w * 3, "pixel buffer length mismatch");
    let n = h * w;
    let mut out = vec![0.0f32; 3 * n];
    for (idx, pixel) in pixels_hwc.chunks_exact(3).enumerate() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        // Normalise: x/127.5 − 1.0
        out[idx]         = r * (1.0 / 127.5) - 1.0; // channel 0
        out[n + idx]     = g * (1.0 / 127.5) - 1.0; // channel 1
        out[2 * n + idx] = b * (1.0 / 127.5) - 1.0; // channel 2
    }
    out
}

// ─── Bilinear resize ─────────────────────────────────────────────────────────

/// Bilinear resize a CHW f32 image.
///
/// All `C` channels are resized independently using the same sampling grid.
///
/// # Arguments
/// * `src_chw`         – `[C × src_h × src_w]` source pixels.
/// * `c`, `src_h`, `src_w` – source dimensions.
/// * `dst_h`, `dst_w`  – target dimensions.
///
/// # Returns
/// `[C × dst_h × dst_w]` f32 vector.
pub fn bilinear_resize_chw(
    src_chw: &[f32],
    c: usize, src_h: usize, src_w: usize,
    dst_h: usize, dst_w: usize,
) -> Vec<f32> {
    debug_assert_eq!(src_chw.len(), c * src_h * src_w);
    let mut out = vec![0.0f32; c * dst_h * dst_w];

    // Scale factors (map destination pixel centre to source coordinates)
    let scale_h = src_h as f32 / dst_h as f32;
    let scale_w = src_w as f32 / dst_w as f32;

    for ch in 0..c {
        let src_plane = &src_chw[ch * src_h * src_w..];
        let dst_plane = &mut out[ch * dst_h * dst_w..];

        for dr in 0..dst_h {
            // Map destination pixel centre to source space
            let sy = (dr as f32 + 0.5) * scale_h - 0.5;
            let sy0 = sy.floor() as isize;
            let sy1 = sy0 + 1;
            let wy = sy - sy0 as f32; // [0, 1)

            // Clamp row indices
            let sy0c = sy0.max(0).min(src_h as isize - 1) as usize;
            let sy1c = sy1.max(0).min(src_h as isize - 1) as usize;

            for dc in 0..dst_w {
                let sx = (dc as f32 + 0.5) * scale_w - 0.5;
                let sx0 = sx.floor() as isize;
                let sx1 = sx0 + 1;
                let wx = sx - sx0 as f32;

                let sx0c = sx0.max(0).min(src_w as isize - 1) as usize;
                let sx1c = sx1.max(0).min(src_w as isize - 1) as usize;

                let tl = src_plane[sy0c * src_w + sx0c];
                let tr = src_plane[sy0c * src_w + sx1c];
                let bl = src_plane[sy1c * src_w + sx0c];
                let br = src_plane[sy1c * src_w + sx1c];

                let top = tl + (tr - tl) * wx;
                let bot = bl + (br - bl) * wx;
                dst_plane[dr * dst_w + dc] = top + (bot - top) * wy;
            }
        }
    }
    out
}

// ─── Grid selection ───────────────────────────────────────────────────────────

/// Choose the best (rows, cols) tiling grid for `max_slices` tiles.
///
/// Evaluates all `(r, c)` combinations with `r*c ≤ max_slices` and picks the
/// one that minimises wasted area after fitting the image into an `r×c` grid
/// at `IMAGE_SIZE×IMAGE_SIZE` per cell.  Ties are broken by preferring fewer
/// total slices (smaller grids are more efficient on compute).
///
/// Returns `(rows, cols)` where `rows ≥ 1`, `cols ≥ 1`.
pub fn pick_grid(img_h: usize, img_w: usize, max_slices: usize) -> (usize, usize) {
    let aspect = img_w as f32 / img_h as f32;

    let mut best = (1usize, 1usize);
    let mut best_score = f32::MAX;

    for r in 1..=max_slices {
        for c in 1..=max_slices {
            if r * c > max_slices {
                continue;
            }
            // Effective aspect ratio of the grid
            let grid_aspect = c as f32 / r as f32;
            // Difference from image aspect ratio — lower is better
            let diff = (grid_aspect - aspect).abs();
            // Prefer fewer total cells (tie-break)
            let score = diff + (r * c) as f32 * 1e-4;
            if score < best_score {
                best_score = score;
                best = (r, c);
            }
        }
    }
    best
}

// ─── Tile extraction ─────────────────────────────────────────────────────────

/// Crop a tile region from a CHW image and resize it to `IMAGE_SIZE×IMAGE_SIZE`.
///
/// The `(tile_row, tile_col)` are 0-indexed positions in the `(rows, cols)` grid.
fn crop_and_resize_tile(
    src_chw: &[f32],
    src_h: usize, src_w: usize,
    tile_row: usize, tile_col: usize,
    rows: usize, cols: usize,
) -> Vec<f32> {
    let c = 3usize;

    // Tile region in the source image (can have fractional rounding)
    let tile_h_src = src_h / rows;
    let tile_w_src = src_w / cols;

    let y0 = tile_row * tile_h_src;
    let x0 = tile_col * tile_w_src;
    // For last row/col take remaining pixels to avoid off-by-one
    let y1 = if tile_row + 1 == rows { src_h } else { y0 + tile_h_src };
    let x1 = if tile_col + 1 == cols { src_w } else { x0 + tile_w_src };

    let crop_h = y1 - y0;
    let crop_w = x1 - x0;

    // Extract crop into contiguous buffer
    let mut crop = vec![0.0f32; c * crop_h * crop_w];
    for ch in 0..c {
        for r in 0..crop_h {
            let src_row = &src_chw[ch * src_h * src_w + (y0 + r) * src_w + x0..];
            let dst_row = &mut crop[ch * crop_h * crop_w + r * crop_w..];
            dst_row[..crop_w].copy_from_slice(&src_row[..crop_w]);
        }
    }

    // Resize crop to IMAGE_SIZE × IMAGE_SIZE
    bilinear_resize_chw(&crop, c, crop_h, crop_w, IMAGE_SIZE, IMAGE_SIZE)
}

// ─── Public tiling API ───────────────────────────────────────────────────────

/// Tile a normalised CHW f32 image into `IMAGE_SIZE×IMAGE_SIZE` tiles.
///
/// Returns a `Vec` of tiles; each tile is `[3 × IMAGE_SIZE × IMAGE_SIZE]` f32.
///
/// # Layout
/// * Tiles 0..(rows*cols): the grid slices in row-major order.
/// * Tile `rows*cols`:     the thumbnail (whole image scaled to 448×448).
///
/// If the image already fits in one tile (`rows=1, cols=1`) the thumbnail IS
/// the single slice — only one tile is returned to avoid duplicating work.
pub fn tile_image_chw(
    src_chw: &[f32],
    src_h: usize,
    src_w: usize,
    max_slices: usize,
) -> Vec<Vec<f32>> {
    assert_eq!(src_chw.len(), 3 * src_h * src_w);
    let (rows, cols) = pick_grid(src_h, src_w, max_slices);
    let mut tiles: Vec<Vec<f32>> = Vec::with_capacity(rows * cols + 1);

    for r in 0..rows {
        for c_idx in 0..cols {
            tiles.push(crop_and_resize_tile(src_chw, src_h, src_w, r, c_idx, rows, cols));
        }
    }

    // Thumbnail — skip if already a 1×1 grid (tile 0 is already the full image)
    if rows > 1 || cols > 1 {
        let thumb = bilinear_resize_chw(src_chw, 3, src_h, src_w, IMAGE_SIZE, IMAGE_SIZE);
        tiles.push(thumb);
    }

    tiles
}

// ─── Patch embed + positional embed (CPU) ────────────────────────────────────

/// Apply patch embedding and add positional embedding to a 448×448 CHW tile.
///
/// # Arguments
/// * `tile_chw`   – `[3 × 448 × 448]` f32 tile.
/// * `patch_w`    – `[HIDDEN_DIM × 3 × 14 × 14]` conv kernel (f32, from dequanted F16).
/// * `patch_b`    – `[HIDDEN_DIM]` bias.
/// * `pos_embed`  – `[4900 × HIDDEN_DIM]` positional embedding; first N_PATCHES rows used.
///
/// # Returns
/// `[N_PATCHES × HIDDEN_DIM]` = `[1024 × 1152]` f32 ready for `GpuVisionModel::encode_image`.
pub fn patch_embed_with_pos(
    tile_chw: &[f32],
    patch_w: &[f32],
    patch_b: &[f32],
    pos_embed: &[f32],
) -> Vec<f32> {
    use crate::core::tensor::Tensor;
    use crate::ops::reference::vision::patch_embed_f32;

    let img_tensor = Tensor::new(tile_chw.to_vec(), vec![3, IMAGE_SIZE, IMAGE_SIZE])
        .expect("tile shape invalid");
    let w_tensor = Tensor::new(patch_w.to_vec(), vec![HIDDEN_DIM, 3, PATCH_SIZE, PATCH_SIZE])
        .expect("patch_w shape invalid");
    let b_tensor = Tensor::new(patch_b.to_vec(), vec![HIDDEN_DIM])
        .expect("patch_b shape invalid");

    let patches = patch_embed_f32(&img_tensor, &w_tensor, &b_tensor, PATCH_SIZE)
        .expect("patch_embed_f32 failed");
    // patches: [N_PATCHES × HIDDEN_DIM]

    // Add first N_PATCHES rows of pos_embed
    let pos_len = N_PATCHES * HIDDEN_DIM;
    let pos_slice = &pos_embed[..pos_len];

    let mut out = patches.data;
    for (x, p) in out.iter_mut().zip(pos_slice.iter()) {
        *x += p;
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_roundtrip_black() {
        let pixels = vec![0u8; 4 * 4 * 3];
        let out = normalize_hwc_u8_to_chw_f32(&pixels, 4, 4);
        assert_eq!(out.len(), 3 * 4 * 4);
        for &v in &out {
            assert!((v - (-1.0)).abs() < 1e-6, "black pixel should map to -1.0, got {v}");
        }
    }

    #[test]
    fn normalize_white_pixel() {
        let pixels = vec![255u8; 2 * 2 * 3];
        let out = normalize_hwc_u8_to_chw_f32(&pixels, 2, 2);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4, "white pixel should map to ≈1.0, got {v}");
        }
    }

    #[test]
    fn bilinear_resize_noop() {
        // Resize to same size → values should be unchanged
        let src: Vec<f32> = (0..3 * 4 * 4).map(|i| i as f32).collect();
        let out = bilinear_resize_chw(&src, 3, 4, 4, 4, 4);
        for (a, b) in src.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5, "noop resize mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn bilinear_resize_upsample_shape() {
        let src = vec![0.5f32; 3 * 2 * 2];
        let out = bilinear_resize_chw(&src, 3, 2, 2, 8, 8);
        assert_eq!(out.len(), 3 * 8 * 8);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn pick_grid_square_prefers_1x1() {
        // A 448×448 image → no slicing needed
        let (r, c) = pick_grid(448, 448, 9);
        assert_eq!((r, c), (1, 1));
    }

    #[test]
    fn pick_grid_wide_image() {
        // 448×1344 (3:1 aspect ratio) → should get 1×3 or similar
        let (r, c) = pick_grid(448, 1344, 9);
        assert!(c >= r, "wide image should have more cols than rows: ({r},{c})");
        assert!(r * c <= 9);
    }

    #[test]
    fn tile_image_square_single_tile() {
        let src = vec![0.0f32; 3 * 448 * 448];
        let tiles = tile_image_chw(&src, 448, 448, 9);
        assert_eq!(tiles.len(), 1, "square 448×448 should produce exactly 1 tile");
        assert_eq!(tiles[0].len(), 3 * IMAGE_SIZE * IMAGE_SIZE);
    }

    #[test]
    fn tile_image_wide_produces_thumb() {
        let src = vec![0.5f32; 3 * 448 * 896]; // 1:2 aspect
        let tiles = tile_image_chw(&src, 448, 896, 9);
        // Should have at least 2 slices + thumbnail
        assert!(tiles.len() >= 3, "wide image should produce >1 slice + thumb, got {}", tiles.len());
        for t in &tiles {
            assert_eq!(t.len(), 3 * IMAGE_SIZE * IMAGE_SIZE);
        }
    }
}
