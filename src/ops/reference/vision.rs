//! Patch embedding for Vision Transformer (ViT) image tokenization.
//!
//! Converts raw image pixel data into a sequence of patch tokens by
//! applying a non-overlapping 2-D convolution (kernel=14, stride=14 for
//! SigLIP-So400M).  This is equivalent to a standard Conv2d but implemented
//! as a strided matmul for clarity and testability on CPU.

use crate::core::{
    error::{LibshimmyError, Result},
    tensor::Tensor,
};

/// Embed a single image into patch tokens.
///
/// # Arguments
/// * `image`  – `[C, H, W]` float32, values already normalised to the model's
///              mean/std (SigLIP: mean=0.5, std=0.5 → range ≈ [−1, 1]).
/// * `weight` – `[out_ch, C, kH, kW]` conv kernel; for SigLIP-So400M:
///              `[1152, 3, 14, 14]`.
/// * `bias`   – `[out_ch]`; for SigLIP-So400M: `[1152]`.
/// * `patch`  – Patch (kernel) size in pixels; must divide H and W exactly.
///
/// # Returns
/// `[n_patches, out_ch]` where `n_patches = (H/patch) * (W/patch)`.
///
/// # Errors
/// * `ShapeMismatch` if image rank ≠ 3, weight rank ≠ 4, H/W not divisible
///   by `patch`, or channel dimensions are inconsistent.
pub fn patch_embed_f32(
    image: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    patch: usize,
) -> Result<Tensor> {
    // ── Shape validation ──────────────────────────────────────────────────────
    if image.ndim() != 3 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "patch_embed_image".to_string(),
            expected: vec![3],
            got: vec![image.ndim()],
        });
    }
    if weight.ndim() != 4 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "patch_embed_weight".to_string(),
            expected: vec![4],
            got: vec![weight.ndim()],
        });
    }

    let (c_in, h, w) = (image.shape[0], image.shape[1], image.shape[2]);
    let (out_ch, wc, wh, ww) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);

    if wc != c_in {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "patch_embed_weight_channels".to_string(),
            expected: vec![c_in],
            got: vec![wc],
        });
    }
    if wh != patch || ww != patch {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "patch_embed_weight_spatial".to_string(),
            expected: vec![patch, patch],
            got: vec![wh, ww],
        });
    }
    if h % patch != 0 || w % patch != 0 {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "patch_embed_image_spatial_not_divisible".to_string(),
            expected: vec![0], // divisible
            got: vec![h % patch, w % patch],
        });
    }
    if bias.ndim() != 1 || bias.shape[0] != out_ch {
        return Err(LibshimmyError::ShapeMismatch {
            tensor: "patch_embed_bias".to_string(),
            expected: vec![out_ch],
            got: bias.shape.clone(),
        });
    }

    let ph = h / patch; // grid rows
    let pw = w / patch; // grid cols
    let n_patches = ph * pw;
    let kernel_flat = c_in * patch * patch; // flattened kernel size

    // ── Flatten weight: [out_ch, kernel_flat] ─────────────────────────────────
    // weight is stored [out_ch, c_in, kH, kW]; row i is one output filter.
    // Already contiguous in that order — we treat it as-is.
    let w_data = &weight.data; // length = out_ch * kernel_flat

    // ── Output buffer ─────────────────────────────────────────────────────────
    let mut out = vec![0.0f32; n_patches * out_ch];

    for pr in 0..ph {
        for pc in 0..pw {
            let patch_idx = pr * pw + pc;

            // Extract and flatten the [C, patch, patch] patch from image
            let mut patch_vec = Vec::with_capacity(kernel_flat);
            for ci in 0..c_in {
                for kr in 0..patch {
                    for kc in 0..patch {
                        let pixel_r = pr * patch + kr;
                        let pixel_c = pc * patch + kc;
                        let img_idx = ci * h * w + pixel_r * w + pixel_c;
                        patch_vec.push(image.data[img_idx]);
                    }
                }
            }

            // Dot each output filter with the patch vector, add bias
            for o in 0..out_ch {
                let filter = &w_data[o * kernel_flat..(o + 1) * kernel_flat];
                let dot: f32 = filter
                    .iter()
                    .zip(patch_vec.iter())
                    .map(|(&f, &p)| f * p)
                    .sum();
                out[patch_idx * out_ch + o] = dot + bias.data[o];
            }
        }
    }

    Tensor::new(out, vec![n_patches, out_ch])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_embed_zero_image_zero_weight() {
        // All zeros in → all zeros out (bias also zero)
        let image  = Tensor::zeros(vec![3, 28, 28]);
        let weight = Tensor::zeros(vec![16, 3, 14, 14]);
        let bias   = Tensor::zeros(vec![16]);
        let result = patch_embed_f32(&image, &weight, &bias, 14).unwrap();
        assert_eq!(result.shape, vec![4, 16]); // (28/14)^2 = 4 patches, 16-d
        assert!(result.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_patch_embed_output_shape_448() {
        // SigLIP-So400M dimensions: 448×448, patch=14, out_ch=1152
        // n_patches = (448/14)^2 = 1024
        let image  = Tensor::zeros(vec![3, 448, 448]);
        let weight = Tensor::zeros(vec![1152, 3, 14, 14]);
        let bias   = Tensor::zeros(vec![1152]);
        let result = patch_embed_f32(&image, &weight, &bias, 14).unwrap();
        assert_eq!(result.shape, vec![1024, 1152]);
    }

    #[test]
    fn test_patch_embed_known_value() {
        // Single-channel 2×2 image, patch=1, out_ch=1
        // weight = [[1.0]], bias = [0.5]
        // Each pixel → 1 patch; output[i] = pixel[i] * 1.0 + 0.5
        let image  = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
        let weight = Tensor::new(vec![1.0], vec![1, 1, 1, 1]).unwrap();
        let bias   = Tensor::new(vec![0.5], vec![1]).unwrap();
        let result = patch_embed_f32(&image, &weight, &bias, 1).unwrap();
        assert_eq!(result.shape, vec![4, 1]);
        assert!((result.data[0] - 1.5).abs() < 1e-6);
        assert!((result.data[1] - 2.5).abs() < 1e-6);
        assert!((result.data[2] - 3.5).abs() < 1e-6);
        assert!((result.data[3] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_patch_embed_bad_spatial() {
        // H=30 not divisible by patch=14 → error
        let image  = Tensor::zeros(vec![3, 30, 28]);
        let weight = Tensor::zeros(vec![16, 3, 14, 14]);
        let bias   = Tensor::zeros(vec![16]);
        assert!(patch_embed_f32(&image, &weight, &bias, 14).is_err());
    }
}
