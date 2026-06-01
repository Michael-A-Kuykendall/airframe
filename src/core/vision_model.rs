//! VisionModel — weight store for MiniCPM-V-2.6 mmproj (ViT + Resampler).
//!
//! Loads `mmproj-model-f16.gguf`, maps every tensor to a `WeightId`, and
//! provides typed accessors used by [`crate::family::vit::SigLipEncoder`] and
//! [`crate::family::resampler::Resampler`].
//!
//! This is intentionally separate from `model.rs` (which targets the LLM backbone):
//! the mmproj GGUF has no LLM metadata, so `model_spec_from_metadata` would fail.

use std::collections::HashMap;
use std::path::Path;

use crate::core::{
    error::{LibshimmyError, Result},
    model::{GgufTensorInfo, load_mmproj_gguf_raw},
    tensor::Tensor,
    weight_id::WeightId,
};

// ─── VisionModel ─────────────────────────────────────────────────────────────

/// Weight container for the vision encoder (ViT + Resampler).
///
/// Only stores the tensors mapped by [`parse_vision_tensor_name`]; unknown
/// tensors (e.g. `clip.vision.*` metadata duplicates) are silently skipped.
pub struct VisionModel {
    /// All loaded tensors, keyed by canonical WeightId.
    pub weights: HashMap<WeightId, Tensor>,
    /// Number of ViT blocks detected at load time.
    pub n_vit_layers: usize,
}

impl VisionModel {
    /// Load from a `mmproj-model-f16.gguf` file.
    ///
    /// ```no_run
    /// let vm = VisionModel::from_mmproj_gguf("D:/models/mmproj-model-f16.gguf")?;
    /// assert!(vm.weights.contains_key(&WeightId::ResamplerQuery));
    /// ```
    pub fn from_mmproj_gguf<P: AsRef<Path>>(path: P) -> Result<Self> {
        let (tensor_infos, mmap, tensor_data_base_offset) =
            load_mmproj_gguf_raw(path.as_ref())?;

        let mut weights: HashMap<WeightId, Tensor> = HashMap::new();
        let mut max_vit_layer: Option<usize> = None;

        for tensor_info in &tensor_infos {
            let id = match parse_vision_tensor_name(&tensor_info.name) {
                Some(id) => id,
                None => {
                    // Unknown tensor — skip (common for CLIP-side metadata duplicates)
                    continue;
                }
            };

            // Track the maximum ViT block index seen
            if let WeightId::VisionBlkAttnQWeight(layer) = id {
                max_vit_layer = Some(max_vit_layer.map_or(layer, |m: usize| m.max(layer)));
            }

            let tensor = crate::core::model::load_vision_tensor(&tensor_info, &mmap, tensor_data_base_offset)?;
            weights.insert(id, tensor);
        }

        let n_vit_layers = max_vit_layer.map(|l| l + 1).unwrap_or(0);

        println!(
            "✅ VisionModel loaded: {} tensors, {} ViT layers",
            weights.len(),
            n_vit_layers
        );

        Ok(VisionModel { weights, n_vit_layers })
    }

    /// Retrieve a required weight or return a `WeightMissing` error.
    pub fn require(&self, id: &WeightId) -> Result<&Tensor> {
        self.weights.get(id).ok_or_else(|| LibshimmyError::WeightMissing {
            weight_id: format!("{:?}", id),
        })
    }

    /// Check that all core ViT and Resampler weights are present.
    pub fn validate(&self) -> Result<()> {
        let required: Vec<WeightId> = [
            WeightId::VisionPatchEmbedWeight,
            WeightId::VisionPatchEmbedBias,
            WeightId::VisionPosEmbedWeight,
            WeightId::VisionPostNormWeight,
            WeightId::VisionPostNormBias,
            WeightId::ResamplerQuery,
            WeightId::ResamplerKvWeight,
            WeightId::ResamplerLnQWeight,
            WeightId::ResamplerLnQBias,
            WeightId::ResamplerLnKvWeight,
            WeightId::ResamplerLnKvBias,
            WeightId::ResamplerAttnQWeight,
            WeightId::ResamplerAttnQBias,
            WeightId::ResamplerAttnKWeight,
            WeightId::ResamplerAttnKBias,
            WeightId::ResamplerAttnVWeight,
            WeightId::ResamplerAttnVBias,
            WeightId::ResamplerAttnOutWeight,
            WeightId::ResamplerAttnOutBias,
            WeightId::ResamplerPosEmbedK,
            WeightId::ResamplerLnPostWeight,
            WeightId::ResamplerLnPostBias,
            WeightId::ResamplerProjWeight,
        ]
        .into();

        for id in &required {
            if !self.weights.contains_key(id) {
                return Err(LibshimmyError::WeightMissing {
                    weight_id: format!("{:?}", id),
                });
            }
        }

        // Check all ViT blocks present
        for layer in 0..self.n_vit_layers {
            for id in vit_block_weight_ids(layer) {
                if !self.weights.contains_key(&id) {
                    return Err(LibshimmyError::WeightMissing {
                        weight_id: format!("{:?}", id),
                    });
                }
            }
        }

        Ok(())
    }
}

// ─── Tensor name → WeightId mapping ─────────────────────────────────────────

/// Map an mmproj GGUF tensor name to its canonical `WeightId`.
///
/// Returns `None` for tensors that have no mapping (e.g. CLIP metadata).
pub fn parse_vision_tensor_name(name: &str) -> Option<WeightId> {
    // ── ViT patch embedding ───────────────────────────────────────────────────
    if name == "v.patch_embd.weight"    { return Some(WeightId::VisionPatchEmbedWeight); }
    if name == "v.patch_embd.bias"      { return Some(WeightId::VisionPatchEmbedBias); }
    if name == "v.position_embd.weight" { return Some(WeightId::VisionPosEmbedWeight); }
    if name == "v.post_ln.weight"       { return Some(WeightId::VisionPostNormWeight); }
    if name == "v.post_ln.bias"         { return Some(WeightId::VisionPostNormBias); }

    // ── ViT per-block weights: prefix "v.blk.N." ─────────────────────────────
    if let Some(rest) = name.strip_prefix("v.blk.") {
        // rest = "N.attn_q.weight", "N.ln1.bias", etc.
        if let Some(dot) = rest.find('.') {
            let layer_str = &rest[..dot];
            let suffix    = &rest[dot + 1..]; // e.g. "attn_q.weight"
            if let Ok(layer) = layer_str.parse::<usize>() {
                return match suffix {
                    "attn_q.weight"   => Some(WeightId::VisionBlkAttnQWeight(layer)),
                    "attn_q.bias"     => Some(WeightId::VisionBlkAttnQBias(layer)),
                    "attn_k.weight"   => Some(WeightId::VisionBlkAttnKWeight(layer)),
                    "attn_k.bias"     => Some(WeightId::VisionBlkAttnKBias(layer)),
                    "attn_v.weight"   => Some(WeightId::VisionBlkAttnVWeight(layer)),
                    "attn_v.bias"     => Some(WeightId::VisionBlkAttnVBias(layer)),
                    "attn_out.weight" => Some(WeightId::VisionBlkAttnOutWeight(layer)),
                    "attn_out.bias"   => Some(WeightId::VisionBlkAttnOutBias(layer)),
                    "ln1.weight"      => Some(WeightId::VisionBlkLn1Weight(layer)),
                    "ln1.bias"        => Some(WeightId::VisionBlkLn1Bias(layer)),
                    "ln2.weight"      => Some(WeightId::VisionBlkLn2Weight(layer)),
                    "ln2.bias"        => Some(WeightId::VisionBlkLn2Bias(layer)),
                    "ffn_up.weight"   => Some(WeightId::VisionBlkFfnUpWeight(layer)),
                    "ffn_up.bias"     => Some(WeightId::VisionBlkFfnUpBias(layer)),
                    "ffn_down.weight" => Some(WeightId::VisionBlkFfnDownWeight(layer)),
                    "ffn_down.bias"   => Some(WeightId::VisionBlkFfnDownBias(layer)),
                    _ => None,
                };
            }
        }
    }

    // ── Resampler weights: prefix "resampler." ────────────────────────────────
    if let Some(rest) = name.strip_prefix("resampler.") {
        return match rest {
            "query"           => Some(WeightId::ResamplerQuery),
            "kv.weight"       => Some(WeightId::ResamplerKvWeight),
            "ln_q.weight"     => Some(WeightId::ResamplerLnQWeight),
            "ln_q.bias"       => Some(WeightId::ResamplerLnQBias),
            "ln_kv.weight"    => Some(WeightId::ResamplerLnKvWeight),
            "ln_kv.bias"      => Some(WeightId::ResamplerLnKvBias),
            "attn.q.weight"   => Some(WeightId::ResamplerAttnQWeight),
            "attn.q.bias"     => Some(WeightId::ResamplerAttnQBias),
            "attn.k.weight"   => Some(WeightId::ResamplerAttnKWeight),
            "attn.k.bias"     => Some(WeightId::ResamplerAttnKBias),
            "attn.v.weight"   => Some(WeightId::ResamplerAttnVWeight),
            "attn.v.bias"     => Some(WeightId::ResamplerAttnVBias),
            "attn.out.weight" => Some(WeightId::ResamplerAttnOutWeight),
            "attn.out.bias"   => Some(WeightId::ResamplerAttnOutBias),
            "pos_embed_k"     => Some(WeightId::ResamplerPosEmbedK),
            "ln_post.weight"  => Some(WeightId::ResamplerLnPostWeight),
            "ln_post.bias"    => Some(WeightId::ResamplerLnPostBias),
            "proj.weight"     => Some(WeightId::ResamplerProjWeight),
            _ => None,
        };
    }

    None
}

/// All WeightId variants for a single ViT block layer.
fn vit_block_weight_ids(layer: usize) -> Vec<WeightId> {
    vec![
        WeightId::VisionBlkAttnQWeight(layer),
        WeightId::VisionBlkAttnQBias(layer),
        WeightId::VisionBlkAttnKWeight(layer),
        WeightId::VisionBlkAttnKBias(layer),
        WeightId::VisionBlkAttnVWeight(layer),
        WeightId::VisionBlkAttnVBias(layer),
        WeightId::VisionBlkAttnOutWeight(layer),
        WeightId::VisionBlkAttnOutBias(layer),
        WeightId::VisionBlkLn1Weight(layer),
        WeightId::VisionBlkLn1Bias(layer),
        WeightId::VisionBlkLn2Weight(layer),
        WeightId::VisionBlkLn2Bias(layer),
        WeightId::VisionBlkFfnUpWeight(layer),
        WeightId::VisionBlkFfnUpBias(layer),
        WeightId::VisionBlkFfnDownWeight(layer),
        WeightId::VisionBlkFfnDownBias(layer),
    ]
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vision_tensor_name_static() {
        assert_eq!(parse_vision_tensor_name("v.patch_embd.weight"), Some(WeightId::VisionPatchEmbedWeight));
        assert_eq!(parse_vision_tensor_name("v.patch_embd.bias"),   Some(WeightId::VisionPatchEmbedBias));
        assert_eq!(parse_vision_tensor_name("v.position_embd.weight"), Some(WeightId::VisionPosEmbedWeight));
        assert_eq!(parse_vision_tensor_name("v.post_ln.weight"),    Some(WeightId::VisionPostNormWeight));
        assert_eq!(parse_vision_tensor_name("v.post_ln.bias"),      Some(WeightId::VisionPostNormBias));
    }

    #[test]
    fn test_parse_vision_tensor_name_vit_blocks() {
        assert_eq!(parse_vision_tensor_name("v.blk.0.attn_q.weight"), Some(WeightId::VisionBlkAttnQWeight(0)));
        assert_eq!(parse_vision_tensor_name("v.blk.26.attn_q.bias"),  Some(WeightId::VisionBlkAttnQBias(26)));
        assert_eq!(parse_vision_tensor_name("v.blk.3.ln1.weight"),    Some(WeightId::VisionBlkLn1Weight(3)));
        assert_eq!(parse_vision_tensor_name("v.blk.7.ffn_down.bias"), Some(WeightId::VisionBlkFfnDownBias(7)));
        assert_eq!(parse_vision_tensor_name("v.blk.12.attn_out.weight"), Some(WeightId::VisionBlkAttnOutWeight(12)));
    }

    #[test]
    fn test_parse_vision_tensor_name_resampler() {
        assert_eq!(parse_vision_tensor_name("resampler.query"),           Some(WeightId::ResamplerQuery));
        assert_eq!(parse_vision_tensor_name("resampler.kv.weight"),       Some(WeightId::ResamplerKvWeight));
        assert_eq!(parse_vision_tensor_name("resampler.attn.q.weight"),   Some(WeightId::ResamplerAttnQWeight));
        assert_eq!(parse_vision_tensor_name("resampler.attn.out.weight"), Some(WeightId::ResamplerAttnOutWeight));
        assert_eq!(parse_vision_tensor_name("resampler.pos_embed_k"),     Some(WeightId::ResamplerPosEmbedK));
        assert_eq!(parse_vision_tensor_name("resampler.proj.weight"),     Some(WeightId::ResamplerProjWeight));
    }

    #[test]
    fn test_parse_vision_tensor_name_unknown_is_none() {
        assert_eq!(parse_vision_tensor_name("clip.vision.something"), None);
        assert_eq!(parse_vision_tensor_name("blk.0.attn_q.weight"),   None); // missing "v." prefix
        assert_eq!(parse_vision_tensor_name(""),                       None);
    }
}
