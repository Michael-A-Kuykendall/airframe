//! Mathematical-certainty layout gate for CPU→GPU uniform structs.
//!
//! Every Rust `#[repr(C)]` params struct that is uploaded to the GPU
//! (`bytemuck::bytes_of`) is read back on the GPU side through a WGSL `struct`
//! declaration. If the field ORDER (hence byte offset) diverges between the two,
//! the shader silently reads the wrong bytes — e.g. B3b's NaN, where the WGSL
//! `formula_*` fields sat in the middle of the struct while Rust had them at the
//! end, so every field after them read a wrong offset.
//!
//! This test PROVES equality by computation rather than by running inference:
//!   * Rust side  — `core::mem::offset_of!` gives the repr(C) byte offset per field.
//!   * WGSL side  — naga parses the shader and reports each member's byte offset
//!                  per the WGSL layout rules.
//! We assert the two `(name, offset)` sequences are identical, field for field.
//!
//! All members of these structs are 4-byte scalars (u32/f32), so both layouts
//! are dense (offset[i] == 4*i) and the equality is exact and unambiguous.

use airframe::backend::bindless::pipeline::{DequantAnyParams, HeadBlobParams, LayerParams};
use core::mem::offset_of;

/// Extract the ordered `(member_name, byte_offset)` list for a named struct
/// from a WGSL source, using naga's spec-compliant layout computation.
fn wgsl_member_offsets(src: &str, struct_name: &str) -> Vec<(String, u32)> {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("naga failed to parse WGSL: {e:?}"));

    for (_handle, ty) in module.types.iter() {
        if ty.name.as_deref() == Some(struct_name) {
            if let naga::TypeInner::Struct { members, .. } = &ty.inner {
                return members
                    .iter()
                    .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
                    .collect();
            }
            panic!("`{struct_name}` in WGSL is not a struct");
        }
    }
    panic!("`{struct_name}` not found in WGSL source");
}

/// Assert a Rust struct's `offset_of!` layout matches the WGSL struct exactly.
fn assert_layout_matches(label: &str, wgsl_src: &str, wgsl_struct: &str, rust: &[(&str, u32)]) {
    let wgsl = wgsl_member_offsets(wgsl_src, wgsl_struct);

    assert_eq!(
        wgsl.len(),
        rust.len(),
        "{label}: field count differs — WGSL has {}, Rust has {}\nWGSL: {:#?}",
        wgsl.len(),
        rust.len(),
        wgsl,
    );

    for (i, ((w_name, w_off), (r_name, r_off))) in wgsl.iter().zip(rust.iter()).enumerate() {
        assert_eq!(
            w_name, r_name,
            "{label}: field #{i} name mismatch — WGSL `{w_name}` vs Rust `{r_name}`"
        );
        assert_eq!(
            *w_off, *r_off,
            "{label}: field `{w_name}` offset mismatch — WGSL {w_off} vs Rust {r_off}"
        );
    }
}

const SH_LAYER_V1: &str = include_str!("../src/backend/bindless/sh_layer_v1.wgsl");
const SH_LAYER_V1_INT4: &str = include_str!("../src/backend/bindless/sh_layer_v1_int4.wgsl");
const SH_HEAD_BLOB: &str = include_str!("../src/backend/bindless/sh_head_blob.wgsl");
const SH_DEQUANT_ANY: &str = include_str!("../src/backend/bindless/sh_dequant_any.wgsl");

fn layer_params_rust_layout() -> Vec<(&'static str, u32)> {
    vec![
        ("dim", offset_of!(LayerParams, dim) as u32),
        ("head_count", offset_of!(LayerParams, head_count) as u32),
        (
            "head_count_kv",
            offset_of!(LayerParams, head_count_kv) as u32,
        ),
        ("head_dim", offset_of!(LayerParams, head_dim) as u32),
        ("rope_dim", offset_of!(LayerParams, rope_dim) as u32),
        ("rms_eps", offset_of!(LayerParams, rms_eps) as u32),
        ("ffn_dim", offset_of!(LayerParams, ffn_dim) as u32),
        ("temp_stride", offset_of!(LayerParams, temp_stride) as u32),
        ("quant_qk", offset_of!(LayerParams, quant_qk) as u32),
        ("quant_v", offset_of!(LayerParams, quant_v) as u32),
        (
            "quant_attn_out",
            offset_of!(LayerParams, quant_attn_out) as u32,
        ),
        (
            "quant_ffn_down",
            offset_of!(LayerParams, quant_ffn_down) as u32,
        ),
        (
            "quant_ffn_gate",
            offset_of!(LayerParams, quant_ffn_gate) as u32,
        ),
        ("quant_ffn_up", offset_of!(LayerParams, quant_ffn_up) as u32),
        (
            "attn_logit_softcap",
            offset_of!(LayerParams, attn_logit_softcap) as u32,
        ),
        (
            "post_norm_enabled",
            offset_of!(LayerParams, post_norm_enabled) as u32,
        ),
        (
            "qk_norm_enabled",
            offset_of!(LayerParams, qk_norm_enabled) as u32,
        ),
        (
            "layer_norm_enabled",
            offset_of!(LayerParams, layer_norm_enabled) as u32,
        ),
        (
            "ffn_kind_policy",
            offset_of!(LayerParams, ffn_kind_policy) as u32,
        ),
        (
            "qkv_layout_policy",
            offset_of!(LayerParams, qkv_layout_policy) as u32,
        ),
        ("batch_offset", offset_of!(LayerParams, batch_offset) as u32),
        ("batch_count", offset_of!(LayerParams, batch_count) as u32),
        ("q_weight_k", offset_of!(LayerParams, q_weight_k) as u32),
        ("k_weight_k", offset_of!(LayerParams, k_weight_k) as u32),
        ("formula_qk", offset_of!(LayerParams, formula_qk) as u32),
        ("formula_v", offset_of!(LayerParams, formula_v) as u32),
        (
            "formula_attn_out",
            offset_of!(LayerParams, formula_attn_out) as u32,
        ),
        (
            "formula_ffn_down",
            offset_of!(LayerParams, formula_ffn_down) as u32,
        ),
        (
            "formula_ffn_gate",
            offset_of!(LayerParams, formula_ffn_gate) as u32,
        ),
        (
            "formula_ffn_up",
            offset_of!(LayerParams, formula_ffn_up) as u32,
        ),
    ]
}

#[test]
fn layer_params_matches_sh_layer_v1() {
    assert_layout_matches(
        "LayerParams vs sh_layer_v1.wgsl",
        SH_LAYER_V1,
        "LayerParams",
        &layer_params_rust_layout(),
    );
}

#[test]
fn layer_params_matches_sh_layer_v1_int4() {
    assert_layout_matches(
        "LayerParams vs sh_layer_v1_int4.wgsl",
        SH_LAYER_V1_INT4,
        "LayerParams",
        &layer_params_rust_layout(),
    );
}

#[test]
fn head_blob_params_matches_sh_head_blob() {
    let rust = vec![
        ("vocab_size", offset_of!(HeadBlobParams, vocab_size) as u32),
        ("dim", offset_of!(HeadBlobParams, dim) as u32),
        ("weight_off", offset_of!(HeadBlobParams, weight_off) as u32),
        (
            "formula_index",
            offset_of!(HeadBlobParams, formula_index) as u32,
        ),
        ("softcap", offset_of!(HeadBlobParams, softcap) as u32),
        ("base_row", offset_of!(HeadBlobParams, base_row) as u32),
        ("_pad", offset_of!(HeadBlobParams, _pad) as u32),
    ];
    assert_layout_matches(
        "HeadBlobParams vs sh_head_blob.wgsl",
        SH_HEAD_BLOB,
        "HeadBlobParams",
        &rust,
    );
}

#[test]
fn dequant_any_params_matches_sh_dequant_any() {
    let rust = vec![
        (
            "offset_bytes",
            offset_of!(DequantAnyParams, offset_bytes) as u32,
        ),
        ("count", offset_of!(DequantAnyParams, count) as u32),
        (
            "formula_index",
            offset_of!(DequantAnyParams, formula_index) as u32,
        ),
        ("pad", offset_of!(DequantAnyParams, pad) as u32),
    ];
    assert_layout_matches(
        "DequantAnyParams vs sh_dequant_any.wgsl",
        SH_DEQUANT_ANY,
        "DequantAnyParams",
        &rust,
    );
}
