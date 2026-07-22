//! Canonical GGML/GGUF quant dequantization formulas — the spec-cited math core.
//!
//! # AUTHORITATIVE SOURCE
//!
//! Every formula here is transcribed from the **GGML quantization specification**
//! (ggerganov/ggml, `ggml.h` — block type definitions and `dequantize_row_*`).
//! This is the referenced mathematical core the inference stack stands on.
//! It is **NOT** derived from candle / llama.cpp / airframe-CPU output — those
//! are at most optional cross-checks (see `docs/beads-fabric-core.md` golden rule 1).
//!
//! The registry is keyed by the raw GGML type id
//! (0=F32, 1=F16, 2=Q4_0, 6=Q5_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K).
//! Each entry carries the canonical element dequant function plus a doc-comment
//! citing the exact block layout from the GGML spec.
//!
//! This module is the single source of truth the dispatch control plane hangs off
//! of (see `airframe::core::routing::ModelRoutePlan`). The WGSL `if qt==` ladder in
//! `sh_layer_v1.wgsl::dequant_dispatch` is replaced by the `FormulaSlot` this
//! registry assigns (B3b).

/// Minimal, dependency-free IEEE-754 fp16 → fp32 conversion.
///
/// Mirrors GGML's `GGML_FP16_TO_FP32`. Inlined so this module stays
/// self-contained (no `half` dependency in the observation crate).
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;

    let sign_bit = (sign as u32) << 31;

    if exp == 0 {
        if mant == 0 {
            // zero (signed)
            f32::from_bits(sign_bit)
        } else {
            // subnormal: (-1)^sign * 2^(-14) * (mant / 2^10)
            let val = (mant as f32) * 2f32.powi(-24);
            if sign == 1 {
                -val
            } else {
                val
            }
        }
    } else if exp == 0x1f {
        // inf / nan
        let m = (mant as u32) << 13;
        f32::from_bits(sign_bit | 0x7f80_0000 | m)
    } else {
        // normal: rebuild with fp32 bias (127) and shift mantissa up 13 bits
        let e = (exp as i32 + (127 - 15)) as u32;
        let m = (mant as u32) << 13;
        f32::from_bits(sign_bit | (e << 23) | m)
    }
}

/// Read `n` little-endian bytes from a block slice as a uNN.
#[inline]
fn rd_u16(block: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([block[off], block[off + 1]])
}

#[inline]
fn rd_u32(block: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([block[off], block[off + 1], block[off + 2], block[off + 3]])
}

/// Unpack the K-quant 6-bit scale (sc) and min (m) pair for scale index `j`.
///
/// GGML `get_scale_min_k4` — the 12-byte `scales` array packs 8 six-bit
/// (sign+magnitude) values across 12 bytes; indices 0..4 are stored directly,
/// indices 4..8 borrow the top 2 bits of their lower-/upper-neighbors. (GGML spec.)
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u32, u32) {
    if j < 4 {
        let sc = (scales[j] & 63) as u32;
        let m = (scales[j + 4] & 63) as u32;
        (sc, m)
    } else {
        let sc = ((scales[j + 4] & 0x0f) as u32) | (((scales[j - 4] >> 6) & 3) as u32) << 4;
        let m = ((scales[j + 4] >> 4) & 0x0f) as u32 | (((scales[j] >> 6) & 3) as u32) << 4;
        (sc, m)
    }
}

/// Canonical dequant function: given the raw bytes of ONE quant block and the
/// element index within that block (0-based), return the dequantized f32.
pub type DequantFn = fn(block: &[u8], elem_in_block: usize) -> f32;

/// A single canonical quant formula entry, spec-cited and auditable.
pub struct QuantFormula {
    /// Raw GGML type id (0=F32, 1=F16, 2=Q4_0, 6=Q5_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K).
    pub type_id: u32,
    /// Symbolic name.
    pub name: &'static str,
    /// Elements per (super)block.
    pub block_elems: usize,
    /// Bytes per (super)block.
    pub block_bytes: usize,
    /// Canonical element dequant function (GGML-spec math).
    pub dequant: DequantFn,
}

// ─────────────────────────────────────────────────────────────────────────────
// F32 (type 0) — direct fp32, 4 bytes per element (no quantization).
// GGML: GGML_TYPE_F32, block = 1 element, 4 bytes.
// ─────────────────────────────────────────────────────────────────────────────
fn dequant_f32(block: &[u8], elem: usize) -> f32 {
    f32::from_le_bytes([
        block[elem * 4],
        block[elem * 4 + 1],
        block[elem * 4 + 2],
        block[elem * 4 + 3],
    ])
}

// ─────────────────────────────────────────────────────────────────────────────
// F16 (type 1) — fp16, 2 bytes per element.
// GGML: GGML_TYPE_F16, block = 1 element, 2 bytes.
// ─────────────────────────────────────────────────────────────────────────────
fn dequant_f16(block: &[u8], elem: usize) -> f32 {
    f16_to_f32(rd_u16(block, elem * 2))
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_0 (type 2) — 32 elements/block, 18 bytes/block.
// Layout: [0..1] d (fp16 scale); [2..18] qs (16 bytes, 2 nibbles/byte).
// GGML: val = (nibble - 8) * d, nibble ∈ 0..15.
// ─────────────────────────────────────────────────────────────────────────────
fn dequant_q4_0(block: &[u8], elem: usize) -> f32 {
    let d = f16_to_f32(rd_u16(block, 0));
    let qs_byte = block[2 + (elem % 16)];
    let nib = if elem < 16 {
        qs_byte & 0x0f
    } else {
        qs_byte >> 4
    };
    (nib as f32 - 8.0) * d
}

// ─────────────────────────────────────────────────────────────────────────────
// Q5_0 (type 6) — 32 elements/block, 22 bytes/block.
// Layout: [0..1] d (fp16); [2..6] qh (u32, bit i = 5th bit of element i);
//         [6..22] qs (16 bytes, low nibble=elem<16, high nibble=elem>=16).
// GGML: val = (qs + (qh>>i & 1)*16 - 16) * d.
// ─────────────────────────────────────────────────────────────────────────────
fn dequant_q5_0(block: &[u8], elem: usize) -> f32 {
    let d = f16_to_f32(rd_u16(block, 0));
    let qh = rd_u32(block, 2);
    let high_bit = (qh >> elem) & 1;
    let qs_byte = block[6 + (elem % 16)];
    let low = if elem < 16 {
        qs_byte & 0x0f
    } else {
        qs_byte >> 4
    };
    let val5 = low as u32 | (high_bit << 4);
    (val5 as f32 - 16.0) * d
}

// ─────────────────────────────────────────────────────────────────────────────
// Q8_0 (type 8) — 32 elements/block, 34 bytes/block.
// Layout: [0..1] d (fp16 scale); [2..34] qs (32 int8 values).
// GGML: val = qs[i] * d.
// ─────────────────────────────────────────────────────────────────────────────
fn dequant_q8_0(block: &[u8], elem: usize) -> f32 {
    let d = f16_to_f32(rd_u16(block, 0));
    let raw = block[2 + elem] as i8;
    d * raw as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Q4_K (type 12) — 256 elements/superblock, 144 bytes/superblock.
// Layout: [0..1] d (fp16); [2..3] dmin (fp16); [4..16] scales (12 bytes);
//         [16..144] qs (128 nibble bytes).
// GGML: 4 groups of 64; per 32-elem sub-block, sc=d*scale_k4(is).x,
//       m=dmin*scale_k4(is).y; val = sc*nibble - m.
// ─────────────────────────────────────────────────────────────────────────────
fn dequant_q4_k(block: &[u8], elem: usize) -> f32 {
    let d = f16_to_f32(rd_u16(block, 0));
    let dmin = f16_to_f32(rd_u16(block, 2));
    let scales = &block[4..16];
    let qs = &block[16..144];

    let group = elem / 64;
    let in_grp = elem % 64;
    let sub = in_grp / 32;
    let l = in_grp % 32;
    let is = group * 2 + sub;

    let (sc, m) = get_scale_min_k4(is, scales);
    let sc_val = d * sc as f32;
    let m_val = dmin * m as f32;

    let qs_byte = qs[group * 32 + l];
    let nibble = if sub == 0 {
        qs_byte & 0x0f
    } else {
        qs_byte >> 4
    };
    sc_val * nibble as f32 - m_val
}

// ─────────────────────────────────────────────────────────────────────────────
// Q5_K (type 13) — 256 elements/superblock, 176 bytes/superblock.
// Layout: [0..1] d; [2..3] dmin; [4..16] scales (12B);
//         [16..48] qh (32B, high bit = (qh[l] >> (elem/32)) & 1);
//         [48..176] qs (128B, low nibble per element).
// GGML: q5 = nibble | (high_bit<<4); val = d*sc*q5 - dmin*m.
// ─────────────────────────────────────────────────────────────────────────────
fn dequant_q5_k(block: &[u8], elem: usize) -> f32 {
    let d = f16_to_f32(rd_u16(block, 0));
    let dmin = f16_to_f32(rd_u16(block, 2));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let group = elem / 64;
    let in_grp = elem % 64;
    let sub = in_grp / 32;
    let l = in_grp % 32;
    let is = group * 2 + sub;

    let (sc, m) = get_scale_min_k4(is, scales);
    let sc_val = d * sc as f32;
    let m_val = dmin * m as f32;

    let qs_byte = qs[group * 32 + l];
    let nibble = if sub == 0 {
        qs_byte & 0x0f
    } else {
        qs_byte >> 4
    };

    let high_bit = (qh[l] >> (elem / 32)) & 1;
    let q5 = nibble as u32 | (high_bit as u32) << 4;
    sc_val * q5 as f32 - m_val
}

// ─────────────────────────────────────────────────────────────────────────────
// Q6_K (type 14) — 256 elements/superblock, 210 bytes/superblock.
// Layout: [0..128] ql (low 4 bits); [128..192] qh (high 2 bits);
//         [192..208] scales (16 int8); [208..210] d (fp16).
// GGML: 6-bit signed value q6 = ql_low | (qh_bits<<4), range -32..31;
//       val = d * sc[q6_scale_idx] * q6.
// ─────────────────────────────────────────────────────────────────────────────
fn dequant_q6_k(block: &[u8], elem: usize) -> f32 {
    let d = f16_to_f32(rd_u16(block, 208));

    let half = elem / 128; // 0 or 1
    let half_e = elem % 128;
    let l = half_e % 32;
    let quarter = half_e / 32; // 0..3

    // ql: quarters 0&2 share ql[half*64 + l]; quarters 1&3 share ql[half*64+32+l]
    let ql_rel = if quarter == 0 || quarter == 2 {
        half * 64 + l
    } else {
        half * 64 + l + 32
    };
    let ql_byte = block[ql_rel];
    let lower4 = if quarter < 2 {
        ql_byte & 0x0f
    } else {
        ql_byte >> 4
    };

    // qh: one byte per l within a half
    let qh_byte = block[128 + half * 32 + l];
    let upper2 = (qh_byte >> (quarter * 2)) & 3;

    let q6 = lower4 as i32 | ((upper2 as i32) << 4); // 0..63
    let signed_q = q6 - 32; // -32..31

    // scale index: 16 int8 scales, 8 per half
    let sc_idx = half * 8 + (l / 16) + quarter * 2;
    let sc_raw = block[192 + sc_idx] as i8;

    d * sc_raw as f32 * signed_q as f32
}

/// The ordered, spec-cited registry. Array index == `FormulaSlot` discriminant.
///
/// The slot is intentionally distinct from the GGML type id so the registry —
/// not the GGML numbering — is the single source of truth for dispatch.
pub const QUANT_FORMULAS: &[QuantFormula] = &[
    QuantFormula {
        type_id: 0,
        name: "F32",
        block_elems: 1,
        block_bytes: 4,
        dequant: dequant_f32,
    },
    QuantFormula {
        type_id: 1,
        name: "F16",
        block_elems: 1,
        block_bytes: 2,
        dequant: dequant_f16,
    },
    QuantFormula {
        type_id: 2,
        name: "Q4_0",
        block_elems: 32,
        block_bytes: 18,
        dequant: dequant_q4_0,
    },
    QuantFormula {
        type_id: 6,
        name: "Q5_0",
        block_elems: 32,
        block_bytes: 22,
        dequant: dequant_q5_0,
    },
    QuantFormula {
        type_id: 8,
        name: "Q8_0",
        block_elems: 32,
        block_bytes: 34,
        dequant: dequant_q8_0,
    },
    QuantFormula {
        type_id: 12,
        name: "Q4_K",
        block_elems: 256,
        block_bytes: 144,
        dequant: dequant_q4_k,
    },
    QuantFormula {
        type_id: 13,
        name: "Q5_K",
        block_elems: 256,
        block_bytes: 176,
        dequant: dequant_q5_k,
    },
    QuantFormula {
        type_id: 14,
        name: "Q6_K",
        block_elems: 256,
        block_bytes: 210,
        dequant: dequant_q6_k,
    },
];

/// Stable dispatch slot the shader consumes. Distinct from GGML type id so the
/// registry owns the mapping (B3b retires the WGSL `if qt==` ladder).
// GGML naming convention uses underscores (Q4_0, Q6_K, etc.)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FormulaSlot {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q5_0 = 3,
    Q8_0 = 4,
    Q4_K = 5,
    Q5_K = 6,
    Q6_K = 7,
}

impl FormulaSlot {
    /// Raw `u32` discriminant (the value passed to shaders as `formula_index`).
    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// Resolve a GGML type id to its registry slot (if supported).
pub fn slot_for_type(type_id: u32) -> Option<FormulaSlot> {
    match type_id {
        0 => Some(FormulaSlot::F32),
        1 => Some(FormulaSlot::F16),
        2 => Some(FormulaSlot::Q4_0),
        6 => Some(FormulaSlot::Q5_0),
        8 => Some(FormulaSlot::Q8_0),
        12 => Some(FormulaSlot::Q4_K),
        13 => Some(FormulaSlot::Q5_K),
        14 => Some(FormulaSlot::Q6_K),
        _ => None,
    }
}

/// Look up the canonical formula entry for a GGML type id.
pub fn formula_for_type(type_id: u32) -> Option<&'static QuantFormula> {
    QUANT_FORMULAS.iter().find(|f| f.type_id == type_id)
}

/// Number of elements in one (super)block of the given GGML type.
pub fn block_elems(type_id: u32) -> Option<usize> {
    formula_for_type(type_id).map(|f| f.block_elems)
}

/// Bytes per (super)block of the given GGML type.
pub fn block_bytes(type_id: u32) -> Option<usize> {
    formula_for_type(type_id).map(|f| f.block_bytes)
}

/// Dequantize exactly one element of a tensor of the given GGML type.
///
/// `block_base` must point at the start of the element's quant (super)block;
/// `elem_in_block` is the 0-based index within that block. Returns `None` for
/// unsupported types.
pub fn dequant_elem(type_id: u32, block: &[u8], elem_in_block: usize) -> Option<f32> {
    let f = formula_for_type(type_id)?;
    Some((f.dequant)(block, elem_in_block))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-computed Q6_K block (210 bytes) — verifies the registry fn matches
    // the GGML spec math independently of any shader / external engine.
    //
    // Construct a degenerate-but-valid block:
    //   d = fp16(1.0)            -> bytes [208..210] = 0x3C00
    //   all ql high nibble=0, low nibble=0  -> q6 = 0 -> signed_q = -32
    //   all scales = 1 (int8)    -> sc = 1
    //   => val = d * sc * signed_q = 1.0 * 1 * -32 = -32.0  for every element
    #[test]
    fn q6_k_hand_computed_block() {
        let mut block = vec![0u8; 210];
        // d = 1.0 fp16
        block[208] = 0x00;
        block[209] = 0x3c;
        // scales = 1
        for i in 192..208 {
            block[i] = 1;
        }
        // ql/qh all zero -> q6 = 0 -> signed_q = -32
        let f = formula_for_type(14).expect("Q6_K registered");
        for e in 0..256 {
            let v = (f.dequant)(&block, e);
            assert!((v + 32.0).abs() < 1e-6, "elem {} = {}", e, v);
        }
    }

    // Distinct Q6_K block: scale byte at index 192 (half 0, quarter 0, l 0) = 2,
    // ql byte 0 = 0x01 (low nibble = 1 -> q6 = 1, signed = 1-32 = -31),
    // d = 1.0.
    //   elem 0: sc=2 -> 1.0 * 2 * (-31) = -62.0
    //   elem 1: ql[1]=0 -> signed -32, sc idx 192=2 -> -64.0
    #[test]
    fn q6_k_scale_index_selects_correct_subblock() {
        let mut block = vec![0u8; 210];
        block[208] = 0x00;
        block[209] = 0x3c; // d = 1.0
        for i in 192..208 {
            block[i] = 1;
        }
        block[192] = 2; // scale for (half0, quarter0, l0) group
        block[0] = 0x01; // ql[0]: low nibble = 1
        let f = formula_for_type(14).expect("Q6_K registered");
        assert!(
            ((f.dequant)(&block, 0) + 62.0).abs() < 1e-6,
            "got {}",
            (f.dequant)(&block, 0)
        );
        assert!(
            ((f.dequant)(&block, 1) + 64.0).abs() < 1e-6,
            "got {}",
            (f.dequant)(&block, 1)
        );
    }

    #[test]
    fn f32_roundtrip() {
        let mut block = 3.14159f32.to_le_bytes().to_vec();
        block.resize(4, 0);
        let v = dequant_elem(0, &block, 0).unwrap();
        assert!((v - 3.14159).abs() < 1e-6);
    }

    #[test]
    fn q4_0_known_value() {
        // d = 1.0 fp16; qs[0]=0x0A (low nibble 10 -> (10-8)*1 = 2.0)
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3c;
        block[2] = 0x0a;
        let v = dequant_elem(2, &block, 0).unwrap();
        assert!((v - 2.0).abs() < 1e-6, "got {}", v);
    }

    #[test]
    fn slot_mapping_covers_all_eight() {
        for (tid, slot) in [
            (0u32, 0u32),
            (1, 1),
            (2, 2),
            (6, 3),
            (8, 4),
            (12, 5),
            (13, 6),
            (14, 7),
        ] {
            let s = slot_for_type(tid).expect("supported");
            assert_eq!(s.as_u32(), slot);
            assert_eq!(formula_for_type(tid).unwrap().type_id, tid);
        }
        assert!(slot_for_type(99).is_none());
    }

    #[test]
    fn f16_helper_matches_reference() {
        // 0x3C00 = 1.0, 0xC000 = -2.0, 0x3E00 = 1.5
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3E00), 1.5);
    }
}
