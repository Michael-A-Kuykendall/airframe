use half::f16;

#[inline]
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_subnormal_min() {
        // 0x0001 = smallest FP16 subnormal = 2^-24 ≈ 5.96046448e-8
        let v = f16_bits_to_f32(0x0001);
        assert!((v - 5.96046448e-8).abs() < 1e-12);
    }

    #[test]
    fn test_f16_subnormal_max() {
        // 0x03FF = largest FP16 subnormal = (1023/1024) * 2^-14 = 1023 * 2^-24
        let v = f16_bits_to_f32(0x03FF);
        assert!((v - 6.09755516e-5).abs() < 1e-10);
    }

    #[test]
    fn test_f16_smallest_normal() {
        // 0x0400 = smallest FP16 normal = 2^-14 ≈ 6.1035156e-5
        let v = f16_bits_to_f32(0x0400);
        assert!((v - 6.1035156e-5).abs() < 1e-10);
    }

    #[test]
    fn test_f16_basic_values() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x8000), -0.0);
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0xBC00), -1.0);
        assert_eq!(f16_bits_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_bits_to_f32(0x7E00).is_nan());
    }
}
