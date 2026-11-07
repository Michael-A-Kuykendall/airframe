//! Dense attention query/context live ranges. Mirrors V1 WGSL max(dim, Q-width).
pub(super) fn query_offset(hidden: u32, query_width: u32, stride: u32) -> Result<u32, String> {
    if hidden == 0 || query_width == 0 {
        return Err("attention scratch requires nonzero dimensions".into());
    }
    let offset = hidden.max(query_width);
    let end = offset
        .checked_add(query_width)
        .ok_or("attention query scratch overflow")?;
    if end > stride {
        return Err("attention query scratch exceeds token stride".into());
    }
    Ok(offset)
}

#[cfg(test)]
mod tests {
    use super::query_offset;

    #[test]
    fn query_and_context_are_disjoint_for_supported_geometries() {
        for (hidden, query, stride, expected) in [
            (1024, 2048, 11264, 2048), // Qwen3-0.6B: old query/context overlap.
            (3072, 3072, 28672, 3072), // Llama: unchanged query placement.
            (2304, 2048, 28672, 2304), // Geometry only; not Gemma certification.
            (256, 16384, 32768, 16384),
        ] {
            let start = query_offset(hidden, query, stride).unwrap();
            assert_eq!(start, expected);
            assert!(start >= query, "context writes overlap live query reads");
            assert!(start >= hidden, "QKV writes overlap normalized input");
            assert!(start + query <= stride);
        }
    }

    #[test]
    fn invalid_or_insufficient_scratch_fails_before_dispatch() {
        assert!(query_offset(0, 128, 1024).is_err());
        assert!(query_offset(128, 0, 1024).is_err());
        assert!(query_offset(1024, 2048, 3072).is_err());
        assert!(query_offset(1, u32::MAX, u32::MAX).is_err());
    }

    #[test]
    fn disjoint_live_ranges_hold_across_width_boundaries() {
        for hidden in [1, 127, 128, 255, 256, 1024, 2048, 3072] {
            for query in [1, 127, 128, 255, 256, 1024, 2048, 4096] {
                let stride = hidden + 2 * query;
                let offset = query_offset(hidden, query, stride).unwrap();
                assert!((0..query).all(|out| out < offset || out >= offset + query));
                assert_eq!(offset, if query > hidden { query } else { hidden });
            }
        }
    }
}
