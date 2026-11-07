//! Diagnostic selection only; no inference mathematics or dispatch selection.
pub(super) fn layer_matches(value: &str, layer: usize) -> Result<bool, String> {
    if value == "all" {
        return Ok(true);
    }
    value
        .parse::<usize>()
        .map(|selected| selected == layer)
        .map_err(|_| "invalid AIRFRAME_CAPTURE_FULL_LAYER".into())
}
pub(super) fn position_matches(value: Option<&str>, position: u32) -> Result<bool, String> {
    value
        .map(|v| {
            v.parse::<u32>()
                .map(|selected| selected == position)
                .map_err(|_| "invalid AIRFRAME_CAPTURE_POSITION".into())
        })
        .unwrap_or(Ok(true))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_capture_is_explicitly_selected() {
        assert!(layer_matches("all", 27).unwrap());
        assert!(layer_matches("0", 0).unwrap());
        assert!(!layer_matches("0", 1).unwrap());
        assert!(layer_matches("", 0).is_err());
        assert!(layer_matches("-1", 0).is_err());
    }
    #[test]
    fn position_filter_is_checked_and_default_preserves_capture() {
        assert!(position_matches(None, 2031).unwrap());
        assert!(position_matches(Some("2031"), 2031).unwrap());
        assert!(!position_matches(Some("2031"), 63).unwrap());
        assert!(position_matches(Some("4294967296"), 0).is_err());
    }
}
