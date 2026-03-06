use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Produce a canonical, deterministic projection string from arbitrary posture text.
///
/// Behavior:
/// - If `posture_text` parses as JSON, recursively sort object keys and serialize with
///   `serde_json::to_string` to produce a stable representation.
/// - Otherwise, collapse whitespace and normalize newlines to produce a stable string.
pub fn canonical_projection(posture_text: &str) -> String {
    // First try JSON parse & canonicalize
    if let Ok(val) = serde_json::from_str::<Value>(posture_text) {
        let canon = canonicalize_json(val);
        return serde_json::to_string(&canon).expect("canonical serialization should succeed");
    }

    // Fallback: collapse whitespace and normalize newlines
    let collapsed = posture_text
        .replace("\r\n", "\n")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    collapsed
}

fn canonicalize_json(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut bmap = BTreeMap::new();
            for (k, v) in map.into_iter() {
                bmap.insert(k, canonicalize_json(v));
            }
            // Build a Map from ordered BTreeMap
            let mut ordered = Map::new();
            for (k, v) in bmap.into_iter() {
                ordered.insert(k, v);
            }
            Value::Object(ordered)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(canonicalize_json).collect()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn test_projection_hash_stable_same_input() {
        let input = "{ \"b\": 2, \"a\": 1 }";
        let p1 = canonical_projection(input);
        let p2 = canonical_projection(input);
        assert_eq!(
            p1, p2,
            "canonical projection must be identical for same input"
        );

        let h1 = Sha256::digest(p1.as_bytes());
        let h2 = Sha256::digest(p2.as_bytes());
        assert_eq!(h1[..], h2[..], "hashes must match");
    }

    #[test]
    fn test_projection_canonicalizes_whitespace() {
        let a = "Hello\n\n  world";
        let b = "Hello world";
        assert_eq!(canonical_projection(a), canonical_projection(b));
    }
}
