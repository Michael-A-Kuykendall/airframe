//! Shared test helpers for shimmy-console tests

use shimmy_console::ToolArgs;

/// Build a ToolArgs from key-value string pairs
pub fn make_str_args(pairs: &[(&str, &str)]) -> ToolArgs {
    let mut args = ToolArgs::new();
    for (k, v) in pairs {
        args.args.insert(k.to_string(), serde_json::Value::String(v.to_string()));
    }
    args
}

/// Build a ToolArgs from key-value JSON pairs
pub fn make_json_args(pairs: &[(&str, serde_json::Value)]) -> ToolArgs {
    let mut args = ToolArgs::new();
    for (k, v) in pairs {
        args.args.insert(k.to_string(), v.clone());
    }
    args
}
