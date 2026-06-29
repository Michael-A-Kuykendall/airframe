//! Integration tests for shimmy-console
//!
//! These tests verify real behavior against the actual API.
//! No phantom methods, no invented fields.

use shimmy_console::{
    ToolRegistry, ToolArgs,
    tools::ToolError,
};

// ── ToolRegistry ─────────────────────────────────────────────────────────────

#[test]
fn test_registry_with_defaults_has_tools() {
    let registry = ToolRegistry::with_defaults();
    let names: Vec<&str> = registry.names().collect();
    assert!(!names.is_empty(), "Registry should have default tools");
    // Verify core tools are registered
    assert!(registry.get("read_file").is_some());
    assert!(registry.get("write_file").is_some());
    assert!(registry.get("list_files").is_some());
    assert!(registry.get("shell_command").is_some());
    assert!(registry.get("system_info").is_some());
    assert!(registry.get("git_status").is_some());
}

#[test]
fn test_registry_get_nonexistent_tool() {
    let registry = ToolRegistry::with_defaults();
    assert!(registry.get("this_tool_does_not_exist").is_none());
}

#[test]
fn test_tool_args_new_is_empty() {
    let args = ToolArgs::new();
    assert!(args.args.is_empty());
}

#[test]
fn test_tool_args_get_str() {
    let mut args = ToolArgs::new();
    args.args.insert("key".to_string(), serde_json::Value::String("value".to_string()));
    assert_eq!(args.get_str("key"), Some("value"));
    assert_eq!(args.get_str("missing"), None);
}

#[test]
fn test_tool_args_require_str_missing() {
    let args = ToolArgs::new();
    let result = args.require_str("required_key");
    assert!(matches!(result, Err(ToolError::MissingArgument(_))));
}

#[test]
fn test_tool_args_get_bool_default() {
    let args = ToolArgs::new();
    assert_eq!(args.get_bool("flag", true), true);
    assert_eq!(args.get_bool("flag", false), false);
}

// ── Tool execution ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_read_file_missing_path_arg() {
    let registry = ToolRegistry::with_defaults();
    let tool = registry.get("read_file").expect("read_file should exist");
    let result = tool.execute(ToolArgs::new()).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ToolError::MissingArgument(arg) => assert_eq!(arg, "path"),
        e => panic!("Expected MissingArgument(path), got: {:?}", e),
    }
}

#[tokio::test]
async fn test_read_file_nonexistent() {
    let registry = ToolRegistry::with_defaults();
    let tool = registry.get("read_file").expect("read_file should exist");
    let mut args = ToolArgs::new();
    args.args.insert("path".to_string(), serde_json::json!("nonexistent_shimmy_test_xyz_12345.txt"));
    let result = tool.execute(args).await;
    assert!(result.is_err() || !result.unwrap().success);
}

#[tokio::test]
async fn test_list_files_current_dir() {
    let registry = ToolRegistry::with_defaults();
    let tool = registry.get("list_files").expect("list_files should exist");
    let mut args = ToolArgs::new();
    args.args.insert("path".to_string(), serde_json::json!("."));
    let result = tool.execute(args).await;
    // Should succeed — current dir always exists
    assert!(result.is_ok());
    assert!(result.unwrap().success);
}

#[tokio::test]
async fn test_system_info_returns_data() {
    let registry = ToolRegistry::with_defaults();
    let tool = registry.get("system_info").expect("system_info should exist");
    let result = tool.execute(ToolArgs::new()).await;
    assert!(result.is_ok());
    let tr = result.unwrap();
    assert!(tr.success);
    assert!(!tr.output.is_empty());
}

#[tokio::test]
async fn test_shell_command_echo() {
    let registry = ToolRegistry::with_defaults();
    let tool = registry.get("shell_command").expect("shell_command should exist");
    let mut args = ToolArgs::new();
    // Use a simple cross-platform command
    #[cfg(windows)]
    args.args.insert("command".to_string(), serde_json::json!("echo hello"));
    #[cfg(not(windows))]
    args.args.insert("command".to_string(), serde_json::json!("echo hello"));
    let result = tool.execute(args).await;
    assert!(result.is_ok());
    let tr = result.unwrap();
    assert!(tr.success);
    assert!(tr.output.contains("hello"));
}

// ── ToolCall struct ───────────────────────────────────────────────────────────

#[test]
fn test_toolcall_struct_fields() {
    use shimmy_console::tools::ToolCall;
    let call = ToolCall {
        name: "read_file".to_string(),
        arguments: ToolArgs::new(),
        call_id: Some("test-id".to_string()),
    };
    assert_eq!(call.name, "read_file");
    assert!(call.call_id.is_some());
}

// ── Config ────────────────────────────────────────────────────────────────────

#[test]
fn test_config_default_values() {
    use shimmy_console::Config;
    let config = Config::default();
    assert_eq!(config.default_theme, "arcade");
    assert_eq!(config.backend_url, "http://localhost:11435");
    assert!(config.default_model.is_none());
    assert!(config.default_model_path.is_none());
}

#[test]
fn test_config_has_local_model_false_when_no_path() {
    use shimmy_console::Config;
    let config = Config::default();
    assert!(!config.has_local_model());
}

#[test]
fn test_config_all_model_dirs_includes_standard_paths() {
    use shimmy_console::Config;
    let config = Config::default();
    let dirs = config.all_model_dirs();
    assert!(!dirs.is_empty());
}

#[test]
fn test_discover_gguf_files_nonexistent_dir() {
    use shimmy_console::config::discover_gguf_files;
    use std::path::PathBuf;
    let results = discover_gguf_files(&[PathBuf::from("nonexistent_dir_xyz_12345")]);
    assert!(results.is_empty());
}
