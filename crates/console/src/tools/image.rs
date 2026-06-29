use async_trait::async_trait;
use std::path::Path;
use crate::tools::{Tool, ToolArgs, ToolResult, ToolError};

/// Read image files, return base64 data and metadata
pub struct ReadImageTool;

#[async_trait]
impl Tool for ReadImageTool {
    fn name(&self) -> &'static str {
        "read_image"
    }

    fn description(&self) -> &'static str {
        "Read an image file from disk, returning base64-encoded data, MIME type, dimensions, and optional OCR text. \
         Parameters: path (required), mode (optional: 'base64', 'meta', 'ocr'; default 'base64')"
    }

    fn requires_license(&self) -> bool {
        false // Public tool, no license required
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult, ToolError> {
        let path_str = args.args.get("path").and_then(|v| v.as_str()).ok_or_else(|| ToolError::MissingArgument("path".to_string()))?;
        let mode = args.args.get("mode").and_then(|s| s.as_str()).unwrap_or("base64");
        if !["base64", "meta", "ocr"].contains(&mode) {
            return Err(ToolError::InvalidArgument("mode".to_string(), format!("Invalid mode {}", mode)));
        }
        // Resolve path (absolute or relative to working directory)
        let path = if Path::new(path_str).is_absolute() {
            Path::new(path_str).to_path_buf()
        } else {
            std::env::current_dir().unwrap_or(std::path::PathBuf::from(".")).join(path_str)
        };
        
        // Read file bytes
        let bytes = std::fs::read(&path)
            .map_err(|e| ToolError::ExecutionFailed(
                format!("Failed to read image file '{}': {}", path.display(), e)
            ))?;
        
        // Determine MIME type using infer crate
        let mime = infer::get(&bytes)
            .map(|info| info.mime_type())
            .unwrap_or("application/octet-stream");
        
        // Decode image to get dimensions
        let image_result = image::load_from_memory(&bytes);
        let (width, height) = if let Ok(img) = image_result {
            (img.width(), img.height())
        } else {
            // Not a valid image or unsupported format
            return Err(ToolError::ExecutionFailed(
                format!("Failed to decode image '{}': unsupported format or corrupted file", path.display())
            ));
        };
        
        // Base64 encode the image
        use base64::Engine as _;
        let base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        
        // Build response based on mode
        match mode {
            "base64" => {
                // Full base64 data + metadata
                Ok(ToolResult {
                    success: true,
                    output: format!("Image read successfully: {} ({}x{})", mime, width, height),
                    data: Some(serde_json::json!({
                        "mime": mime,
                        "width": width,
                        "height": height,
                        "base64": base64,
                        "size_bytes": bytes.len(),
                    })),
                })
            }
            "meta" => {
                // Metadata only (no base64)
                Ok(ToolResult {
                    success: true,
                    output: format!("Image metadata: {} ({}x{}), {} bytes", mime, width, height, bytes.len()),
                    data: Some(serde_json::json!({
                        "mime": mime,
                        "width": width,
                        "height": height,
                        "size_bytes": bytes.len(),
                    })),
                })
            }
            "ocr" => {
                // OCR mode - extract text from image
                // For now, return a placeholder indicating OCR is not implemented
                // In production, would use leptess or similar OCR crate
                Ok(ToolResult {
                    success: true,
                    output: format!("Image read with OCR (OCR not yet implemented): {} ({}x{})", mime, width, height),
                    data: Some(serde_json::json!({
                        "mime": mime,
                        "width": width,
                        "height": height,
                        "base64": base64,
                        "text": "[OCR feature not yet implemented]",
                        "size_bytes": bytes.len(),
                    })),
                })
            }
            _ => unreachable!(), // Already validated above
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_args(pairs: &[(&str, &str)]) -> ToolArgs {
        let mut args = ToolArgs::new();
        for (k, v) in pairs {
            args.args.insert(k.to_string(), serde_json::Value::String(v.to_string()));
        }
        args
    }

    #[tokio::test]
    async fn test_read_image_missing_path() {
        let tool = ReadImageTool;
        let result = tool.execute(ToolArgs::new()).await;
        assert!(result.is_err());
        match result {
            Err(ToolError::MissingArgument(msg)) => assert!(msg.contains("path")),
            _ => panic!("Expected MissingArgument error"),
        }
    }

    #[tokio::test]
    async fn test_read_image_invalid_mode() {
        let tool = ReadImageTool;
        let args = make_args(&[("path", "test.png"), ("mode", "invalid")]);
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result {
            Err(ToolError::InvalidArgument(_, msg)) => assert!(msg.contains("invalid")),
            _ => panic!("Expected InvalidArgument error"),
        }
    }

    #[tokio::test]
    async fn test_read_image_file_not_found() {
        let tool = ReadImageTool;
        let args = make_args(&[("path", "nonexistent_shimmy_test_12345.png")]);
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result {
            Err(ToolError::ExecutionFailed(msg)) => assert!(msg.contains("Failed to read")),
            _ => panic!("Expected ExecutionFailed error"),
        }
    }

    #[tokio::test]
    async fn test_read_image_success() {
        // Use a real valid PNG — create it via the image crate
        let temp_dir = std::env::temp_dir();
        let test_image_path = temp_dir.join("test_image_shimmy_valid.png");

        // Create a 2x2 red image using the image crate (already a dependency)
        let img = image::RgbImage::from_fn(2, 2, |_, _| image::Rgb([255u8, 0u8, 0u8]));
        img.save(&test_image_path).expect("Should save test image");

        let tool = ReadImageTool;
        let args = make_args(&[("path", &test_image_path.to_string_lossy())]);
        let result = tool.execute(args).await;
        assert!(result.is_ok(), "Expected Ok but got: {:?}", result);
        let tr = result.unwrap();
        assert!(tr.success);
        assert!(tr.data.is_some());
        let data = tr.data.unwrap();
        assert!(data.get("mime").is_some());
        assert!(data.get("base64").is_some());
        assert_eq!(data.get("width").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(data.get("height").and_then(|v| v.as_u64()), Some(2));

        fs::remove_file(&test_image_path).ok();
    }
}
