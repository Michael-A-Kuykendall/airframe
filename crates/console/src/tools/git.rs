use async_trait::async_trait;
use std::process::Stdio;
use tokio::process::Command;
use crate::tools::{Tool, ToolArgs, ToolResult, ToolError};

pub struct GitStatusTool;
pub struct GitDiffTool;
pub struct GitCommitTool;
pub struct GitLogTool;

#[async_trait]
impl Tool for GitStatusTool {
    fn name(&self) -> &str {
        "git_status"
    }

    fn description(&self) -> &str {
        "Get git repository status"
    }

    fn requires_license(&self) -> bool {
        false
    }

    async fn execute(&self, _args: ToolArgs) -> Result<ToolResult, ToolError> { 
        let output = Command::new("git")
            .args(&["status", "--porcelain"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Git command failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(ToolResult::success_with_data(
                stdout.to_string(),
                serde_json::json!({
                    "status": "success",
                    "files": Self::parse_git_status(&stdout)
                })
            ))
        } else {
            Err(ToolError::ExecutionFailed(stderr.to_string()))
        }
    }
}

impl GitStatusTool {
    pub fn parse_git_status(output: &str) -> Vec<serde_json::Value> {
        output.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let status = &line[..2];
                let file = &line[3..];
                serde_json::json!({
                    "status": status,
                    "file": file
                })
            })
            .collect()
    }
}

#[async_trait]
impl Tool for GitDiffTool {
    fn name(&self) -> &str {
        "git_diff"
    }

    fn description(&self) -> &str {
        "Show git diff for specified file or all changes"
    }

    fn requires_license(&self) -> bool {
        true
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult, ToolError> {  
        let file_path = args.get_str("file");

        let mut cmd = Command::new("git");
        cmd.arg("diff");

        if let Some(file) = file_path {
            cmd.arg(file);
        }

        let output = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Git diff failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(ToolResult::success_with_data(
                stdout.to_string(),
                serde_json::json!({
                    "diff": stdout,
                    "file": file_path
                })
            ))
        } else {
            Err(ToolError::ExecutionFailed(stderr.to_string()))
        }
    }
}

#[async_trait]
impl Tool for GitCommitTool {
    fn name(&self) -> &str {
        "git_commit"
    }

    fn description(&self) -> &str {
        "Create a git commit with specified message"
    }

    fn requires_license(&self) -> bool {
        true
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult, ToolError> {  
        let message = args.require_str("message")?;

        // First add all changes
        let add_output = Command::new("git")
            .args(&["add", "."])
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Git add failed: {}", e)))?;

        if !add_output.status.success() {
            return Err(ToolError::ExecutionFailed("Failed to stage changes".to_string()));
        }

        // Then commit
        let commit_output = Command::new("git")
            .args(&["commit", "-m", message])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Git commit failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&commit_output.stdout);
        let stderr = String::from_utf8_lossy(&commit_output.stderr);

        if commit_output.status.success() {
            Ok(ToolResult::success_with_data(
                format!("Commit created: {}", stdout),
                serde_json::json!({
                    "message": message,
                    "output": stdout
                })
            ))
        } else {
            Err(ToolError::ExecutionFailed(stderr.to_string()))
        }
    }
}

#[async_trait]
impl Tool for GitLogTool {
    fn name(&self) -> &str {
        "git_log"
    }

    fn description(&self) -> &str {
        "Show git commit history"
    }

    fn requires_license(&self) -> bool {
        false
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult, ToolError> {  
        let limit = args.get_i64("limit").unwrap_or(10) as u32;

        let output = Command::new("git")
            .args(&["log", "--oneline", &format!("-{}", limit)])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("Git log failed: {}", e)))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if output.status.success() {
            Ok(ToolResult::success_with_data(
                stdout.to_string(),
                serde_json::json!({
                    "commits": Self::parse_git_log(&stdout),
                    "limit": limit
                })
            ))
        } else {
            Err(ToolError::ExecutionFailed(stderr.to_string()))
        }
    }
}

impl GitLogTool {
    pub fn parse_git_log(output: &str) -> Vec<serde_json::Value> {
        output.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let parts: Vec<&str> = line.splitn(2, ' ').collect();
                serde_json::json!({
                    "hash": parts.get(0).unwrap_or(&""),
                    "message": parts.get(1).unwrap_or(&"")
                })
            })
            .collect()
    }
}
