use async_trait::async_trait;
use crate::tools::{Tool, ToolArgs, ToolResult, ToolError};

pub struct ExplainCommandTool;
pub struct GetHelpTool;

#[async_trait]
impl Tool for ExplainCommandTool {
    fn name(&self) -> &'static str {
        "explain_command"
    }

    fn description(&self) -> &'static str {
        "Explain what a command does and how to use it"
    }

    fn requires_license(&self) -> bool {
        false
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult, ToolError> {  
        let command = args.require_str("command")?;
        let explanation = format!("Explanation for command: {}", command);      

        Ok(ToolResult::success_with_data(
            explanation,
            serde_json::json!({
                "command": command,
                "explanation": format!("Command '{}' explanation", command)     
            })
        ))
    }
}

#[async_trait]
impl Tool for GetHelpTool {
    fn name(&self) -> &'static str {
        "get_help"
    }

    fn description(&self) -> &'static str {
        "Get help information for available tools and commands"
    }

    fn requires_license(&self) -> bool {
        false
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult, ToolError> {  
        let topic = args.get_str("topic").unwrap_or("general");

        let help_content = format!("Help information for topic: {}", topic);    

        Ok(ToolResult::success_with_data(
            help_content,
            serde_json::json!({
                "topic": topic,
                "help_available": true
            })
        ))
    }
}
