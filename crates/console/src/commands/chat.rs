use clap::Args;
use std::io::{self, Write};
use crate::adapters::LocalInferenceAdapter;
use crate::websocket::InferenceBackend;
use crate::ToolRegistry;
use serde_json::Value;

#[derive(Debug, Args)]
pub struct ChatCommand {
    #[arg(short, long)]
    pub model: Option<String>,
    pub message: Option<String>,
    #[arg(short, long)]
    pub session: Option<String>,
    #[arg(long, default_value = "true")]
    pub stream: bool,
}

impl ChatCommand {
    pub async fn run(&self) -> anyhow::Result<()> {
        let default_model = "C:/Users/micha/repos/llama.cpp/models/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf".to_string();
        let model_path = self.model.clone().unwrap_or(default_model);

        println!("Loading agentic interface with TinyLlama...");
        let adapter = LocalInferenceAdapter::new(model_path);
        let registry = ToolRegistry::with_defaults();
        
        let tool_descriptions = registry
            .all()
            .map(|t| format!("- {}: {}", t.name(), t.description()))
            .collect::<Vec<_>>()
            .join("\n");

        let system_prompt = format!(
            "<|system|>\nYou are Shimmy, an AI agent. You have tools:\n{}\nTo use a tool, output a JSON block wrapped in <tool_call>...</tool_call>. Format: {{\"name\":\"tool_name\",\"arguments\":{{\"arg\":\"val\"}}}}</s>\n",
            tool_descriptions
        );

        let mut history = system_prompt.clone();
        
        println!("System ready. Type your message below.");
        println!("Available tools: \n{}", tool_descriptions);

        loop {
            print!("\nYou: ");
            io::stdout().flush()?;
            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() || input.trim() == "exit" {
                break;
            }
            if input.trim().is_empty() { continue; }

            history.push_str(&format!("<|user|>\n{}</s>\n<|assistant|>\n", input.trim()));

            print!("Shimmy: ");
            io::stdout().flush()?;

            let response = adapter.generate_response("local", &history).await?;
            println!("{}", response);

            history.push_str(&response);
            history.push_str("</s>\n");

            // Look for tool calls in response
            if let Some(start) = response.find("<tool_call>") {
                if let Some(end) = response[start..].find("</tool_call>") {
                    let json_str = &response[start + 11..start + end];
                    println!("\n[Agent requested tool execution parsing: {}]", json_str);
                    
                    if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                        if let (Some(name), Some(args_val)) = (v["name"].as_str(), v["arguments"].as_object()) {
                            if let Some(tool) = registry.get(name) {
                                let mut tool_args = crate::ToolArgs::new();
                                for (k, val) in args_val {
                                    tool_args.args.insert(k.clone(), val.clone());
                                }
                                println!("[Executing tool: {}]", name);
                                match tool.execute(tool_args).await {
                                    Ok(res) => {
                                        let out = format!("\n<|user|>\nTool result:\n{}</s>\n<|assistant|>\n", res.output);
                                        println!("{}", out);
                                        history.push_str(&out);
                                        let next = adapter.generate_response("local", &history).await?;
                                        println!("{}", next);
                                        history.push_str(&next);
                                        history.push_str("</s>\n");
                                    }
                                    Err(e) => {
                                        println!("[Error: {}]", e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
