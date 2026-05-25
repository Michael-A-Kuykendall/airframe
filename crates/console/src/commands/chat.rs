use clap::Args;
use std::io::{self, Write};
use crate::adapters::ShimmyServerAdapter;
use crate::websocket::InferenceBackend;
use crate::ToolRegistry;
use serde_json::Value;
use uuid::Uuid;

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
        let default_model = "D:/shimmy-test-models/gguf_collection/TinyLlama-1.1B-Chat-v1.0.Q4_0.gguf".to_string();
        let _model_path = self.model.clone().unwrap_or(default_model);
        let session_id = self
            .session
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        println!("Connecting to Shimmy GPU Server at http://127.0.0.1:8080...");
        println!("Session: {}", session_id);
        let adapter = ShimmyServerAdapter::new("http://127.0.0.1:8080".to_string(), session_id);
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
        let mut first_turn = true;
        
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

            let request_prompt = if first_turn {
                format!(
                    "{}<|user|>\n{}</s>\n<|assistant|>\n",
                    system_prompt,
                    input.trim()
                )
            } else {
                format!("<|user|>\n{}</s>\n<|assistant|>\n", input.trim())
            };

            history.push_str(&format!("<|user|>\n{}</s>\n<|assistant|>\n", input.trim()));

            print!("Shimmy: ");
            io::stdout().flush()?;

            let mut response = String::new();
            let (tx, mut rx) = tokio::sync::mpsc::channel(32);
            
            let adapter_clone = adapter.clone();
            let prompt_clone = request_prompt.clone();
            let handle = tokio::spawn(async move {
                let _ = adapter_clone.generate_stream("local", &prompt_clone, tx).await;
            });

            while let Some(token) = rx.recv().await {
                print!("{}", token);
                io::stdout().flush()?;
                response.push_str(&token);
            }
            let _ = handle.await;
            println!();
            history.push_str(&response);
            history.push_str("</s>\n");
            first_turn = false;

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
                                        let tool_prompt = out.clone();
                                        
                                        print!("Shimmy: ");
                                        io::stdout().flush()?;
                            
                                        let mut next = String::new();
                                        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
                                        
                                        let adapter_clone = adapter.clone();
                                        let prompt_clone = tool_prompt.clone();
                                        let handle = tokio::spawn(async move {
                                            let _ = adapter_clone.generate_stream("local", &prompt_clone, tx).await;
                                        });
                            
                                        while let Some(token) = rx.recv().await {
                                            print!("{}", token);
                                            io::stdout().flush()?;
                                            next.push_str(&token);
                                        }
                                        let _ = handle.await;
                                        println!();

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
