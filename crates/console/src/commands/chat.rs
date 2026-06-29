use clap::Args;
use std::io::{self, Write};
use crate::adapters::{HttpInferenceAdapter, LocalInferenceAdapter};
use crate::config::{Config, discover_gguf_files};
use crate::websocket::InferenceBackend;
use crate::ToolRegistry;
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Args)]
pub struct ChatCommand {
    /// Path to a specific GGUF model file (overrides config)
    #[arg(short, long)]
    pub model: Option<String>,
    /// Optional single message (non-interactive mode)
    pub message: Option<String>,
    /// Resume a previous session by ID
    #[arg(short, long)]
    pub session: Option<String>,
    /// Connect to a running shimmy server instead of loading locally
    #[arg(long)]
    pub remote: bool,
    /// Save chosen model as default in ~/.shimmy/config.toml
    #[arg(long)]
    pub save_default: bool,
}

impl ChatCommand {
    pub async fn run(&self) -> anyhow::Result<()> {
        let config = Config::from_env();
        let session_id = self
            .session
            .clone()
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        // ── Adapter selection ─────────────────────────────────────────────────
        // Priority:
        //   1. --model flag (explicit path)
        //   2. --remote flag (use HttpInferenceAdapter)
        //   3. config.default_model_path (saved local path)
        //   4. Model chooser (interactive, scans discovered paths)
        //   5. Fall back to HttpInferenceAdapter at config.backend_url

        let model_path: Option<String> = if let Some(ref m) = self.model {
            // --model flag: could be a name or a full path
            if std::path::Path::new(m).exists() {
                Some(m.clone())
            } else {
                // Try to find it in known dirs
                find_model_by_name(m, &config)
            }
        } else if self.remote {
            None // Force remote
        } else if config.has_local_model() {
            config.default_model_path.as_ref().map(|p| p.to_string_lossy().to_string())
        } else {
            // No explicit model — run chooser
            run_model_chooser(&config, self.save_default)?
        };

        // Build the right adapter
        if let Some(ref path) = model_path {
            println!("🎯 Session: {}", session_id);
            run_local_chat(path.clone(), session_id, &config).await
        } else {
            println!("🌐 Connecting to shimmy server at {}", config.backend_url);
            println!("🎯 Session: {}", session_id);
            run_remote_chat(config.backend_url.clone(), session_id).await
        }
    }
}

/// Interactive model chooser — shows all discovered models, user picks one
fn run_model_chooser(config: &Config, save_default: bool) -> anyhow::Result<Option<String>> {
    let all_dirs = config.all_model_dirs();
    let mut models = discover_gguf_files(&all_dirs);

    if models.is_empty() {
        println!("⚠️  No .gguf models found in any search path.");
        println!("   Add model directories to ~/.shimmy/config.toml:");
        println!("   model_dirs = [\"D:/shimmy-test-models/gguf_collection\"]");
        println!();
        println!("   Or point to a running server with --remote");
        println!("   Or specify a model directly with --model /path/to/model.gguf");
        return Ok(None);
    }

    // Sort by name for consistent display
    models.sort_by(|a, b| a.0.cmp(&b.0));

    println!();
    println!("┌─────────────────────────────────────────────┐");
    println!("│           shimmy console — arcade           │");
    println!("└─────────────────────────────────────────────┘");
    println!();
    println!("Available models:");
    println!();
    for (i, (name, path)) in models.iter().enumerate() {
        let size_mb = std::fs::metadata(path)
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);
        println!("  [{:>2}]  {} ({}MB)", i + 1, name, size_mb);
    }
    println!();
    print!("Select model [1-{}]: ", models.len());
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let choice: usize = input.trim().parse().unwrap_or(0);

    if choice == 0 || choice > models.len() {
        println!("Invalid selection. Exiting.");
        return Ok(None);
    }

    let (name, path) = &models[choice - 1];
    let path_str = path.to_string_lossy().to_string();
    println!("✅ Selected: {}", name);

    if save_default {
        let mut cfg = config.clone();
        cfg.default_model_path = Some(path.clone());
        cfg.default_model = Some(name.clone());
        if let Err(e) = cfg.save() {
            eprintln!("⚠️  Could not save config: {}", e);
        } else {
            println!("💾 Saved as default in ~/.shimmy/config.toml");
        }
    } else {
        println!("💡 Tip: run with --save-default to skip this chooser next time");
    }

    println!();
    Ok(Some(path_str))
}

/// Find a model by name (partial match) across all configured dirs
fn find_model_by_name(name: &str, config: &Config) -> Option<String> {
    let all_dirs = config.all_model_dirs();
    let models = discover_gguf_files(&all_dirs);
    let name_lower = name.to_lowercase();
    models.into_iter()
        .find(|(n, _)| n.to_lowercase().contains(&name_lower))
        .map(|(_, p)| p.to_string_lossy().to_string())
}

/// Run chat loop with local airframe engine
async fn run_local_chat(
    model_path: String,
    session_id: String,
    _config: &Config,
) -> anyhow::Result<()> {
    let adapter = LocalInferenceAdapter::new(model_path);
    let registry = ToolRegistry::with_defaults();

    let tool_descriptions = registry
        .all()
        .map(|t| format!("- {}: {}", t.name(), t.description()))
        .collect::<Vec<_>>()
        .join("\n");

    let system_prompt = format!(
        "<|system|>\nYou are Shimmy, an AI assistant running locally via Airframe. \
You have access to tools:\n{}\n\
To use a tool, output a JSON block in <tool_call>...</tool_call> tags.\n\
Format: {{\"name\":\"tool_name\",\"arguments\":{{\"arg\":\"val\"}}}}</s>\n",
        tool_descriptions
    );

    println!("─────────────────────────────────────────────");
    println!("  Model:   {}", adapter.model_name());
    println!("  Session: {}", &session_id[..8]);
    println!("  Backend: airframe (local)");
    println!("  Tools:   {} available", registry.all().count());
    println!("─────────────────────────────────────────────");
    println!("Type your message. 'exit' to quit.");
    println!();

    run_chat_loop(adapter, registry, system_prompt, session_id).await
}

/// Run chat loop connected to a remote shimmy server
async fn run_remote_chat(backend_url: String, session_id: String) -> anyhow::Result<()> {
    let adapter = HttpInferenceAdapter::new(backend_url.clone());
    let registry = ToolRegistry::with_defaults();

    let tool_descriptions = registry
        .all()
        .map(|t| format!("- {}: {}", t.name(), t.description()))
        .collect::<Vec<_>>()
        .join("\n");

    let system_prompt = format!(
        "<|system|>\nYou are Shimmy, an AI assistant. You have tools:\n{}\n\
To use a tool, output a JSON block in <tool_call>...</tool_call> tags.\n\
Format: {{\"name\":\"tool_name\",\"arguments\":{{\"arg\":\"val\"}}}}</s>\n",
        tool_descriptions
    );

    println!("─────────────────────────────────────────────");
    println!("  Server:  {}", backend_url);
    println!("  Session: {}", &session_id[..8]);
    println!("  Backend: shimmy server (remote)");
    println!("  Tools:   {} available", registry.all().count());
    println!("─────────────────────────────────────────────");
    println!("Type your message. 'exit' to quit.");
    println!();

    run_chat_loop(adapter, registry, system_prompt, session_id).await
}

/// Core REPL loop — shared between local and remote adapters
async fn run_chat_loop<A: InferenceBackend>(
    adapter: A,
    registry: ToolRegistry,
    system_prompt: String,
    _session_id: String,
) -> anyhow::Result<()> {
    let mut history = system_prompt.clone();
    let mut first_turn = true;

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() || input.trim() == "exit" {
            println!("\n👋 Goodbye.");
            break;
        }
        if input.trim().is_empty() {
            continue;
        }

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
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);

        let handle = {
            // Use a reference-counted adapter clone via wrapper
            let prompt_clone = request_prompt.clone();
            // We can't clone the trait object directly, so we spawn with a channel
            // and let the caller drive the stream
            tokio::spawn({
                let tx = tx.clone();
                async move {
                    // tx is moved in; prompt_clone drives the generation
                    let _ = tx.send(prompt_clone).await; // sentinel: send prompt as first msg
                }
            })
        };
        let _ = handle.await;

        // Actually drive generation — pull prompt from channel sentinel
        let prompt_to_use = rx.recv().await.unwrap_or_default();
        let (gen_tx, mut gen_rx) = tokio::sync::mpsc::channel::<String>(64);

        // Drive generation inline (adapter not Clone-able via trait object)
        // Use a oneshot to get the result
        let gen_result = adapter.generate_stream("local", &prompt_to_use, gen_tx).await;

        // Drain tokens
        while let Ok(token) = gen_rx.try_recv() {
            print!("{}", token);
            io::stdout().flush()?;
            response.push_str(&token);
        }

        if let Err(e) = gen_result {
            eprintln!("\n⚠️  Generation error: {}", e);
        }

        println!();
        history.push_str(&response);
        history.push_str("</s>\n");
        first_turn = false;

        // Tool call detection and execution
        if let Some(tool_response) = try_execute_tool_call(&response, &registry).await {
            let tool_prompt = format!(
                "\n<|user|>\nTool result:\n{}</s>\n<|assistant|>\n",
                tool_response
            );
            println!("[Tool executed]");
            history.push_str(&tool_prompt);

            print!("Shimmy: ");
            io::stdout().flush()?;

            let mut followup = String::new();
            let (ft_tx, mut ft_rx) = tokio::sync::mpsc::channel::<String>(64);
            let ft_result = adapter.generate_stream("local", &tool_prompt, ft_tx).await;

            while let Ok(token) = ft_rx.try_recv() {
                print!("{}", token);
                io::stdout().flush()?;
                followup.push_str(&token);
            }
            if let Err(e) = ft_result {
                eprintln!("\n⚠️  Generation error: {}", e);
            }
            println!();
            history.push_str(&followup);
            history.push_str("</s>\n");
        }
    }

    Ok(())
}

/// Parse and execute a tool call if present in the response
async fn try_execute_tool_call(response: &str, registry: &ToolRegistry) -> Option<String> {
    let start = response.find("<tool_call>")?;
    let end = response[start..].find("</tool_call>")?;
    let json_str = &response[start + 11..start + end];

    let v: Value = serde_json::from_str(json_str).ok()?;
    let name = v["name"].as_str()?;
    let args_obj = v["arguments"].as_object()?;
    let tool = registry.get(name)?;

    let mut tool_args = crate::ToolArgs::new();
    for (k, val) in args_obj {
        tool_args.args.insert(k.clone(), val.clone());
    }

    println!("\n[🔧 Tool: {}]", name);
    match tool.execute(tool_args).await {
        Ok(result) => Some(result.output),
        Err(e) => {
            eprintln!("[Tool error: {}]", e);
            None
        }
    }
}
