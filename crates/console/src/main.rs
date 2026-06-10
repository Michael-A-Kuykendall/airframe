use clap::{Parser, Subcommand};
use shimmy_console::commands::{
    analyze::AnalyzeCommand,
    chat::ChatCommand,
    config::ConfigCommand,
    edit::EditCommand,
    license::LicenseCommand,
};
use shimmy_console::{ToolRegistry, ToolArgs};
use tokio::io::AsyncWriteExt;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(name = "shimmy", about = "Shimmy Console CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Chat with the model
    Chat(ChatCommand),
    /// Analyze files or projects
    Analyze(AnalyzeCommand),
    /// Edit files
    Edit(EditCommand),
    /// Manage configuration
    Config(ConfigCommand),
    /// License management
    #[command(subcommand)]
    License(LicenseCommand),
    
    /// Execute a tool directly (e.g. file_ops, git, cmd)
    Tool {
        /// Name of the tool to execute
        name: String,
        
        /// Arguments for the tool in key=value format (e.g. path=.)
        #[arg(short, long)]
        arg: Vec<String>,
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Ensure tracing is initialized
    let _ = tracing_subscriber::fmt::try_init();

    // Initialize the tool registry with all tools
    let registry = ToolRegistry::with_defaults();

    // Now parse flags
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat(cmd) => {
            cmd.run().await?;
        }
        Commands::Analyze(cmd) => {
            cmd.run().await?;
        }
        Commands::Edit(cmd) => {
            cmd.run().await?;
        }
        Commands::Config(cmd) => {
            cmd.run().await?;
        }
        Commands::License(cmd) => {
            cmd.execute().await?;
        }
        Commands::Tool { name, arg } => {
            if let Some(tool) = registry.get(&name) {
                let mut tool_args = ToolArgs::new();
                for a in arg {
                    if let Some((k, v)) = a.split_once('=') {
                        // try parsed as json, if it fails, insert as string.
                        let val: Value = serde_json::from_str(v).unwrap_or(Value::String(v.to_string()));
                        tool_args.args.insert(k.to_string(), val);
                    }
                }
                
                let result = tool.execute(tool_args).await?;
                if result.success {
                    println!("Success:\n{}", result.output);
                } else {
                    println!("Error:\n{}", result.output);
                }
            } else {
                println!("Error: Tool '{}' not found in registry.", name);
                println!("Available tools:");
                for tool_name in registry.names() {
                    println!(" - {}", tool_name);
                }
            }
        }
    }
    
    // Strict flush for rustchain 
    tokio::io::stdout().flush().await?;

    Ok(())
}
