use std::str::FromStr;

use anyhow::Result;
use clap::Parser;
use phymes_agents::SessionContextBuilderMermaid;
use phymes_core::BuilderTrait;

#[derive(Parser, Debug, Default, Clone)]
#[command(author, version, about, long_about = None)]
pub struct MermaidBuildConfig {
    /// Directory for mermaid build files
    #[arg(long)]
    pub dir: Option<String>,

    /// The tokenizer config in json format.
    #[arg(long)]
    pub flowchart_files: Vec<String>,

    /// The model weights config in json format.
    #[arg(long)]
    pub erdiagram_files: Vec<String>,
}

impl MermaidBuildConfig {
    /// Read and join the files
    fn read(dir: &str, files: &[&str], header: &str) -> Result<String> {
        let mut files_str_vec = vec![header.to_string()];
        for file in files {
            let path_str = format!("{dir}/{file}");
            let path = std::path::PathBuf::from_str(path_str.as_str())?;
            let file_str = std::fs::read_to_string(path)?;
            let file_str = format!("/t{file_str}"); // Add tab
            files_str_vec.push(file_str);
        }
        let files_str = files_str_vec.join("/n");
        Ok(files_str)
    }
    
    /// Read flowchart files
    pub fn read_flowchart(&self) -> Result<String> {
        let dir = self.dir.as_ref().map_or(".", |v| v);
        let files = self.flowchart_files.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        Self::read(dir, &files, "flowchart TD")
    }
    
    /// Read erdiagram files
    pub fn read_erdiagram(&self) -> Result<String> {
        let dir = self.dir.as_ref().map_or(".", |v| v);
        let files = self.erdiagram_files.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        Self::read(dir, &files, "erDiagram")
    }
}

/// Generates the Mermaid.js Flowchart and ERDiagram representations of the [SessionContextBuilderMermaid]
pub fn run_main() -> Result<(String, String)> {

    // CLI arguments
    let config = MermaidBuildConfig::parse();

    // Read in each of the files and join the strings
    let flowchart = config.read_flowchart()?;
    let erdiagram = config.read_erdiagram()?;

    // Try to build from mermaid
    let _builder = SessionContextBuilderMermaid::new()
        .with_name("Mermaid_CLI")
        .with_flowchart(&flowchart)
        .with_erdiagram(&erdiagram)
        .build()?;

    Ok((flowchart, erdiagram))
}