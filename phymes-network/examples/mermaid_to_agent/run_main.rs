use std::{io::Write, str::FromStr};

use anyhow::Result;
use clap::Parser;
use phymes_subject::BuilderTrait;
use phymes_network::SessionContextBuilderMermaid;

#[derive(Parser, Debug, Default, Clone)]
#[command(author, version, about, long_about = None)]
pub struct MermaidBuildConfig {
    /// Directory for mermaid build files
    #[arg(long)]
    pub dir: Option<String>,

    /// The input files for the flowchart diagram
    #[arg(long)]
    pub flowchart_files: Vec<String>,

    /// The input files for the ER diagram
    #[arg(long)]
    pub erdiagram_files: Vec<String>,

    /// The output filename for the flowchart diagram
    #[arg(long)]
    pub flowchart_out: Option<String>,

    /// The output filename for the ER diagram diagram
    #[arg(long)]
    pub erdiagram_out: Option<String>,
}

impl MermaidBuildConfig {
    /// Read and join the files
    fn read(dir: &str, files: &[&str], header: &str, indentation: &str) -> Result<String> {
        let mut files_str_vec = vec![header.to_string()];
        for file in files {
            let path_str = format!("{dir}/{file}");
            let path = std::path::PathBuf::from_str(path_str.as_str())?;
            let file_str = std::fs::read_to_string(path)?;
            let file_str_indent = file_str
                .lines()
                .map(|l| format!("{indentation}{l}"))
                .collect::<Vec<_>>()
                .join("\n");
            files_str_vec.push(file_str_indent);
        }
        let files_str = files_str_vec.join("\n");
        Ok(files_str)
    }

    /// Read flowchart files
    pub fn read_flowchart(&self) -> Result<String> {
        let dir = self.dir.as_ref().map_or(".", |v| v);
        let files = self
            .flowchart_files
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        Self::read(dir, &files, "flowchart TD", "\t")
    }

    /// Read erdiagram files
    pub fn read_erdiagram(&self) -> Result<String> {
        let dir = self.dir.as_ref().map_or(".", |v| v);
        let files = self
            .erdiagram_files
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>();
        Self::read(dir, &files, "erDiagram", "\t")
    }

    /// Write the file
    fn write(dir: &str, file: &str, contents: &str) -> Result<()> {
        let path_str = format!("{dir}/{file}");
        let path = std::path::PathBuf::from_str(path_str.as_str())?;
        let mut file = std::fs::File::create(&path)?;
        file.write_all(contents.as_bytes())?;
        Ok(())
    }

    /// Write flowchart
    pub fn write_flowchart(&self, contents: &str) -> Result<()> {
        let dir = self.dir.as_ref().map_or(".", |v| v);
        let file = self.flowchart_out.as_ref().map_or("out.flowchart", |v| v);
        Self::write(dir, file, contents)
    }

    /// Write flowchart
    pub fn write_erdiagram(&self, contents: &str) -> Result<()> {
        let dir = self.dir.as_ref().map_or(".", |v| v);
        let file = self.erdiagram_out.as_ref().map_or("out.erdiagram", |v| v);
        Self::write(dir, file, contents)
    }
}

/// Generates the Mermaid.js Flowchart and ERDiagram representations of the [SessionContextBuilderMermaid]
pub fn run_main() -> Result<()> {
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

    // Write each file to disk
    config.write_flowchart(&flowchart)?;
    config.write_erdiagram(&erdiagram)?;

    Ok(())
}
