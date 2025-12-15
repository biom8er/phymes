use std::{env, fmt::Display};

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use phymes_core::{MappableTrait, Table, TableTrait};
use phymes_diagnostics::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use crate::DataConfigTrait;

/// Command runners
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum CommandSandboxRunners {
    /// Run the command in a docker container
    #[default]
    #[value(name = "Docker")]
    Docker,
    /// Run the command using wasmtime
    #[value(name = "Wasmtime")]
    Wasmtime,
    #[value(skip)]
    Custom(String),
}
impl Display for CommandSandboxRunners {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Docker => write!(f, "Docker"),
            Self::Wasmtime => write!(f, "Wasmtime"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Command environments
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum CommandSandboxEnvironments {
    /// Python coding environment
    #[value(name = "Python")]
    Python,
    /// Rust coding environment
    #[default]
    #[value(name = "Rust")]
    Rust,
    /// WASM component or module environment
    #[value(name = "WASM")]
    WASM,
    #[value(skip)]
    Custom(String),
}
impl Display for CommandSandboxEnvironments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Python => write!(f, "Python"),
            Self::Rust => write!(f, "Rust"),
            Self::WASM => write!(f, "WASM"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Data transfer methods
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataIOMethod {
    /// Transfer [RecordBatch]es as bytes over the stdio interface
    /// 
    /// The [RecordBatch]es will be serialized as IPC and added as a named argument `lhs_args` to the CLI arguments
    #[default]
    #[value(name = "Stdio")]
    Stdio,
    /// Write [RecordBatch]es as IPC bytes to a temporary file 
    /// 
    /// The [RecordBatch]es will be serialized as IPC and written to a named temporary file called `lhs_args.ipc`
    #[value(name = "TempFile")]
    TempFile,
    /// Use the config and ignore the batches
    #[value(name = "None")]
    None,
}
impl Display for DataIOMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stdio => write!(f, "Stdio"),
            Self::TempFile => write!(f, "TempFile"),
            Self::None => write!(f, "None"),
        }
    }
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct CommandSandboxConfig {
    /// The sandboxed runner
    #[arg(long)]
    pub runner: CommandSandboxRunners,

    /// The sandboxed environment
    #[arg(long)]
    pub environment: CommandSandboxEnvironments,

    /// Container image or WASM component/module
    #[arg(long)]
    pub container_image: String,

    /// Data transfer method
    #[arg(long)]
    pub data_io: DataIOMethod,

    /// The command to run inside container or the wasm module to invoke
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,

    /// The timeout in seconds
    /// 
    /// # Notes
    /// * Not yet implemented
    #[arg(long, default_value_t = 15)]
    pub timeout: usize,

    /// Project directory
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_dir: Option<String>,

    /// Container project directory
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_project_dir: Option<String>,

    /// Entry script
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_script: Option<String>,

    /// List of arguments for the container in addition to the environment-specific defaults
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_args: Option<Vec<String>>,

    /// List of arguments for running the command
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cli_args: Option<Vec<String>>,

    /// List of environmental variables for running the command
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_args: Option<Vec<String>>,

    /// Temporary input file
    /// 
    /// # Notes
    /// 
    /// * Not implemented
    /// * See `container_input_path` for how it could be implemented
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_input: Option<String>,

    /// Temporary output file
    /// 
    /// # Notes
    /// 
    /// * Not implemented
    /// * See `container_input_path` for how it could be implemented
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_output: Option<String>,

    /// Host input path
    /// 
    /// # Notes
    /// 
    /// * Not implemented
    /// * See `container_input_path` for how it could be implemented
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_input_path: Option<String>,

    /// Container input path
    /// 
    /// # Notes
    /// 
    /// * Not implemented
    /// * e.g., ```
    /// use tempfile::NamedTempFile;
    /// // Create a temporary input file (auto-deletes on drop)
    /// let mut temp_input = NamedTempFile::new().expect("Failed to create temp input file");
    /// writeln!(temp_input, "42\n99\n123").expect("Failed to write to temp input file");
    /// 
    /// let host_input_path = temp_input.path().to_str().unwrap();
    /// let container_input_path = "/home/sandbox/input.txt";
    /// 
    /// // Container paths
    /// let container_project_dir = "/home/sandbox/project";
    /// let container_entry = format!("{}/{}", container_project_dir, entry_script);
    /// ```
    /// 
    /// Then add to the command arguments as
    /// `"-v", &format!("{}:{}:ro", host_input_path, container_input_path), // Input file read-only`
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_input_path: Option<String>,
}

impl CommandSandboxConfig {
    /// Return environmental arguments as key/value pairs
    pub fn env_args(&self) -> Result<HashMap<String, String>> {
        let mut map = HashMap::<String, String>::new();
        if let Some(env_args) = &self.env_args {
            for env_var in env_args {
                match env::var(env_var) {
                    Ok(key) => {
                        map.insert(env_var.to_string(), key);
                    },
                    Err(e) => {
                        return Err(anyhow!("{e:?}"));
                    },
                }
            }
        }
        Ok(map)
    }
}

impl DataConfigTrait for CommandSandboxConfig {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Self::default())
    }
    fn from_table(table: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // Check for the required fields
        let column_names = table
            .get_schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<HashSet<_>>();
        if !(column_names.contains("timeout") && column_names.contains("runner") && column_names.contains("environment") && column_names.contains("container_image")
        && column_names.contains("data_io")) {
            return Err(anyhow!(
                "Table {} is missing required Field for `timeout`, `runner`, `environment`, `container_image`, and `data_io` in CommandSandboxConfig.",
                table.get_name()
            ));
        }

        // Try to build the config
        match table.to_struct::<CommandSandboxConfig>() {
            Ok(config_vec) => match config_vec.first() {
                Some(config) => Ok(config.to_owned()),
                None => Err(anyhow!(
                    "No config data found for CommandSandboxConfig with subject {}",
                    table.get_name()
                )),
            },
            Err(err) => Err(anyhow!(
                "CommandSandboxConfig could not be built for subject {}. {err}",
                table.get_name()
            )),
        }
    }
}