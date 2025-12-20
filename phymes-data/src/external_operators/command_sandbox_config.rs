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
    /// Run the command in a sandboxed docker container
    #[default]
    #[value(name = "Docker")]
    Docker,
    /// Run the command in a non-sandboxed (unsafe) docker container
    #[value(name = "DockerUnsafe")]
    DockerUnsafe,
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
            Self::DockerUnsafe => write!(f, "DockerUnsafe"),
            Self::Wasmtime => write!(f, "Wasmtime"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Command environments
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum CommandSandboxEnvironments {
    /// Basic Bash shell to run commands
    /// 
    /// # Notes
    /// * Not intended for use except to setup Docker resources
    ///   e.g., `docker pull`, `docker run python pip install ...`, `docker run git pull ...`, etc.
    #[value(name = "Bash")]
    Bash,
    /// Python coding environment
    #[value(name = "Python")]
    Python,
    /// Rust coding environment
    #[default]
    #[value(name = "Rust")]
    Rust,
    /// WASM module (without using the component model)
    /// 
    /// e.g., `wasmtime run foo.wasm` or `wasmtime run foo.wat` with optional CLI arguments for `foo` following
    #[value(name = "WasmModule")]
    WasmModule,
    /// WASM component (using WAVE syntax)
    /// 
    /// e.g., `wasmtime run --invoke 'add(1, 2)' foo.wasm` with no CLI arguments included using WAVE syntax
    /// where `add` is the command
    #[value(name = "WasmComponent")]
    WasmComponent,
    #[value(skip)]
    Custom(String),
}
impl Display for CommandSandboxEnvironments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bash => write!(f, "Bash"),
            Self::Python => write!(f, "Python"),
            Self::Rust => write!(f, "Rust"),
            Self::WasmModule => write!(f, "WasmModule"),
            Self::WasmComponent => write!(f, "WasmComponent"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Data transfer methods
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default, PartialEq)]
pub enum DataIOMethod {
    /// Transfer [RecordBatch]es as bytes over the stdio interface
    /// 
    /// The [RecordBatch]es will be serialized as JSON and added as a named argument `lhs_args` to the CLI arguments
    /// and the output will be deserialized from JSON
    /// 
    /// # Notes
    /// * The schema between input and output data must be the same since JSON is used and we need to know the schema
    ///   to correctly interpret the JSON data types
    #[default]
    #[value(name = "Stdio")]
    Stdio,
    /// Write [RecordBatch]es as IPC bytes to a temporary file 
    /// 
    /// The [RecordBatch]es will be serialized as IPC and written to a named temporary file called `lhs_args.ipc`
    /// and the output will be deserialized from IPC from the same temporary file
    #[value(name = "TempFile")]
    TempFile,
    /// Use the config and ignore the batches
    /// 
    /// The output will be read in from the Stdout and packaged as a message
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

    /// Data input transfer method
    #[arg(long)]
    pub data_i: DataIOMethod,

    /// Data output transfer method
    #[arg(long)]
    pub data_o: DataIOMethod,

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

    /// Initialization script
    /// 
    /// # Notes
    /// * Used during the initialization phase to install or setup additional resources
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initialization_script: Option<String>,

    /// Run script
    /// 
    /// # Notes
    /// * Used during the run phase
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_script: Option<String>,

    /// List of arguments for the container in addition to the environment-specific defaults
    /// 
    /// # Examples
    /// * `--rm` to remove the container after execution
    /// * `-v` to mount directories and files
    /// * `-e` include environmental arguments
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

    /// Container input path
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
        && column_names.contains("data_i") && column_names.contains("data_o")) {
            return Err(anyhow!(
                "Table {} is missing required Field for `timeout`, `runner`, `environment`, `container_image`, `data_i`, and `data_o` in CommandSandboxConfig.",
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