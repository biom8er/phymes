use std::{env, fmt::Display};

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait, Subject, SubjectBuilderTrait, SubjectTrait,
    WorkspaceSubject, create_workspace_batch,
};
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
    #[value(name = "Bash")]
    Bash,
    /// Python coding environment
    ///
    /// # Notes
    /// * A project directory that follows normal convention below is assumed
    ///
    /// my_python_project/
    /// ├── install.sh
    /// ├── requirements.txt
    /// ├── src/
    /// │   └── main.py
    /// └── .venv/
    #[value(name = "Python")]
    Python,
    /// Rust coding environment
    ///
    /// # Notes
    /// * A project directory that follows normal convention below is assumed
    ///
    /// my_python_project/
    /// ├── Cargo.toml
    /// ├── examples/
    /// │   └── example/
    /// │       └── main.rs
    /// ├── install.sh
    /// └── src/
    ///     ├── lib.rs
    ///     └── main.rs
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

impl CommandSandboxEnvironments {
    /// To workspace
    pub fn to_workspace(
        &self,
        workspace_name: Option<&str>,
        workspace_contents: Option<&[WorkspaceSubject]>,
    ) -> Result<Subject> {
        if let Some(workspace_contents) = workspace_contents {
            let (path, content): (Vec<String>, Vec<String>) = workspace_contents
                .iter()
                .map(|w| (w.path.to_owned(), w.content.to_owned()))
                .unzip();
            let batch = create_workspace_batch(path, content)?;
            Subject::get_builder()
                .with_name(self.to_string().as_str())
                .with_record_batches(vec![batch])?
                .build()
        } else {
            self.to_default_workspace(workspace_name)
        }
    }
    /// To default workspace
    pub fn to_default_workspace(&self, workspace_name: Option<&str>) -> Result<Subject> {
        let root = if let Some(workspace_name) = workspace_name {
            format!("{workspace_name}/")
        } else {
            String::new()
        };
        match self {
            Self::Bash => {
                let path = [format!("{root}src/main.sh"), format!("{root}install.sh")]
                    .into_iter()
                    .collect::<Vec<_>>();
                let content = [
                    r#"#!/usr/bin/env bash

[[ -n "$1" ]] && echo "$1"
[[ -n "$2" ]] && echo "$2"
[[ -n "$3" ]] && echo "$3""#,
                    r#"#!/bin/sh
apk add --no-cache bash
chmod +x ./src/main.sh"#,
                ]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
                let batch = create_workspace_batch(path, content)?;
                Subject::get_builder()
                    .with_name(self.to_string().as_str())
                    .with_record_batches(vec![batch])?
                    .build()
            }
            Self::Python => {
                let path = [
                    format!("{root}requirements.txt"),
                    format!("{root}src/main.py"),
                    format!("{root}install.sh"),
                ]
                .into_iter()
                .collect::<Vec<_>>();
                let content = [
                    r#"pandas==2.2.3
pyarrow==17.0.0"#,
                    r#"#!/usr/bin/env python3
import argparse
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    args = parser.parse_args()

    # Read using PyArrow IPC file reader (works with Rust arrow FileWriter output)
    with open(args.input_file, "rb") as f:
        table = ipc.open_file(f).read_all()

    # Convert to pandas for your transformation
    df = table.to_pandas()

    # Your transformation
    df['age'] = df['age'] + 10

    # Convert pandas back to Arrow
    table_out = pa.Table.from_pandas(df)

    # One-liner to make all fields non-nullable
    new_schema = pa.schema([pa.field(f.name, f.type, nullable=False) for f in table_out.schema])

    # Cast the table to the new schema
    # This will fail if there are actual nulls in the data
    try:
        table_non_nullable = table_out.cast(new_schema)
    except pa.ArrowInvalid as e:
        raise ValueError(
            "Cannot cast to non-nullable schema because null values exist in the data."
        ) from e

    # Write Arrow IPC File format (Rust-compatible)
    with pa.OSFile(args.output_file, "wb") as f:
        writer = ipc.RecordBatchFileWriter(f, table_non_nullable.schema)
        writer.write_table(table_non_nullable)
        writer.close()"#,
                    r#"#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt"#,
                ]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
                let batch = create_workspace_batch(path, content)?;
                Subject::get_builder()
                    .with_name(self.to_string().as_str())
                    .with_record_batches(vec![batch])?
                    .build()
            }
            Self::Rust => {
                let path = [
                    format!("{root}Cargo.toml"),
                    format!("{root}src/main.rs"),
                    format!("{root}install.sh"),
                ]
                .into_iter()
                .collect::<Vec<_>>();
                let content = [
                    r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = { version = "1", default-features = false }
arrow = "58.0.0"
serde_json = "1.0.133"
serde = { version = "1.0.215", features = ["derive"] }
clap = { version = "4.5.4", features = ["derive"] }"#,
                    r#"use arrow::array::{ArrayRef, StringArray, UInt32Array};
use arrow::error::{ArrowError, Result};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use clap::Parser;
use std::fs::File;
use std::sync::Arc;

/// Minimal Feather/IPC editor using a single CLI argument.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the input Feather/IPC file
    #[arg(long)]
    input_file: String,
    /// Path to the output Feather/IPC file
    #[arg(long)]
    output_file: String,
}

/// Helper function to add 10 to years to the age column in a RecordBatch
fn add_10_yrs_to_age(batch: &RecordBatch) -> arrow::error::Result<RecordBatch> {
    let names = batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap_or_default().to_string())
        .collect::<Vec<String>>();
    let ages = batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
        .iter()
        .map(|s| s.unwrap_or_default() + 10)
        .collect::<Vec<u32>>();
    let names_arr: ArrayRef = Arc::new(StringArray::from(names));
    let ages_arr: ArrayRef = Arc::new(UInt32Array::from(ages));
    RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_path = args.input_file.as_str();
    let output_path = args.output_file.as_str();

    // --- Step 1: Read Feather/IPC file ---
    let file = File::open(input_path)?;
    let reader = FileReader::try_new(file, None)?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<_>>()?;

    // --- Step 2: Modify the batches ---
    let batches = batches.into_iter()
        .map(|batch| add_10_yrs_to_age(&batch).unwrap())
        .collect::<Vec<_>>();

    // --- Step 3: Write modified batches to output file ---
    let file = File::create(output_path)?;
    let mut writer = FileWriter::try_new(file, &batches[0].schema())?;

    for batch in batches {
        writer.write(&batch)?;
    }

    writer.finish()?;

    Ok(())
}"#,
                    r#"#!/usr/bin/env bash
apt update
apt install --assume-yes protobuf-compiler clang"#,
                ]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
                let batch = create_workspace_batch(path, content)?;
                Subject::get_builder()
                    .with_name(self.to_string().as_str())
                    .with_record_batches(vec![batch])?
                    .build()
            }
            Self::WasmModule => {
                let path = [format!("{root}src/main.wat")]
                    .into_iter()
                    .collect::<Vec<_>>();
                let content = [r#"(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )
)"#]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
                let batch = create_workspace_batch(path, content)?;
                Subject::get_builder()
                    .with_name(self.to_string().as_str())
                    .with_record_batches(vec![batch])?
                    .build()
            }
            Self::WasmComponent => {
                let path = [format!("{root}src/main.wat")]
                    .into_iter()
                    .collect::<Vec<_>>();
                let content = [r#"(component
  ;; Define a core module
  (core module $math
    (func $add (param $a i32) (param $b i32) (result i32)
      local.get $a
      local.get $b
      i32.add
    )
    (export "add" (func $add))
  )

  ;; Instantiate the core module
  (core instance $math-inst
    (instantiate $math)
  )

  ;; Lift the core function into the component world
  (func (export "add") (param "a" u32) (param "b" u32) (result u32)
    (canon lift (core func $math-inst "add"))
  )
)"#]
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>();
                let batch = create_workspace_batch(path, content)?;
                Subject::get_builder()
                    .with_name(self.to_string().as_str())
                    .with_record_batches(vec![batch])?
                    .build()
            }
            _ => Err(anyhow!(
                "The CommandSandboxEnvironments `{self}` is not yet supported."
            )),
        }
    }
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
    /// 
    /// [RecordBatch]: arrow::array::RecordBatch
    #[default]
    #[value(name = "Stdio")]
    Stdio,
    /// Write [RecordBatch]es as IPC bytes to a temporary file
    ///
    /// The [RecordBatch]es will be serialized as IPC and written to a named temporary file called `lhs_args.ipc`
    /// and the output will be deserialized from IPC from the same temporary file
    /// 
    /// [RecordBatch]: arrow::array::RecordBatch
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
    /// * Can either be the name of the file in the project directory or the text for the script that will be created on the fly
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initialization_script: Option<String>,

    /// Initialization file
    ///
    /// # Notes
    /// * Used during the initialization phase to install or setup additional resources
    /// * Name of the file at the root level of the directory
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initialization_file: Option<String>,

    /// Run script
    ///
    /// # Notes
    /// * Used during the run phase
    /// * Can either be the name of the file in the project directory or the text for the script that will be created on the fly
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_script: Option<String>,

    /// Run script
    ///
    /// # Notes
    /// * Used during the run phase
    /// * Name of the file within the /src folder (for most projects) of the directory
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_file: Option<String>,

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
    /// The CLI arguments MUST be specified if the `subject_name` is not specified!
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cli_args: Option<Vec<String>>,

    /// List of environmental variables for running the command
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_args: Option<Vec<String>>,

    /// The name of the streaming subject with the data to run with the command
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none", alias = "lhs_name")]
    pub subject_name: Option<String>,

    /// The name of the streaming workspace with the files needed by the command
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none", alias = "rhs_name")]
    pub workspace_name: Option<String>,
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
                    }
                    Err(e) => {
                        return Err(anyhow!("{e:?}"));
                    }
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
    fn from_table(table: &Subject) -> Result<Self>
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
        if !(column_names.contains("timeout")
            && column_names.contains("runner")
            && column_names.contains("environment")
            && column_names.contains("container_image")
            && column_names.contains("data_i")
            && column_names.contains("data_o"))
        {
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
