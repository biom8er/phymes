use std::{
    fs::{self, File},
    io::Write,
    path::Path,
    pin::Pin,
    process::Output,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{FutureExt, Stream, StreamExt};
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait, RecordBatchStream, RuntimeEnv,
    SendableRecordBatchStream, Subject, SubjectBuilder, SubjectBuilderTrait,
    SubjectTrait,
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, create_timestamp_micros,
};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_schemas::{
    AvailableSchemaTrait, AvailableSubjects, create_bytes_fields, create_chat_record_batch, create_values_fields,
};
use phymes_processor::ProcessorTrait;
use serde_json::Value;
use tempfile::NamedTempFile;
use tokio::process::Command;

use crate::{
    DataConfigTrait, WorkspaceEditor,
    external_operators::{
        command_sandbox_config::{
            CommandSandboxConfig, CommandSandboxEnvironments, CommandSandboxRunners, DataIOMethod,
        },
        http_client_processor::error_report,
    },
};

/// The state of the command stream
///
/// # Notes
/// * We need to capture each stage of the request so that the connection
///   is not dropped during repeated polling of the stream.
pub enum CommandSandboxStreamState {
    NewPoll,
    ExistingPoll,
    Output(Pin<Box<dyn Future<Output = std::io::Result<Output>> + Send + 'static>>),
    Done,
}

/// Information needed by the runner to run
#[derive(Default, Clone, Debug)]
pub struct CommandSandboxRunnerInfo {
    /// Name of the runner (i.e., docker --name)
    name: Option<String>,
    /// Input file path
    input_file: Option<String>,
    /// Output file path
    output_file: Option<String>,
    /// Content
    content: Option<String>,
    /// Installation script file
    initialization_file: Option<String>,
    /// Run script file
    run_file: Option<String>,
}

impl CommandSandboxRunnerInfo {
    pub fn new() -> Self {
        CommandSandboxRunnerInfo::default()
    }
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    pub fn with_input_file(mut self, input_file: &str) -> Self {
        self.input_file = Some(input_file.to_string());
        self
    }
    pub fn with_output_file(mut self, output_file: &str) -> Self {
        self.output_file = Some(output_file.to_string());
        self
    }
    pub fn with_content(mut self, content: &str) -> Self {
        self.content = Some(content.to_string());
        self
    }
    pub fn with_initialization_file(mut self, initialization_file: &str) -> Self {
        self.initialization_file = Some(initialization_file.to_string());
        self
    }
    pub fn with_run_file(mut self, run_file: &str) -> Self {
        self.run_file = Some(run_file.to_string());
        self
    }
}

/// The state of the runner
///
/// # Notes
/// * We need to capture each stage of the request so that the connection
///   is not dropped during repeated polling of the stream.
#[derive(Debug)]
pub enum CommandSandboxRunnerState {
    NotStarted,
    /// Starting the runner
    Starting(CommandSandboxRunnerInfo),
    /// Initializing runner running environment and installing any additional dependencies
    Initializing(CommandSandboxRunnerInfo),
    /// Running the runner for each each streaming batch
    Running(CommandSandboxRunnerInfo),
    /// Cleanup all resources
    Done(CommandSandboxRunnerInfo),
}

pub struct CommandSandboxStream {
    /// Output schema
    schema: SchemaRef,
    /// The messages containing the lhs and rhs
    /// which we cannot determine until we intialize the config
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The candle assets needed for inference
    _runtime_env: Arc<RuntimeEnv>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<CommandSandboxConfig>,
    /// State of the Command Stream
    stream_state: CommandSandboxStreamState,
    /// State of the Runner
    runner_state: CommandSandboxRunnerState,
    /// The inbox of messages to processes
    message_inbox: Option<Subject>,
    /// The inbox of CLI args to processes
    from_cli_args: bool,
    /// The inbox of workspace files to setup
    workspace_inbox: Option<Subject>,
}

impl CommandSandboxStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::Messages.to_schema(),
            messages,
            diagnostic_builder,
            config_stream,
            _runtime_env: runtime_env,
            config: None,
            stream_state: CommandSandboxStreamState::NewPoll,
            runner_state: CommandSandboxRunnerState::NotStarted,
            message_inbox: None,
            from_cli_args: false,
            workspace_inbox: None,
        })
    }
}

impl Stream for CommandSandboxStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Iterate through each state until the Command queue is completed
        match &mut self.stream_state {
            CommandSandboxStreamState::NewPoll => {
                // Initialize the config
                if self.config.is_none() {
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                        batches.push(batch);
                    }
                    let config_table = SubjectBuilder::new()
                        .with_name("config")
                        .with_record_batches(batches)?
                        .build()?;
                    if config_table
                        .get_schema()
                        .fields()
                        .contains(&create_values_fields())
                    {
                        let config_json = config_table.get_column_as_vec_str("values").join("");
                        let config = serde_json::from_str::<CommandSandboxConfig>(&config_json)?;
                        self.config.replace(config);
                    } else if config_table
                        .get_schema()
                        .fields()
                        .contains(&create_bytes_fields())
                    {
                        let config_json = config_table
                            .get_column_as_vec_nested_primitive::<u8>("bytes")?
                            .into_iter()
                            .map(|b| String::from_utf8(b).unwrap())
                            .collect::<Vec<_>>()
                            .join("");
                        let config = serde_json::from_str::<CommandSandboxConfig>(&config_json)?;
                        self.config.replace(config);
                    } else {
                        let config = CommandSandboxConfig::from_table(&config_table)?;
                        self.config.replace(config);
                    }
                }

                // collect the workspace
                if self.workspace_inbox.is_none()
                    && let Some(workspace) = self.config.as_ref().unwrap().workspace_name.clone()
                {
                    match remove_message_by_subject(&workspace, &mut self.messages) {
                        // Poll the next batches
                        Some(mut fut) => {
                            // DM: where we will specify to stream batch by batch or collect all batches
                            let mut batches = Vec::new();
                            while let Some(Ok(batch)) =
                                ready!(fut.get_message_mut().poll_next_unpin(cx))
                            {
                                // DM: need to add a check for the expected workspace schema
                                if batch.num_rows() > 0 {
                                    batches.push(batch);
                                    break;
                                }
                            }
                            self.messages.insert(fut.get_name().to_string(), fut);

                            // Replace the inbox
                            if !batches.is_empty() {
                                let table = Subject::get_builder()
                                    .with_name("workspace")
                                    .with_record_batches(batches)?
                                    .build()?;
                                self.schema = table.get_schema();
                                self.workspace_inbox.replace(table);
                            }
                        }
                        None => {
                            self.stream_state = CommandSandboxStreamState::Done;
                            let error = Err(anyhow!(
                                "Subject `{workspace}` was not found in the messages. The available message subjects are `{:?}`",
                                self.messages.keys()
                            ));
                            return Poll::Ready(Some(error));
                        }
                    }
                }

                // Collect the next batch or continue processing the current batch
                if self.message_inbox.is_none()
                    && !self.from_cli_args
                    && let Some(subject_name) = self.config.as_ref().unwrap().subject_name.clone()
                {
                    match remove_message_by_subject(&subject_name, &mut self.messages) {
                        // Poll the next batches
                        Some(mut fut) => {
                            // DM: where we will specify to stream batch by batch or collect all batches
                            let mut batches = Vec::new();
                            while let Some(Ok(batch)) =
                                ready!(fut.get_message_mut().poll_next_unpin(cx))
                            {
                                if batch.num_rows() > 0 {
                                    batches.push(batch);
                                    break;
                                }
                            }
                            self.messages.insert(fut.get_name().to_string(), fut);

                            // Replace the inbox
                            if !batches.is_empty() {
                                let table = Subject::get_builder()
                                    .with_name("messages")
                                    .with_record_batches(batches)?
                                    .build()?;
                                self.schema = table.get_schema();
                                self.message_inbox.replace(table);
                            }
                        }
                        None => {
                            self.stream_state = CommandSandboxStreamState::Done;
                            let error = Err(anyhow!(
                                "Subject `{subject_name}` was not found in the messages. The available message subjects are `{:?}`",
                                self.messages.keys()
                            ));
                            return Poll::Ready(Some(error));
                        }
                    }
                } else if self.message_inbox.is_none()
                    && !self.from_cli_args
                    && self.config.as_ref().unwrap().cli_args.is_some()
                {
                    // "Poll" the config
                    self.from_cli_args = true;
                } else if self.from_cli_args {
                    // The config has already been "polled"
                    self.from_cli_args = false;
                }

                // The poll ends when there are no more batches
                if self.message_inbox.is_none() && !self.from_cli_args {
                    self.stream_state = CommandSandboxStreamState::Done;
                    match &self.runner_state {
                        CommandSandboxRunnerState::NotStarted => return Poll::Ready(None),
                        CommandSandboxRunnerState::Starting(runner_info)
                        | CommandSandboxRunnerState::Initializing(runner_info)
                        | CommandSandboxRunnerState::Running(runner_info)
                        | CommandSandboxRunnerState::Done(runner_info) => {
                            // Cleanup resources
                            self.runner_state =
                                CommandSandboxRunnerState::Done(runner_info.to_owned());
                        }
                    }
                }

                self.stream_state = CommandSandboxStreamState::ExistingPoll;
                self.poll_next(cx)
            }
            CommandSandboxStreamState::ExistingPoll => {
                // Build the `CommandSandboxRunnerInfo` for the command
                // NOTE!: we have to declare and initialize ALL mount directories and files when the runner is first called
                match &self.runner_state {
                    CommandSandboxRunnerState::NotStarted => {
                        // Make a random name for the runner
                        let mut buf = [0u8; 16];
                        getrandom::fill(&mut buf)?;
                        let hash = u128::from_ne_bytes(buf);
                        let name = format!("phymes-sandbox_{hash}");

                        // Create the workspace, initialization, and runner files
                        let (
                            run_file_path,
                            initialization_file_path,
                            input_file_path,
                            output_file_path,
                        ) = if let Some(project_dir) =
                            self.config.as_ref().unwrap().project_dir.clone()
                        {
                            // Check the directory
                            if !Path::new(&project_dir).exists() {
                                let err_str =
                                    format!("Project folder '{project_dir}' does not exist.");
                                self.stream_state = CommandSandboxStreamState::Done;
                                return Poll::Ready(Some(Err(anyhow!(err_str))));
                            }

                            // Create the workspace
                            // DM: whether we create the workspace here or in the "container" the files will need to persist on disk
                            if let Some(workspace) = self.workspace_inbox.take() {
                                let workspace_editor = WorkspaceEditor::new(&project_dir);
                                let paths_vec = workspace.get_column_as_vec_str("path");
                                let contents_vec = workspace.get_column_as_vec_str("content");
                                for (path, content) in
                                    paths_vec.into_iter().zip(contents_vec.into_iter())
                                {
                                    workspace_editor
                                        .create_file(std::path::Path::new(path), content)?;
                                }
                            }

                            // Check the run file
                            let run_file_path = if let Some(run_file) =
                                self.config.as_ref().unwrap().run_file.as_ref()
                            {
                                let run_file_path = format!("{project_dir}/src/{run_file}");
                                if !Path::new(&run_file_path).exists() {
                                    let err_str = format!(
                                        "Run script '{run_file}' does not exist in the project src folder '{project_dir}/src'."
                                    );
                                    self.stream_state = CommandSandboxStreamState::Done;
                                    return Poll::Ready(Some(Err(anyhow!(err_str))));
                                }
                                Some(run_file_path)
                            } else {
                                // Create the run file from script
                                if let Some(run_script) =
                                    self.config.as_ref().unwrap().run_script.as_ref()
                                {
                                    let run_file_path = match self
                                        .config
                                        .as_ref()
                                        .unwrap()
                                        .environment
                                    {
                                        CommandSandboxEnvironments::Python => {
                                            format!("{project_dir}/src/main.py")
                                        }
                                        CommandSandboxEnvironments::Rust => {
                                            format!("{project_dir}/src/main.rs")
                                        }
                                        CommandSandboxEnvironments::Bash => {
                                            format!("{project_dir}/src/main.sh")
                                        }
                                        _ => {
                                            let err_str = format!(
                                                "Sandbox Environment '{}' does not yet support a run script.",
                                                self.config.as_ref().unwrap().environment
                                            );
                                            self.stream_state = CommandSandboxStreamState::Done;
                                            return Poll::Ready(Some(Err(anyhow!(err_str))));
                                        }
                                    };
                                    let mut file = File::create(&run_file_path)?;
                                    let _ = file.write(run_script.as_bytes())?;
                                    file.flush()?;
                                    Some(run_file_path)
                                } else {
                                    None
                                }
                            };

                            // Check the initialization file
                            let initialization_file_path = if let Some(initialization_file) =
                                self.config.as_ref().unwrap().initialization_file.as_ref()
                            {
                                let initialization_file_path =
                                    format!("{project_dir}/{initialization_file}");
                                if !Path::new(&initialization_file_path).exists() {
                                    let err_str = format!(
                                        "Initialization script '{initialization_file}' does not exist in the project folder '{project_dir}'."
                                    );
                                    self.stream_state = CommandSandboxStreamState::Done;
                                    return Poll::Ready(Some(Err(anyhow!(err_str))));
                                }
                                Some(initialization_file_path)
                            } else {
                                // Create the initialization file from script
                                if let Some(initialization_script) =
                                    self.config.as_ref().unwrap().initialization_script.as_ref()
                                {
                                    // DM: update based on environment
                                    let initialization_file_path =
                                        format!("{project_dir}/install.sh");
                                    let mut file = File::create(&initialization_file_path)?;
                                    let _ = file.write(initialization_script.as_bytes())?;
                                    file.flush()?;
                                    Some(initialization_file_path)
                                } else {
                                    None
                                }
                            };

                            // Create the input and output file paths
                            let input_file_path = format!("{project_dir}/input.ipc");
                            let output_file_path = format!("{project_dir}/output.ipc");

                            (
                                run_file_path,
                                initialization_file_path,
                                Some(input_file_path),
                                Some(output_file_path),
                            )
                        } else {
                            (None, None, None, None)
                        };

                        // Create the temporary input and output files with content or stdin content
                        let mut runner_info = match (
                            &self.config.as_ref().unwrap().data_i,
                            &self.config.as_ref().unwrap().data_o,
                        ) {
                            (DataIOMethod::None, DataIOMethod::None)
                            | (DataIOMethod::None, DataIOMethod::Stdio) => {
                                let _ = self.message_inbox.take();
                                CommandSandboxRunnerInfo::new().with_name(&name)
                            }
                            (DataIOMethod::Stdio, DataIOMethod::None)
                            | (DataIOMethod::Stdio, DataIOMethod::Stdio) => {
                                let content =
                                    self.message_inbox.take().unwrap().to_json_object()?;
                                let content = serde_json::to_string(&content)?;
                                CommandSandboxRunnerInfo::new()
                                    .with_name(&name)
                                    .with_content(&content)
                            }
                            (DataIOMethod::TempFile, DataIOMethod::None)
                            | (DataIOMethod::TempFile, DataIOMethod::Stdio) => {
                                let input_file = NamedTempFile::new()?;
                                let input_persist_path = input_file_path.ok_or(anyhow!("Missing input file path for data input method {} and data output method {}.", &self.config.as_ref().unwrap().data_i, &self.config.as_ref().unwrap().data_o))?;
                                let mut input_file = input_file.persist(&input_persist_path)?;
                                self.message_inbox
                                    .take()
                                    .unwrap()
                                    .to_ipc_file(&mut input_file)?;
                                CommandSandboxRunnerInfo::new()
                                    .with_name(&name)
                                    .with_input_file(&input_persist_path)
                            }
                            (DataIOMethod::None, DataIOMethod::TempFile) => {
                                let output_file = NamedTempFile::new()?;
                                let output_persist_path = output_file_path.ok_or(anyhow!("Missing output file path for data input method {} and data output method {}.", &self.config.as_ref().unwrap().data_i, &self.config.as_ref().unwrap().data_o))?;
                                let _output_file = output_file.persist(&output_persist_path)?;
                                CommandSandboxRunnerInfo::new()
                                    .with_name(&name)
                                    .with_output_file(&output_persist_path)
                            }
                            (DataIOMethod::TempFile, DataIOMethod::TempFile) => {
                                let input_file = NamedTempFile::new()?;
                                let input_persist_path = input_file_path.ok_or(anyhow!("Missing input file path for data input method {} and data output method {}.", &self.config.as_ref().unwrap().data_i, &self.config.as_ref().unwrap().data_o))?;
                                let output_file = NamedTempFile::new()?;
                                let output_persist_path = output_file_path.ok_or(anyhow!("Missing output file path for data input method {} and data output method {}.", &self.config.as_ref().unwrap().data_i, &self.config.as_ref().unwrap().data_o))?;
                                let mut input_file = input_file.persist(&input_persist_path)?;
                                self.message_inbox
                                    .take()
                                    .unwrap()
                                    .to_ipc_file(&mut input_file)?;
                                let _output_file = output_file.persist(&output_persist_path)?;
                                CommandSandboxRunnerInfo::new()
                                    .with_name(&name)
                                    .with_input_file(&input_persist_path)
                                    .with_output_file(&output_persist_path)
                            }
                            (DataIOMethod::Stdio, DataIOMethod::TempFile) => {
                                let output_file = NamedTempFile::new()?;
                                let output_persist_path = output_file_path.ok_or(anyhow!("Missing output file path for data input method {} and data output method {}.", &self.config.as_ref().unwrap().data_i, &self.config.as_ref().unwrap().data_o))?;
                                let messages = self.message_inbox.take().unwrap();
                                let content = messages.to_json_object()?;
                                let content = serde_json::to_string(&content)?;
                                let _output_file = output_file.persist(&output_persist_path)?;
                                CommandSandboxRunnerInfo::new()
                                    .with_name(&name)
                                    .with_content(&content)
                                    .with_output_file(&output_persist_path)
                            }
                        };

                        // Update the runner info with the init and run filepaths
                        if let Some(run_file) = run_file_path {
                            runner_info = runner_info.with_run_file(&run_file);
                        }
                        if let Some(initialization_file) = initialization_file_path {
                            runner_info =
                                runner_info.with_initialization_file(&initialization_file);
                        }

                        // Determine the runner state
                        match &self.config.as_ref().unwrap().runner {
                            CommandSandboxRunners::Docker | CommandSandboxRunners::DockerUnsafe => {
                                self.runner_state =
                                    CommandSandboxRunnerState::Starting(runner_info);
                            }
                            CommandSandboxRunners::Wasmtime => {
                                self.runner_state = CommandSandboxRunnerState::Running(runner_info);
                            }
                            CommandSandboxRunners::Custom(_) => unimplemented!(),
                        }
                    }
                    CommandSandboxRunnerState::Starting(runner_info) => {
                        if runner_info.initialization_file.is_none() {
                            self.runner_state =
                                CommandSandboxRunnerState::Running(runner_info.to_owned());
                        } else {
                            self.runner_state =
                                CommandSandboxRunnerState::Initializing(runner_info.to_owned());
                        }
                    }
                    CommandSandboxRunnerState::Initializing(runner_info) => {
                        self.runner_state =
                            CommandSandboxRunnerState::Running(runner_info.to_owned());
                    }
                    CommandSandboxRunnerState::Running(runner_info) => {
                        // Clear the temporary input file and/or create the stdin content
                        let runner_info = match (
                            &self.config.as_ref().unwrap().data_i,
                            &self.config.as_ref().unwrap().data_o,
                        ) {
                            (DataIOMethod::None, DataIOMethod::None)
                            | (DataIOMethod::None, DataIOMethod::Stdio) => runner_info.to_owned(),
                            (DataIOMethod::Stdio, DataIOMethod::None)
                            | (DataIOMethod::Stdio, DataIOMethod::Stdio) => {
                                // Update the content
                                let runner_info = runner_info.to_owned();
                                let content =
                                    self.message_inbox.take().unwrap().to_json_object()?;
                                let content = serde_json::to_string(&content)?;
                                runner_info.with_content(&content)
                            }
                            (DataIOMethod::TempFile, DataIOMethod::None)
                            | (DataIOMethod::TempFile, DataIOMethod::Stdio) => {
                                if let Some(input_file_path) = runner_info.input_file.as_ref() {
                                    let runner_info = runner_info.to_owned();
                                    // Update the file
                                    let input_file = NamedTempFile::new()?;
                                    let mut input_file = input_file.persist(input_file_path)?;
                                    self.message_inbox
                                        .take()
                                        .unwrap()
                                        .to_ipc_file(&mut input_file)?;
                                    runner_info
                                } else {
                                    self.stream_state = CommandSandboxStreamState::Done;
                                    return Poll::Ready(Some(Err(anyhow!(
                                        "Missing TempFile for runner."
                                    ))));
                                }
                            }
                            (DataIOMethod::TempFile, DataIOMethod::TempFile) => {
                                if let (Some(input_file_path), Some(output_file_path)) = (
                                    runner_info.input_file.as_ref(),
                                    runner_info.output_file.as_ref(),
                                ) {
                                    let runner_info = runner_info.to_owned();
                                    // Update the file
                                    let input_file = NamedTempFile::new()?;
                                    let mut input_file = input_file.persist(input_file_path)?;

                                    // Truncate the file
                                    let output_file = NamedTempFile::new()?;
                                    let _output_file = output_file.persist(output_file_path)?;
                                    self.message_inbox
                                        .take()
                                        .unwrap()
                                        .to_ipc_file(&mut input_file)?;
                                    runner_info
                                } else {
                                    self.stream_state = CommandSandboxStreamState::Done;
                                    return Poll::Ready(Some(Err(anyhow!(
                                        "Missing TempFile for runner."
                                    ))));
                                }
                            }
                            (DataIOMethod::None, DataIOMethod::TempFile) => {
                                if let Some(output_file_path) = runner_info.output_file.as_ref() {
                                    // Truncate the file
                                    let output_file = NamedTempFile::new()?;
                                    let _output_file = output_file.persist(output_file_path)?;
                                    runner_info.to_owned()
                                } else {
                                    self.stream_state = CommandSandboxStreamState::Done;
                                    return Poll::Ready(Some(Err(anyhow!(
                                        "Missing TempFile for runner."
                                    ))));
                                }
                            }
                            (DataIOMethod::Stdio, DataIOMethod::TempFile) => {
                                if let Some(output_file_path) = runner_info.output_file.as_ref() {
                                    let runner_info = runner_info.to_owned();
                                    // Truncate the file
                                    let output_file = NamedTempFile::new()?;
                                    let _output_file = output_file.persist(output_file_path)?;

                                    // Update the content
                                    let content =
                                        self.message_inbox.take().unwrap().to_json_object()?;
                                    let content = serde_json::to_string(&content)?;
                                    runner_info.with_content(&content)
                                } else {
                                    self.stream_state = CommandSandboxStreamState::Done;
                                    return Poll::Ready(Some(Err(anyhow!(
                                        "Missing TempFile for runner."
                                    ))));
                                }
                            }
                        };

                        // Change to running state
                        self.runner_state = CommandSandboxRunnerState::Running(runner_info);
                    }
                    CommandSandboxRunnerState::Done(_runner_info) => {
                        // Do nothing
                    }
                }

                // Execute the command
                // DM: A future optimization maybe to treat each row as a parallel Command
                let fut = match self.config.as_ref().unwrap().runner {
                    CommandSandboxRunners::Docker | CommandSandboxRunners::DockerUnsafe => {
                        // Build Docker args
                        let mut command_args =
                            match (&self.config.as_ref().unwrap().runner, &self.runner_state) {
                                (
                                    CommandSandboxRunners::Docker,
                                    CommandSandboxRunnerState::Starting(runner_info),
                                ) => {
                                    let mut command_args = vec![
                                        "run".to_string(),
                                        "--name".to_string(), // Name the container for later calls
                                        runner_info
                                            .name
                                            .as_ref()
                                            .expect("Missing name for runner.")
                                            .to_string(),
                                        "--network".to_string(),
                                        "none".to_string(), // No network
                                        "--memory".to_string(),
                                        "128m".to_string(), // Memory limit
                                        "--cpus".to_string(),
                                        "0.5".to_string(),         // CPU limit
                                        "--read-only".to_string(), // Entire container FS read-only
                                        "--pids-limit".to_string(),
                                        "50".to_string(), // Process limit
                                        "-d".to_string(), // Datach to run in the background
                                        "-i".to_string(), // Interactive for subsequent calls
                                    ];

                                    // Mount the project dir if it exists
                                    if let (Some(project_dir), Some(container_project_dir)) = (
                                        self.config.as_ref().unwrap().project_dir.as_ref(),
                                        self.config
                                            .as_ref()
                                            .unwrap()
                                            .container_project_dir
                                            .as_ref(),
                                    ) {
                                        command_args.push("-v".to_string());
                                        command_args.push(format!(
                                            "{project_dir}:{container_project_dir}:ro"
                                        )); // Project folder read-only
                                        command_args.push("-w".to_string());
                                        command_args.push(container_project_dir.to_string());
                                    }
                                    command_args
                                }
                                (
                                    CommandSandboxRunners::DockerUnsafe,
                                    CommandSandboxRunnerState::Starting(runner_info),
                                ) => {
                                    let mut command_args = vec![
                                        "run".to_string(),
                                        "--name".to_string(), // Name the container for later calls
                                        runner_info
                                            .name
                                            .as_ref()
                                            .expect("Missing name for runner.")
                                            .to_string(),
                                        "-d".to_string(), // Datach to run in the background
                                        "-i".to_string(), // Interactive for subsequent calls
                                    ];

                                    // User defined container arguments allowed only in an unsafe environment
                                    if let Some(args) =
                                        self.config.as_ref().unwrap().container_args.as_ref()
                                    {
                                        for arg in args {
                                            command_args.push(arg.to_string());
                                        }
                                    }

                                    // Mount the project dir if it exists
                                    if let (Some(project_dir), Some(container_project_dir)) = (
                                        self.config.as_ref().unwrap().project_dir.as_ref(),
                                        self.config
                                            .as_ref()
                                            .unwrap()
                                            .container_project_dir
                                            .as_ref(),
                                    ) {
                                        command_args.push("-v".to_string());
                                        command_args
                                            .push(format!("{project_dir}:{container_project_dir}"));
                                        command_args.push("-w".to_string());
                                        command_args.push(container_project_dir.to_string());
                                    }
                                    command_args
                                }
                                (
                                    CommandSandboxRunners::Docker,
                                    CommandSandboxRunnerState::Initializing(_runner_info),
                                )
                                | (
                                    CommandSandboxRunners::DockerUnsafe,
                                    CommandSandboxRunnerState::Initializing(_runner_info),
                                )
                                | (
                                    CommandSandboxRunners::Docker,
                                    CommandSandboxRunnerState::Running(_runner_info),
                                )
                                | (
                                    CommandSandboxRunners::DockerUnsafe,
                                    CommandSandboxRunnerState::Running(_runner_info),
                                ) => {
                                    let mut command_args = vec![
                                        "exec".to_string(),
                                        "-i".to_string(), // Interactive mode to keep STDIN open
                                    ];

                                    // Mount the project dir if it exists
                                    if let Some(container_project_dir) =
                                        self.config.as_ref().unwrap().container_project_dir.as_ref()
                                    {
                                        command_args.push("-w".to_string());
                                        command_args.push(container_project_dir.to_string());
                                    }
                                    command_args
                                }
                                (_, CommandSandboxRunnerState::Done(_runner_info)) => {
                                    vec![
                                        "rm".to_string(),
                                        "-f".to_string(), // DM: not best practices but can be done in one step
                                    ]
                                }
                                _ => unreachable!(),
                            };

                        // Add environment variables to command args
                        for (k, v) in self.config.as_ref().unwrap().env_args()? {
                            command_args.push("-e".to_string());
                            command_args.push(format!("{k}={v}"));
                        }

                        // Add docker image/command CLI arguments depending upon the runner state
                        match &self.runner_state {
                            CommandSandboxRunnerState::NotStarted => unreachable!(),
                            CommandSandboxRunnerState::Starting(_runner_info) => {
                                // Add docker image and command
                                command_args.push(
                                    self.config.as_ref().unwrap().container_image.to_string(),
                                );
                            }
                            CommandSandboxRunnerState::Initializing(runner_info) => {
                                // Add container name, initialization/run script, and optional command
                                command_args.push(
                                    runner_info
                                        .name
                                        .as_ref()
                                        .expect("Missing name for runner.")
                                        .to_string(),
                                );
                                match self.config.as_ref().unwrap().environment {
                                    CommandSandboxEnvironments::Python => {
                                        if let (
                                            Some(initialization_file),
                                            Some(container_project_dir),
                                        ) = (
                                            runner_info.initialization_file.as_ref(),
                                            self.config
                                                .as_ref()
                                                .unwrap()
                                                .container_project_dir
                                                .as_ref(),
                                        ) {
                                            command_args.push("bash".to_string());
                                            command_args.push("-c".to_string());
                                            let initialization_path =
                                                Path::new(initialization_file);
                                            command_args.push(format!(
                                                "chmod +x {container_project_dir}/{} && {container_project_dir}/{}",
                                                initialization_path
                                                    .file_name()
                                                    .unwrap()
                                                    .to_str()
                                                    .unwrap(),
                                                initialization_path
                                                    .file_name()
                                                    .unwrap()
                                                    .to_str()
                                                    .unwrap()
                                            ));
                                        } else {
                                            command_args.push("python3".to_string());
                                            if let Some(command) =
                                                self.config.as_ref().unwrap().command.as_ref()
                                            {
                                                command_args.push(command.to_string());
                                            }
                                        }
                                    }
                                    CommandSandboxEnvironments::Rust => {
                                        if let (
                                            Some(initialization_file),
                                            Some(container_project_dir),
                                        ) = (
                                            runner_info.initialization_file.as_ref(),
                                            self.config
                                                .as_ref()
                                                .unwrap()
                                                .container_project_dir
                                                .as_ref(),
                                        ) {
                                            command_args.push("bash".to_string());
                                            command_args.push("-c".to_string());
                                            let initialization_path =
                                                Path::new(initialization_file);
                                            command_args.push(format!(
                                                "chmod +x {container_project_dir}/{} && {container_project_dir}/{}",
                                                initialization_path
                                                    .file_name()
                                                    .unwrap()
                                                    .to_str()
                                                    .unwrap(),
                                                initialization_path
                                                    .file_name()
                                                    .unwrap()
                                                    .to_str()
                                                    .unwrap()
                                            ));
                                        } else {
                                            command_args.push("cargo".to_string());
                                            if let Some(command) =
                                                self.config.as_ref().unwrap().command.as_ref()
                                            {
                                                command_args.push(command.to_string());
                                            }
                                        }
                                    }
                                    CommandSandboxEnvironments::Bash => {
                                        if let (
                                            Some(initialization_file),
                                            Some(container_project_dir),
                                        ) = (
                                            runner_info.initialization_file.as_ref(),
                                            self.config
                                                .as_ref()
                                                .unwrap()
                                                .container_project_dir
                                                .as_ref(),
                                        ) {
                                            command_args.push("sh".to_string());
                                            command_args.push("-c".to_string());
                                            let initialization_path =
                                                Path::new(initialization_file);
                                            command_args.push(format!(
                                                "chmod +x {container_project_dir}/{} && {container_project_dir}/{}",
                                                initialization_path
                                                    .file_name()
                                                    .unwrap()
                                                    .to_str()
                                                    .unwrap(),
                                                initialization_path
                                                    .file_name()
                                                    .unwrap()
                                                    .to_str()
                                                    .unwrap()
                                            ));
                                        } else {
                                            command_args.push("bash".to_string());
                                            command_args.push("-c".to_string());
                                            if let Some(command) =
                                                self.config.as_ref().unwrap().command.as_ref()
                                            {
                                                command_args.push(command.to_string());
                                            }
                                        }
                                    }
                                    _ => {
                                        self.stream_state = CommandSandboxStreamState::Done;
                                        return Poll::Ready(Some(Err(anyhow!(
                                            "Environment type {} is not yet supported for Runner type {}.",
                                            self.config.as_ref().unwrap().environment,
                                            self.config.as_ref().unwrap().runner
                                        ))));
                                    }
                                }

                                // // User defined CLI arguments
                                // if let Some(args) = self.config.as_ref().unwrap().cli_args.as_ref()
                                // {
                                //     for arg in args {
                                //         command_args.push(arg.to_string());
                                //     }
                                // }

                                // Add the data
                                if runner_info.initialization_file.is_none() {
                                    match (
                                        &self.config.as_ref().unwrap().data_i,
                                        &self.config.as_ref().unwrap().data_o,
                                        self.config
                                            .as_ref()
                                            .unwrap()
                                            .container_project_dir
                                            .as_ref(),
                                    ) {
                                        (
                                            DataIOMethod::Stdio,
                                            DataIOMethod::TempFile,
                                            Some(container_project_dir),
                                        ) => {
                                            command_args.push("--input".to_string());
                                            command_args.push(
                                                runner_info
                                                    .content
                                                    .as_ref()
                                                    .expect("Missing content for runner.")
                                                    .to_string(),
                                            );
                                            command_args.push("--output-file".to_string());
                                            let output_path = Path::new(runner_info.output_file.as_ref().ok_or(anyhow!("Container output path must be provided for data output method {}.", self.config.as_ref().unwrap().data_o))?);
                                            command_args.push(format!(
                                                "{}/{}",
                                                container_project_dir,
                                                output_path.file_name().unwrap().to_str().unwrap()
                                            ));
                                        }
                                        (DataIOMethod::Stdio, DataIOMethod::Stdio, _)
                                        | (DataIOMethod::Stdio, DataIOMethod::None, _) => {
                                            command_args.push("--input".to_string());
                                            command_args.push(
                                                runner_info
                                                    .content
                                                    .as_ref()
                                                    .expect("Missing content for runner.")
                                                    .to_string(),
                                            );
                                        }
                                        (
                                            DataIOMethod::TempFile,
                                            DataIOMethod::TempFile,
                                            Some(container_project_dir),
                                        ) => {
                                            command_args.push("--input-file".to_string());
                                            let input_path = Path::new(runner_info.input_file.as_ref().ok_or(anyhow!("Container input path must be provided for data output method {}.", self.config.as_ref().unwrap().data_i))?);
                                            command_args.push(format!(
                                                "{}/{}",
                                                container_project_dir,
                                                input_path.file_name().unwrap().to_str().unwrap()
                                            ));
                                            command_args.push("--output-file".to_string());
                                            let output_path = Path::new(runner_info.output_file.as_ref().ok_or(anyhow!("Container output path must be provided for data output method {}.", self.config.as_ref().unwrap().data_o))?);
                                            command_args.push(format!(
                                                "{}/{}",
                                                container_project_dir,
                                                output_path.file_name().unwrap().to_str().unwrap()
                                            ));
                                        }
                                        (
                                            DataIOMethod::TempFile,
                                            DataIOMethod::Stdio,
                                            Some(container_project_dir),
                                        )
                                        | (
                                            DataIOMethod::TempFile,
                                            DataIOMethod::None,
                                            Some(container_project_dir),
                                        ) => {
                                            command_args.push("--input-file".to_string());
                                            let input_path = Path::new(runner_info.input_file.as_ref().ok_or(anyhow!("Container input path must be provided for data output method {}.", self.config.as_ref().unwrap().data_i))?);
                                            command_args.push(format!(
                                                "{}/{}",
                                                container_project_dir,
                                                input_path.file_name().unwrap().to_str().unwrap()
                                            ));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            CommandSandboxRunnerState::Running(runner_info) => {
                                // Add docker container name, run script, and optional command
                                command_args.push(
                                    runner_info
                                        .name
                                        .as_ref()
                                        .expect("Missing name for runner.")
                                        .to_string(),
                                );
                                match self.config.as_ref().unwrap().environment {
                                    CommandSandboxEnvironments::Python => {
                                        if let (Some(run_file), Some(container_project_dir)) = (
                                            runner_info.run_file.as_ref(),
                                            self.config
                                                .as_ref()
                                                .unwrap()
                                                .container_project_dir
                                                .as_ref(),
                                        ) {
                                            command_args.push(format!(
                                                "{container_project_dir}/.venv/bin/python"
                                            ));
                                            let run_path = Path::new(run_file);
                                            command_args.push(format!(
                                                "{container_project_dir}/src/{}",
                                                run_path.file_name().unwrap().to_str().unwrap()
                                            ));
                                        } else {
                                            command_args.push("python3".to_string());
                                            if let Some(command) =
                                                self.config.as_ref().unwrap().command.as_ref()
                                            {
                                                command_args.push(command.to_string());
                                            }
                                        }
                                    }
                                    CommandSandboxEnvironments::Rust => {
                                        if let (Some(_run_file), Some(_container_project_dir)) = (
                                            runner_info.run_file.as_ref(),
                                            self.config
                                                .as_ref()
                                                .unwrap()
                                                .container_project_dir
                                                .as_ref(),
                                        ) {
                                            command_args.push("cargo".to_string());
                                            command_args.push("run".to_string());
                                            // let run_path = Path::new(run_file);
                                            // command_args.push(format!("{}/src/{}", container_project_dir, run_path.file_name().unwrap().to_str().unwrap()));
                                        } else {
                                            command_args.push("cargo".to_string());
                                            if let Some(command) =
                                                self.config.as_ref().unwrap().command.as_ref()
                                            {
                                                command_args.push(command.to_string());
                                            }
                                        }
                                    }
                                    CommandSandboxEnvironments::Bash => {
                                        if let (Some(run_file), Some(container_project_dir)) = (
                                            runner_info.run_file.as_ref(),
                                            self.config
                                                .as_ref()
                                                .unwrap()
                                                .container_project_dir
                                                .as_ref(),
                                        ) {
                                            // command_args.push("chmod".to_string());
                                            // command_args.push("+x".to_string());
                                            let run_path = Path::new(run_file);
                                            // command_args.push(format!(
                                            //     "{container_project_dir}/src/{}",
                                            //     run_path.file_name().unwrap().to_str().unwrap(),
                                            // ));
                                            // command_args.push("&&".to_string());
                                            command_args.push(format!(
                                                "{container_project_dir}/src/{}",
                                                run_path.file_name().unwrap().to_str().unwrap(),
                                            ));
                                        } else if let Some(command) =
                                            self.config.as_ref().unwrap().command.as_ref()
                                        {
                                            command_args.push(command.to_string());
                                        }
                                    }
                                    _ => {
                                        self.stream_state = CommandSandboxStreamState::Done;
                                        return Poll::Ready(Some(Err(anyhow!(
                                            "Environment type {} is not yet supported for Runner type {}.",
                                            self.config.as_ref().unwrap().environment,
                                            self.config.as_ref().unwrap().runner
                                        ))));
                                    }
                                }

                                // User defined ClI arguments
                                if let Some(args) = self.config.as_ref().unwrap().cli_args.as_ref()
                                {
                                    for arg in args {
                                        command_args.push(arg.to_string());
                                    }
                                }

                                // Add the data
                                match (
                                    &self.config.as_ref().unwrap().data_i,
                                    &self.config.as_ref().unwrap().data_o,
                                    self.config.as_ref().unwrap().container_project_dir.as_ref(),
                                ) {
                                    (
                                        DataIOMethod::Stdio,
                                        DataIOMethod::TempFile,
                                        Some(container_project_dir),
                                    ) => {
                                        command_args.push("--input".to_string());
                                        command_args.push(
                                            runner_info
                                                .content
                                                .as_ref()
                                                .expect("Missing content for runner.")
                                                .to_string(),
                                        );
                                        command_args.push("--output-file".to_string());
                                        let output_path = Path::new(runner_info.output_file.as_ref().ok_or(anyhow!("Container output path must be provided for data output method {}.", self.config.as_ref().unwrap().data_o))?);
                                        command_args.push(format!(
                                            "{container_project_dir}/{}",
                                            output_path.file_name().unwrap().to_str().unwrap()
                                        ));
                                    }
                                    (DataIOMethod::Stdio, DataIOMethod::Stdio, _)
                                    | (DataIOMethod::Stdio, DataIOMethod::None, _) => {
                                        command_args.push("--input".to_string());
                                        command_args.push(
                                            runner_info
                                                .content
                                                .as_ref()
                                                .expect("Missing content for runner.")
                                                .to_string(),
                                        );
                                    }
                                    (
                                        DataIOMethod::TempFile,
                                        DataIOMethod::TempFile,
                                        Some(container_project_dir),
                                    ) => {
                                        command_args.push("--input-file".to_string());
                                        let input_path = Path::new(runner_info.input_file.as_ref().ok_or(anyhow!("Container input path must be provided for data output method {}.", self.config.as_ref().unwrap().data_i))?);
                                        command_args.push(format!(
                                            "{container_project_dir}/{}",
                                            input_path.file_name().unwrap().to_str().unwrap()
                                        ));
                                        command_args.push("--output-file".to_string());
                                        let output_path = Path::new(runner_info.output_file.as_ref().ok_or(anyhow!("Container output path must be provided for data output method {}.", self.config.as_ref().unwrap().data_o))?);
                                        command_args.push(format!(
                                            "{container_project_dir}/{}",
                                            output_path.file_name().unwrap().to_str().unwrap()
                                        ));
                                    }
                                    (
                                        DataIOMethod::TempFile,
                                        DataIOMethod::Stdio,
                                        Some(container_project_dir),
                                    )
                                    | (
                                        DataIOMethod::TempFile,
                                        DataIOMethod::None,
                                        Some(container_project_dir),
                                    ) => {
                                        command_args.push("--input-file".to_string());
                                        let input_path = Path::new(runner_info.input_file.as_ref().ok_or(anyhow!("Container input path must be provided for data output method {}.", self.config.as_ref().unwrap().data_i))?);
                                        command_args.push(format!(
                                            "{container_project_dir}/{}",
                                            input_path.file_name().unwrap().to_str().unwrap()
                                        ));
                                    }
                                    _ => {}
                                }
                            }
                            CommandSandboxRunnerState::Done(runner_info) => {
                                // Add container name
                                command_args.push(
                                    runner_info
                                        .name
                                        .as_ref()
                                        .expect("Missing name for runner.")
                                        .to_string(),
                                );
                            }
                        }

                        // Run the command
                        // DM: useful for debugging
                        dbg!(&command_args);
                        Command::new("docker").args(&command_args).output()
                    }
                    CommandSandboxRunners::Wasmtime => {
                        // Build wasmtime args
                        let command_args = match &self.runner_state {
                            CommandSandboxRunnerState::NotStarted
                            | CommandSandboxRunnerState::Starting(_)
                            | CommandSandboxRunnerState::Initializing(_) => unreachable!(),
                            CommandSandboxRunnerState::Running(runner_info) => {
                                let mut command_args = Vec::new();

                                // Add run for the component model
                                if let CommandSandboxEnvironments::WasmComponent =
                                    self.config.as_ref().unwrap().environment
                                {
                                    command_args.push("run".to_string());
                                }

                                // User defined container arguments
                                if let Some(args) =
                                    self.config.as_ref().unwrap().container_args.as_ref()
                                {
                                    for arg in args {
                                        command_args.push(arg.to_string());
                                    }
                                }

                                // Mount the project dir if it exists
                                if let Some(project_dir) =
                                    self.config.as_ref().unwrap().project_dir.as_ref()
                                {
                                    command_args.push(format!("--dir={project_dir}"));
                                }

                                // Add the input file
                                if DataIOMethod::TempFile == self.config.as_ref().unwrap().data_i {
                                    let path = Path::new(
                                        runner_info
                                            .input_file
                                            .as_ref()
                                            .expect("Missing tempfile for runner."),
                                    );
                                    let host_input_dir =
                                        path.parent().unwrap().to_str().unwrap().to_string();
                                    command_args.push(format!("--dir={host_input_dir}")); // Input file read-only
                                }

                                // Add the output file
                                if DataIOMethod::TempFile == self.config.as_ref().unwrap().data_o {
                                    let path = Path::new(
                                        runner_info
                                            .output_file
                                            .as_ref()
                                            .expect("Missing tempfile for runner."),
                                    );
                                    let host_output_dir =
                                        path.parent().unwrap().to_str().unwrap().to_string();
                                    command_args.push(format!("--dir={host_output_dir}"));
                                }

                                // Add environment variables to command args
                                for (k, v) in self.config.as_ref().unwrap().env_args()? {
                                    command_args.push(format!("--env={k}={v}"));
                                }

                                // Add command for the component model
                                match self.config.as_ref().unwrap().environment {
                                    CommandSandboxEnvironments::WasmComponent => {
                                        command_args.push("--invoke".to_string());
                                        let command = self.config.as_ref().unwrap().command.as_ref().ok_or(anyhow!("Command to run must be defined when using the {} environment.", self.config.as_ref().unwrap().environment))?;
                                        let mut args_vec = if let Some(args) =
                                            self.config.as_ref().unwrap().cli_args.as_ref()
                                        {
                                            args.to_owned()
                                        } else {
                                            Vec::new()
                                        };

                                        // Extract out the message and add as the last argument
                                        match self.config.as_ref().unwrap().data_i {
                                            DataIOMethod::Stdio => {
                                                args_vec.push(
                                                    runner_info
                                                        .content
                                                        .as_ref()
                                                        .expect("Missing content for runner.")
                                                        .to_string(),
                                                );
                                            }
                                            DataIOMethod::TempFile => {
                                                let host_input_path = runner_info
                                                    .input_file
                                                    .as_ref()
                                                    .expect("Missing tempfile for runner.")
                                                    .to_string();
                                                args_vec.push(host_input_path);
                                            }
                                            DataIOMethod::None => {}
                                        };
                                        command_args
                                            .push(format!("{command}({})", args_vec.join(" ,")));
                                    }
                                    _ => {
                                        if let Some(command) =
                                            self.config.as_ref().unwrap().command.as_ref()
                                        {
                                            command_args.push(command.to_string());
                                        }
                                    }
                                }

                                // Add the container image
                                // DM: this is a bit confusing since wasmtime does not have a "container"
                                //  however, WASM treats executables and "container"s since they are intended to
                                //  be resusable components...
                                command_args.push(
                                    self.config.as_ref().unwrap().container_image.to_string(),
                                );

                                // User defined CLI arguments for modules
                                if let CommandSandboxEnvironments::WasmModule =
                                    self.config.as_ref().unwrap().environment
                                {
                                    if let Some(args) =
                                        self.config.as_ref().unwrap().cli_args.as_ref()
                                    {
                                        for arg in args {
                                            command_args.push(arg.to_string());
                                        }
                                    }

                                    // Extract out the message and add as CLI arguments
                                    match self.config.as_ref().unwrap().data_i {
                                        DataIOMethod::Stdio => {
                                            command_args.push("--input".to_string());
                                            command_args.push(
                                                runner_info
                                                    .content
                                                    .as_ref()
                                                    .expect("Missing content for runner.")
                                                    .to_string(),
                                            );
                                        }
                                        DataIOMethod::TempFile => {
                                            command_args.push("--input-file".to_string());
                                            let host_input_path = runner_info
                                                .input_file
                                                .as_ref()
                                                .expect("Missing input tempfile for runner.")
                                                .to_string();
                                            command_args.push(host_input_path);
                                        }
                                        DataIOMethod::None => {}
                                    };
                                    // Add data
                                    match (
                                        &self.config.as_ref().unwrap().data_i,
                                        &self.config.as_ref().unwrap().data_o,
                                    ) {
                                        (DataIOMethod::Stdio, DataIOMethod::TempFile) => {
                                            command_args.push("--input".to_string());
                                            command_args.push(
                                                runner_info
                                                    .content
                                                    .as_ref()
                                                    .expect("Missing content for runner.")
                                                    .to_string(),
                                            );
                                            command_args.push("--output-file".to_string());
                                            command_args.push(
                                                runner_info
                                                    .output_file
                                                    .as_ref()
                                                    .expect("Missing output tempfile for runner.")
                                                    .to_string(),
                                            );
                                        }
                                        (DataIOMethod::Stdio, DataIOMethod::Stdio)
                                        | (DataIOMethod::Stdio, DataIOMethod::None) => {
                                            command_args.push("--input".to_string());
                                            command_args.push(
                                                runner_info
                                                    .content
                                                    .as_ref()
                                                    .expect("Missing content for runner.")
                                                    .to_string(),
                                            );
                                        }
                                        (DataIOMethod::TempFile, DataIOMethod::TempFile) => {
                                            command_args.push("--input-file".to_string());
                                            command_args.push(
                                                runner_info
                                                    .input_file
                                                    .as_ref()
                                                    .expect("Missing input tempfile for runner.")
                                                    .to_string(),
                                            );
                                            command_args.push("--output-file".to_string());
                                            command_args.push(
                                                runner_info
                                                    .output_file
                                                    .as_ref()
                                                    .expect("Missing output tempfile for runner.")
                                                    .to_string(),
                                            );
                                        }
                                        (DataIOMethod::TempFile, DataIOMethod::Stdio)
                                        | (DataIOMethod::TempFile, DataIOMethod::None) => {
                                            command_args.push("--input-file".to_string());
                                            command_args.push(
                                                runner_info
                                                    .input_file
                                                    .as_ref()
                                                    .expect("Missing input tempfile for runner.")
                                                    .to_string(),
                                            );
                                        }
                                        _ => {}
                                    }
                                }
                                command_args
                            }
                            CommandSandboxRunnerState::Done(_runner_info) => {
                                self.stream_state = CommandSandboxStreamState::Done;
                                return self.poll_next(cx);
                            }
                        };

                        // Run the command
                        dbg!(&command_args);
                        Command::new("wasmtime").args(&command_args).output()
                    }
                    _ => {
                        self.stream_state = CommandSandboxStreamState::Done;
                        return Poll::Ready(Some(Err(anyhow!(
                            "Runner type {} is not supported yet.",
                            self.config.as_ref().unwrap().runner
                        ))));
                    }
                };

                // Update the request state and poll next
                self.stream_state = CommandSandboxStreamState::Output(Box::pin(fut));
                self.poll_next(cx)
            }
            CommandSandboxStreamState::Output(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
                Ok(output) => {
                    // Initialize the metrics
                    let baseline_metrics =
                        if let Some(diagnostic_builder) = &self.diagnostic_builder {
                            Some(
                                diagnostic_builder
                                    .clone()
                                    .to_child("CommandSandboxStream")?
                                    .baseline_metrics(line!(), file!(), "poll_next"),
                            )
                        } else {
                            None
                        };
                    let _timer = baseline_metrics
                        .as_ref()
                        .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

                    // Check for an error
                    if let Some(exit_code) = &output.status.code()
                        && exit_code != &1
                        && !output.status.success()
                    {
                        self.stream_state = CommandSandboxStreamState::Done;
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        return Poll::Ready(Some(Err(anyhow!(
                            "Command exited with code {}, stderr {}, and stdout {}.",
                            output.status.code().unwrap_or(0),
                            stderr,
                            stdout
                        ))));
                    }
                    // DM: useful for debugging
                    {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        dbg!(stdout);
                    }

                    // Parse the response if running and skip if starting, initializing, or done
                    let (batch, stream_state) =
                        match (&self.config.as_ref().unwrap().data_o, &self.runner_state) {
                            (
                                DataIOMethod::None,
                                CommandSandboxRunnerState::Running(_runner_info),
                            ) => {
                                let stdout = String::from_utf8_lossy(&output.stdout);
                                let batch = create_chat_record_batch(
                                    vec!["tool".to_string()],
                                    vec![stdout.to_string()],
                                    vec![create_timestamp_micros()],
                                )?;
                                (Some(batch), CommandSandboxStreamState::NewPoll)
                            }
                            (
                                DataIOMethod::Stdio,
                                CommandSandboxRunnerState::Running(_runner_info),
                            ) => {
                                let json_values =
                                    serde_json::from_slice::<Vec<Value>>(&output.stdout)?;
                                let table = SubjectBuilder::new()
                                    .with_name("sandbox_stdio_running")
                                    .with_schema(self.schema.clone())
                                    .with_json_values(&json_values)?
                                    .build()?;
                                let batch = table.get_record_batches_own().pop().unwrap();
                                (Some(batch), CommandSandboxStreamState::NewPoll)
                            }
                            (
                                DataIOMethod::TempFile,
                                CommandSandboxRunnerState::Running(runner_info),
                            ) => {
                                let file = fs::File::open(
                                    runner_info
                                        .output_file
                                        .as_ref()
                                        .expect("Missing output TempFile from runner."),
                                )?;
                                let table = SubjectBuilder::new_from_ipc_file(file)?
                                    .with_name("sandbox_tempfile_running")
                                    .build()?;
                                let batch = table.get_record_batches_own().pop().unwrap();
                                (Some(batch), CommandSandboxStreamState::NewPoll)
                            }
                            (
                                DataIOMethod::None,
                                CommandSandboxRunnerState::Initializing(_runner_info),
                            )
                            | (
                                DataIOMethod::Stdio,
                                CommandSandboxRunnerState::Initializing(_runner_info),
                            )
                            | (
                                DataIOMethod::TempFile,
                                CommandSandboxRunnerState::Initializing(_runner_info),
                            ) => (None, CommandSandboxStreamState::ExistingPoll),
                            (
                                DataIOMethod::None,
                                CommandSandboxRunnerState::Starting(_runner_info),
                            )
                            | (
                                DataIOMethod::Stdio,
                                CommandSandboxRunnerState::Starting(_runner_info),
                            )
                            | (
                                DataIOMethod::TempFile,
                                CommandSandboxRunnerState::Starting(_runner_info),
                            ) => (None, CommandSandboxStreamState::ExistingPoll),
                            (_, CommandSandboxRunnerState::Done(_runner_info)) => {
                                (None, CommandSandboxStreamState::Done)
                            }
                            _ => unreachable!(),
                        };

                    // Record the poll
                    self.stream_state = stream_state;
                    if let Some(batch) = batch {
                        let poll = Poll::Ready(Some(Ok(batch)));
                        if let Some(baseline_metrics) = &baseline_metrics {
                            baseline_metrics.record_poll(poll)
                        } else {
                            poll
                        }
                    } else {
                        self.poll_next(cx)
                    }
                }
                Err(err) => {
                    self.stream_state = CommandSandboxStreamState::Done;
                    Poll::Ready(Some(Err(anyhow!(error_report(&err)))))
                }
            },
            CommandSandboxStreamState::Done => {
                match &self.runner_state {
                    CommandSandboxRunnerState::NotStarted => Poll::Ready(None),
                    CommandSandboxRunnerState::Starting(runner_info)
                    | CommandSandboxRunnerState::Initializing(runner_info)
                    | CommandSandboxRunnerState::Running(runner_info) => {
                        self.runner_state = CommandSandboxRunnerState::Done(runner_info.to_owned());
                        self.poll_next(cx)
                    }
                    CommandSandboxRunnerState::Done(runner_info) => {
                        // Remove the temporary input/output file
                        if let Some(input_file) = runner_info.input_file.as_ref()
                            && fs::metadata(input_file).is_ok()
                        {
                            fs::remove_file(input_file)?;
                        }
                        if let Some(output_file) = runner_info.output_file.as_ref()
                            && fs::metadata(output_file).is_ok()
                        {
                            fs::remove_file(output_file)?;
                        }

                        // End the poll
                        Poll::Ready(None)
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl RecordBatchStream for CommandSandboxStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}