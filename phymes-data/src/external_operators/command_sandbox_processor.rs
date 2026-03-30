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
    AvailableSchemaTrait, AvailableSubjects, BuildableTrait, BuilderTrait, MappableTrait,
    MessageBuilderTrait, MessageTrait, ProcessorTrait, RecordBatchStream, RuntimeEnv,
    SendableRecordBatchStream, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, Subject, SubjectBuilder, SubjectBuilderTrait,
    SubjectTrait, WorkspaceEditor, create_bytes_fields, create_chat_record_batch,
    create_values_fields, remove_message_by_subject,
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, create_timestamp_micros,
};
use serde_json::Value;
use tempfile::NamedTempFile;
use tokio::process::Command;

use crate::{
    DataConfigTrait,
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

/// Runs commands in a sandboxed environment and returns the result or error message
///
/// # Notes
/// * Current sandboxed environments include docker and wasmtime
/// * Stderr messages are routed to `SessionError`s
#[derive(Debug)]
pub struct CommandSandboxProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for CommandSandboxProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for CommandSandboxProcessor {
    fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
        }
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn line_and_file(&self) -> (u32, String) {
        (line!(), file!().to_string())
    }

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Run the stream
        let out = Box::pin(CommandSandboxStream::new(
            message,
            config,
            Arc::clone(&runtime_env),
            diagnostic_builder.cloned(),
        )?);

        // Prepare the message builder
        let mut builder_map = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
        let builder = SendableRecordBatchStreamMessage::get_builder()
            .with_name(self.get_name())
            .with_message(out);
        let _ = builder_map.insert(self.get_name().to_string(), builder);

        Ok(builder_map)
    }
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

pub mod test_command_sandbox_processor {
    use super::*;
    use arrow::array::{ArrayRef, StringArray, UInt32Array};

    pub fn create_messages() -> Result<RecordBatch> {
        let names = vec!["Alice", "Bob"];
        let names_arr: ArrayRef = Arc::new(StringArray::from(names));
        let ages = vec![30, 25];
        let ages_arr: ArrayRef = Arc::new(UInt32Array::from(ages));
        let batch = RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, StringArray, UInt32Array};
    use futures::TryStreamExt;
    use phymes_core::{ChatBuilderTraitExt, Publication, SubjectBuilder};
    use phymes_diagnostics::{Diagnostics, SpanBuilder};
    use std::{fs::File, io::Write};
    use tempfile::TempDir;

    use crate::external_operators::command_sandbox_config::DataIOMethod;

    use super::*;

    #[tokio::test]
    async fn test_command_sandbox_processor_wasmtime_no_workspace_no_messages() -> Result<()> {
        let name = "CommandSandboxProcessor";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // --- From config with wasm module env ---

        // Create the wasm module
        let wasm_module_str = r#"(module
  (func (export "add") (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )
)"#;
        let mut wasm_module_file = NamedTempFile::new()?;
        writeln!(wasm_module_file, "{wasm_module_str}")?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            timeout: 5,
            runner: CommandSandboxRunners::Wasmtime,
            environment: CommandSandboxEnvironments::WasmModule,
            container_image: wasm_module_file.path().to_str().unwrap().to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("add".to_string()), // DM: mimic module style CLI without WAVE
            container_args: Some(vec!["run".to_string(), "--invoke".to_string()]), // DM: mimic module style CLI without WAVE
            cli_args: Some(vec!["1".to_string(), "2".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["3\n"]);

        // --- From config with wasm component env ---

        // Create the wasm module
        let wasm_module_str = r#"(component
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
)"#;
        let mut wasm_module_file = NamedTempFile::new()?;
        writeln!(wasm_module_file, "{wasm_module_str}")?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Wasmtime,
            environment: CommandSandboxEnvironments::WasmComponent,
            container_image: wasm_module_file.path().to_str().unwrap().to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("add".to_string()),
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["1".to_string(), "2".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["3\n"]);

        // --- From stdio ---
        // DM: requires named arguments which is only possible with .wasm
        // DM, todo: create a small wasm example that takes a RecordBatch, modifies it, and returns the modified RecordBatch

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_wasmtime_workspace_no_messages() -> Result<()> {
        let name = "CommandSandboxProcessor";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Create project directory
        let project_name = "phymes-wasm-workspace";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = fs::create_dir(&project_dir);

        // --- From config with wasm module env ---

        // Create the workspace
        let workspace_table = CommandSandboxEnvironments::WasmModule.to_default_workspace(None)?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            timeout: 5,
            runner: CommandSandboxRunners::Wasmtime,
            environment: CommandSandboxEnvironments::WasmModule,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_image: project_dir
                .join("src/main.wat")
                .as_path()
                .to_str()
                .unwrap()
                .to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: None,
            container_args: Some(vec![
                "run".to_string(),
                "--invoke".to_string(),
                "add".to_string(),
            ]), // DM: mimic module style CLI without WAVE
            cli_args: Some(vec!["1".to_string(), "2".to_string()]),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["3\n"]);

        // --- From config with wasm component env ---
        let workspace_table =
            CommandSandboxEnvironments::WasmComponent.to_default_workspace(None)?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Wasmtime,
            environment: CommandSandboxEnvironments::WasmComponent,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_image: project_dir
                .join("src/main.wat")
                .as_path()
                .to_str()
                .unwrap()
                .to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("add".to_string()),
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["1".to_string(), "2".to_string()]),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["3\n"]);

        // --- From stdio ---
        // DM: requires named arguments which is only possible with .wasm
        // DM, todo: create a small wasm example that takes a RecordBatch, modifies it, and returns the modified RecordBatch

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_bash_no_workspace() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // --- From config ---
        // based on docker run --rm alpine echo "Hello from Docker!"

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Bash,
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("echo".to_string()),
            timeout: 5,
            cli_args: Some(vec!["Hello from Docker!".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains("Hello from Docker"));

        // --- From Stdio ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Bash,
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::None,
            command: Some("echo".to_string()),
            timeout: 5,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .append_new_user_query_str("Hello from Docker!", "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains(
            "--input [{\"content\":\"Hello from Docker!\",\"role\":\"user\",\"timestamp\":"
        ));

        // --- From TempFile ---

        // Create project directory
        let project_name = "phymes-bash-project";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = fs::create_dir(&project_dir);

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Bash,
            container_image: "alpine".to_string(),
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::None,
            command: Some("echo".to_string()),
            timeout: 5,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(
            result
                .first()
                .unwrap()
                .contains("--input-file /home/sandbox/input.ipc")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_bash_workspace() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Create project directory
        let project_name = "phymes-bash-workspace";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = fs::create_dir(&project_dir);

        // --- From config ---
        // based on docker run --rm alpine echo "Hello from Docker!"

        // Create the workspace
        let workspace_table = CommandSandboxEnvironments::Bash.to_default_workspace(None)?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Bash,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.sh".to_string()),
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            cli_args: Some(vec!["Hello from Docker!".to_string()]),
            timeout: 5,
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_bash_workspace From config")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains("Hello from Docker"));

        // --- From Stdio ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Bash,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.sh".to_string()),
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::None,
            timeout: 5,
            subject_name: Some(messages.to_string()),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = SubjectBuilder::new()
            .with_name(messages)
            .append_new_user_query_str("Hello from Docker!", "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_bash_workspace From Stdio")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains(
            "--input\n[{\"content\":\"Hello from Docker!\",\"role\":\"user\",\"timestamp\":"
        ));

        // --- From TempFile ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Bash,
            container_image: "alpine".to_string(),
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.sh".to_string()),
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::None,
            timeout: 5,
            subject_name: Some(messages.to_string()),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_bash_workspace From Tempfile")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(
            result
                .first()
                .unwrap()
                .contains("--input-file\n/home/sandbox/input.ipc")
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py_no_workspace_no_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // --- From config ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("-c".to_string()),
            timeout: 5,
            cli_args: Some(vec![
                "import json, sys; data=json.loads(sys.argv[1]); print(json.dumps([{\"name\": item[\"name\"]} for item in data]))".to_string(),
                "[{\"name\": \"Alice\", \"age\": 30}, {\"name\": \"Bob\", \"age\": 25}]".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_run from Config")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(
            result
                .first()
                .unwrap()
                .contains("[{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]")
        );

        // --- From Stdio ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::Stdio,
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            command: Some("-c".to_string()),
            timeout: 5,
            cli_args: Some(vec!["import json, argparse; \
                parser = argparse.ArgumentParser(); \
                parser.add_argument('--input', required=True); \
                args = parser.parse_args(); \
                data = json.loads(args.input); \
                print(json.dumps([{\"name\": item[\"name\"], \"age\": item[\"age\"] + 10} for item in data]))".to_string()]),
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_run from STDIO")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py_no_workspace_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Create project directory
        let project_name = "phymes-py-project";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = fs::create_dir(&project_dir);

        // Create src directory
        let src_path = format!("{}/src", project_dir.as_path().to_str().unwrap());
        let _ = fs::create_dir(&src_path);

        // Create the requirements.txt
        let requirements_file_path = format!(
            "{}/requirements.txt",
            project_dir.as_path().to_str().unwrap()
        );
        let mut requirements_file =
            File::create(&requirements_file_path).expect("Failed to create requirements.txt");
        let requirements_str = r#"pandas==2.2.3
pyarrow==17.0.0"#;
        let _ = requirements_file.write(requirements_str.as_bytes())?;
        requirements_file.flush()?;

        // Create the initialization script
        let initialization_str = r#"#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt"#;

        // Create the run script
        let run_str = r#"#!/usr/bin/env python3
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
    
    # Write back out as Feather v2 (fully compatible with Pandas)
    #df.to_feather(args.output_file, version=2)

    # Convert pandas back to Arrow
    table_out = pa.Table.from_pandas(df)

    # Write Arrow IPC File format (Rust-compatible)
    with pa.OSFile(args.output_file, "wb") as f:
        writer = ipc.RecordBatchFileWriter(f, table_out.schema)
        writer.write_table(table_out)
        writer.close()"#;

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: Some(initialization_str.to_string()),
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: None,
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_install")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py_workspace_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Create project directory
        let project_name = "phymes-py-workspace";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = fs::create_dir(&project_dir);

        // Create the workspace
        let workspace_table = CommandSandboxEnvironments::Python.to_default_workspace(None)?;

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.py".to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: None,
            subject_name: Some(messages.to_string()),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_install")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_rs_no_workspace_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Create project directory
        let project_name = "phymes-rs-project";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = fs::create_dir(&project_dir);

        // Create src directory
        let src_path = format!("{}/src", project_dir.as_path().to_str().unwrap());
        let _ = fs::create_dir(&src_path);

        // Create the cargo.toml
        let requirements_file_path =
            format!("{}/Cargo.toml", project_dir.as_path().to_str().unwrap());
        let mut requirements_file =
            File::create(&requirements_file_path).expect("Failed to create Cargo.toml");
        let requirements_str = r#"[package]
name = "phymes_rs"
version = "0.1.0"
edition = "2024"

[dependencies]
anyhow = { version = "1", default-features = false }
arrow = "58.0.0"
serde_json = "1.0.133"
serde = { version = "1.0.215", features = ["derive"] }
clap = { version = "4.5.4", features = ["derive"] }"#;
        let _ = requirements_file.write(requirements_str.as_bytes())?;
        requirements_file.flush()?;

        // Create the run script
        let run_str = r#"use std::io::Write;
use anyhow::Result;
use clap::Parser;

/// Minimal Feather/IPC editor using a single CLI argument.
#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
struct Args {
    /// Input string
    #[arg(long)]
    input: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Data {
    /// Name
    name: String,
    /// Age
    age: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = serde_json::from_str::<Vec<Data>>(&args.input)?;
    let modified = input.into_iter()
        .map(|d| Data { name: d.name, age: d.age + 10 })
        .collect::<Vec<_>>();
    let serialized = serde_json::to_string(&modified)?;
    let _bytes = std::io::stdout().write(serialized.as_bytes())?;
    std::io::stdout().flush().unwrap();

    Ok(())
}"#;

        // --- from Config, no initialization, and Error ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            project_dir: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: None,
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec![
                "--release".to_string(),
                "--".to_string(),
                "--input".to_string(),
                "[{\"name\": \"Alice\", \"age\": 30}, {\"name\": \"Bob\", \"age\": 25}]"
                    .to_string(),
            ]),
            subject_name: None,
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await;
        if let Err(err) = &result {
            assert!(
                err.to_string()
                    .contains("error: cannot find derive macro `Serialize` in this scope")
            );
            assert!(
                err.to_string()
                    .contains("error: cannot find derive macro `Deserialize` in this scope")
            );
            assert!(err.to_string().contains(
                "error[E0277]: the trait bound `Data: serde::Deserialize<'de>` is not satisfied"
            ));
        } else {
            panic!("Should have generated an Error.")
        }

        // Create the run script
        let run_str = r#"use std::io::Write;
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

/// Minimal Feather/IPC editor using a single CLI argument.
#[derive(Parser, Debug, Serialize, Deserialize)]
#[command(author, version, about)]
struct Args {
    /// Input string
    #[arg(long)]
    input: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Data {
    /// Name
    name: String,
    /// Age
    age: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input = serde_json::from_str::<Vec<Data>>(&args.input)?;
    let modified = input.into_iter()
        .map(|d| Data { name: d.name, age: d.age + 10 })
        .collect::<Vec<_>>();
    let serialized = serde_json::to_string(&modified)?;
    let _bytes = std::io::stdout().write(serialized.as_bytes())?;
    std::io::stdout().flush().unwrap();

    Ok(())
}"#;

        // --- from Config, no initialization ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            project_dir: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: None,
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec![
                "--release".to_string(),
                "--".to_string(),
                "--input".to_string(),
                "[{\"name\": \"Alice\", \"age\": 30}, {\"name\": \"Bob\", \"age\": 25}]"
                    .to_string(),
            ]),
            subject_name: None,
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(
            result,
            ["[{\"name\":\"Alice\",\"age\":40},{\"name\":\"Bob\",\"age\":35}]"]
        );

        // --- from StdIO, no initialization ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::Stdio,
            project_dir: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: None,
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch.clone()])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        // --- from TempFile, no initialization ---

        // Create the run script
        let run_str = r#"use arrow::array::{ArrayRef, StringArray, UInt32Array};
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
}"#;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.clone().as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: None,
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch.clone()])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        // --- from TempFile (multiple batches), no initialization ---

        // State for the command processor config
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch_1 = test_command_sandbox_processor::create_messages()?;
        let names = vec!["Joe"];
        let names_arr: ArrayRef = Arc::new(StringArray::from(names));
        let ages = vec![40];
        let ages_arr: ArrayRef = Arc::new(UInt32Array::from(ages));
        let batch_2 = RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch_1, batch_2])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob", "Joe"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35, 50]);

        // --- from TempFile, initialization ---

        // Create the initialization script
        let initialization_str = r#"#!/usr/bin/env bash
apt update
apt install --assume-yes protobuf-compiler clang"#;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_script: Some(initialization_str.to_string()),
            run_script: Some(run_str.to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            subject_name: Some(messages.to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    #[tokio::test]
    async fn test_command_sandbox_processor_docker_rs_workspace_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Create project directory
        let project_name = "phymes-rs-workspace";
        let tmp_dir = TempDir::new()?;
        let project_dir = tmp_dir.path().join(project_name);
        let _ = fs::create_dir(&project_dir);

        // --- from TempFile, initialization ---

        // Create the workspace
        let workspace_table = CommandSandboxEnvironments::Rust.to_default_workspace(None)?;

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            // We need to mount the directories when we first run the container
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            project_dir: Some(project_dir.as_path().to_str().unwrap().to_string()),
            container_project_dir: Some("/home/sandbox".to_string()),
            initialization_file: Some("install.sh".to_string()),
            run_file: Some("main.rs".to_string()),
            runner: CommandSandboxRunners::DockerUnsafe,
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            subject_name: Some(messages.to_string()),
            workspace_name: Some(workspace_table.get_name().to_string()),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let batch = test_command_sandbox_processor::create_messages()?;

        let message_table = SubjectBuilder::new()
            .with_record_batches(vec![batch])?
            .with_name(messages)
            .build()?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            workspace_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(workspace_table.get_name())
                .with_publisher("")
                .with_subject(workspace_table.get_name())
                .with_update(&Publication::None)
                .with_message(workspace_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&Publication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor =
            CommandSandboxProcessor::new(name, CommandSandboxProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(name)
            .unwrap()
            .message
            .take()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await?;
        let table = SubjectBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }
}
