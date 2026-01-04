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
    BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait,
    RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, Table,
    TableBuilder, TableBuilderTrait, TablePublication, TableSubscribePolicyTrait,
    TableSubscription, TableTrait, create_chat_record_batch, remove_message_by_subject,
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, TraceBuilderTrait,
    create_timestamp_micros,
};
use serde_json::Value;
use tempfile::NamedTempFile;
use tokio::process::Command;
use tracing::{Level, event};

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
    NotStarted,
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
    /// Initializing runner and installing any dependencies
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

impl PublishAndSubscribeTrait for CommandSandboxProcessor {
    fn get_publications(&self) -> Vec<&TablePublication> {
        self.publications.iter().collect::<Vec<_>>()
    }

    fn get_subscriptions(&self) -> Vec<&TableSubscription> {
        self.subscriptions.iter().collect::<Vec<_>>()
    }
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        self.subscribe_policy
            .check_subscriptions(&self.subscriptions, updates, state)
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

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        // Trace the inbox
        let trace = if let Some(diagnostic_builder) = diagnostic_builder {
            let trace_builder = diagnostic_builder.clone().to_child(self.get_name())?;
            let trace = trace_builder
                .clone()
                .messages(line!(), file!(), self.get_name());
            trace.enter(&message.values().collect::<Vec<_>>());
            Some((trace, trace_builder))
        } else {
            None
        };

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Extract out the subscribed messages
        let mut subscriptions = Vec::new();
        for subs in self.subscriptions.iter() {
            if subs.get_table_name() != self.get_name() {
                match remove_message_by_subject(subs.get_table_name(), &mut message) {
                    Some(m) => {
                        subscriptions.push(m);
                    }
                    None => {
                        event!(
                            Level::WARN,
                            "Subscription {} not provided for {}.",
                            subs.get_table_name(),
                            self.get_name()
                        );
                    }
                }
            }
        }
        if subscriptions.len() > 1 {
            return Err(anyhow!("More than one subscription was found."));
        } else if subscriptions.is_empty() {
            return Err(anyhow!("No subscriptions were found."));
        }

        // Run the stream
        let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
        let out = Box::pin(CommandSandboxStream::new(
            subscriptions.swap_remove(0).get_message_own(),
            config,
            Arc::clone(&runtime_env),
            stream_diagnostic_builder,
        )?);
        let out_m = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher(self.get_name())
            .with_subject(self.publications.first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.publications.first().unwrap())
            .make_name()?
            .build()?;
        let _ = message.insert(out_m.get_name().to_string(), out_m);

        // Trace the outbox
        if let Some(trace) = trace {
            trace.0.exit(&message.values().collect::<Vec<_>>());
        }
        Ok(message)
    }
}

pub struct CommandSandboxStream {
    /// Output schema
    schema: SchemaRef,
    /// The input message to process
    message_stream: SendableRecordBatchStream,
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
    message_inbox: Option<Table>,
}

impl CommandSandboxStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: message_stream.schema(),
            message_stream,
            diagnostic_builder,
            config_stream,
            _runtime_env: runtime_env,
            config: None,
            stream_state: CommandSandboxStreamState::NotStarted,
            runner_state: CommandSandboxRunnerState::NotStarted,
            message_inbox: None,
        })
    }

    /// Initialize the config for text generation inference
    fn init_config(&mut self, config_table: Table) -> Result<()> {
        if self.config.is_none() {
            let config = CommandSandboxConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }
}

impl Stream for CommandSandboxStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Iterate through each state until the API request is completed
        match &mut self.stream_state {
            CommandSandboxStreamState::NotStarted => {
                // Initialize the config
                if self.config.is_none() {
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                        batches.push(batch);
                    }
                    let config_table = Table::get_builder()
                        .with_name("config")
                        .with_record_batches(batches)?
                        .build()?;
                    self.init_config(config_table)?;
                }

                // Collect the next batch or continue processing the current batch
                if self.message_inbox.is_none() {
                    // Collect the message data in a streaming fashion
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(self.message_stream.poll_next_unpin(cx)) {
                        if batch.num_rows() > 0 {
                            batches.push(batch);
                            break;
                        }
                    }

                    // The poll ends when there are no more batches
                    let messages = if batches.is_empty() {
                        self.stream_state = CommandSandboxStreamState::Done;
                        match &self.runner_state {
                            CommandSandboxRunnerState::NotStarted => return Poll::Ready(None),
                            CommandSandboxRunnerState::Initializing(runner_info)
                            | CommandSandboxRunnerState::Running(runner_info)
                            | CommandSandboxRunnerState::Done(runner_info) => {
                                // Cleanup resources
                                self.runner_state =
                                    CommandSandboxRunnerState::Done(runner_info.to_owned());
                                Table::default()
                            }
                        }
                    } else {
                        Table::get_builder()
                            .with_name("messages")
                            .with_record_batches(batches)?
                            .build()?
                    };

                    self.message_inbox.replace(messages);
                }

                // Create the input/output file or content for the next batch based on the runner state
                match &self.runner_state {
                    CommandSandboxRunnerState::NotStarted => {
                        // Make a random name for the runner
                        let mut buf = [0u8; 16];
                        getrandom::fill(&mut buf)?;
                        let hash = u128::from_ne_bytes(buf);
                        let name = format!("phymes-sandbox_{hash}");

                        // Create the initialization and runner files
                        let (
                            run_file_path,
                            initialization_file_path,
                            input_file_path,
                            output_file_path,
                        ) = if let Some(project_dir) =
                            self.config.as_ref().unwrap().project_dir.as_ref()
                        {
                            // Check the directory
                            if !Path::new(project_dir).exists() {
                                let err_str =
                                    format!("Project folder '{project_dir}' does not exist.");
                                self.stream_state = CommandSandboxStreamState::Done;
                                return Poll::Ready(Some(Err(anyhow!(err_str))));
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

                        // Create the temporary input and output files with content or stdin content ONLY if there is no initialization script else just create an empty temporary file
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
                                if initialization_file_path.is_none() {
                                    let content =
                                        self.message_inbox.take().unwrap().to_json_object()?;
                                    let content = serde_json::to_string(&content)?;
                                    CommandSandboxRunnerInfo::new()
                                        .with_name(&name)
                                        .with_content(&content)
                                } else {
                                    CommandSandboxRunnerInfo::new().with_name(&name)
                                }
                            }
                            (DataIOMethod::TempFile, DataIOMethod::None)
                            | (DataIOMethod::TempFile, DataIOMethod::Stdio) => {
                                let input_file = NamedTempFile::new()?;
                                let input_persist_path = input_file_path.ok_or(anyhow!("Missing input file path for data input method {} and data output method {}.", &self.config.as_ref().unwrap().data_i, &self.config.as_ref().unwrap().data_o))?;
                                if initialization_file_path.is_none() {
                                    let mut input_file = input_file.persist(&input_persist_path)?;
                                    self.message_inbox
                                        .take()
                                        .unwrap()
                                        .to_ipc_file(&mut input_file)?;
                                    CommandSandboxRunnerInfo::new()
                                        .with_name(&name)
                                        .with_input_file(&input_persist_path)
                                } else {
                                    let _input_file = input_file.persist(&input_persist_path)?;
                                    CommandSandboxRunnerInfo::new()
                                        .with_name(&name)
                                        .with_input_file(&input_persist_path)
                                }
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
                                if initialization_file_path.is_none() {
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
                                } else {
                                    let _input_file = input_file.persist(&input_persist_path)?;
                                    let _output_file = output_file.persist(&output_persist_path)?;
                                    CommandSandboxRunnerInfo::new()
                                        .with_name(&name)
                                        .with_input_file(&input_persist_path)
                                        .with_output_file(&output_persist_path)
                                }
                            }
                            (DataIOMethod::Stdio, DataIOMethod::TempFile) => {
                                let output_file = NamedTempFile::new()?;
                                let output_persist_path = output_file_path.ok_or(anyhow!("Missing output file path for data input method {} and data output method {}.", &self.config.as_ref().unwrap().data_i, &self.config.as_ref().unwrap().data_o))?;
                                if initialization_file_path.is_none() {
                                    let messages = self.message_inbox.take().unwrap();
                                    let content = messages.to_json_object()?;
                                    let content = serde_json::to_string(&content)?;
                                    let _output_file = output_file.persist(&output_persist_path)?;
                                    CommandSandboxRunnerInfo::new()
                                        .with_name(&name)
                                        .with_content(&content)
                                        .with_output_file(&output_persist_path)
                                } else {
                                    let _output_file = output_file.persist(&output_persist_path)?;
                                    CommandSandboxRunnerInfo::new()
                                        .with_name(&name)
                                        .with_output_file(&output_persist_path)
                                }
                            }
                        };

                        // Change to initialize state
                        if let Some(initialization_file) = initialization_file_path {
                            runner_info =
                                runner_info.with_initialization_file(&initialization_file);
                        }
                        if let Some(run_file) = run_file_path {
                            runner_info = runner_info.with_run_file(&run_file);
                        }
                        self.runner_state = CommandSandboxRunnerState::Initializing(runner_info);
                    }
                    CommandSandboxRunnerState::Initializing(runner_info)
                    | CommandSandboxRunnerState::Running(runner_info) => {
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
                dbg!(&self.runner_state);

                // Execute the command
                // DM: A future optimization maybe to treat each row as a parallel Command
                let fut = match self.config.as_ref().unwrap().runner {
                    CommandSandboxRunners::Docker | CommandSandboxRunners::DockerUnsafe => {
                        // Build Docker args
                        let mut command_args =
                            match (&self.config.as_ref().unwrap().runner, &self.runner_state) {
                                (
                                    CommandSandboxRunners::Docker,
                                    CommandSandboxRunnerState::Initializing(runner_info),
                                ) => {
                                    let mut command_args = vec![
                                        "run".to_string(),
                                        "--name".to_string(), // Name the container for later calls
                                        runner_info
                                            .name
                                            .as_ref()
                                            .expect("Missing name for runner.")
                                            .to_string(),
                                        //"--rm".to_string(), // Remove the container after exit
                                        "--network".to_string(),
                                        "none".to_string(), // No network
                                        "--memory".to_string(),
                                        "128m".to_string(), // Memory limit
                                        "--cpus".to_string(),
                                        "0.5".to_string(),         // CPU limit
                                        "--read-only".to_string(), // Entire container FS read-only
                                        "--pids-limit".to_string(),
                                        "50".to_string(), // Process limit
                                    ];

                                    // Detach for subsequent calls
                                    if runner_info.initialization_file.is_some() {
                                        command_args.push("-d".to_string());
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
                                    CommandSandboxRunnerState::Initializing(runner_info),
                                ) => {
                                    let mut command_args = vec![
                                        "run".to_string(),
                                        "--name".to_string(), // Name the container for later calls
                                        runner_info
                                            .name
                                            .as_ref()
                                            .expect("Missing name for runner.")
                                            .to_string(),
                                    ];

                                    // Detach for subsequent calls
                                    if runner_info.initialization_file.is_some() {
                                        command_args.push("-d".to_string());
                                    }

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
                                    CommandSandboxRunnerState::Running(_runner_info),
                                )
                                | (
                                    CommandSandboxRunners::DockerUnsafe,
                                    CommandSandboxRunnerState::Running(_runner_info),
                                ) => {
                                    let mut command_args = vec![
                                        "exec".to_string(),
                                        "-it".to_string(), // Interactive mode to keep STDIN open
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
                            CommandSandboxRunnerState::Initializing(runner_info) => {
                                // Add docker image, initialization/run script, and optional command
                                command_args.push(
                                    self.config.as_ref().unwrap().container_image.to_string(),
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
                                            let initialization_path =
                                                Path::new(initialization_file);
                                            command_args.push(format!(
                                                "{container_project_dir}/{}",
                                                initialization_path
                                                    .file_name()
                                                    .unwrap()
                                                    .to_str()
                                                    .unwrap()
                                            ));
                                        } else if let (
                                            Some(run_file),
                                            Some(container_project_dir),
                                        ) = (
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
                                            let initialization_path =
                                                Path::new(initialization_file);
                                            command_args.push(format!(
                                                "{container_project_dir}/{}",
                                                initialization_path
                                                    .file_name()
                                                    .unwrap()
                                                    .to_str()
                                                    .unwrap()
                                            ));
                                        } else if let (
                                            Some(_run_file),
                                            Some(_container_project_dir),
                                        ) = (
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
                                            // command_args.push(format!("/{}/src/{}", container_project_dir, run_path.file_name().unwrap().to_str().unwrap()));
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
                                        if let Some(command) =
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

                                // User defined CLI arguments
                                if let Some(args) = self.config.as_ref().unwrap().cli_args.as_ref()
                                {
                                    for arg in args {
                                        command_args.push(arg.to_string());
                                    }
                                }

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
                                        if let Some(command) =
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
                        dbg!(&command_args);
                        Command::new("docker").args(&command_args).output()
                    }
                    CommandSandboxRunners::Wasmtime => {
                        // Build wasmtime args
                        let command_args = match &self.runner_state {
                            CommandSandboxRunnerState::NotStarted => unreachable!(),
                            CommandSandboxRunnerState::Initializing(runner_info)
                            | CommandSandboxRunnerState::Running(runner_info) => {
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
                    if let Some(exit_code) = output.status.code()
                        && exit_code != 1
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
                    {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        dbg!(stdout);
                    }

                    // Parse the response if running and skip if initializing or done
                    let batch = match (&self.config.as_ref().unwrap().data_o, &self.runner_state) {
                        (DataIOMethod::None, CommandSandboxRunnerState::Running(_runner_info)) => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            create_chat_record_batch(
                                vec!["tool".to_string()],
                                vec![stdout.to_string()],
                                vec![create_timestamp_micros()],
                            )?
                        }
                        (DataIOMethod::Stdio, CommandSandboxRunnerState::Running(_runner_info)) => {
                            let json_values = serde_json::from_slice::<Vec<Value>>(&output.stdout)?;
                            let table = TableBuilder::new()
                                .with_name("sandbox_stdio_running")
                                .with_schema(self.schema.clone())
                                .with_json_values(&json_values)?
                                .build()?;
                            table.get_record_batches_own().pop().unwrap()
                        }
                        (
                            DataIOMethod::TempFile,
                            CommandSandboxRunnerState::Running(runner_info),
                        ) => {
                            dbg!(&runner_info);
                            let file = fs::File::open(
                                runner_info
                                    .output_file
                                    .as_ref()
                                    .expect("Missing output TempFile from runner."),
                            )?;
                            let table = TableBuilder::new_from_ipc_file(file)?
                                .with_name("sandbox_tempfile_running")
                                .build()?;
                            dbg!(&table);
                            table.get_record_batches_own().pop().unwrap()
                        }
                        (
                            DataIOMethod::None,
                            CommandSandboxRunnerState::Initializing(_runner_info),
                        ) => {
                            if self
                                .config
                                .as_ref()
                                .unwrap()
                                .initialization_script
                                .is_some()
                            {
                                self.stream_state = CommandSandboxStreamState::NotStarted;
                                return self.poll_next(cx);
                            } else {
                                let stdout = String::from_utf8_lossy(&output.stdout);
                                create_chat_record_batch(
                                    vec!["tool".to_string()],
                                    vec![stdout.to_string()],
                                    vec![create_timestamp_micros()],
                                )?
                            }
                        }
                        (
                            DataIOMethod::Stdio,
                            CommandSandboxRunnerState::Initializing(_runner_info),
                        ) => {
                            if self
                                .config
                                .as_ref()
                                .unwrap()
                                .initialization_script
                                .is_some()
                            {
                                self.stream_state = CommandSandboxStreamState::NotStarted;
                                return self.poll_next(cx);
                            } else {
                                let json_values =
                                    serde_json::from_slice::<Vec<Value>>(&output.stdout)?;
                                let table = TableBuilder::new()
                                    .with_name("sandbox_stdio_initializing")
                                    .with_schema(self.schema.clone())
                                    .with_json_values(&json_values)?
                                    .build()?;
                                table.get_record_batches_own().pop().unwrap()
                            }
                        }
                        (
                            DataIOMethod::TempFile,
                            CommandSandboxRunnerState::Initializing(runner_info),
                        ) => {
                            if self
                                .config
                                .as_ref()
                                .unwrap()
                                .initialization_script
                                .is_some()
                            {
                                self.stream_state = CommandSandboxStreamState::NotStarted;
                                return self.poll_next(cx);
                            } else {
                                let file = fs::File::open(
                                    runner_info
                                        .output_file
                                        .as_ref()
                                        .expect("Missing output TempFile from runner."),
                                )?;
                                let table = TableBuilder::new_from_ipc_file(file)?
                                    .with_name("sandbox_tempfile_initializing")
                                    .build()?;
                                table.get_record_batches_own().pop().unwrap()
                            }
                        }
                        (_, CommandSandboxRunnerState::Done(_runner_info)) => {
                            self.stream_state = CommandSandboxStreamState::Done;
                            return self.poll_next(cx);
                        }
                        _ => unreachable!(),
                    };

                    // Reset the state to poll the next batch
                    self.stream_state = CommandSandboxStreamState::NotStarted;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    if let Some(baseline_metrics) = &baseline_metrics {
                        baseline_metrics.record_poll(poll)
                    } else {
                        poll
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
                    CommandSandboxRunnerState::Initializing(runner_info)
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

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, StringArray, UInt32Array};
    use futures::TryStreamExt;
    use phymes_core::{
        AvailableTableSubscribePolicies, ChatBuilderTraitExt, RuntimeEnvTrait, TableBuilder,
    };
    use phymes_diagnostics::{Diagnostics, SpanBuilder};
    use std::{fs::File, io::Write};

    use crate::external_operators::command_sandbox_config::DataIOMethod;

    use super::*;

    /// WASM component and module example
    #[tokio::test]
    async fn test_command_sandbox_processor_wasmtime() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

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
        let command_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str("Hello from WASM!", "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
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
        let command_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
            .with_name(messages)
            .append_new_user_query_str("Hello from WASM!", "user")?;

        // Build the current message state
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            messages.to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(messages)
                .with_publisher("")
                .with_subject(messages)
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
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

    /// Docker CLI example
    #[tokio::test]
    async fn test_command_sandbox_processor_docker_echo() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

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
        let command_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the system prompt and add the user query
        let message_builder = TableBuilder::new()
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
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["Hello from Docker!\n"]);

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
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = TableBuilder::new()
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
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
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
        let project_name = "phymes-echo-project";
        let project_dir = std::env::temp_dir().join(project_name);
        let _ = fs::remove_dir_all(&project_dir); // Doesn't matter if it is an error
        fs::create_dir(&project_dir).expect("Failed to create project directory");

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
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = TableBuilder::new()
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
                .with_update(&TablePublication::None)
                .with_message(message_builder.clone().build()?.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let _ = fs::remove_dir_all(&project_dir); // Doesn't matter if it is an error
        let table = TableBuilder::new()
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

    /// Python code execution example
    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py_run() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

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
        let command_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let names = vec!["Alice", "Bob"];
        let names_arr: ArrayRef = Arc::new(StringArray::from(names));
        let ages = vec![30, 25];
        let ages_arr: ArrayRef = Arc::new(UInt32Array::from(ages));
        let batch = RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])?;

        let message_table = TableBuilder::new()
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
                .with_update(&TablePublication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert_eq!(result, ["[{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]\n"]);

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
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = TableBuilder::new()
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
                .with_update(&TablePublication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        let result = table.get_column_as_vec_primitive::<u32>("age")?;
        assert_eq!(result, [40, 35]);

        Ok(())
    }

    /// Python code execution example
    #[ignore = "Pip cache not updating within the container resulting in `Command exited with code 127, stderr , and stdout OCI runtime exec failed: exec failed: unable to start container process: exec: \"/home/sandbox/.venv/bin/python\": stat /home/sandbox/.venv/bin/python: no such file or directory: unknown`."]
    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Create project directory
        let project_name = "phymes-py-project";
        let project_dir = std::env::temp_dir().join(project_name);
        let _ = fs::remove_dir_all(&project_dir); // Doesn't matter if it is an error
        fs::create_dir(&project_dir).expect("Failed to create project directory");

        // Create src directory
        let src_path = format!("{}/src", project_dir.as_path().to_str().unwrap());
        fs::create_dir(&src_path).expect("Failed to create src directory");

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
        // DM: add `sleep infinity` to keep the terminal open to inspect with docker
        let initialization_str = r#"#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install --no-cache-dir -r requirements.txt
sleep infinity"#;

        // Create the run script
        let run_str = r#"#!/bin/bash
import pandas as pd
import argparse
import json
if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument('--input-file', required=True);
    parser.add_argument('--output-file', required=True);
    args = parser.parse_args();
    df = pd.read_feather(args.input_file);
    df = df.drop(columns=['age']);
    df.to_feather(args.output_file);"#;

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

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
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let names = vec!["Alice", "Bob"];
        let names_arr: ArrayRef = Arc::new(StringArray::from(names));
        let ages = vec![30, 25];
        let ages_arr: ArrayRef = Arc::new(UInt32Array::from(ages));
        let batch = RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])?;

        let message_table = TableBuilder::new()
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
                .with_update(&TablePublication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        fs::remove_dir_all(project_dir)?;
        let table = TableBuilder::new()
            .with_name("test_command_sandbox_processor_docker_py_install")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        assert!(table.get_schema().column_with_name("age").is_none());

        Ok(())
    }

    /// Rust code execution example
    /// DM, examples: code execution loop which requires diff'ing https://github.com/AnubhabB/diff-match-patch-rs
    #[tokio::test]
    async fn test_command_sandbox_processor_docker_rs_install() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Create project directory
        let project_name = "phymes-rs-project";
        let project_dir = std::env::temp_dir().join(project_name);
        let _ = fs::remove_dir_all(&project_dir); // Doesn't matter if it is an error
        fs::create_dir(&project_dir).expect("Failed to create project directory");

        // Create src directory
        let src_path = format!("{}/src", project_dir.as_path().to_str().unwrap());
        fs::create_dir(&src_path).expect("Failed to create src directory");

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
arrow = "53.0.0"
clap = { version = "4.5.4", features = ["derive"] }"#;
        let _ = requirements_file.write(requirements_str.as_bytes())?;
        requirements_file.flush()?;

        // Create the initialization script
        let initialization_str = r#"#!/bin/bash
apt update
apt install --assume-yes protobuf-compiler clang
rustup toolchain install stable --target x86_64-unknown-linux-gnu
rustup default stable
curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash"#;

        // Create the run script
        let run_str = r#"use arrow::array::ArrayRef;
use arrow::error::{ArrowError, Result};
use arrow::datatypes::{Field, Schema};
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

/// Helper function to remove a RecordBatch column by name
fn remove_column_by_name(batch: &RecordBatch, col_name: &str) -> arrow::error::Result<RecordBatch> {
    // Find the index of the column to remove
    let idx = batch
        .schema()
        .index_of(col_name)
        .map_err(|_| ArrowError::SchemaError(format!("Column '{}' not found", col_name)))?;

    // Keep all fields except the one to remove
    let new_fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(i, f)| if i != idx { Some((**f).clone()) } else { None })
        .collect();

    // Keep all arrays except the one to remove
    let new_columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .enumerate()
        .filter_map(|(i, col)| if i != idx { Some(col.clone()) } else { None })
        .collect();

    // Create new schema and RecordBatch
    let new_schema = Arc::new(Schema::new(new_fields));
    RecordBatch::try_new(new_schema, new_columns)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let input_path = args.input_file.as_str();
    let output_path = args.output_file.as_str();

    // --- Step 1: Read Feather/IPC file ---
    let file = File::open(input_path)?;
    let mut reader = FileReader::try_new(file, None)?;
    let mut batches: Vec<RecordBatch> = reader.collect::<Result<_>>()?;

    // --- Step 2: Modify the batches ---
    let batches = batches.into_iter()
        .map(|batch| remove_column_by_name(&batch, "age").unwrap())
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

        // Runtime env
        let rt_env = Arc::new(RuntimeEnv::new().with_name("rt"));

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
            environment: CommandSandboxEnvironments::Rust,
            container_image: "amd64/rust".to_string(),
            command: None,
            timeout: 5,
            container_args: None,
            cli_args: Some(vec!["--release".to_string(), "--".to_string()]),
            ..Default::default()
        };
        let command_config_json = serde_json::to_vec(&command_config)?;
        let command_config_table = TableBuilder::new()
            .with_name(name)
            .with_json(&command_config_json, 1)?
            .build()?;

        // Make the input data for the script
        let names = vec!["Alice", "Bob"];
        let names_arr: ArrayRef = Arc::new(StringArray::from(names));
        let ages = vec![30, 25];
        let ages_arr: ArrayRef = Arc::new(UInt32Array::from(ages));
        let batch = RecordBatch::try_from_iter(vec![("name", names_arr), ("age", ages_arr)])?;

        let message_table = TableBuilder::new()
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
                .with_update(&TablePublication::None)
                .with_message(message_table.to_record_batch_stream())
                .build()?,
        );
        let _ = message.insert(
            command_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(command_config_table.get_name())
                .with_publisher("")
                .with_subject(command_config_table.get_name())
                .with_update(&TablePublication::None)
                .with_message(command_config_table.to_record_batch_stream())
                .build()?,
        );

        // Build the command processor
        let processor = CommandSandboxProcessor::new(
            name,
            CommandSandboxProcessor::get_static_name(),
            &[TablePublication::Replace {
                table_name: messages.to_string(),
            }],
            &[
                TableSubscription::OnUpdateFullTable {
                    table_name: messages.to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: command_config_table.get_name().to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env.clone())?;

        // Check the response
        let result = stream
            .remove(&format!("from_{name}_on_{messages}"))
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;
        let _ = fs::remove_dir_all(project_dir);
        let table = TableBuilder::new()
            .with_name("test_command_sandbox_processor_docker_rs")
            .with_record_batches(result)?
            .build()?;

        let result = table.get_column_as_vec_str("name");
        assert_eq!(result, ["Alice", "Bob"]);
        assert!(table.get_schema().column_with_name("age").is_none());

        Ok(())
    }
}
