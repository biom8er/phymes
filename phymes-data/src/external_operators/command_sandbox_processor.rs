use std::{
    path::Path, pin::Pin, process::Output, sync::Arc, task::{Context, Poll, ready}
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{FutureExt, Stream, StreamExt};
use parking_lot::Mutex;
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait, PublishAndSubscribeTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, Table, TableBuilder, TableBuilderTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription, TableTrait, create_chat_record_batch, remove_message_by_subject
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, TraceBuilderTrait,
    create_timestamp_micros,
};
use serde_json::Value;
use tempfile::NamedTempFile;
use tokio::process::Command;
use tracing::{Level, event};

use crate::{DataConfigTrait, external_operators::{command_sandbox_config::{CommandSandboxConfig, CommandSandboxEnvironments, CommandSandboxRunners, DataIOMethod}, http_client_processor::error_report}};

/// The state of the Command
///
/// # Notes
/// * We need to capture each stage of the request so that the connection 
///   is not dropped during repeated polling of the stream.
pub enum CommandSandboxState {
    NotStarted,
    Output(Pin<Box<dyn Future<Output = std::io::Result<Output>> + Send + 'static>>),
    Done,
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
    publications: Vec<TablePublication>,
    subscriptions: Vec<TableSubscription>,
    subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
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
    fn new(
        name: &str,
        r#type: &str,
        publications: &[TablePublication],
        subscriptions: &[TableSubscription],
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    ) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            subscribe_policy,
        }
    }

    fn get_subscribe_policy(&self) -> &dyn TableSubscribePolicyTrait {
        self.subscribe_policy.as_ref()
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
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
    _runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<CommandSandboxConfig>,
    /// State of the OpenAI API request
    state: CommandSandboxState,
    /// Input file
    input_file: Option<NamedTempFile>,
}

impl CommandSandboxStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: message_stream.schema(),
            message_stream,
            diagnostic_builder,
            config_stream,
            _runtime_env: runtime_env,
            config: None,
            state: CommandSandboxState::NotStarted,
            input_file: None,
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
        match &mut self.state {
            CommandSandboxState::NotStarted => {
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

                // Collect the message data in a streaming fashion
                let mut batches = Vec::new();
                while let Some(Ok(batch)) = ready!(self.message_stream.poll_next_unpin(cx)) {
                    if batch.num_rows() > 0 {
                        batches.push(batch);
                        break;
                    }                    
                }

                // The poll ends when there are no more batches
                if batches.is_empty() {
                    self.state = CommandSandboxState::Done;
                    return Poll::Ready(None)
                }
                let messages = Table::get_builder()
                    .with_name("messages")
                    .with_record_batches(batches)?
                    .build()?;

                // Validate the config paths
                let err_str = if let Some(project_dir) = self.config.as_ref().unwrap().project_dir.as_ref() {
                    // Validate the directory
                    if !Path::new(project_dir).exists() {
                        Some("Project folder '{project_dir}' does not exist.")
                    } else {
                        // Validate the entry script
                        if let Some(entry_script) = self.config.as_ref().unwrap().entry_script.as_ref() {
                            if !Path::new(&format!("{project_dir}/{entry_script}")).exists() {
                                Some("Entry script '{entry_script}' does not exist in the project folder '{project_dir}'.")
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                } else {
                    None
                };
                if let Some(err) = err_str {
                    self.state = CommandSandboxState::Done;
                    return Poll::Ready(Some(Err(anyhow!(err))))
                }

                // Create the temporary input file
                // DM: needed later but this could be optimized to prevent the write operation...
                let mut file = NamedTempFile::new()?;
                let host_input_dir = file.path().parent().unwrap().to_str().unwrap().to_string();
                let host_input_path = file.path().to_str().unwrap().to_string();

                // Execute the command
                // DM: A future optimization maybe to treat each row as a parallel Command
                let fut = match self.config.as_ref().unwrap().runner {
                    CommandSandboxRunners::Docker => {
                        // Build Docker args
                        let mut command_args = vec![
                            "run".to_string(),
                            "--rm".to_string(),
                            "--network".to_string(), "none".to_string(), // No network
                            "--memory".to_string(), "128m".to_string(), // Memory limit
                            "--cpus".to_string(), "0.5".to_string(), // CPU limit
                            "--read-only".to_string(), // Entire container FS read-only
                            "--pids-limit".to_string(), "50".to_string(), // Process limit
                            "-v".to_string(), "sandbox_tmp:/tmp".to_string(), // Writable /tmp inside container
                        ];

                        // Add the input/output file
                        if let Some(container_input_path) = self.config.as_ref().unwrap().container_input_path.as_ref()
                            && DataIOMethod::TempFile == self.config.as_ref().unwrap().data_i {
                            command_args.push("-v".to_string());
                            command_args.push(format!("{host_input_path}:{container_input_path}"));
                        }

                        // User defined container arguments
                        if let Some(args) = self.config.as_ref().unwrap().container_args.as_ref() {
                            for arg in args {
                                command_args.push(arg.to_string());
                            }                            
                        }

                        // Mount the project dir if it exists
                        if let (Some(project_dir), Some(container_project_dir)) = (self.config.as_ref().unwrap().project_dir.as_ref(), self.config.as_ref().unwrap().container_project_dir.as_ref()) {
                            command_args.push("-v".to_string());
                            command_args.push(format!("{project_dir}:{container_project_dir}:ro")); // Project folder read-only
                        }

                        // Add environment variables to command args
                        for (k, v) in self.config.as_ref().unwrap().env_args()? {
                            command_args.push("-e".to_string());
                            command_args.push(format!("{}={}", k, v));
                        }

                        // Add docker image and command
                        command_args.push(self.config.as_ref().unwrap().container_image.to_string());
                        command_args.push(self.config.as_ref().unwrap().command.as_ref().ok_or(anyhow!("Missing container image for {} runner and {} environment.", self.config.as_ref().unwrap().runner, self.config.as_ref().unwrap().environment))?.to_string());
                            
                        // Entry script
                        if let (Some(entry_script), Some(container_project_dir)) = (self.config.as_ref().unwrap().entry_script.as_ref(), self.config.as_ref().unwrap().container_project_dir.as_ref()) {
                            command_args.push(format!("{}/{}", container_project_dir, entry_script));
                        }

                        // User defined ClI arguments
                        if let Some(args) = self.config.as_ref().unwrap().cli_args.as_ref() {
                            for arg in args {
                                command_args.push(arg.to_string());
                            }                            
                        }

                        // Extract out the message and run the command
                        match self.config.as_ref().unwrap().data_i {
                            DataIOMethod::Stdio => {
                                let content = messages.to_json_object()?;
                                let arg = serde_json::to_string(&content)?;
                                command_args.push("--lhs-args".to_string());
                                command_args.push(arg);
                            }
                            DataIOMethod::TempFile => {
                                messages.to_ipc_file(&mut file)?;
                                command_args.push("--lhs-args".to_string());
                                command_args.push(self.config.as_ref().unwrap().container_input_path.as_ref().ok_or(anyhow!("Container input path must be provided for data input method {}.", self.config.as_ref().unwrap().data_i))?.to_string());
                            }
                            DataIOMethod::None => {}
                        };

                        // Run the command
                        Command::new("docker").args(&command_args).output()
                    },
                    CommandSandboxRunners::Wasmtime => {
                        // Build wasmtime args
                        let mut command_args = Vec::new();

                        // Add run for the component model
                        match self.config.as_ref().unwrap().environment {
                            CommandSandboxEnvironments::WasmComponent => {
                                command_args.push("run".to_string());
                            }
                            _ => {}
                        }

                        // User defined container arguments
                        if let Some(args) = self.config.as_ref().unwrap().container_args.as_ref() {
                            for arg in args {
                                command_args.push(arg.to_string());
                            }                            
                        }

                        // Mount the project dir if it exists
                        if let Some(project_dir) = self.config.as_ref().unwrap().project_dir.as_ref() {
                            command_args.push(format!("--dir={project_dir}"));
                        }

                        // Add the input file
                        if DataIOMethod::TempFile == self.config.as_ref().unwrap().data_i {
                            command_args.push(format!("--dir={host_input_dir}")); // Input file read-only
                        }

                        // Add environment variables to command args
                        for (k, v) in self.config.as_ref().unwrap().env_args()? {
                            command_args.push(format!("--env={}={}", k, v));
                        }

                        // Add command for the component model
                        match self.config.as_ref().unwrap().environment {
                            CommandSandboxEnvironments::WasmComponent => {
                                command_args.push("--invoke".to_string());
                                let command = self.config.as_ref().unwrap().command.as_ref().ok_or(anyhow!("Command to run must be defined when using the {} environment.", self.config.as_ref().unwrap().environment))?;
                                let mut args_vec =  if let Some(args) = self.config.as_ref().unwrap().cli_args.as_ref() {
                                    args.to_owned()                   
                                } else {
                                    Vec::new()
                                };

                                // Extract out the message and add as the last argument
                                match self.config.as_ref().unwrap().data_i {
                                    DataIOMethod::Stdio => {
                                        let content = messages.to_json_object()?;
                                        let arg = serde_json::to_string(&content)?;
                                        args_vec.push(arg);
                                    }
                                    DataIOMethod::TempFile => {
                                        messages.to_ipc_file(&mut file)?;
                                        args_vec.push(host_input_path.clone());
                                    }
                                    DataIOMethod::None => {}
                                };
                                command_args.push(format!("{command}({})", args_vec.join(" ,")));
                            }
                            _ => {                                
                                if let Some(command) = self.config.as_ref().unwrap().command.as_ref() {
                                    command_args.push(command.to_string());                                
                                }
                            }
                        }
                        command_args.push(self.config.as_ref().unwrap().container_image.to_string());

                        // User defined CII arguments for modules                        
                        match self.config.as_ref().unwrap().environment {
                            CommandSandboxEnvironments::WasmModule => {
                                if let Some(args) = self.config.as_ref().unwrap().cli_args.as_ref() {
                                    for arg in args {
                                        command_args.push(arg.to_string());
                                    }                            
                                }

                                // Extract out the message and add as CLI arguments
                                match self.config.as_ref().unwrap().data_i {
                                    DataIOMethod::Stdio => {
                                        let content = messages.to_json_object()?;
                                        let arg = serde_json::to_string(&content)?;
                                        command_args.push("--lhs-args".to_string());
                                        command_args.push(arg);
                                    }
                                    DataIOMethod::TempFile => {
                                        messages.to_ipc_file(&mut file)?;
                                        command_args.push("--lhs-args".to_string());
                                        command_args.push(host_input_path);
                                    }
                                    DataIOMethod::None => {}
                                };
                            }
                            _ => {}
                        }

                        // Run the command
                        Command::new("wasmtime").args(&command_args).output()
                    },
                    _ => {
                        self.state = CommandSandboxState::Done;
                        return Poll::Ready(Some(Err(anyhow!("Runner type {} is not supported yet.", self.config.as_ref().unwrap().runner))))
                    }
                };

                // Update the request state and poll next
                self.input_file.replace(file);
                self.state = CommandSandboxState::Output(Box::pin(fut));
                self.poll_next(cx)
            }
            CommandSandboxState::Output(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
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

                    // Check the status
                    if !output.status.success() {
                        self.state = CommandSandboxState::Done;
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        return Poll::Ready(Some(Err(anyhow!("Command exited with code {} and error {}.", output.status.code().unwrap_or(0), stderr))));
                    }

                    // Parse the response
                    let batch = match self.config.as_ref().unwrap().data_o {
                        DataIOMethod::Stdio => {
                            let json_values = serde_json::from_slice::<Vec<Value>>(&output.stdout)?;
                            let table = TableBuilder::new()
                                .with_schema(self.schema.clone())
                                .with_json_values(&json_values)?
                                .with_name("cmd_sandbox")
                                .build()?;
                            table.get_record_batches_own().pop().unwrap()
                        }
                        DataIOMethod::TempFile => {
                            let file = self.input_file.take().ok_or(anyhow!("Temporary input file was not found when reading in the response!"))?;
                            let table = TableBuilder::new_from_ipc_file(file)?
                                .with_name("cmd_sandbox")
                                .build()?;
                            table.get_record_batches_own().pop().unwrap()
                        }
                        DataIOMethod::None => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            create_chat_record_batch(
                                vec!["tool".to_string()],
                                vec![stdout.to_string()],
                                vec![create_timestamp_micros()],
                            )?
                        }
                    };                    

                    // Reset the state to poll the next batch
                    self.state = CommandSandboxState::NotStarted;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    if let Some(baseline_metrics) = &baseline_metrics {
                        baseline_metrics.record_poll(poll)
                    } else {
                        poll
                    }
                }
                Err(err) => {
                    self.state = CommandSandboxState::Done;
                    Poll::Ready(Some(Err(anyhow!(error_report(&err)))))
                }
            },
            CommandSandboxState::Done => Poll::Ready(None),
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
    use std::io::Write;
    use arrow::array::{ArrayRef, StringArray, UInt32Array};
    use futures::TryStreamExt;
    use phymes_core::{AvailableTableSubscribePolicies, ChatBuilderTraitExt, RuntimeEnvTrait, TableBuilder};
    use phymes_diagnostics::{Diagnostics, SpanBuilder};

    use crate::external_operators::command_sandbox_config::DataIOMethod;

    use super::*;

    /// WASM component and module example
    #[tokio::test]
    async fn test_command_sandbox_processor_wasmtime() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt")));

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
            .append_new_user_query_str(
                "Hello from WASM!",
                "user",
            )?;

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
            .append_new_user_query_str(
                "Hello from WASM!",
                "user",
            )?;

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
        let rt_env = Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt")));

        // Metrics to compute time and rows
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // --- From config ---
        // based on docker run --rm alpine echo "Hello from Docker!"

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Custom("bash".to_string()),
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::None,
            data_o: DataIOMethod::None,
            command: Some("echo".to_string()),
            timeout: 5,
            container_args: Some(vec!["--rm".to_string()]),
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
            .append_new_user_query_str(
                "Hello from Docker!",
                "user",
            )?;

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
        assert_eq!(*result.first().unwrap(), "Hello from Docker!\n");

        // --- From Stdio ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Custom("bash".to_string()),
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::Stdio,
            data_o: DataIOMethod::Stdio,
            command: Some("echo".to_string()),
            timeout: 5,
            container_args: Some(vec!["--rm".to_string()]),
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
        assert!(result.first().unwrap().contains("--lhs-args [{\"content\":\"Hello from Docker!\",\"role\":\"user\",\"timestamp\":"));

        // --- From TempFile ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Custom("bash".to_string()),
            container_image: "alpine".to_string(),
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::None,
            command: Some("echo".to_string()),
            timeout: 5,
            container_args: Some(vec!["--rm".to_string()]),
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
        let table = TableBuilder::new()
            .with_record_batches(result)?
            .with_name("")
            .build()?;

        let result = table.get_column_as_vec_str("role");
        assert_eq!(result, ["tool"]);
        let result = table.get_column_as_vec_str("content");
        assert!(result.first().unwrap().contains("--lhs-args /tmp/."));

        Ok(())
    }

    /// Python code execution example
    #[tokio::test]
    async fn test_command_sandbox_processor_docker_py() -> Result<()> {
        let name = "CommandSandboxProcessor";
        let messages = "messages";

        // Runtime env
        let rt_env = Arc::new(Mutex::new(RuntimeEnv::new().with_name("rt")));

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
            command: Some("python".to_string()),
            timeout: 5,
            container_args: Some(vec!["--rm".to_string()]),
            cli_args: Some(vec!["-c".to_string(),
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
        let batch = RecordBatch::try_from_iter(vec![
            ("name",  names_arr),
            ("age", ages_arr)
        ])?;

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
            command: Some("python".to_string()),
            timeout: 5,
            container_args: Some(vec!["--rm".to_string()]),
            cli_args: Some(vec!["-c".to_string(),
                "import json, argparse; \
                parser = argparse.ArgumentParser(); \
                parser.add_argument('--lhs-args', required=True); \
                args = parser.parse_args(); \
                data = json.loads(args.lhs_args); \
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

        // --- From TempFile ---

        // State for the command processor config
        let command_config = CommandSandboxConfig {
            data_i: DataIOMethod::TempFile,
            data_o: DataIOMethod::TempFile,
            container_input_path: Some("/home/sandbox/input.ipc".to_string()),
            runner: CommandSandboxRunners::Docker,
            environment: CommandSandboxEnvironments::Python,
            container_image: "python:3.12-slim-trixie".to_string(),
            command: Some("bash".to_string()),
            timeout: 5,
            container_args: Some(vec!["--rm".to_string()]),
            cli_args: Some(vec!["-c".to_string(),
                r#""pip3 install pandas pyarrow >/dev/null && \
                python -c \"import pandas as pd, argparse, json; \
                parser = argparse.ArgumentParser(); \
                parser.add_argument('--lhs-args', required=True); \
                args = parser.parse_args(); \
                df = pd.read_feather(args.lhs_args); \
                df = df.drop(columns=['age']); \
                df.to_feather(args.lhs_args);\"""#.to_string()]),
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
        let mut stream = processor.process(message, Some(&diagnostic_builder), rt_env)?;

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
        assert_eq!(result, ["alice", "bob"]);

        Ok(())
    }

    /// Rust code execution example
    #[tokio::test]
    async fn test_command_sandbox_processor_docker_rs() -> Result<()> {
        // DM, todo: create a small rust example that takes a RecordBatch, modifies it, and returns the modified RecordBatch
        // DM, examples: code execution loop which requires diff'ing https://github.com/AnubhabB/diff-match-patch-rs
        Ok(())
    }
}