use std::{
    path::Path, pin::Pin, process::Output, sync::Arc, task::{Context, Poll, ready}
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{FutureExt, Stream, StreamExt};
use parking_lot::Mutex;
use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait, PublishAndSubscribeTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, Table, TableBuilderTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription, TableTrait, create_chat_record_batch, remove_message_by_subject
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, TraceBuilderTrait,
    create_timestamp_micros,
};
use serde_json::{Map, Value};
use tokio::process::Command;
use tracing::{Level, event};

use crate::{DataConfigTrait, external_operators::{command_sandbox_config::{CommandSandboxConfig, CommandSandboxEnvironments, CommandSandboxRunners}, http_client_processor::error_report}};

/// The state of the Command
///
/// # Notes
/// * We need to capture each stage of the request so that the connection 
///   is not dropped during repeated polling of the stream.
pub enum CommandSandboxState {
    NotStarted,
    Started(Pin<Box<dyn Future<Output = std::io::Result<Output>> + Send + 'static>>),
    Done,
}

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
}

impl CommandSandboxStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::Messages.to_schema(),
            message_stream,
            diagnostic_builder,
            config_stream,
            _runtime_env: runtime_env,
            config: None,
            state: CommandSandboxState::NotStarted,
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

                        // Run the command
                        Command::new("docker").args(&command_args).output()
                    },
                    CommandSandboxRunners::Wasmtime => {
                        // Build wasmtime args
                        let mut command_args = Vec::new();

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

                        // Add environment variables to command args
                        for (k, v) in self.config.as_ref().unwrap().env_args()? {
                            command_args.push(format!("--env={}={}", k, v));
                        }

                        // Add command and wasm component
                        command_args.push(self.config.as_ref().unwrap().command.as_ref().ok_or(anyhow!("Missing container image for {} runner and {} environment.", self.config.as_ref().unwrap().runner, self.config.as_ref().unwrap().environment))?.to_string());
                        command_args.push(self.config.as_ref().unwrap().container_image.to_string());

                        // User defined ClI arguments
                        if let Some(args) = self.config.as_ref().unwrap().cli_args.as_ref() {
                            for arg in args {
                                command_args.push(arg.to_string());
                            }                            
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
                self.state = CommandSandboxState::Started(Box::pin(fut));
                self.poll_next(cx)
            }
            CommandSandboxState::Started(fut) => match ready!(fut.as_mut().poll_unpin(cx)) {
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

                    // Parse the response
                    if !output.status.success() {
                        self.state = CommandSandboxState::Done;
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        return Poll::Ready(Some(Err(anyhow!("Command exited with code {:?} and error {}.", output.status.code(), stderr))));
                    }
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let batch = create_chat_record_batch(
                        vec!["tool".to_string()],
                        vec![stdout.to_string()],
                        vec![create_timestamp_micros()],
                    )?;

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
    use super::*;

    /// WASM component and module example
    #[tokio::test]
    async fn test_command_io_processor_wasmtime() -> Result<()> {
        Ok(())
    }

    /// Docker CLI example
    #[tokio::test]
    async fn test_command_io_processor_docker_echo() -> Result<()> {
        //docker run --rm alpine echo "Hello from Docker!"
        Ok(())
    }

    /// Python code execution example
    #[tokio::test]
    async fn test_command_io_processor_docker_py() -> Result<()> {
        Ok(())
    }

    /// Rust code execution example
    #[tokio::test]
    async fn test_command_io_processor_docker_rs() -> Result<()> {
        Ok(())
    }
}