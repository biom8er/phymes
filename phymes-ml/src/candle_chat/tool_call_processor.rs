use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, BuildableTrait, BuilderTrait, MappableTrait,
    MessageBuilderTrait, MessageTrait, ProcessorTrait, RecordBatchStream, RuntimeEnv,
    SendableRecordBatchStream, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, Subject, SubjectBuilderTrait, SubjectTrait, create_bytes_fields,
    create_session_tasks_subscribe_publish_batch, create_values_fields, remove_message_by_subject,
};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, HashSet, MetricBuilderTrait,
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{Stream, StreamExt};
use serde_json::{Map, Value};
use tracing::{Level, event, instrument};

use crate::candle_chat::tool_call_config::ToolCallConfig;

/// Processor that parses a [ProcessorTrait] configuration subject and
///   creates an on-the-fly `SessionTasksSubscribePublish` subject which calls
///   the [ProcessorTrait] with subscriptions provided in the configuration subject
///
/// # Notes
///
/// - This processor MUST subscribe to a `ViewTasksSubscribePublishAggregated` subject
/// - It is assumed that the name of the configuration subject is the SAME as the processor
#[derive(Debug)]
pub struct ToolCallProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for ToolCallProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for ToolCallProcessor {
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

    #[instrument(skip(self, message, diagnostic_builder, runtime_env))]
    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Extract the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let out = Box::pin(ToolCallStream::new(
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

#[allow(dead_code)]
pub struct ToolCallStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The messages to parse
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters tool calling
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    runtime_env: Arc<RuntimeEnv>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters tool calling after polling
    config: Option<ToolCallConfig>,
    /// `ViewTasksSubscribePublishAggregated` subject after polling
    subject_name: Option<Subject>,
    /// The tables of processor configurations after polling
    tool_calls: Vec<Subject>,
}

impl ToolCallStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::SessionTasksSubscribePublish.to_schema(),
            messages,
            config_stream,
            runtime_env,
            diagnostic_builder,
            config: None,
            subject_name: None,
            tool_calls: Vec::new(),
        })
    }

    fn init_config(&mut self, config_table: Subject) -> Result<()> {
        if self.config.is_none() {
            let config = ToolCallConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }
}

impl Stream for ToolCallStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.config.is_some() {
            // The config is set to None before first iteration,
            // and then set to Some after the first iteration
            // breaking the loop
            Poll::Ready(None)
        } else {
            // Initialize the metrics
            let baseline_metrics = if let Some(diagnostic_builder) = &self.diagnostic_builder {
                Some(
                    diagnostic_builder
                        .clone()
                        .to_child("ToolCallStream")?
                        .baseline_metrics(line!(), file!(), "poll_next"),
                )
            } else {
                None
            };
            let _timer = baseline_metrics
                .as_ref()
                .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

            // Initialize the config
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let config_table = Subject::get_builder()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;

            // Collect task/publisher subscriptions and publications as a HashMap
            let all_subscribe_publish_subject_name =
                self.config.as_ref().unwrap().subject_name.clone();
            let mut all_subscribe_publish_map = {
                let mut message_stream = if let Some(s) = remove_message_by_subject(
                    &all_subscribe_publish_subject_name,
                    &mut self.messages,
                ) {
                    s.get_message_own()
                } else {
                    return Poll::Ready(Some(Err(anyhow!(
                        "All task/publisher subscriptions and publications subject was not found in the message stream. Available messages are {:?}",
                        self.messages.keys()
                    ))));
                };
                let mut batches = Vec::new();
                while let Some(Ok(batch)) = ready!(message_stream.poll_next_unpin(cx)) {
                    batches.push(batch);
                }
                let table = Subject::get_builder()
                    .with_name("task/publisher subscriptions and publications")
                    .with_record_batches(batches)?
                    .build()?;
                let session_names = table.get_column_as_vec_string("session_name")?;
                let task_names = table.get_column_as_vec_string("task_name")?;
                let processor_names = table.get_column_as_vec_string("processor_name")?;
                let processor_types = table.get_column_as_vec_string("processor_type")?;
                let subscription_names =
                    table.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
                let subscription_table_names = table
                    .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
                let publication_names =
                    table.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
                let publication_table_names = table
                    .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
                session_names
                    .into_iter()
                    .zip(task_names)
                    .zip(processor_names)
                    .zip(processor_types)
                    .zip(subscription_names)
                    .zip(subscription_table_names)
                    .zip(publication_names)
                    .zip(publication_table_names)
                    .map(
                        |(
                            (
                                (
                                    (
                                        (
                                            ((session_name, task_name), processor_name),
                                            processor_type,
                                        ),
                                        subscription_name,
                                    ),
                                    subscription_table_name,
                                ),
                                publication_name,
                            ),
                            publication_table_name,
                        )| {
                            (
                                processor_name,
                                (
                                    session_name,
                                    task_name,
                                    processor_type,
                                    subscription_name,
                                    subscription_table_name,
                                    publication_name,
                                    publication_table_name,
                                ),
                            )
                        },
                    )
                    .collect::<HashMap<_, _>>()
            };

            // Collect tool call configuration record batches from the rest of the messages
            let subscription_table_names_set = self
                .config
                .as_ref()
                .unwrap()
                .subscription_table_names
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<_>>();
            let subscription_name_default = self
                .config
                .as_ref()
                .unwrap()
                .subscription_name
                .clone()
                .unwrap_or("AlwaysFullTable".to_string());
            let subject_names = self.config.as_ref().unwrap().subject_names.clone();
            let batch = {
                let mut session_names = Vec::new();
                let mut task_names = Vec::new();
                let mut processor_names = Vec::new();
                let mut processor_types = Vec::new();
                let mut subscription_names = Vec::new();
                let mut subscription_table_names = Vec::new();
                let mut publication_names = Vec::new();
                let mut publication_table_names = Vec::new();
                for subject_name in subject_names {
                    // Extract the message
                    let mut message_stream = if let Some(s) =
                        remove_message_by_subject(&subject_name, &mut self.messages)
                    {
                        s.get_message_own()
                    } else {
                        return Poll::Ready(Some(Err(anyhow!(
                            "Tool call subscription subject was not found in the message stream. Available messages are {:?}",
                            self.messages.keys()
                        ))));
                    };
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(message_stream.poll_next_unpin(cx)) {
                        batches.push(batch);
                    }

                    // Extract the subscription configuration batches
                    let table = Subject::get_builder()
                        .with_name("Tool call subscription subject")
                        .with_record_batches(batches)?
                        .build()?;

                    // Extract the tool call subscription table names directly from the table or
                    //  or from values/bytes schemas used with tool calls
                    let tool_call_subject_names = if table
                        .get_schema()
                        .fields()
                        .contains(&create_values_fields())
                    {
                        table
                            .get_column_as_vec_str("values")
                            .last()
                            .map(|b| {
                                serde_json::from_str::<Map<String, Value>>(b)
                                    .unwrap()
                                    .into_iter()
                                    .filter_map(|(k, v)| {
                                        if subscription_table_names_set.contains(&k) {
                                            Some(v.as_str().unwrap().to_string())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .ok_or(anyhow!(
                                "Missing `values` for subscription `{}` in tool_call_processor",
                                table.get_name()
                            ))?
                    } else if table.get_schema().fields().contains(&create_bytes_fields()) {
                        table
                            .get_column_as_vec_nested_primitive::<u8>("bytes")?
                            .last()
                            .map(|b| {
                                serde_json::from_slice::<Map<String, Value>>(b)
                                    .unwrap()
                                    .into_iter()
                                    .filter_map(|(k, v)| {
                                        if subscription_table_names_set.contains(&k) {
                                            Some(v.as_str().unwrap().to_string())
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .ok_or(anyhow!(
                                "Missing `bytes` for subscription `{}` in tool_call_processor",
                                table.get_name()
                            ))?
                    } else {
                        table
                            .get_schema()
                            .fields()
                            .iter()
                            .filter_map(|field| {
                                if subscription_table_names_set.contains(field.name()) {
                                    let subscription_table_name =
                                        table.get_column_as_vec_str(field.name());
                                    subscription_table_name.last().map(|name| name.to_string())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<String>>()
                    };

                    // Create the `SessionTasksSubscribePublish` batches
                    if let Some(all_subscribe_publish) =
                        all_subscribe_publish_map.remove(&subject_name)
                    {
                        session_names.push(all_subscribe_publish.0);
                        task_names.push(all_subscribe_publish.1);
                        processor_types.push(all_subscribe_publish.2);
                        let (sub, name): (Vec<_>, Vec<_>) = tool_call_subject_names
                            .into_iter()
                            .chain([subject_name.to_string()])
                            .map(|t| {
                                let (sub, name): (Vec<_>, Vec<_>) =
                                    all_subscribe_publish
                                        .3
                                        .iter()
                                        .zip(all_subscribe_publish.4.iter())
                                        .filter_map(|(sub, name)| {
                                            if name == &t { Some((sub, name)) } else { None }
                                        })
                                        .unzip();
                                if let (Some(sub), Some(name)) = (sub.first(), name.first()) {
                                    (sub.to_string(), name.to_string())
                                } else {
                                    (subscription_name_default.to_owned(), t)
                                }
                            })
                            .unzip();
                        subscription_names.push(sub);
                        subscription_table_names.push(name);
                        publication_names.push(all_subscribe_publish.5);
                        publication_table_names.push(all_subscribe_publish.6);
                        processor_names.push(subject_name);
                    } else {
                        return Poll::Ready(Some(Err(anyhow!(
                            "Tool call subscription subject `{subject_name}` was not found in the All task/publisher subscriptions and publications. Available subscription subjects are {:?}",
                            all_subscribe_publish_map.keys()
                        ))));
                    };
                }
                create_session_tasks_subscribe_publish_batch(
                    session_names,
                    task_names,
                    processor_names,
                    processor_types,
                    subscription_names,
                    subscription_table_names,
                    publication_names,
                    publication_table_names,
                )?
            };

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            if let Some(baseline_metrics) = &baseline_metrics {
                baseline_metrics.record_poll(poll)
            } else {
                poll
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }
}

impl RecordBatchStream for ToolCallStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{SubjectBuilder, Publication, create_bytes_record_batch};
    use phymes_data::DataConfig;
    use phymes_diagnostics::{Diagnostics, SpanBuilder};

    use super::*;

    #[tokio::test]
    async fn test_tool_call_processor_from_struct() -> Result<()> {
        let name = "tool_call_processor";

        // Make the diagnostics and runtime env
        let span = SpanBuilder::default().with_span(name).build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let runtime_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Make the tool_call_processor config
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let tool_call_processor_config = ToolCallConfig {
            subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            subject_names: vec!["processor_1".to_string(), "processor_2".to_string()],
            subscription_table_names: vec!["lhs_name".to_string(), "rhs_name".to_string()],
            ..Default::default()
        };
        let tool_call_processor_config_json = serde_json::to_vec(&tool_call_processor_config)?;
        let tool_call_processor_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&tool_call_processor_config_json, 1)?
            .build()?;
        let _ = message.insert(
            tool_call_processor_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(tool_call_processor_config_table.get_name())
                .with_publisher("")
                .with_subject(tool_call_processor_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(tool_call_processor_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the dummy processor configs
        let processor_1_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            ..Default::default()
        };
        let processor_1_config_json = serde_json::to_vec(&processor_1_config)?;
        let processor_1_config_table = SubjectBuilder::new()
            .with_name("processor_1")
            .with_json(&processor_1_config_json, 1)?
            .build()?;
        let _ = message.insert(
            processor_1_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_1_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_1_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_1_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor_2_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            ..Default::default()
        };
        let processor_2_config_json = serde_json::to_vec(&processor_2_config)?;
        let processor_2_config_table = SubjectBuilder::new()
            .with_name("processor_2")
            .with_json(&processor_2_config_json, 1)?
            .build()?;
        let _ = message.insert(
            processor_2_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_2_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_2_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_2_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the mock subject_name table
        let task_names = vec!["task_1", "task_2", "task_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec!["processor_1", "processor_2", "processor_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", "Join", "Filter"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![
            vec!["OnUpdateFullTable", "AlwaysLastRecordBatch"],
            vec!["OnUpdateFullTable", "AlwaysLastRecordBatch"],
            vec!["OnUpdateFullTable", "AlwaysLastRecordBatch"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec!["state_1", "processor_1"],
            vec!["state_2", "processor_2"],
            vec!["state_3", "processor_3"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], vec!["Replace"], vec!["Replace"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![vec!["state_1"], vec!["state_2"], vec!["state_3"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| "session_1".to_string())
            .collect::<Vec<_>>();
        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Subject::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message.insert(
            table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(table.get_name())
                .with_publisher("")
                .with_subject(table.get_name())
                .with_update(&Publication::None)
                .with_message(table.to_record_batch_stream())
                .build()?,
        );

        // Create the processor and run
        let processor = ToolCallProcessor::new(name, ToolCallProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let table_reading = SubjectBuilder::new_from_sendable_record_batch_stream(
            stream.remove(name).unwrap().message.take().unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, ["session_1", "session_1"]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, ["processor_1", "processor_2"]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, ["GroupBy", "Join"]);
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "OnUpdateFullTable",
                "AlwaysLastRecordBatch",
                "AlwaysFullTable",
                "OnUpdateFullTable",
                "AlwaysLastRecordBatch"
            ]
        );
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "state_1",
                "processor_1",
                "state_1",
                "state_2",
                "processor_2"
            ]
        );
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["Replace", "Replace"]);
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["state_1", "state_2"]);

        Ok(())
    }

    #[tokio::test]
    async fn test_tool_call_processor_from_bytes() -> Result<()> {
        let name = "tool_call_processor";

        // Make the diagnostics and runtime env
        let span = SpanBuilder::default().with_span(name).build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let runtime_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Make the tool_call_processor config
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let tool_call_processor_config = ToolCallConfig {
            subject_name: AvailableSubjects::SessionTasksSubscribePublish.to_string(),
            subject_names: vec!["processor_1".to_string(), "processor_2".to_string()],
            subscription_table_names: vec!["lhs_name".to_string(), "rhs_name".to_string()],
            ..Default::default()
        };
        let tool_call_processor_config_json = serde_json::to_vec(&tool_call_processor_config)?;
        let tool_call_processor_config_table = SubjectBuilder::new()
            .with_name(name)
            .with_json(&tool_call_processor_config_json, 1)?
            .build()?;
        let _ = message.insert(
            tool_call_processor_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(tool_call_processor_config_table.get_name())
                .with_publisher("")
                .with_subject(tool_call_processor_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(tool_call_processor_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the dummy processor configs
        let processor_1_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            ..Default::default()
        };
        let processor_1_config_json = serde_json::to_vec(&processor_1_config)?;
        let processor_1_config_batches = create_bytes_record_batch(vec![processor_1_config_json])?;
        let processor_1_config_table = SubjectBuilder::new()
            .with_name("processor_1")
            .with_record_batches(vec![processor_1_config_batches])?
            .build()?;
        let _ = message.insert(
            processor_1_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_1_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_1_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_1_config_table.to_record_batch_stream())
                .build()?,
        );
        let processor_2_config = DataConfig {
            cpu: false,
            lhs_name: Some("state_1".to_string()),
            rhs_name: Some("state_2".to_string()),
            ..Default::default()
        };
        let processor_2_config_json = serde_json::to_vec(&processor_2_config)?;
        let processor_2_config_batches = create_bytes_record_batch(vec![processor_2_config_json])?;
        let processor_2_config_table = SubjectBuilder::new()
            .with_name("processor_2")
            .with_record_batches(vec![processor_2_config_batches])?
            .build()?;
        let _ = message.insert(
            processor_2_config_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(processor_2_config_table.get_name())
                .with_publisher("")
                .with_subject(processor_2_config_table.get_name())
                .with_update(&Publication::None)
                .with_message(processor_2_config_table.to_record_batch_stream())
                .build()?,
        );

        // Make the mock subject_name table
        let task_names = vec!["task_1", "task_2", "task_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_names = vec!["processor_1", "processor_2", "processor_3"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let processor_types = vec!["GroupBy", "Join", "Filter"]
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let subscription_names = vec![
            vec!["OnUpdateFullTable", "AlwaysLastRecordBatch"],
            vec!["OnUpdateFullTable", "AlwaysLastRecordBatch"],
            vec!["OnUpdateFullTable", "AlwaysLastRecordBatch"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let subscription_table_names = vec![
            vec!["state_1", "processor_1"],
            vec!["state_2", "processor_2"],
            vec!["state_3", "processor_3"],
        ]
        .into_iter()
        .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
        let publication_names = vec![vec!["Replace"], vec!["Replace"], vec!["Replace"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let publication_table_names = vec![vec!["state_1"], vec!["state_2"], vec!["state_3"]]
            .into_iter()
            .map(|v| v.into_iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let session_names = task_names
            .iter()
            .map(|_| "session_1".to_string())
            .collect::<Vec<_>>();
        let batch = create_session_tasks_subscribe_publish_batch(
            session_names,
            task_names,
            processor_names,
            processor_types,
            subscription_names,
            subscription_table_names,
            publication_names,
            publication_table_names,
        )?;
        let table = Subject::get_builder()
            .with_name(
                AvailableSubjects::SessionTasksSubscribePublish
                    .to_string()
                    .as_str(),
            )
            .with_record_batches(vec![batch])?
            .build()?;
        let _ = message.insert(
            table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(table.get_name())
                .with_publisher("")
                .with_subject(table.get_name())
                .with_update(&Publication::None)
                .with_message(table.to_record_batch_stream())
                .build()?,
        );

        // Create the processor and run
        let processor = ToolCallProcessor::new(name, ToolCallProcessor::get_static_name());
        let mut stream = processor.process(message, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let table_reading = SubjectBuilder::new_from_sendable_record_batch_stream(
            stream.remove(name).unwrap().message.take().unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        let column = table_reading.get_column_as_vec_str("session_name");
        assert_eq!(column, ["session_1", "session_1"]);
        let column = table_reading.get_column_as_vec_str("processor_name");
        assert_eq!(column, ["processor_1", "processor_2"]);
        let column = table_reading.get_column_as_vec_str("processor_type");
        assert_eq!(column, ["GroupBy", "Join"]);
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("subscription_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "OnUpdateFullTable",
                "AlwaysLastRecordBatch",
                "AlwaysFullTable",
                "OnUpdateFullTable",
                "AlwaysLastRecordBatch"
            ]
        );
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("subscription_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(
            flattened,
            [
                "state_1",
                "processor_1",
                "state_1",
                "state_2",
                "processor_2"
            ]
        );
        let column =
            table_reading.get_column_as_vec_nested_nonprimitive::<String>("publication_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["Replace", "Replace"]);
        let column = table_reading
            .get_column_as_vec_nested_nonprimitive::<String>("publication_table_names")?;
        let flattened = column.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(flattened, ["state_1", "state_2"]);

        Ok(())
    }
}
