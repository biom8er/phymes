use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_subject::{
    BuildableTrait, BuilderTrait, MappableTrait, RecordBatchStream, RuntimeEnv,
    SendableRecordBatchStream, Subject, SubjectBuilderTrait, SubjectTrait,
};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, HashSet, MetricBuilderTrait,
};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_schemas::{
    AvailableSchemaTrait, AvailableSubjects, create_bytes_fields,
    create_session_tasks_subscribe_publish_batch, create_values_fields,
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{Stream, StreamExt};
use serde_json::{Map, Value};

use crate::ToolCallConfig;

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
                .unwrap_or("AlwaysAllRecordBatches".to_string());
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
