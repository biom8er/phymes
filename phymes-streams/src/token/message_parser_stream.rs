use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    BuildableTrait, BuilderTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    Subject, SubjectBuilderTrait, SubjectTrait,
};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, MetricBuilderTrait, create_timestamp_micros,
};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_ml::CandleChatConfig;
use phymes_schemas::{
    AvailableSchemaTrait, AvailableSubjects, DataFormat, ToolCall, create_bytes_record_batch,
    create_chat_record_batch, create_route_bytes_record_batch,
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{Stream, StreamExt};
use tracing::{Level, event};

use crate::{extract_tool_calls_str, token::tool_parser::format_tool_calls_str};

#[allow(dead_code)]
pub struct MessageParserStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The messages to parse
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    runtime_env: Arc<RuntimeEnv>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<CandleChatConfig>,
}

impl MessageParserStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::RouteBytes.to_schema(),
            messages,
            config_stream,
            runtime_env,
            diagnostic_builder,
            config: None,
        })
    }

    fn init_config(&mut self, config_table: Subject) -> Result<()> {
        if self.config.is_none() {
            let config = CandleChatConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }
}

impl Stream for MessageParserStream {
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
                        .to_child("MessageParserStream")?
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

            // Collect the messages
            let messages_table = self.config.as_ref().unwrap().messages.clone();
            let mut message_stream = if let Some(s) =
                remove_message_by_subject(&messages_table, &mut self.messages)
            {
                s.get_message_own()
            } else {
                return Poll::Ready(Some(Err(anyhow!(
                    "Message history subject was not found in the message stream. Available messages are {:?}",
                    self.messages.keys()
                ))));
            };
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(message_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }

            // Concatenate into a single record batch
            let message = Subject::get_builder()
                .with_name("MessageParserStream")
                .with_record_batches(batches)?
                .build()?
                .concat_record_batches()?;
            // ... and then a single string
            let content = message.get_column_as_vec_str("content").join("");
            event!(Level::DEBUG, "Extracted content: {}", content.as_str());

            // Extract out the function arguments
            // 1. try the OpenAI `ToolCall` schema and
            // 2. try the Candle `serde_json::Value` schema after parsing the raw content
            let batch = match serde_json::from_str::<Vec<ToolCall>>(&content) {
                Ok(tool_calls) => {
                    event!(Level::DEBUG, "ToolCall content: {:?}", &tool_calls);

                    // Wrap the parsed content into a record batch
                    let mut names_vec = Vec::new();
                    let mut publishers_vec = Vec::new();
                    let mut subjects_vec = Vec::new();
                    let mut formats_vec = Vec::new();
                    let mut values_vec = Vec::new();
                    for tool_call in tool_calls.iter() {
                        names_vec.push(
                            tool_call
                                .function
                                .name
                                .as_ref()
                                .unwrap()
                                .as_str()
                                .to_string(),
                        );
                        publishers_vec.push("message_parser_processor".to_string());
                        subjects_vec.push(
                            tool_call
                                .function
                                .name
                                .as_ref()
                                .unwrap()
                                .as_str()
                                .to_string(),
                        );
                        formats_vec.push(DataFormat::Bytes.to_string());

                        // Parse the arguments and rebuild the as a `serde_json::Value`
                        //  that is compatible with `DataConfig`-like subject targets
                        let mut values =
                            serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
                                tool_call.function.arguments.as_ref().unwrap().as_str(),
                            )?;
                        let _ = values.insert(
                            "operator".to_string(),
                            serde_json::Value::String(
                                tool_call
                                    .function
                                    .name
                                    .as_ref()
                                    .unwrap()
                                    .as_str()
                                    .to_string(),
                            ),
                        );

                        // Wrap into a `Bytes` record batch
                        let batch = create_bytes_record_batch(vec![serde_json::to_vec(&values)?])?;
                        let bytes = Subject::get_builder()
                            .with_name("message_parser_processor serde_json::Value")
                            .with_record_batches(vec![batch])?
                            .build()?
                            .to_bytes()?
                            .to_vec();
                        values_vec.push(bytes);
                    }
                    create_route_bytes_record_batch(
                        names_vec,
                        publishers_vec,
                        subjects_vec,
                        formats_vec,
                        values_vec,
                    )?
                }
                Err(_e) => {
                    // Parse for Qwen
                    let content = extract_tool_calls_str(
                        content.as_str(),
                        Some("<tool_call>\n"),
                        Some("\n</tool_call>"),
                    );

                    // Parse for Llama
                    let content = content
                        .replace("}}<|python_tag|>{", "}},{")
                        .replace("<|python_tag|>", "")
                        .replace("|>", "")
                        // DM: convert Llama-style to OpenAI-style
                        .replace("\"parameters\":", "\"arguments\":")
                        .replace("\"function\":", "\"name\":");

                    // Clean up into a proper JSON list
                    let content = format_tool_calls_str(content.as_str());
                    match serde_json::from_str::<Vec<serde_json::Value>>(&content) {
                        Ok(json_values) => {
                            event!(Level::DEBUG, "JSON Values content: {:?}", &json_values);
                            // Wrap the parsed content into a record batch
                            let mut names_vec = Vec::new();
                            let mut publishers_vec = Vec::new();
                            let mut subjects_vec = Vec::new();
                            let mut formats_vec = Vec::new();
                            let mut values_vec = Vec::new();
                            for json_value in json_values.into_iter() {
                                let name = json_value
                                    .get("name")
                                    .unwrap()
                                    .as_str()
                                    .unwrap()
                                    .to_string();
                                names_vec.push(name.to_owned());
                                publishers_vec.push("message_parser_processor".to_string());
                                subjects_vec.push(name.to_owned());
                                formats_vec.push(DataFormat::Bytes.to_string());

                                // Parse the arguments and rebuild the as a `serde_json::Value`
                                //  that is compatible with `DataConfig`-like subject targets
                                let s = json_value.get("arguments").ok_or(anyhow!("Missing object for `arguments` when parsing JSON message response `{json_value}`."))?;
                                let mut map = serde_json::from_value::<
                                    serde_json::Map<String, serde_json::Value>,
                                >(s.to_owned())?;
                                let _ = map.insert(
                                    "operator".to_string(),
                                    serde_json::Value::String(name),
                                );

                                // Wrap into a `Bytes` record batch
                                let batch =
                                    create_bytes_record_batch(vec![serde_json::to_vec(&map)?])?;
                                let bytes = Subject::get_builder()
                                    .with_name("message_parser_processor serde_json::Value")
                                    .with_record_batches(vec![batch])?
                                    .build()?
                                    .to_bytes()?
                                    .to_vec();
                                values_vec.push(bytes);
                            }
                            create_route_bytes_record_batch(
                                names_vec,
                                publishers_vec,
                                subjects_vec,
                                formats_vec,
                                values_vec,
                            )?
                        }
                        Err(e) => {
                            // Cannot be parsed, send back to publisher
                            event!(Level::ERROR, "Unparsable content: {}", e.to_string());
                            self.schema = message.get_schema();

                            // and append error message for next try
                            let mut timestamp_vec = Vec::new();
                            let mut role_vec = Vec::new();
                            let mut content_vec = Vec::new();

                            // Assistant content
                            timestamp_vec.push(create_timestamp_micros());
                            role_vec.push(
                                message
                                    .get_column_as_vec_str("role")
                                    .first()
                                    .unwrap()
                                    .to_string(),
                            );
                            content_vec.push(content.to_string());

                            // // Mock user content
                            // DM: accuracy of SLMs is such that many will not use the HITL template to respond
                            // timestamp_vec.push(create_timestamp_micros());
                            // role_vec.push("user".to_string());
                            // content_vec.push(format!("Reformat your response to follow the tool schemas. If trying to respond to the user, use the human-in-the-loop tool."));
                            create_chat_record_batch(role_vec, content_vec, timestamp_vec)?
                        }
                    }
                }
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

impl RecordBatchStream for MessageParserStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
