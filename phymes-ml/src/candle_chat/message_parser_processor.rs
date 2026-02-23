use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, BuildableTrait, BuilderTrait, DataFormat,
    MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait, RecordBatchStream,
    RuntimeEnv, SendableRecordBatchStream, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, Table, TableBuilderTrait, TableTrait, ToolCall,
    create_bytes_record_batch, create_chat_record_batch, create_route_bytes_record_batch,
    remove_message_by_subject,
};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, create_timestamp_micros,
};

use anyhow::{Result, anyhow};
use arrow::{array::RecordBatch, datatypes::SchemaRef};
use futures::{Stream, StreamExt};
use tracing::{Level, event, instrument};

use crate::candle_chat::{chat_config::CandleChatConfig, tool_parser::format_tool_calls_str};

use super::tool_parser::extract_tool_calls_str;

/// Processor that takes an unstructured chat response
///   and attempts to convert to a structured output
///
/// # Notes
///
/// - Supports OpenAI and Llama tool response formats
/// - Parsed messages are routed based on the function call
/// - Messages that cannot be parsed are sent the default publish subject
///
/// # Todo
///
/// - Better support different tool response formats through the config
#[derive(Debug)]
pub struct MessageParserProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for MessageParserProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for MessageParserProcessor {
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
        let out = Box::pin(MessageParserStream::new(
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

    fn init_config(&mut self, config_table: Table) -> Result<()> {
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
            let config_table = Table::get_builder()
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
            let message = Table::get_builder()
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
                        let bytes = Table::get_builder()
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
                                let bytes = Table::get_builder()
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
                            content_vec.push(content);

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

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, StringArray};
    use phymes_core::{TableBuilder, TablePublication};
    use phymes_diagnostics::{Diagnostics, SpanBuilder};

    use crate::AvailableCandleAssets;

    use super::*;

    #[tokio::test]
    async fn test_message_parser_processor_candle() -> Result<()> {
        // Create the input
        let role: ArrayRef = Arc::new(StringArray::from(vec![
            "assistant".to_string(),
            "assistant".to_string(),
            "assistant".to_string(),
        ]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![
            "\n<tool_call>\n{\"name\": \"get_current_",
            "weather\", \"arguments\": {\"location\": \"San Francisco, CA\", \"format\": \"celsius\"}}, {\"name\":",
            "\"get_weather\", \"arguments\": {\"location\": \"Santa Ana, CA\", \"time\": \"08:00\"}}\n</tool_call><|im_end|>\n",
        ]));
        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content)])?;
        let mut message_map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message_map.insert(
            "messages".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("messages")
                .with_subject("messages")
                .with_publisher("s1")
                .with_update(&TablePublication::None)
                .with_message(
                    Table::get_builder()
                        .with_name("messages")
                        .with_record_batches(vec![batch])?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );
        let _ = message_map.insert(
            "message_processor".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("message_processor")
                .with_subject("message_processor")
                .with_publisher("message_processor")
                .with_update(&TablePublication::None)
                .with_message(
                    Table::get_builder()
                        .with_name("message_processor")
                        .with_json(
                            &serde_json::to_vec(&CandleChatConfig {
                                messages: "messages".to_string(),
                                max_tokens: 1000,
                                temperature: 0.8,
                                seed: 299792458,
                                repeat_penalty: 1.1,
                                repeat_last_n: 64,
                                candle_asset: Some(AvailableCandleAssets::default()),
                                ..Default::default()
                            })?,
                            1,
                        )?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        let runtime_env = Arc::new(RuntimeEnv {
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        });

        // Create the processor and run
        let processor = MessageParserProcessor::new("message_processor", "");
        let mut stream = processor.process(message_map, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            stream
                .remove("message_processor")
                .unwrap()
                .message
                .take()
                .unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 2);
        assert_eq!(
            partitions.get_column_as_vec_str("name"),
            ["get_current_weather", "get_weather"]
        );
        assert_eq!(
            partitions.get_column_as_vec_str("publisher"),
            ["message_parser_processor", "message_parser_processor"]
        );
        assert_eq!(
            partitions.get_column_as_vec_str("subject"),
            ["get_current_weather", "get_weather"]
        );
        assert_eq!(
            partitions.get_column_as_vec_str("format"),
            [DataFormat::Bytes.to_string(), DataFormat::Bytes.to_string()]
        );
        let test: Vec<String> = partitions
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flat_map(|b| {
                Table::get_builder()
                    .with_name("test_message_parser")
                    .with_schema(AvailableSubjects::Bytes.to_schema())
                    .with_bytes(&b)
                    .unwrap()
                    .build()
                    .unwrap()
                    .get_column_as_vec_nested_primitive::<u8>("bytes")
                    .unwrap()
                    .into_iter()
                    .map(|b| String::from_utf8(b).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            test,
            [
                "{\"format\":\"celsius\",\"location\":\"San Francisco, CA\",\"operator\":\"get_current_weather\"}",
                "{\"location\":\"Santa Ana, CA\",\"operator\":\"get_weather\",\"time\":\"08:00\"}"
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_message_parser_processor_openai() -> Result<()> {
        // Create the input
        let role: ArrayRef = Arc::new(StringArray::from(vec!["assistant".to_string()]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![
            "[{\"id\":\"fc_12345xyz\",\"type\":\"function\",\"function\":{\"name\":\"get_current_weather\",\"arguments\":\"{\\\"location\\\":\\\"San Francisco, CA\\\",\\\"format\\\":\\\"celsius\\\"}\"}}]",
        ]));
        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content)])?;
        let mut message_map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message_map.insert(
            "messages".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("messages")
                .with_subject("messages")
                .with_publisher("s1")
                .with_update(&TablePublication::None)
                .with_message(
                    Table::get_builder()
                        .with_name("messages")
                        .with_record_batches(vec![batch])?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );
        let _ = message_map.insert(
            "message_processor".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("message_processor")
                .with_subject("message_processor")
                .with_publisher("message_processor")
                .with_update(&TablePublication::None)
                .with_message(
                    Table::get_builder()
                        .with_name("message_processor")
                        .with_json(
                            &serde_json::to_vec(&CandleChatConfig {
                                messages: "messages".to_string(),
                                max_tokens: 1000,
                                temperature: 0.8,
                                seed: 299792458,
                                repeat_penalty: 1.1,
                                repeat_last_n: 64,
                                candle_asset: Some(AvailableCandleAssets::default()),
                                ..Default::default()
                            })?,
                            1,
                        )?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        let runtime_env = Arc::new(RuntimeEnv {
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        });

        // Create the processor and run
        let processor = MessageParserProcessor::new("message_processor", "");
        let mut stream = processor.process(message_map, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            stream
                .remove("message_processor")
                .unwrap()
                .message
                .take()
                .unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(
            partitions.get_column_as_vec_str("name"),
            ["get_current_weather"]
        );
        assert_eq!(
            partitions.get_column_as_vec_str("publisher"),
            ["message_parser_processor"]
        );
        assert_eq!(
            partitions.get_column_as_vec_str("subject"),
            ["get_current_weather"]
        );
        assert_eq!(
            partitions.get_column_as_vec_str("format"),
            [DataFormat::Bytes.to_string()]
        );
        let test: Vec<String> = partitions
            .get_column_as_vec_nested_primitive::<u8>("bytes")?
            .into_iter()
            .flat_map(|b| {
                Table::get_builder()
                    .with_name("test_message_parser")
                    .with_schema(AvailableSubjects::Bytes.to_schema())
                    .with_bytes(&b)
                    .unwrap()
                    .build()
                    .unwrap()
                    .get_column_as_vec_nested_primitive::<u8>("bytes")
                    .unwrap()
                    .into_iter()
                    .map(|b| String::from_utf8(b).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(
            test,
            [
                "{\"format\":\"celsius\",\"location\":\"San Francisco, CA\",\"operator\":\"get_current_weather\"}"
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_message_parser_processor_error() -> Result<()> {
        // Create the input
        let role: ArrayRef = Arc::new(StringArray::from(vec!["assistant"]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![
            "<get_current_weather location=\"Boston, MA\" unit=\"fahrenheit\">",
        ]));
        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content)])?;
        let mut message_map = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message_map.insert(
            "messages".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("messages")
                .with_subject("messages")
                .with_publisher("s1")
                .with_update(&TablePublication::None)
                .with_message(
                    Table::get_builder()
                        .with_name("messages")
                        .with_record_batches(vec![batch])?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );
        let _ = message_map.insert(
            "message_processor".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("message_processor")
                .with_subject("message_processor")
                .with_publisher("message_processor")
                .with_update(&TablePublication::None)
                .with_message(
                    Table::get_builder()
                        .with_name("message_processor")
                        .with_json(
                            &serde_json::to_vec(&CandleChatConfig {
                                messages: "messages".to_string(),
                                max_tokens: 1000,
                                temperature: 0.8,
                                seed: 299792458,
                                repeat_penalty: 1.1,
                                repeat_last_n: 64,
                                candle_asset: Some(AvailableCandleAssets::default()),
                                ..Default::default()
                            })?,
                            1,
                        )?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        let runtime_env = Arc::new(RuntimeEnv {
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        });

        // Create the processor and run
        let processor = MessageParserProcessor::new("message_processor", "");
        let mut stream = processor.process(message_map, Some(&diagnostic_builder), runtime_env)?;

        // DM: this will result in an error because the schema is dynamically updated
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            stream
                .remove("message_processor")
                .unwrap()
                .message
                .take()
                .unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(partitions.get_column_as_vec_str("role"), ["assistant"]);
        assert_eq!(
            partitions.get_column_as_vec_str("content"),
            ["<get_current_weather location=\"Boston, MA\" unit=\"fahrenheit\">"]
        );

        Ok(())
    }
}
