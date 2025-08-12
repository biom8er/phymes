use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics, HashMap},
    schemas::{
        chat_completion::ToolCall,
        message_history::{
            create_messages_record_batch, create_timestamp_micros, create_values_record_batch,
        },
    },
    session::{
        common_traits::{
            BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap, StateMap,
        },
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::{AllTableNamesSubscribe, ArrowTableSubscribe, SubscribeTrait},
        stream::{RecordBatchStream, SendableRecordBatchStream},
    },
    task::{
        arrow_message::{
            ArrowMessageBuilderTrait, ArrowOutgoingMessage, ArrowOutgoingMessageBuilderTrait,
            ArrowOutgoingMessageTrait,
        },
        arrow_processor::ArrowProcessorTrait,
        publish_subscribe::PubSubTrait,
    },
};

use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use serde_json::json;
use tracing::{Level, event, instrument};

use crate::candle_chat::{chat_config::CandleChatConfig, tool_parser::format_tool_calls_str};

use super::tool_parser::extract_tool_calls_str;

/// Processor that takes an unstructured chat response
///   and attempts to convert to a structured output
///
/// # Notes
///
/// - The user will need to implement their custom logic
///   to determine how messages should be split and
///   to determine the destination that messages should be sent
/// - Please use the below as a template
///
/// # Assumptions
///
/// - There needs to be a subject called `Error` that any
///   message content that cannot be parsed can be sent to
///   for a retry
#[derive(Debug)]
pub struct MessageParserProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    subscribe: Box<dyn SubscribeTrait>,
}

impl MappableTrait for MessageParserProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PubSubTrait for MessageParserProcessor {
    fn get_publications(&self) -> Vec<&ArrowTablePublish> {
        self.publications.iter().collect()
    }
    fn get_subscriptions(&self) -> Vec<&ArrowTableSubscribe> {
        self.subscriptions.iter().collect()
    }
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        self.subscribe
            .check_subscriptions(&self.subscriptions, updates, state)
    }
}

impl ArrowProcessorTrait for MessageParserProcessor {
    fn new_arc_with_pub_sub(
        name: &str,
        publications: &[ArrowTablePublish],
        subscriptions: &[ArrowTableSubscribe],
        subscribe: Box<dyn SubscribeTrait>,
    ) -> Arc<dyn ArrowProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            subscribe,
        })
    }

    fn new_arc(name: &str) -> Arc<dyn ArrowProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: vec![ArrowTablePublish::None],
            subscriptions: vec![ArrowTableSubscribe::None],
            subscribe: AllTableNamesSubscribe::new_box(),
        })
    }

    #[instrument(skip(self, message, metrics, runtime_env))]
    fn process(
        &self,
        mut message: OutgoingMessageMap,
        metrics: ArrowTaskMetricsSet,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<OutgoingMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Extract out the messages and config
        let messages =
            match message.remove(self.get_subscriptions().first().unwrap().get_table_name()) {
                Some(i) => i.get_message_own(),
                None => return Err(anyhow!("Messages not provided for {}.", self.get_name())),
            };
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let out = Box::pin(MessageParserStream::new(
            messages,
            config,
            Arc::clone(&runtime_env),
            BaselineMetrics::new(&metrics, self.get_name()),
        )?);

        // By default, we send back to the publisher in case of any errors which the publisher
        //  can correct before moving on.
        // DM: this is not rigorously tested yet...
        let out_m = ArrowOutgoingMessage::get_builder()
            .with_name(self.get_publications().first().unwrap().get_table_name())
            .with_publisher(self.get_name())
            .with_subject(self.get_publications().first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.get_publications().first().unwrap())
            .build()?;
        let _ = message.insert(out_m.get_name().to_string(), out_m);
        Ok(message)
    }
}

#[allow(dead_code)]
pub struct MessageParserStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    message_stream: SendableRecordBatchStream,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    baseline_metrics: BaselineMetrics,
    /// Parameters for chat inference
    config: Option<CandleChatConfig>,
}

impl MessageParserStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        baseline_metrics: BaselineMetrics,
    ) -> Result<Self> {
        // Output schema
        let field_names = ["name", "publisher", "subject", "values"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields_vec));

        Ok(Self {
            schema,
            message_stream,
            config_stream,
            runtime_env,
            baseline_metrics,
            config: None,
        })
    }

    fn init_config(&mut self, config_table: ArrowTable) -> Result<()> {
        if self.config.is_none() {
            let config: CandleChatConfig = serde_json::from_value(serde_json::Value::Object(
                config_table.to_json_object()?.first().unwrap().to_owned(),
            ))?;
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
            let metrics = self.baseline_metrics.clone();
            let _timer = metrics.elapsed_compute().timer();

            // Initialize the config
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let config_table = ArrowTable::get_builder()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;

            // Collect the messages
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.message_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }

            // Concatenate into a single record batch
            let message = ArrowTable::get_builder()
                .with_name("")
                .with_record_batches(batches)?
                .build()?
                .concat_record_batches()?;
            // ... and then a single string
            let content = message.get_column_as_vec_str("content").join("");
            event!(Level::DEBUG, "Extracted content: {}", content.as_str());

            // Extract out the function arguments
            // First try the OpenAI `ToolCall` schema and
            // Second try the Candle `serde_json::Value` schema after parsing the raw content
            let batch = match serde_json::from_str::<Vec<ToolCall>>(&content) {
                Ok(tool_calls) => {
                    event!(Level::DEBUG, "ToolCall content: {:?}", &tool_calls);

                    // Wrap the parsed content into a record batch
                    let mut names_vec = Vec::new();
                    let mut publishers_vec = Vec::new();
                    let mut subjects_vec = Vec::new();
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

                        // Parse the arguments and rebuild the as a `serde_json::Value`
                        let arguments = serde_json::from_str::<serde_json::Value>(
                            tool_call.function.arguments.as_ref().unwrap().as_str(),
                        )?;
                        let values = json!({
                            "name": tool_call.function.name.as_ref().unwrap().as_str(),
                            "arguments": arguments
                        });
                        values_vec.push(serde_json::to_string(&values)?);
                    }
                    create_values_record_batch(names_vec, publishers_vec, subjects_vec, values_vec)?
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
                            let mut values_vec = Vec::new();
                            for json_value in json_values.into_iter() {
                                names_vec.push(
                                    json_value
                                        .get("name")
                                        .unwrap()
                                        .as_str()
                                        .unwrap()
                                        .to_string(),
                                );
                                publishers_vec.push("message_parser_processor".to_string());
                                subjects_vec.push(
                                    json_value
                                        .get("name")
                                        .unwrap()
                                        .as_str()
                                        .unwrap()
                                        .to_string(),
                                );
                                values_vec.push(serde_json::to_string(&json_value)?);
                            }
                            create_values_record_batch(
                                names_vec,
                                publishers_vec,
                                subjects_vec,
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
                            create_messages_record_batch(role_vec, content_vec, timestamp_vec)?
                        }
                    }
                }
            };

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            metrics.record_poll(poll)
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
    use phymes_core::{
        metrics::HashMap,
        table::{arrow_table::ArrowTableBuilder, arrow_table_publish::ArrowTablePublish},
    };

    use super::*;

    #[tokio::test]
    async fn test_message_processor_candle() -> Result<()> {
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
        let mut message_map = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message_map.insert(
            "messages".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("messages")
                .with_subject("messages")
                .with_publisher("s1")
                .with_update(&ArrowTablePublish::None)
                .with_message(
                    ArrowTable::get_builder()
                        .with_name("messages")
                        .with_record_batches(vec![batch])?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );
        let _ = message_map.insert(
            "message_processor".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("message_processor")
                .with_subject("message_processor")
                .with_publisher("message_processor")
                .with_update(&ArrowTablePublish::None)
                .with_message(
                    ArrowTable::get_builder()
                        .with_name("message_processor")
                        .with_json(
                            &serde_json::to_vec(&CandleChatConfig {
                                ..Default::default()
                            })?,
                            1,
                        )?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );

        let metrics = ArrowTaskMetricsSet::new();

        let runtime_env = Arc::new(Mutex::new(RuntimeEnv {
            token_service: None,
            tensor_service: None,
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        }));

        // Create the processor and run
        let processor = MessageParserProcessor::new_arc_with_pub_sub(
            "message_processor",
            &[ArrowTablePublish::ExtendChunks {
                table_name: "messages".to_string(),
                col_name: "content".to_string(),
            }],
            &[ArrowTableSubscribe::AlwaysFullTable {
                table_name: "messages".to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        );
        let mut stream = processor.process(message_map, metrics.clone(), runtime_env)?;

        // Wrap the results in a table
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            stream.remove("messages").unwrap().get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 2);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 2);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 10);
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
            partitions.get_column_as_vec_str("values"),
            [
                "{\"arguments\":{\"format\":\"celsius\",\"location\":\"San Francisco, CA\"},\"name\":\"get_current_weather\"}",
                "{\"arguments\":{\"location\":\"Santa Ana, CA\",\"time\":\"08:00\"},\"name\":\"get_weather\"}"
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_message_processor_openai() -> Result<()> {
        // Create the input
        let role: ArrayRef = Arc::new(StringArray::from(vec!["assistant".to_string()]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![
            "[{\"id\":\"fc_12345xyz\",\"type\":\"function\",\"function\":{\"name\":\"get_current_weather\",\"arguments\":\"{\\\"location\\\":\\\"San Francisco, CA\\\",\\\"format\\\":\\\"celsius\\\"}\"}}]",
        ]));
        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content)])?;
        let mut message_map = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message_map.insert(
            "messages".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("messages")
                .with_subject("messages")
                .with_publisher("s1")
                .with_update(&ArrowTablePublish::None)
                .with_message(
                    ArrowTable::get_builder()
                        .with_name("messages")
                        .with_record_batches(vec![batch])?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );
        let _ = message_map.insert(
            "message_processor".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("message_processor")
                .with_subject("message_processor")
                .with_publisher("message_processor")
                .with_update(&ArrowTablePublish::None)
                .with_message(
                    ArrowTable::get_builder()
                        .with_name("message_processor")
                        .with_json(
                            &serde_json::to_vec(&CandleChatConfig {
                                ..Default::default()
                            })?,
                            1,
                        )?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );

        let metrics = ArrowTaskMetricsSet::new();

        let runtime_env = Arc::new(Mutex::new(RuntimeEnv {
            token_service: None,
            tensor_service: None,
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        }));

        // Create the processor and run
        let processor = MessageParserProcessor::new_arc_with_pub_sub(
            "message_processor",
            &[ArrowTablePublish::ExtendChunks {
                table_name: "messages".to_string(),
                col_name: "content".to_string(),
            }],
            &[ArrowTableSubscribe::AlwaysFullTable {
                table_name: "messages".to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        );
        let mut stream = processor.process(message_map, metrics.clone(), runtime_env)?;

        // Wrap the results in a table
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            stream.remove("messages").unwrap().get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 1);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 10);
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
            partitions.get_column_as_vec_str("values"),
            [
                "{\"arguments\":{\"format\":\"celsius\",\"location\":\"San Francisco, CA\"},\"name\":\"get_current_weather\"}"
            ]
        );

        Ok(())
    }

    #[ignore = "dynamic schema update breaks `RecordBatchStream` in tests"]
    #[tokio::test]
    async fn test_message_processor_error() -> Result<()> {
        // Create the input
        let role: ArrayRef = Arc::new(StringArray::from(vec!["assistant"]));
        let content: ArrayRef = Arc::new(StringArray::from(vec![
            "<get_current_weather location=\"Boston, MA\" unit=\"fahrenheit\">",
        ]));
        let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content)])?;
        let mut message_map = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message_map.insert(
            "messages".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("messages")
                .with_subject("messages")
                .with_publisher("s1")
                .with_update(&ArrowTablePublish::None)
                .with_message(
                    ArrowTable::get_builder()
                        .with_name("messages")
                        .with_record_batches(vec![batch])?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );
        let _ = message_map.insert(
            "message_processor".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("message_processor")
                .with_subject("message_processor")
                .with_publisher("message_processor")
                .with_update(&ArrowTablePublish::None)
                .with_message(
                    ArrowTable::get_builder()
                        .with_name("message_processor")
                        .with_json(
                            &serde_json::to_vec(&CandleChatConfig {
                                ..Default::default()
                            })?,
                            1,
                        )?
                        .build()?
                        .to_record_batch_stream(),
                )
                .build()?,
        );

        let metrics = ArrowTaskMetricsSet::new();

        let runtime_env = Arc::new(Mutex::new(RuntimeEnv {
            token_service: None,
            tensor_service: None,
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        }));

        // Create the processor and run
        let processor = MessageParserProcessor::new_arc_with_pub_sub(
            "message_processor",
            &[ArrowTablePublish::ExtendChunks {
                table_name: "messages".to_string(),
                col_name: "content".to_string(),
            }],
            &[ArrowTableSubscribe::AlwaysFullTable {
                table_name: "messages".to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        );
        let mut stream = processor.process(message_map, metrics.clone(), runtime_env)?;

        // DM: this will result in an error because the schema is dynamically updated
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            stream.remove("messages").unwrap().get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 1);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 10);
        assert_eq!(partitions.get_column_as_vec_str("role"), ["assistant"]);
        assert_eq!(
            partitions.get_column_as_vec_str("content"),
            ["<get_current_weather location=\"Boston, MA\" unit=\"fahrenheit\">"]
        );

        Ok(())
    }
}
