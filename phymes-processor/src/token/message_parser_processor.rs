use std::sync::Arc;

use crate::ProcessorTrait;
use anyhow::{Result, anyhow};
use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, RuntimeEnv};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_streams::MessageParserStream;
use tracing::{Level, event, instrument};

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

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use phymes_core::{Subject, SubjectBuilder, SubjectBuilderTrait, SubjectTrait};
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};
    use phymes_event::Publication;
    use phymes_ml::{AvailableCandleAssets, CandleChatConfig};
    use phymes_schemas::{AvailableSchemaTrait, AvailableSubjects, DataFormat};

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
                .with_update(&Publication::None)
                .with_message(
                    Subject::get_builder()
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
                .with_update(&Publication::None)
                .with_message(
                    Subject::get_builder()
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

        let runtime_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Create the processor and run
        let processor = MessageParserProcessor::new("message_processor", "");
        let mut stream = processor.process(message_map, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let partitions = SubjectBuilder::new_from_sendable_record_batch_stream(
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
                Subject::get_builder()
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
                .with_update(&Publication::None)
                .with_message(
                    Subject::get_builder()
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
                .with_update(&Publication::None)
                .with_message(
                    Subject::get_builder()
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

        let runtime_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Create the processor and run
        let processor = MessageParserProcessor::new("message_processor", "");
        let mut stream = processor.process(message_map, Some(&diagnostic_builder), runtime_env)?;

        // Wrap the results in a table
        let partitions = SubjectBuilder::new_from_sendable_record_batch_stream(
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
                Subject::get_builder()
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
                .with_update(&Publication::None)
                .with_message(
                    Subject::get_builder()
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
                .with_update(&Publication::None)
                .with_message(
                    Subject::get_builder()
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

        let runtime_env = Arc::new(RuntimeEnv::get_builder().with_name("rt").build()?);

        // Create the processor and run
        let processor = MessageParserProcessor::new("message_processor", "");
        let mut stream = processor.process(message_map, Some(&diagnostic_builder), runtime_env)?;

        // DM: this will result in an error because the schema is dynamically updated
        let partitions = SubjectBuilder::new_from_sendable_record_batch_stream(
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
