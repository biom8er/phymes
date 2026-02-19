use std::sync::Arc;

use phymes_core::{
    AvailableSchemaTrait, AvailableSubjects, BuildableTrait, BuilderTrait, MappableTrait,
    MessageBuilderTrait, MessageTrait, ProcessorTrait, RuntimeEnv,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageBuilder,
    SendableRecordBatchStreamMessageBuilderMap, SendableRecordBatchStreamMessageMap,
    create_chat_fields, remove_message_by_subject,
};

use anyhow::{Result, anyhow};
use phymes_data::{AggregatorStream, collect_messages_by_schema};
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use tracing::{Level, event, instrument};

/// Processor that aggregates messages
///
/// # Notes
///
/// - There is no guarantee that the order of incoming messages is preserved
/// - All incoming meessages MUST have the same (chat) schema
#[derive(Debug)]
pub struct MessageAggregatorProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for MessageAggregatorProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for MessageAggregatorProcessor {
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

        // Extract the messages with the messages schema
        let input = collect_messages_by_schema(&mut message, &create_chat_fields());

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Run the aggregator stream
        let out = Box::pin(AggregatorStream::new(
            AvailableSubjects::Messages.to_schema(),
            input,
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
    use phymes_core::{
        TableBuilder, TableBuilderTrait, TablePublication, TableTrait,
        test_table::{make_test_table, make_test_table_chat},
    };
    use phymes_data::{AvailableCandleOperators, DataConfig};
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};

    use super::*;

    #[tokio::test]
    async fn test_message_aggregator_processor() -> Result<()> {
        // Create the input
        let mut message_1 = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message_1.insert(
            "m1".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m1")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&TablePublication::None)
                .with_message(make_test_table_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m2".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m2")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&TablePublication::None)
                .with_message(make_test_table_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m3".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m3")
                .with_publisher("s3")
                .with_subject("messages")
                .with_update(&TablePublication::None)
                .with_message(make_test_table("t1", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );

        // Make the config
        let config = DataConfig {
            lhs_values: Some(vec!["timestamp".to_string()]),
            op_kwargs: Some("{\"asc\": true}".to_string()),
            operator: AvailableCandleOperators::Sort,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = TableBuilder::new()
            .with_name("aggregator_processor")
            .with_json(&config_json, 1)?
            .build()?;
        let _ = message_1.insert(
            "aggregator_processor".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("aggregator_processor")
                .with_publisher("")
                .with_subject("aggregator_processor")
                .with_update(&TablePublication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let runtime_env = RuntimeEnv {
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        };
        let runtime_env = Arc::new(runtime_env);

        // Create the aggregator and run
        let agg_arc_1 = MessageAggregatorProcessor::new("aggregator_processor", "");
        let mut agg_stream =
            agg_arc_1.process(message_1, Some(&diagnostic_builder), runtime_env)?;
        assert_eq!(agg_stream.len(), 1);
        assert!(agg_stream.get("aggregator_processor").is_some());

        // Wrap the results in a table
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            agg_stream
                .remove("aggregator_processor")
                .unwrap()
                .message
                .take()
                .unwrap(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 8);
        assert_eq!(
            partitions.get_column_as_vec_str("role"),
            &[
                "user",
                "user",
                "assistant",
                "assistant",
                "user",
                "user",
                "assistant",
                "assistant"
            ]
        );
        assert_eq!(
            partitions.get_column_as_vec_str("content"),
            &[
                "Hi!",
                "Hi!",
                "magic!",
                "magic!",
                "What is Deep Learning?",
                "What is Deep Learning?",
                "Hello how can I help?",
                "Hello how can I help?"
            ]
        );
        assert_eq!(
            partitions
                .get_column_as_vec_primitive::<i64>("timestamp")
                .unwrap(),
            &[
                1754224496, 1754224496, 1754311256, 1754311256, 1754398256, 1754398256, 1754484956,
                1754484956
            ]
        );

        Ok(())
    }
}
