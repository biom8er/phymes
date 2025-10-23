use std::sync::Arc;

use phymes_core::{
    AllTableNamesSubscribe, AvailableSubjects, AvailableSubjectsTrait, BuildableTrait,
    BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait, PubSubTrait,
    RuntimeEnv, SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap,
    SubscribeTrait, TablePublish, TableSubscribe, create_chat_fields,
};

use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use phymes_data::{AggregatorStream, collect_messages_by_schema};
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, TraceBuilderTrait};
use tracing::{Level, event, instrument};

/// Processor that aggregates messages
///
/// # Notes
///
/// - There is no guarantee that the order of incoming
///   messages is preserved
/// - All incoming meessages MUST have the same schema
#[derive(Debug)]
pub struct MessageAggregatorProcessor {
    name: String,
    publications: Vec<TablePublish>,
    subscriptions: Vec<TableSubscribe>,
    subscribe: Box<dyn SubscribeTrait>,
}

impl MappableTrait for MessageAggregatorProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PubSubTrait for MessageAggregatorProcessor {
    fn get_publications(&self) -> Vec<&TablePublish> {
        self.publications.iter().collect()
    }
    fn get_subscriptions(&self) -> Vec<&TableSubscribe> {
        self.subscriptions.iter().collect()
    }
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        self.subscribe
            .check_subscriptions(&self.subscriptions, updates, state)
    }
}

impl ProcessorTrait for MessageAggregatorProcessor {
    fn new_arc_with_pub_sub(
        name: &str,
        publications: &[TablePublish],
        subscriptions: &[TableSubscribe],
        subscribe: Box<dyn SubscribeTrait>,
    ) -> Arc<dyn ProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            subscribe,
        })
    }

    fn new_arc(name: &str) -> Arc<dyn ProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: vec![TablePublish::Extend {
                table_name: "messages".to_string(),
            }],
            subscriptions: vec![TableSubscribe::None],
            subscribe: AllTableNamesSubscribe::new_box(),
        })
    }

    fn get_subscribe(&self) -> &dyn SubscribeTrait {
        self.subscribe.as_ref()
    }

    fn get_type(&self) -> &str {
        Self::get_static_name()
    }

    #[instrument(skip(self, message, diagnostic_builder, runtime_env))]
    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

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

        // Collect the messages with the messages schema
        let input = collect_messages_by_schema(&mut message, &create_chat_fields());

        // Extract out the config
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
        let out = Box::pin(AggregatorStream::new(
            AvailableSubjects::Messages.to_schema(),
            input,
            config,
            Arc::clone(&runtime_env),
            stream_diagnostic_builder,
        )?);
        let out_m = SendableRecordBatchStreamMessage::get_builder()
            .with_name(self.get_publications().first().unwrap().get_table_name())
            .with_publisher(self.get_name())
            .with_subject(self.get_publications().first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.get_publications().first().unwrap())
            .build()?;
        let _ = message.insert(out_m.get_name().to_string(), out_m);

        // Trace the outbox
        if let Some(trace) = trace {
            trace.0.exit(&message.values().collect::<Vec<_>>());
        }
        Ok(message)
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{
        TableBuilder, TableBuilderTrait, TableTrait, device,
        test_table::{make_test_table, make_test_table_chat},
    };
    use phymes_data::{AvailableCandleOperators, CandleTensorService, DataConfig};
    use phymes_diagnostics::{Diagnostics, SpanBuilder};

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
                .with_update(&TablePublish::None)
                .with_message(make_test_table_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m2".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m2")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&TablePublish::None)
                .with_message(make_test_table_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m3".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m3")
                .with_publisher("s3")
                .with_subject("messages")
                .with_update(&TablePublish::None)
                .with_message(make_test_table("t1", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );

        // Make the config
        let config = DataConfig {
            lhs_name: "".to_string(),
            lhs_pk: "".to_string(),
            lhs_fk: "".to_string(),
            lhs_values: vec!["timestamp".to_string()],
            op_kwargs: Some("{\"asc\": true}".to_string()),
            operator: AvailableCandleOperators::SortColumnAndIndices,
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
                .with_subject("")
                .with_update(&TablePublish::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let device = device(config.cpu)?;
        let service = CandleTensorService::new(device);
        let runtime_env = RuntimeEnv {
            token_service: None,
            tensor_service: Some(Box::new(service)),
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        };
        let runtime_env = Arc::new(Mutex::new(runtime_env));

        // Create the aggregator and run
        let agg_arc_1 = MessageAggregatorProcessor::new_arc("aggregator_processor");
        let mut agg_stream =
            agg_arc_1.process(message_1, Some(&diagnostic_builder), runtime_env)?;
        assert_eq!(agg_stream.len(), 2);
        assert!(agg_stream.get("messages").is_some());
        assert!(agg_stream.get("m3").is_some());

        // Wrap the results in a table
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            agg_stream.remove("messages").unwrap().get_message_own(),
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
