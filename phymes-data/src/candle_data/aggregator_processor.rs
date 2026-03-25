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
    SendableRecordBatchStreamMessageMap, Subject, SubjectBuilder, SubjectBuilderTrait,
    remove_message_by_subject,
};

use crate::{
    CandleTensorService, DataConfig, DataConfigTrait, DataOperatorTrait, TensorProcessorTrait,
    device,
};
use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Fields, SchemaRef},
};
use futures::{Stream, StreamExt};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, EventBuilderTrait, HashMap, MetricBuilderTrait,
};
use tracing::{Level, event, instrument};

/// Collect messages that match a given schema
///
/// # Arguments
/// * `messages` - The messages to process
/// * `fields` - The fields of the schema that need to match
///
/// # Returns
/// Vec of extracted messages
pub fn collect_messages_by_schema(
    message: &mut SendableRecordBatchStreamMessageMap,
    fields: &Fields,
) -> Vec<Pin<Box<dyn RecordBatchStream + Send>>> {
    message
        .extract_if(|_msg_name, msg| msg.get_message().schema().fields().contains(fields))
        .map(|(_msg_name, msg)| msg.get_message_own())
        .collect::<Vec<_>>()
}

/// Processor that aggregates messages
///
/// # Notes
///
/// - There is no guarantee that the order of incoming
///   messages is preserved
/// - All incoming meessages MUST have the same schema
#[derive(Debug)]
pub struct AggregatorProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for AggregatorProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for AggregatorProcessor {
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

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Run the aggregator stream
        let out = Box::pin(AggregatorStream::new(
            AvailableSubjects::Attachments.to_schema(),
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
pub struct AggregatorStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    input: SendableRecordBatchStreamMessageMap,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// Parameters for tensor operations
    config: Option<DataConfig>,
    /// The service for operating over tensors
    tensor_service: Option<Box<dyn TensorProcessorTrait>>,
    /// The data operator to run
    data_operator: Option<Box<dyn DataOperatorTrait>>,
    /// The Candle model assets needed for inference
    runtime_env: Arc<RuntimeEnv>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
}

impl AggregatorStream {
    pub fn new(
        schema: SchemaRef,
        input: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema,
            input,
            config_stream,
            config: None,
            tensor_service: None,
            data_operator: None,
            runtime_env,
            diagnostic_builder,
        })
    }

    #[instrument(skip(self))]
    fn init_config(&mut self, config_table: Subject) -> Result<()> {
        if self.config.is_none() {
            let config = DataConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        Ok(())
    }

    fn init_tensor_service(&mut self) -> Result<()> {
        if let Some(ref config) = self.config {
            if self.tensor_service.is_none() {
                let device = device(config.cpu)?;
                let service = CandleTensorService::new(device);
                let _ = self.tensor_service.replace(Box::new(service));
            }
        } else {
            return Err(anyhow!(
                "The config for Ops processor needs to be initialized before trying to initialize the tensor service."
            ));
        }
        Ok(())
    }
}

impl Stream for AggregatorStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.input.is_empty() {
            Poll::Ready(None)
        } else {
            // Initialize the metrics
            let baseline_metrics = if let Some(diagnostic_builder) = &self.diagnostic_builder {
                Some(
                    diagnostic_builder
                        .clone()
                        .to_child("DataSummaryStream")?
                        .baseline_metrics(line!(), file!(), "poll_next"),
                )
            } else {
                None
            };
            let _timer = baseline_metrics
                .as_ref()
                .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

            // initialize the config and tensor services
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let config_table = SubjectBuilder::new()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;
            self.init_tensor_service()?;

            // Build the data operator
            if self.data_operator.is_none() {
                let operator = self
                    .config
                    .as_ref()
                    .unwrap()
                    .operator
                    .build(self.config.as_ref().unwrap())?;
                self.data_operator.replace(operator);
            }

            // Collect the input
            let mut batches = Vec::new();
            for (_k, mut v) in self.input.drain() {
                while let Some(Ok(batch)) = ready!(v.get_message_mut().poll_next_unpin(cx)) {
                    // Skip empty batches
                    if batch.num_rows() > 0 {
                        batches.push(batch);
                    }
                }
            }

            // Clear the input so that any subsequent pools will return None
            self.input.clear();

            // Sort the record batches by timestamp and concatenate
            let batch = self.data_operator.as_ref().unwrap().forward(
                &batches,
                None,
                self.tensor_service.as_ref().unwrap().get_device(),
            )?;
            if batch.num_rows() == 0
                && let Some(diagnostic_builder) = &self.diagnostic_builder
            {
                let event = diagnostic_builder
                    .clone()
                    .to_child("AggregatorStream")?
                    .warn(line!(), file!(), "poll_next");
                event.insert("empty_batch", &serde_json::Value::String(
                        format!("The result of the data operator {} with config {:?} was an empty RecordBatch.", 
                            self.data_operator.as_ref().unwrap().get_name(),
                            self.config.as_ref().unwrap())));
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

impl RecordBatchStream for AggregatorStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AvailableCandleOperators, DataConfig};
    use phymes_core::{
        Publication, SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
        test_subject::{make_test_subject, make_test_subject_chat},
    };
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};

    use super::*;

    #[tokio::test]
    async fn test_aggregator_processor_schema_match() -> Result<()> {
        // Create the input
        let mut message_1 = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message_1.insert(
            "m1".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m1")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&Publication::None)
                .with_message(make_test_subject_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m2".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m2")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&Publication::None)
                .with_message(make_test_subject_chat("messages")?.to_record_batch_stream())
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
        let config_table = SubjectBuilder::new()
            .with_name("aggregator_processor")
            .with_json(&config_json, 1)?
            .build()?;
        let _ = message_1.insert(
            "aggregator_processor".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("aggregator_processor")
                .with_publisher("")
                .with_subject("aggregator_processor")
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let runtime_env = RuntimeEnv::get_builder().with_name("rt").build()?;
        let runtime_env = Arc::new(runtime_env);

        // Create the aggregator and run
        let agg_arc_1 = AggregatorProcessor::new("aggregator_processor", "");
        let mut agg_stream =
            agg_arc_1.process(message_1, Some(&diagnostic_builder), runtime_env)?;
        assert_eq!(agg_stream.len(), 1);
        assert!(agg_stream.get("aggregator_processor").is_some());

        // Wrap the results in a table
        let partitions = SubjectBuilder::new_from_sendable_record_batch_stream(
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

    #[tokio::test]
    async fn test_aggregator_processor_schema_mismatch_error() -> Result<()> {
        // Create the input
        let mut message_1 = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message_1.insert(
            "m1".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m1")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&Publication::None)
                .with_message(make_test_subject_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m2".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m2")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&Publication::None)
                .with_message(make_test_subject_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m3".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("m3")
                .with_publisher("s3")
                .with_subject("messages")
                .with_update(&Publication::None)
                .with_message(make_test_subject("t1", 4, 8, 3)?.to_record_batch_stream())
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
        let config_table = SubjectBuilder::new()
            .with_name("aggregator_processor")
            .with_json(&config_json, 1)?
            .build()?;
        let _ = message_1.insert(
            "aggregator_processor".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("aggregator_processor")
                .with_publisher("")
                .with_subject("aggregator_processor")
                .with_update(&Publication::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let runtime_env = RuntimeEnv::get_builder().with_name("rt").build()?;
        let runtime_env = Arc::new(runtime_env);

        // Create the aggregator and run
        let agg_arc_1 = AggregatorProcessor::new("aggregator_processor", "");
        let mut agg_stream =
            agg_arc_1.process(message_1, Some(&diagnostic_builder), runtime_env)?;
        assert_eq!(agg_stream.len(), 1);
        assert!(agg_stream.get("aggregator_processor").is_some());

        // Wrap the results in a table
        let subject_builder = SubjectBuilder::new_from_sendable_record_batch_stream(
            agg_stream
                .remove("aggregator_processor")
                .unwrap()
                .message
                .take()
                .unwrap(),
        )
        .await;
        assert!(subject_builder.is_err());

        Ok(())
    }
}
