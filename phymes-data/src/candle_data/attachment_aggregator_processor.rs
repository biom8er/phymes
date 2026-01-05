use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    AvailableSubjects, AvailableSubjectsTrait, BuildableTrait, BuilderTrait, MappableTrait,
    MessageBuilderTrait, MessageTrait, ProcessorTrait, RecordBatchStream,
    RuntimeEnv, SendableRecordBatchStream, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageMap, StateMap, Table, TableBuilder, TableBuilderTrait,
    TablePublication, TableSubscribePolicyTrait, TableSubscription, create_blob_fields,
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
    TraceBuilderTrait,
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
pub struct AttachmentAggregatorProcessor {
    name: String,
    r#type: String,
}

impl MappableTrait for AttachmentAggregatorProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ProcessorTrait for AttachmentAggregatorProcessor {
    fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
        }
    }

    fn get_type(&self) -> &str {
        &self.r#type
    }

    #[instrument(skip(self, message, diagnostic_builder, runtime_env))]
    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
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
        let input = collect_messages_by_schema(&mut message, &create_blob_fields());

        // Extract out the config
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
        let out = Box::pin(AggregatorStream::new(
            AvailableSubjects::Blob.to_schema(),
            input,
            config,
            Arc::clone(&runtime_env),
            stream_diagnostic_builder,
        )?);
        let out_m = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher(self.get_name())
            .with_subject(
                self.get_publications()
                    .first()
                    .ok_or(anyhow!(
                        "Missing publications for processor {}",
                        self.get_name()
                    ))?
                    .get_table_name(),
            )
            .with_message(out)
            .with_update(self.get_publications().first().ok_or(anyhow!(
                "Missing publications for processor {}",
                self.get_name()
            ))?)
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

#[allow(dead_code)]
pub struct AggregatorStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    input: Vec<SendableRecordBatchStream>,
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
        input: Vec<SendableRecordBatchStream>,
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
    fn init_config(&mut self, config_table: Table) -> Result<()> {
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
            let config_table = TableBuilder::new()
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
            for i in self.input.as_mut_slice().iter_mut() {
                while let Some(Ok(batch)) = ready!(i.poll_next_unpin(cx)) {
                    batches.push(batch);
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
    use super::*;

    #[tokio::test]
    async fn test_attachment_aggregator_processor() -> Result<()> {
        // DM: see `phymes_ml::candle_chat::message_aggregator_processor.rs` for more comprehensive tests
        Ok(())
    }
}
