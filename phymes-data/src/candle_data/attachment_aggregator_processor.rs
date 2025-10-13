use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    metrics::{create_random_id, HashMap, MetricBuilder},
    schemas::{available_subjects::{create_timestamp_micros, AvailableSubjects, AvailableSubjectsTrait}, blob::create_blob_fields},
    session::{
        common_traits::{
            device, BuildableTrait, BuilderTrait, MappableTrait, SendableRecordBatchStreamMessageMap, StateMap
        },
        runtime_env::RuntimeEnv,
    },
    table::{
        stream::{RecordBatchStream, SendableRecordBatchStream}, table_publish::TablePublish, table_subscribe::{AllTableNamesSubscribe, SubscribeTrait, TableSubscribe}, table_trait::{Table, TableBuilder, TableBuilderTrait, TableTrait}
    },
    task::{
        message::{MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage},
        processor::ProcessorTrait,
        publish_subscribe::PubSubTrait,
    },
};

use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Fields, SchemaRef},
};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use crate::{
    candle_data::{data_config::DataConfig, tensor_service::CandleTensorService},
    candle_operators::data_operator::DataOperatorTrait,
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
    publications: Vec<TablePublish>,
    subscriptions: Vec<TableSubscribe>,
    subscribe: Box<dyn SubscribeTrait>,
}

impl MappableTrait for AttachmentAggregatorProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PubSubTrait for AttachmentAggregatorProcessor {
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

impl ProcessorTrait for AttachmentAggregatorProcessor {
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
        diagnostic_builder: &DiagnosticBuilder,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Collect the messages with the messages schema
        let input = collect_messages_by_schema(&mut message, &create_blob_fields());

        // Extract out the config
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let out = Box::pin(AggregatorStream::new(
            AvailableSubjects::Blob.to_schema(),
            input,
            config,
            Arc::clone(&runtime_env),
            diagnostic_builder.clone().to_child().with_span(self.get_name(), create_random_id()),
        )?);
        let out_m = SendableRecordBatchStreamMessage::get_builder()
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
pub struct AggregatorStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    input: Vec<SendableRecordBatchStream>,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// Parameters for tensor operations
    config: Option<DataConfig>,
    /// The data operator to run
    data_operator: Option<Box<dyn DataOperatorTrait>>,
    /// The Candle model assets needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    diagnostic_builder: DiagnosticBuilder,
}

impl AggregatorStream {
    pub fn new(
        schema: SchemaRef,
        input: Vec<SendableRecordBatchStream>,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        diagnostic_builder: DiagnosticBuilder,
    ) -> Result<Self> {
        Ok(Self {
            schema,
            input,
            config_stream,
            config: None,
            data_operator: None,
            runtime_env,
            diagnostic_builder,
        })
    }

    #[instrument(skip(self))]
    fn init_config(&mut self, config_table: Table) -> Result<()> {
        if self.config.is_none() {
            let config: DataConfig = serde_json::from_value(serde_json::Value::Object(
                config_table.to_json_object()?.first().unwrap().to_owned(),
            ))?;
            self.config.replace(config);
        }
        Ok(())
    }

    #[instrument(skip(self))]
    fn init_tensor_service(&mut self) -> Result<()> {
        if let Some(ref config) = self.config {
            if self
                .runtime_env
                .lock()
                .tensor_service
                .is_none()
            {
                let device = device(config.cpu)?;
                let service = CandleTensorService::new(device);
                let _ = self
                    .runtime_env
                    .lock()
                    .tensor_service
                    .replace(Box::new(service));
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
            let metrics = self.diagnostic_builder.clone().to_child().with_span("Stream", create_timestamp_micros().try_into().unwrap()).baseline_metrics();
            let _timer = metrics.elapsed_compute().timer();

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
                let operator = self.config.as_ref().unwrap().operator.build(&self.config.as_ref().unwrap());
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
                self.runtime_env
                    .try_lock()
                    .unwrap()
                    .tensor_service
                    .as_ref()
                    .unwrap()
                    .get_device(),
            )?;

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            metrics.record_poll(poll)
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
