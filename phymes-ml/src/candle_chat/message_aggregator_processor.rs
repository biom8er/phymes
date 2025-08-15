use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics, HashMap},
    schemas::message_history::{create_messages_fields, create_messages_schema},
    session::{
        common_traits::{
            BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap, StateMap, device,
        },
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait},
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
    datatypes::{Fields, SchemaRef},
};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use phymes_data::{
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
    message: &mut OutgoingMessageMap,
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
pub struct MessageAggregatorProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    subscribe: Box<dyn SubscribeTrait>,
}


impl MappableTrait for MessageAggregatorProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PubSubTrait for MessageAggregatorProcessor {
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

impl ArrowProcessorTrait for MessageAggregatorProcessor {
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
            publications: vec![ArrowTablePublish::Extend {
                table_name: "messages".to_string(),
            }],
            subscriptions: vec![ArrowTableSubscribe::None],
            subscribe: AllTableNamesSubscribe::new_box(),
        })
    }
    
    fn get_subscribe(&self) -> &Box<dyn SubscribeTrait> {
        &self.subscribe
    }

    fn get_type(&self) -> &str {
        Self::get_static_name()
    }

    #[instrument(skip(self, message, metrics, runtime_env))]
    fn process(
        &self,
        mut message: OutgoingMessageMap,
        metrics: ArrowTaskMetricsSet,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<OutgoingMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Collect the messages with the messages schema
        let input = collect_messages_by_schema(&mut message, &create_messages_fields());

        // Extract out the config
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let out = Box::pin(MessageAggregatorStream::new(
            input,
            config,
            Arc::clone(&runtime_env),
            BaselineMetrics::new(&metrics, self.get_name()),
        )?);
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
pub struct MessageAggregatorStream {
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
    baseline_metrics: BaselineMetrics,
}

impl MessageAggregatorStream {
    pub fn new(
        input: Vec<SendableRecordBatchStream>,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        baseline_metrics: BaselineMetrics,
    ) -> Result<Self> {
        Ok(Self {
            schema: create_messages_schema(),
            input,
            config_stream,
            config: None,
            data_operator: None,
            runtime_env,
            baseline_metrics,
        })
    }

    #[instrument(skip(self))]
    fn init_config(&mut self, config_table: ArrowTable) -> Result<()> {
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
                .try_lock()
                .unwrap()
                .tensor_service
                .is_none()
            {
                let device = device(config.cpu)?;
                let service = CandleTensorService::new(device);
                let _ = self
                    .runtime_env
                    .try_lock()
                    .unwrap()
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

impl Stream for MessageAggregatorStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.input.is_empty() {
            Poll::Ready(None)
        } else {
            // Initialize the metrics
            let metrics = self.baseline_metrics.clone();
            let _timer = metrics.elapsed_compute().timer();

            // initialize the config and tensor services
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let config_table = ArrowTableBuilder::new()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;
            self.init_tensor_service()?;

            // Build the data operator
            if self.data_operator.is_none() {
                let config = self.config.as_ref().unwrap().clone();
                self.data_operator.replace(config.which.build(
                    &config.lhs_pk,
                    &config.lhs_fk,
                    &config.lhs_values,
                    config.rhs_pk.as_deref(),
                    config.rhs_fk.as_deref(),
                    config.rhs_values.as_deref(),
                    config.op_kwargs.as_deref(),
                ));
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

impl RecordBatchStream for MessageAggregatorStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::{
        metrics::HashMap,
        table::arrow_table::{
            ArrowTableBuilder,
            test_table::{make_test_table, make_test_table_chat},
        },
    };
    use phymes_data::candle_operators::available_candle_operators::AvailableCandleOperators;

    use super::*;

    #[tokio::test]
    async fn test_message_aggregator_processor() -> Result<()> {
        // Create the input
        let mut message_1 = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message_1.insert(
            "m1".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("m1")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&ArrowTablePublish::None)
                .with_message(make_test_table_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m2".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("m2")
                .with_publisher("s1")
                .with_subject("messages")
                .with_update(&ArrowTablePublish::None)
                .with_message(make_test_table_chat("messages")?.to_record_batch_stream())
                .build()?,
        );
        let _ = message_1.insert(
            "m3".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("m3")
                .with_publisher("s3")
                .with_subject("messages")
                .with_update(&ArrowTablePublish::None)
                .with_message(make_test_table("t1", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );

        // Make the config
        let config = DataConfig {
            lhs_name: "".to_string(),
            lhs_pk: "".to_string(),
            lhs_fk: "".to_string(),
            lhs_values: "timestamp".to_string(),
            op_kwargs: Some("{\"asc\": true}".to_string()),
            which: AvailableCandleOperators::SortColumnAndIndices,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = ArrowTableBuilder::new()
            .with_name("aggregator_processor")
            .with_json(&config_json, 1)?
            .build()?;
        let _ = message_1.insert(
            "aggregator_processor".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("aggregator_processor")
                .with_publisher("")
                .with_subject("")
                .with_update(&ArrowTablePublish::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let metrics = ArrowTaskMetricsSet::new();

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
        let mut agg_stream = agg_arc_1.process(message_1, metrics.clone(), runtime_env)?;
        assert_eq!(agg_stream.len(), 2);
        assert!(agg_stream.get("messages").is_some());
        assert!(agg_stream.get("m3").is_some());

        // Wrap the results in a table
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
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
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 8);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);

        Ok(())
    }
}
