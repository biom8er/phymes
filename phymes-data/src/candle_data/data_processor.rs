use crate::{CandleTensorService, DataConfig, DataConfigTrait, DataOperatorTrait, DataStreamManager, TensorProcessorTrait, device};
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait,
    PublishAndSubscribeTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, TableBuilder,
    TableBuilderTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription, TableTrait,
    remove_message_by_subject,
};

use arrow::{
    array::StringArray,
    datatypes::{DataType, Field, Fields, Schema, SchemaRef},
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, EventBuilderTrait, HashMap, MetricBuilderTrait,
    TraceBuilderTrait,
};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tracing::{Level, event, instrument};

/// Tensor processor made possible by Candle
///
/// Each operator has a defined input and output schema that calling processors or consuming processors
/// need to adhere to
#[derive(Debug)]
pub struct CandleDataProcessor {
    name: String,
    r#type: String,
    publications: Vec<TablePublication>,
    subscriptions: Vec<TableSubscription>,
    subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
}

impl MappableTrait for CandleDataProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PublishAndSubscribeTrait for CandleDataProcessor {
    fn get_publications(&self) -> Vec<&TablePublication> {
        self.publications.iter().collect()
    }

    fn get_subscriptions(&self) -> Vec<&TableSubscription> {
        self.subscriptions.iter().collect()
    }
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        self.subscribe_policy
            .check_subscriptions(&self.subscriptions, updates, state)
    }
}

impl ProcessorTrait for CandleDataProcessor {
    fn new(
        name: &str,
        r#type: &str,
        publications: &[TablePublication],
        subscriptions: &[TableSubscription],
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    ) -> Self {
        Self {
            name: name.to_string(),
            r#type: r#type.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            subscribe_policy,
        }
    }

    fn get_subscribe_policy(&self) -> &dyn TableSubscribePolicyTrait {
        self.subscribe_policy.as_ref()
    }

    fn get_type(&self) -> &str {
        &self.r#type
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

        // Extract out the config
        // let config = match message.remove(self.get_name()) {
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Remove subscriptions
        let mut subscriptions = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        for subs in self.subscriptions.iter() {
            if subs.get_table_name() != self.get_name() {
                match remove_message_by_subject(subs.get_table_name(), &mut message) {
                    // change from a random key to the subject name as the key to align with the [DataConfig]
                    Some(m) => {
                        subscriptions.insert(m.get_subject().to_string(), m);
                    }
                    None => {
                        return Err(anyhow!(
                            "Subscription {} not provided for {}.",
                            subs.get_table_name(),
                            self.get_name()
                        ));
                    }
                }
            }
        }

        // Run the ops
        let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
        let out = Box::pin(CandleDataStream::new(
            subscriptions,
            config,
            Arc::clone(&runtime_env),
            stream_diagnostic_builder,
        )?);
        let out_m = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher(self.get_name())
            .with_subject(
                self.publications
                    .first()
                    .ok_or(anyhow!(
                        "Missing publications for processor {}",
                        self.get_name()
                    ))?
                    .get_table_name(),
            )
            .with_message(out)
            .with_update(self.publications.first().ok_or(anyhow!(
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

/// Compute the relative similarity score between two embeddings
#[allow(dead_code)]
pub struct CandleDataStream {
    /// The messages containing the lhs and rhs
    /// which we cannot determine until we intialize the config
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for tensor operations
    config_stream: SendableRecordBatchStream,
    /// The tensor services needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for tensor operations
    config: Option<DataConfig>,
    /// The service for operating over tensors
    tensor_service: Option<Box<dyn TensorProcessorTrait>>,
    /// The data operator to run
    data_operator: Option<Box<dyn DataOperatorTrait>>,
    /// The polled record batches from the input
    lhs_inbox: Vec<RecordBatch>,
    /// The polled record batches from the input
    rhs_inbox: Vec<RecordBatch>,
    /// The prepared record batches for the output
    outbox: Vec<RecordBatch>,
    /// Switch to finished polling
    is_finished: bool,
}

impl CandleDataStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            messages,
            config_stream,
            diagnostic_builder,
            runtime_env,
            config: None,
            tensor_service: None,
            data_operator: None,
            lhs_inbox: Vec::new(),
            rhs_inbox: Vec::new(),
            outbox: Vec::new(),
            is_finished: false,
        })
    }

    fn init_tensor_service(&mut self) -> Result<()> {
        if let Some(ref config) = self.config {
            if self.tensor_service.is_none() {
                let device = device(config.cpu)?;
                let service = CandleTensorService::new(device);
                let _ = self.tensor_service
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

impl Stream for CandleDataStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_finished {
            return Poll::Ready(None);
        }

        // Initialize the metrics
        let baseline_metrics = if let Some(diagnostic_builder) = &self.diagnostic_builder {
            Some(
                diagnostic_builder
                    .clone()
                    .to_child("CandleDataStream")?
                    .baseline_metrics(line!(), file!(), "poll_next"),
            )
        } else {
            None
        };
        let _timer = baseline_metrics
            .as_ref()
            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());

        // Intialize the config
        if self.config.is_none() {
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let values = Fields::from_iter(vec![Field::new("values", DataType::Utf8, false)]);
            if batches
                .first()
                .ok_or(anyhow!("Config stream for CandleDataStream is empty"))?
                .schema()
                .fields()
                .contains(&values)
            {
                let config_json = batches
                    .first()
                    .unwrap()
                    .column_by_name("values")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join("");
                let mut config_values: serde_json::Value = serde_json::from_str(&config_json)?;
                config_values["arguments"]["operator"] = config_values["name"].clone();
                let config: DataConfig =
                    serde_json::from_value(config_values.get("arguments").unwrap().clone())?;
                self.config.replace(config);
            } else {
                let config_table = TableBuilder::new()
                    .with_name("config")
                    .with_record_batches(batches)?
                    .build()?;
                let config = DataConfig::from_table(&config_table)?;
                self.config.replace(config);
            }
        }
        // DM: need to implement a trigger for event verbosity
        // if let Some(diagnostic_builder) = &self.diagnostic_builder {
        //     let event = diagnostic_builder
        //         .clone()
        //         .to_child("CandleDataStream")?
        //         .debug(line!(), file!(), "poll_next");
        //     event.insert(
        //         "config",
        //         &serde_json::Value::String(format!("{:?}", &self.config)),
        //     );
        // };

        // Build the data operator
        if self.data_operator.is_none() {
            let operator = self
                .config
                .as_ref()
                .ok_or(anyhow!("Config CandleDataStream is empty"))?
                .operator
                .build(self.config.as_ref().unwrap())?;
            self.data_operator.replace(operator);
        }

        // Collect the LHS batches
        let stream = self.config.as_ref().unwrap().stream.clone();
        if self.lhs_inbox.is_empty() && self.config.as_ref().unwrap().lhs_name.is_some() {
            let lhs_name = self
                .config
                .as_ref()
                .unwrap()
                .lhs_name
                .as_ref()
                .ok_or(anyhow!(
                    "lhs_name was not provided for config {:?}",
                    self.config
                ))?
                .clone();

            // Poll all the LHS batches (accumulation) or the next LHS batch (stream)
            let lhs = match self.messages.get_mut(lhs_name.as_str()) {
                Some(lhs) => match stream {
                    DataStreamManager::AccumulateLHSAccumulateRHS
                    | DataStreamManager::AccumulateLHSStreamRHS => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) =
                            ready!(lhs.get_message_mut().poll_next_unpin(cx))
                        {
                            batches.push(batch);
                        }
                        batches
                    }
                    DataStreamManager::StreamLHSAccumulateRHS
                    | DataStreamManager::StreamLHSStreamRHS => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) =
                            ready!(lhs.get_message_mut().poll_next_unpin(cx))
                        {
                            if batch.num_rows() > 0 && batch.num_columns() > 0 {
                                batches.push(batch);
                                break;
                            }
                        }
                        batches
                    }
                },
                // Check for the LHS in the config (accumulation only)
                None => match stream {
                    DataStreamManager::AccumulateLHSAccumulateRHS
                    | DataStreamManager::AccumulateLHSStreamRHS => {
                        // Extract the input from the config
                        match self.config.as_ref().unwrap().lhs_args.as_ref() {
                            Some(qs) => {
                                let table = TableBuilder::new()
                                    .with_json(qs.as_bytes(), 512)?
                                    .with_name("")
                                    .build()?;
                                table.get_record_batches_own()
                            }
                            None => {
                                self.is_finished = true;
                                return Poll::Ready(Some(Err(anyhow!(
                                    "lhs_name {lhs_name} does not exist. Available options are {:?}",
                                    self.messages.keys()
                                ))));
                            }
                        }
                    }
                    DataStreamManager::StreamLHSAccumulateRHS
                    | DataStreamManager::StreamLHSStreamRHS => {
                        self.is_finished = true;
                        return Poll::Ready(Some(Err(anyhow!(
                            "lhs_name {lhs_name} does not exist. Available options are {:?}",
                            self.messages.keys()
                        ))));
                    }
                },
            };

            // Break if the lhs as been exhausted
            if lhs.is_empty() {
                self.is_finished = true;
                return Poll::Ready(None);
            } else {
                self.lhs_inbox = lhs;
            }
        };
        // DM: need to implement a trigger for event verbosity
        // if let Some(diagnostic_builder) = &self.diagnostic_builder {
        //     let event = diagnostic_builder
        //         .clone()
        //         .to_child("CandleDataStream")?
        //         .debug(line!(), file!(), "poll_next");
        //     event.insert(
        //         "lhs_inbox",
        //         &serde_json::Value::String(format!("{:?}", &self.lhs_inbox)),
        //     );
        // };

        // Collect the RHS batches through accumulating or as stream
        if self.rhs_inbox.is_empty() && self.config.as_ref().unwrap().rhs_name.is_some() {
            let rhs_name = self
                .config
                .as_ref()
                .unwrap()
                .rhs_name
                .as_ref()
                .ok_or(anyhow!(
                    "rhs_name was not provided for config {:?}",
                    self.config
                ))?
                .clone();

            // Poll all the RHS batches (accumulation) or the next RHS batch (stream)
            let rhs = match self.messages.get_mut(rhs_name.as_str()) {
                Some(rhs) => match stream {
                    DataStreamManager::AccumulateLHSAccumulateRHS
                    | DataStreamManager::StreamLHSAccumulateRHS => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) =
                            ready!(rhs.get_message_mut().poll_next_unpin(cx))
                        {
                            batches.push(batch);
                        }
                        batches
                    }
                    DataStreamManager::StreamLHSStreamRHS
                    | DataStreamManager::AccumulateLHSStreamRHS => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) =
                            ready!(rhs.get_message_mut().poll_next_unpin(cx))
                        {
                            if batch.num_rows() > 0 && batch.num_columns() > 0 {
                                batches.push(batch);
                                break;
                            }
                        }
                        batches
                    }
                },
                // Check for the RHS in the config (accumulation only)
                None => match stream {
                    DataStreamManager::AccumulateLHSAccumulateRHS
                    | DataStreamManager::StreamLHSAccumulateRHS => {
                        // Extract the input from the config
                        match self.config.as_ref().unwrap().rhs_args.as_ref() {
                            Some(qs) => {
                                let table = TableBuilder::new()
                                    .with_json(qs.as_bytes(), 512)?
                                    .with_name("")
                                    .build()?;
                                table.get_record_batches_own()
                            }
                            None => {
                                self.is_finished = true;
                                return Poll::Ready(Some(Err(anyhow!(
                                    "rhs_name {rhs_name} does not exist. Available options are {:?}",
                                    self.messages.keys()
                                ))));
                            }
                        }
                    }
                    DataStreamManager::StreamLHSStreamRHS
                    | DataStreamManager::AccumulateLHSStreamRHS => {
                        self.is_finished = true;
                        return Poll::Ready(Some(Err(anyhow!(
                            "rhs_name {rhs_name} does not exist. Available options are {:?}",
                            self.messages.keys()
                        ))));
                    }
                },
            };

            // Break if the rhs has been exhausted
            if rhs.is_empty() {
                self.is_finished = true;
                return Poll::Ready(None);
            } else {
                self.rhs_inbox = rhs;
            }
        }
        // DM: need to implement a trigger for event verbosity
        // if let Some(diagnostic_builder) = &self.diagnostic_builder {
        //     let event = diagnostic_builder
        //         .clone()
        //         .to_child("CandleDataStream")?
        //         .debug(line!(), file!(), "poll_next");
        //     event.insert(
        //         "rhs_inbox",
        //         &serde_json::Value::String(format!("{:?}", &self.rhs_inbox)),
        //     );
        // };

        // Compute the data operator
        self.init_tensor_service()?;
        let batch = match self.data_operator.as_ref().unwrap().forward(
            &self.lhs_inbox,
            Some(&self.rhs_inbox),
            self.tensor_service
                .as_ref()
                .unwrap()
                .get_device(),
        ) {
            Ok(batch) => batch,
            Err(err) => {
                if let Some(diagnostic_builder) = &self.diagnostic_builder {
                    let event = diagnostic_builder
                        .clone()
                        .to_child("CandleDataStream")?
                        .warn(line!(), file!(), "poll_next");
                    event.insert(
                        "data_operator",
                        &serde_json::Value::String(format!(
                            "Data operator {} with config {:?} resulted in an error: {err:?}",
                            self.data_operator.as_ref().unwrap().get_name(),
                            self.config.as_ref().unwrap()
                        )),
                    );
                };
                return Poll::Ready(Some(Err(err)));
            }
        };
        if batch.num_rows() == 0
            && let Some(diagnostic_builder) = &self.diagnostic_builder
        {
            let event = diagnostic_builder
                .clone()
                .to_child("CandleDataStream")?
                .warn(line!(), file!(), "poll_next");
            event.insert(
                "empty_batch",
                &serde_json::Value::String(format!(
                    "The result of the data operator {} with config {:?} was an empty RecordBatch.",
                    self.data_operator.as_ref().unwrap().get_name(),
                    self.config.as_ref().unwrap()
                )),
            );
        };
        // DM: need to implement a trigger for event verbosity
        // if let Some(diagnostic_builder) = &self.diagnostic_builder {
        //     let event = diagnostic_builder
        //         .clone()
        //         .to_child("CandleDataStream")?
        //         .debug(line!(), file!(), "poll_next");
        //     event.insert("result", &serde_json::Value::String(format!("{batch:?}")));
        // };

        // Reset the inboxes for the next poll and return the current poll
        match stream {
            DataStreamManager::AccumulateLHSAccumulateRHS => {
                self.is_finished = true;
                let poll = Poll::Ready(Some(Ok(batch)));
                if let Some(baseline_metrics) = &baseline_metrics {
                    baseline_metrics.record_poll(poll)
                } else {
                    poll
                }
            }
            DataStreamManager::AccumulateLHSStreamRHS => {
                self.rhs_inbox.clear();
                let poll = Poll::Ready(Some(Ok(batch)));
                if let Some(baseline_metrics) = &baseline_metrics {
                    baseline_metrics.record_poll(poll)
                } else {
                    poll
                }
            }
            DataStreamManager::StreamLHSAccumulateRHS => {
                self.lhs_inbox.clear();
                let poll = Poll::Ready(Some(Ok(batch)));
                if let Some(baseline_metrics) = &baseline_metrics {
                    baseline_metrics.record_poll(poll)
                } else {
                    poll
                }
            }
            DataStreamManager::StreamLHSStreamRHS => {
                self.lhs_inbox.clear();
                self.rhs_inbox.clear();
                let poll = Poll::Ready(Some(Ok(batch)));
                if let Some(baseline_metrics) = &baseline_metrics {
                    baseline_metrics.record_poll(poll)
                } else {
                    poll
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl RecordBatchStream for CandleDataStream {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::empty())
    }
}

#[allow(dead_code)]
pub mod test_candle_ops_processor {
    use std::sync::Arc;

    use anyhow::Result;
    use arrow::{
        array::{ArrayData, ArrayRef, FixedSizeListArray, RecordBatch, StringArray, UInt32Array},
        buffer::Buffer,
        datatypes::{DataType, Field},
    };

    pub fn make_embeddings_f32(embeddings: Vec<Vec<f32>>) -> ArrayRef {
        // Parse the embeddings
        let dim_1 = embeddings.len();
        let dim_2 = embeddings.first().unwrap().len();
        let embeddings_flat = embeddings.into_iter().flatten().collect::<Vec<_>>();

        // Make the embeddings array
        let value_data = ArrayData::builder(DataType::Float32)
            .len(dim_1 * dim_2)
            .add_buffer(Buffer::from_slice_ref(embeddings_flat))
            .build()
            .unwrap();
        let list_data_type = DataType::FixedSizeList(
            Arc::new(Field::new_list_field(DataType::Float32, false)),
            dim_2.try_into().unwrap(),
        );
        let list_data = ArrayData::builder(list_data_type.clone())
            .len(dim_1)
            .add_child_data(value_data.clone())
            .build()
            .unwrap();
        Arc::new(FixedSizeListArray::from(list_data))
    }

    pub fn make_embeddings_record_batch_str_f32(
        id_str: &str,
        ids: Vec<&str>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<RecordBatch> {
        let embedding: ArrayRef = make_embeddings_f32(embeddings);
        let ids_ar: ArrayRef = Arc::new(StringArray::from(ids));
        let batch = RecordBatch::try_from_iter(vec![(id_str, ids_ar), ("embedding", embedding)])?;
        Ok(batch)
    }

    pub fn make_embeddings_record_batch_u32_f32(
        id_str: &str,
        ids: Vec<u32>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<RecordBatch> {
        let embedding: ArrayRef = make_embeddings_f32(embeddings);
        let ids_ar: ArrayRef = Arc::new(UInt32Array::from(ids));
        let batch = RecordBatch::try_from_iter(vec![(id_str, ids_ar), ("embedding", embedding)])?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use crate::{DataDistanceOperator, candle_operators::AvailableCandleOperators};
    use arrow::array::Float32Array;
    use futures::TryStreamExt;
    use phymes_core::{AvailableTableSubscribePolicies, Table, TablePublication};
    use phymes_diagnostics::{Diagnostics, SpanBuilder};

    use super::*;

    #[tokio::test]
    async fn test_candle_ops_stream() -> Result<()> {
        // Case 1:  LHS and RHS messages from single stream batch
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "lhs_pk",
            lhs_ids_vec,
            lhs_embeddings_vec,
        )?;
        let lhs_table = Table::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;
        let rhs_ids_vec = vec!["1", "2", "3", "4"];
        let rhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let rhs_batch = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "rhs_pk",
            rhs_ids_vec,
            rhs_embeddings_vec,
        )?;
        let rhs_table = Table::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch])?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the config
        let config = DataConfig {
            lhs_name: Some("lhs_name".to_string()),
            rhs_name: Some("rhs_name".to_string()),
            lhs_pk: Some("lhs_pk".to_string()),
            lhs_fk: Some("lhs_fk".to_string()),
            lhs_values: Some(vec!["embedding".to_string()]),
            rhs_pk: Some("rhs_pk".to_string()),
            rhs_fk: Some("rhs_fk".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
            operator: AvailableCandleOperators::VectorDistance,
            ..Default::default()
        };
        let config_table = Table::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the metrics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let runtime_env = RuntimeEnv {
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        };
        let runtime_env = Arc::new(Mutex::new(runtime_env));

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_table.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        // Expected values
        let lhs_ids_test = vec!["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "3", "4", "1", "2", "3", "4", "1", "2", "3", "4"];
        let scores_test: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 0.70710677, 0.70710677, 0.70710677, 0.70710677, 0.5, 0.5, 0.5, 0.5,
        ];

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("lhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("rhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id, rhs_ids_test);
        let scores = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("score")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(scores, scores_test);

        // Case 2: LHS and RHS from config

        // Make the config
        let config_args = DataConfig {
            operator: AvailableCandleOperators::HumanInTheLoop,
            lhs_name: Some("".to_string()),
            lhs_args: Some("{\"role\": \"assistant\", \"content\": \"RESPONSE\"}".to_string()),
            rhs_args: None,
            ..Default::default()
        };
        let config_args_table = Table::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config_args)?, 1)?
            .build()?;

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            HashMap::<String, SendableRecordBatchStreamMessage>::new(),
            config_args_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("role")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id.first().unwrap(), &"assistant");
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("content")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id.first().unwrap(), &"RESPONSE");

        // Case 3: LHS and RHS messages from multiple stream batch (accumulate LHS and RHS)
        let lhs_ids_vec_1 = vec!["1"];
        let lhs_embeddings_vec_1: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.]];
        let lhs_batch_1 = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "lhs_pk",
            lhs_ids_vec_1,
            lhs_embeddings_vec_1,
        )?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_embeddings_vec_2: Vec<Vec<f32>> = vec![vec![0., 1., 0., 1.], vec![0., 0., 0., 1.]];
        let lhs_batch_2 = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "lhs_pk",
            lhs_ids_vec_2,
            lhs_embeddings_vec_2,
        )?;
        let lhs_table = Table::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch_1, lhs_batch_2])?
            .build()?;
        let rhs_ids_vec_1 = vec!["1", "2"];
        let rhs_embeddings_vec_1: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.], vec![1., 1., 1., 1.]];
        let rhs_batch_1 = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "rhs_pk",
            rhs_ids_vec_1,
            rhs_embeddings_vec_1,
        )?;
        let rhs_ids_vec_2 = vec!["3", "4"];
        let rhs_embeddings_vec_2: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.], vec![1., 1., 1., 1.]];
        let rhs_batch_2 = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "rhs_pk",
            rhs_ids_vec_2,
            rhs_embeddings_vec_2,
        )?;
        let rhs_table = Table::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch_1, rhs_batch_2])?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_table.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        // Expected values
        let lhs_ids_test = vec!["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "3", "4", "1", "2", "3", "4", "1", "2", "3", "4"];
        let scores_test: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 0.70710677, 0.70710677, 0.70710677, 0.70710677, 0.5, 0.5, 0.5, 0.5,
        ];

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("lhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("rhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id, rhs_ids_test);
        let scores = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("score")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(scores, scores_test);

        // Case 4: LHS and RHS messages from multiple stream batch (accumulate LHS and Stream RHS)
        // Make the config
        let config = DataConfig {
            lhs_name: Some("lhs_name".to_string()),
            rhs_name: Some("rhs_name".to_string()),
            lhs_pk: Some("lhs_pk".to_string()),
            lhs_fk: Some("lhs_fk".to_string()),
            lhs_values: Some(vec!["embedding".to_string()]),
            rhs_pk: Some("rhs_pk".to_string()),
            rhs_fk: Some("rhs_fk".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
            operator: AvailableCandleOperators::VectorDistance,
            stream: DataStreamManager::AccumulateLHSStreamRHS,
            ..Default::default()
        };
        let config_table = Table::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_table.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        // Expected values (for RHS streaming)
        let lhs_ids_test = vec!["1", "1", "2", "2", "3", "3", "1", "1", "2", "2", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "1", "2", "1", "2", "3", "4", "3", "4", "3", "4"];
        let scores_test: Vec<f32> = vec![
            1.0, 1.0, 0.70710677, 0.70710677, 0.5, 0.5, 1.0, 1.0, 0.70710677, 0.70710677, 0.5, 0.5,
        ];

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("lhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("rhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id, rhs_ids_test);
        let scores = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("score")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(scores, scores_test);

        // Case 5: LHS and RHS messages from multiple stream batch (Stream LHS and Stream RHS)
        // Make the config
        let config = DataConfig {
            lhs_name: Some("lhs_name".to_string()),
            rhs_name: Some("rhs_name".to_string()),
            lhs_pk: Some("lhs_pk".to_string()),
            lhs_fk: Some("lhs_fk".to_string()),
            lhs_values: Some(vec!["embedding".to_string()]),
            rhs_pk: Some("rhs_pk".to_string()),
            rhs_fk: Some("rhs_fk".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
            operator: AvailableCandleOperators::VectorDistance,
            stream: DataStreamManager::StreamLHSStreamRHS,
            ..Default::default()
        };
        let config_table = Table::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_table.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        // Expected values
        let lhs_ids_test = vec!["1", "1", "2", "2", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "3", "4", "3", "4"];
        let scores_test: Vec<f32> = vec![1.0, 1.0, 0.70710677, 0.70710677, 0.5, 0.5];

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("lhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("rhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id, rhs_ids_test);
        let scores = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("score")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(scores, scores_test);

        // Case 6: LHS and RHS messages from multiple stream batch (Stream LHS and Accumulate RHS)
        // Make the config
        let config = DataConfig {
            lhs_name: Some("lhs_name".to_string()),
            rhs_name: Some("rhs_name".to_string()),
            lhs_pk: Some("lhs_pk".to_string()),
            lhs_fk: Some("lhs_fk".to_string()),
            lhs_values: Some(vec!["embedding".to_string()]),
            rhs_pk: Some("rhs_pk".to_string()),
            rhs_fk: Some("rhs_fk".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
            operator: AvailableCandleOperators::VectorDistance,
            stream: DataStreamManager::StreamLHSAccumulateRHS,
            ..Default::default()
        };
        let config_table = Table::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_table.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        // Expected values
        let lhs_ids_test = vec!["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "3", "4", "1", "2", "3", "4", "1", "2", "3", "4"];
        let scores_test: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 0.70710677, 0.70710677, 0.70710677, 0.70710677, 0.5, 0.5, 0.5, 0.5,
        ];

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("lhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("rhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id, rhs_ids_test);
        let scores = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("score")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(scores, scores_test);

        Ok(())
    }

    #[tokio::test]
    async fn test_candle_ops_processor() -> Result<()> {
        // LHS and RHS messages
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "lhs_pk",
            lhs_ids_vec,
            lhs_embeddings_vec,
        )?;
        let lhs_table = Table::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("lhs_name")
            .with_publisher("")
            .with_subject("lhs_name")
            .with_update(&TablePublication::None)
            .with_message(lhs_table.to_record_batch_stream())
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);
        let rhs_ids_vec = vec!["1", "2", "3", "4"];
        let rhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let rhs_batch = test_candle_ops_processor::make_embeddings_record_batch_str_f32(
            "rhs_pk",
            rhs_ids_vec,
            rhs_embeddings_vec,
        )?;
        let rhs_table = Table::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch])?
            .build()?;
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("rhs_name")
            .with_publisher("")
            .with_subject("rhs_name")
            .with_update(&TablePublication::None)
            .with_message(rhs_table.to_record_batch_stream())
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);

        // Make the config
        let config = DataConfig {
            lhs_name: Some("lhs_name".to_string()),
            rhs_name: Some("rhs_name".to_string()),
            lhs_pk: Some("lhs_pk".to_string()),
            lhs_fk: Some("lhs_fk".to_string()),
            lhs_values: Some(vec!["embedding".to_string()]),
            rhs_pk: Some("rhs_pk".to_string()),
            rhs_fk: Some("rhs_fk".to_string()),
            rhs_values: Some(vec!["embedding".to_string()]),
            dist_operator: Some(DataDistanceOperator::NormalizedDotProduct),
            operator: AvailableCandleOperators::VectorDistance,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = TableBuilder::new()
            .with_name("candle_ops_processor")
            .with_json(&config_json, 1)?
            .build()?;
        let message = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher("")
            .with_subject("candle_ops_processor")
            .with_update(&TablePublication::None)
            .with_message(config_table.to_record_batch_stream())
            .make_random_name()?
            .build()?;
        let _ = messages.insert(message.get_name().to_string(), message);

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let runtime_env = RuntimeEnv {
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        };
        let runtime_env = Arc::new(Mutex::new(runtime_env));

        // Make the stream and run
        let ops_processor = CandleDataProcessor::new(
            "candle_ops_processor",
            "",
            &[TablePublication::Replace {
                table_name: "results".to_string(),
            }],
            &[
                TableSubscription::AlwaysFullTable {
                    table_name: "lhs_name".to_string(),
                },
                TableSubscription::AlwaysFullTable {
                    table_name: "rhs_name".to_string(),
                },
            ],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut ops_stream =
            ops_processor.process(messages, Some(&diagnostic_builder), runtime_env)?;
        let result = ops_stream
            .remove("from_candle_ops_processor_on_results")
            .unwrap()
            .get_message_own()
            .try_collect::<Vec<_>>()
            .await?;

        // Expected values
        let lhs_ids_test = vec!["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "3"];
        let rhs_ids_test = vec!["1", "2", "3", "4", "1", "2", "3", "4", "1", "2", "3", "4"];
        let scores_test: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, 0.70710677, 0.70710677, 0.70710677, 0.70710677, 0.5, 0.5, 0.5, 0.5,
        ];

        let lhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("lhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(lhs_id, lhs_ids_test);
        let rhs_id = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("rhs_pk")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap_or_default())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rhs_id, rhs_ids_test);
        let scores = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("score")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .map(|s| s.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(scores, scores_test);

        Ok(())
    }
}
