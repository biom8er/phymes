use anyhow::{Result, anyhow};
use arrow::{
    datatypes::{Schema, SchemaRef},
    record_batch::RecordBatch,
};
use futures::{Stream, StreamExt};
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait,
    RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, SubjectBuilder, SubjectBuilderTrait, SubjectTrait
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, EventBuilderTrait, HashMap, MetricBuilderTrait,
};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_processor::ProcessorTrait;
use phymes_schemas::{create_bytes_fields, create_values_fields};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tracing::{Level, event, instrument};

use crate::{
    CandleTensorService, DataConfig, DataConfigTrait, DataOperatorTrait, DataStreamManager,
    TensorProcessorTrait, device,
};

/// Data operator stream
pub struct CandleDataStream {
    /// The messages containing the lhs and rhs
    /// which we cannot determine until we intialize the config
    messages: SendableRecordBatchStreamMessageMap,
    /// Parameters for tensor operations
    config_stream: SendableRecordBatchStream,
    /// The tensor services needed for inference
    _runtime_env: Arc<RuntimeEnv>,
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
    /// Switch to finished polling
    is_finished: bool,
}

impl CandleDataStream {
    pub fn new(
        messages: SendableRecordBatchStreamMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            messages,
            config_stream,
            diagnostic_builder,
            _runtime_env: runtime_env,
            config: None,
            tensor_service: None,
            data_operator: None,
            lhs_inbox: Vec::new(),
            rhs_inbox: Vec::new(),
            is_finished: false,
        })
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
            let config_subject = SubjectBuilder::new()
                .with_name("CandleDataStream Config")
                .with_record_batches(batches)?
                .build()?;
            if config_subject
                .get_schema()
                .fields()
                .contains(&create_values_fields())
            {
                let config_json = config_subject.get_column_as_vec_str("values").join("");
                let config = serde_json::from_str::<DataConfig>(&config_json)?;
                self.config.replace(config);
            } else if config_subject
                .get_schema()
                .fields()
                .contains(&create_bytes_fields())
            {
                let config_json = config_subject
                    .get_column_as_vec_nested_primitive::<u8>("bytes")?
                    .into_iter()
                    .map(|b| String::from_utf8(b).unwrap())
                    .collect::<Vec<_>>()
                    .join("");
                let config = serde_json::from_str::<DataConfig>(&config_json)?;
                self.config.replace(config);
            } else {
                let config = DataConfig::from_table(&config_subject)?;
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
        let lhs_stream = self.config.as_ref().unwrap().lhs_stream.clone();
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
            let lhs = match remove_message_by_subject(lhs_name.as_str(), &mut self.messages) {
                Some(mut lhs) => match lhs_stream {
                    DataStreamManager::Accumulate => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) =
                            ready!(lhs.get_message_mut().poll_next_unpin(cx))
                        {
                            batches.push(batch);
                        }
                        self.messages.insert(lhs.get_name().to_string(), lhs);
                        batches
                    }
                    DataStreamManager::Stream => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) =
                            ready!(lhs.get_message_mut().poll_next_unpin(cx))
                        {
                            if batch.num_rows() > 0 && batch.num_columns() > 0 {
                                batches.push(batch);
                                break;
                            }
                        }
                        self.messages.insert(lhs.get_name().to_string(), lhs);
                        batches
                    }
                },
                // Check for the LHS in the config (accumulation only)
                None => match lhs_stream {
                    DataStreamManager::Accumulate => {
                        // Extract the input from the config
                        match self.config.as_ref().unwrap().lhs_args.as_ref() {
                            Some(qs) => {
                                let table = SubjectBuilder::new()
                                    .with_json(qs.as_bytes(), 512)?
                                    .with_name("")
                                    .build()?;
                                table.get_record_batches_own()
                            }
                            None => {
                                self.is_finished = true;
                                return Poll::Ready(Some(Err(anyhow!(
                                    "lhs_name `{lhs_name}` does not exist. Available options are {:?}",
                                    self.messages.keys()
                                ))));
                            }
                        }
                    }
                    DataStreamManager::Stream => {
                        self.is_finished = true;
                        return Poll::Ready(Some(Err(anyhow!(
                            "lhs_name `{lhs_name}` does not exist. Available options are {:?}",
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
        let rhs_stream = self
            .config
            .as_ref()
            .unwrap()
            .rhs_stream
            .clone()
            .unwrap_or_default();
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
            let rhs = match remove_message_by_subject(rhs_name.as_str(), &mut self.messages) {
                Some(mut rhs) => match rhs_stream {
                    DataStreamManager::Accumulate => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) =
                            ready!(rhs.get_message_mut().poll_next_unpin(cx))
                        {
                            batches.push(batch);
                        }
                        self.messages.insert(rhs.get_name().to_string(), rhs);
                        batches
                    }
                    DataStreamManager::Stream => {
                        let mut batches = Vec::new();
                        while let Some(Ok(batch)) =
                            ready!(rhs.get_message_mut().poll_next_unpin(cx))
                        {
                            if batch.num_rows() > 0 && batch.num_columns() > 0 {
                                batches.push(batch);
                                break;
                            }
                        }
                        self.messages.insert(rhs.get_name().to_string(), rhs);
                        batches
                    }
                },
                // Check for the RHS in the config (accumulation only)
                None => match rhs_stream {
                    DataStreamManager::Accumulate => {
                        // Extract the input from the config
                        match self.config.as_ref().unwrap().rhs_args.as_ref() {
                            Some(qs) => {
                                let table = SubjectBuilder::new()
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
                    DataStreamManager::Stream => {
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
            self.tensor_service.as_ref().unwrap().get_device(),
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
        match (lhs_stream, rhs_stream) {
            (DataStreamManager::Accumulate, DataStreamManager::Accumulate) => {
                self.is_finished = true;
                let poll = Poll::Ready(Some(Ok(batch)));
                if let Some(baseline_metrics) = &baseline_metrics {
                    baseline_metrics.record_poll(poll)
                } else {
                    poll
                }
            }
            (DataStreamManager::Accumulate, DataStreamManager::Stream) => {
                self.rhs_inbox.clear();
                let poll = Poll::Ready(Some(Ok(batch)));
                if let Some(baseline_metrics) = &baseline_metrics {
                    baseline_metrics.record_poll(poll)
                } else {
                    poll
                }
            }
            (DataStreamManager::Stream, DataStreamManager::Accumulate) => {
                self.lhs_inbox.clear();
                let poll = Poll::Ready(Some(Ok(batch)));
                if let Some(baseline_metrics) = &baseline_metrics {
                    baseline_metrics.record_poll(poll)
                } else {
                    poll
                }
            }
            (DataStreamManager::Stream, DataStreamManager::Stream) => {
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
    use crate::{DataDistanceOperator, candle_operators::AvailableOperators};
    use arrow::array::{Float32Array, StringArray};
    use futures::TryStreamExt;
    use phymes_core::Subject;
    use phymes_diagnostics::{Diagnostics, SpanBuilder};
    use phymes_event::Publication;

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
        let lhs_table = Subject::get_builder()
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
        let rhs_table = Subject::get_builder()
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
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
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
            operator: AvailableOperators::VectorDistance,
            ..Default::default()
        };
        let config_subject = Subject::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the metrics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the runtime environment
        let runtime_env = RuntimeEnv::get_builder().with_name("rt").build()?;
        let runtime_env = Arc::new(runtime_env);

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
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
            operator: AvailableOperators::HumanInTheLoop,
            lhs_name: Some("".to_string()),
            lhs_args: Some("{\"role\": \"assistant\", \"content\": \"RESPONSE\"}".to_string()),
            rhs_args: None,
            ..Default::default()
        };
        let config_args_table = Subject::get_builder()
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
        let lhs_table = Subject::get_builder()
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
        let rhs_table = Subject::get_builder()
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
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
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
            operator: AvailableOperators::VectorDistance,
            lhs_stream: DataStreamManager::Accumulate,
            rhs_stream: Some(DataStreamManager::Stream),
            ..Default::default()
        };
        let config_subject = Subject::get_builder()
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
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
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
            operator: AvailableOperators::VectorDistance,
            lhs_stream: DataStreamManager::Stream,
            rhs_stream: Some(DataStreamManager::Stream),
            ..Default::default()
        };
        let config_subject = Subject::get_builder()
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
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
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
            operator: AvailableOperators::VectorDistance,
            lhs_stream: DataStreamManager::Stream,
            rhs_stream: Some(DataStreamManager::Accumulate),
            ..Default::default()
        };
        let config_subject = Subject::get_builder()
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
                .with_subject(lhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject(rhs_table.get_name())
                .with_update(&Publication::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the stream and run
        let ops_stream = CandleDataStream::new(
            messages,
            config_subject.clone().to_record_batch_stream(),
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
}
