use anyhow::{Result, anyhow};
use arrow::{
    datatypes::{Schema, SchemaRef},
    record_batch::RecordBatch,
};
use futures::{Stream, StreamExt};
use phymes_subject::{
    BuilderTrait, MappableTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    SubjectBuilder, SubjectBuilderTrait, SubjectTrait,
};
use phymes_data::{DataConfig, DataConfigTrait, DataOperatorTrait, DataStreamManager, device};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, EventBuilderTrait, MetricBuilderTrait,
};
use phymes_message::{
    MessageTrait, SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_ml::{CandleTensorService, TensorStreamTrait};
use phymes_schemas::{create_bytes_fields, create_values_fields};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
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
    tensor_service: Option<Box<dyn TensorStreamTrait>>,
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
