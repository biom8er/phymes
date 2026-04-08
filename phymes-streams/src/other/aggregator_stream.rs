use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Fields, SchemaRef},
};
use futures::{Stream, StreamExt};
use phymes_core::{BuildableTrait, BuilderTrait, MappableTrait, RecordBatchStream, RuntimeEnv,
    SendableRecordBatchStream, Subject, SubjectBuilder, SubjectBuilderTrait,
};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_schemas::{AvailableSchemaTrait, AvailableSubjects};
use phymes_data::{DataConfig, DataConfigTrait, DataOperatorTrait, device};
use phymes_ml::{CandleTensorService, TensorStreamTrait};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, EventBuilderTrait, HashMap, MetricBuilderTrait,
};
use tracing::instrument;

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
    tensor_service: Option<Box<dyn TensorStreamTrait>>,
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