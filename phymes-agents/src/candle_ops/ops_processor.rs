use crate::{candle_assets::device::device, candle_ops::ops_functions::join_inner};

use super::{
    ops_config::CandleOpsConfig,
    ops_functions::{chunk_documents, relative_similarity_scores, sort_scores_and_indices},
    ops_service::CandleOpsService,
    ops_which::WhichCandleOps,
};

use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics, HashMap},
    session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap},
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table::{ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait},
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::ArrowTableSubscribe,
        stream::{RecordBatchStream, SendableRecordBatchStream},
    },
    task::{
        arrow_message::{
            ArrowMessageBuilderTrait, ArrowOutgoingMessage, ArrowOutgoingMessageBuilderTrait,
            ArrowOutgoingMessageTrait,
        },
        arrow_processor::ArrowProcessorTrait,
    },
};

use arrow::{
    array::StringArray,
    datatypes::{DataType, Field, Fields, Schema, SchemaRef},
    record_batch::RecordBatch,
};

use anyhow::{Result, anyhow};
use futures::{Stream, StreamExt};
use parking_lot::{Mutex, RwLock};
use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};
use tracing::{Level, event};

/// Tensor processor made possible by Candle
///
/// Each operator has a defined input and output schema
/// that calling processors or consuming processors
/// need to adhere to
#[derive(Default, Debug)]
pub struct CandleOpProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    forward: Vec<String>,
}

impl CandleOpProcessor {
    pub fn new_with_pub_sub_for(
        name: &str,
        publications: &[ArrowTablePublish],
        subscriptions: &[ArrowTableSubscribe],
        forward: &[&str],
    ) -> Arc<dyn ArrowProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            forward: forward.iter().map(|s| s.to_string()).collect(),
        })
    }
}

impl MappableTrait for CandleOpProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ArrowProcessorTrait for CandleOpProcessor {
    fn new_arc(name: &str) -> Arc<dyn ArrowProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: vec![ArrowTablePublish::None],
            subscriptions: vec![ArrowTableSubscribe::None],
            forward: Vec::new(),
        })
    }

    fn get_publications(&self) -> &[ArrowTablePublish] {
        &self.publications
    }

    fn get_subscriptions(&self) -> &[ArrowTableSubscribe] {
        &self.subscriptions
    }

    fn get_forward_subscriptions(&self) -> &[String] {
        self.forward.as_slice()
    }

    fn process(
        &self,
        mut message: OutgoingMessageMap,
        metrics: ArrowTaskMetricsSet,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<OutgoingMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());

        // Extract out the config
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Make the outbox and move forwarded messages
        let mut outbox = HashMap::<String, ArrowOutgoingMessage>::new();
        for f in self.forward.iter() {
            if let Some(m) = message.remove(f) {
                let _ = outbox.insert(f.to_string(), m);
            }
        }

        // Run the ops
        let out = Box::pin(CandleOpStream::new(
            message,
            config,
            Arc::clone(&runtime_env),
            BaselineMetrics::new(&metrics, self.get_name()),
        )?);
        let out_m = ArrowOutgoingMessage::get_builder()
            .with_name(self.publications.first().unwrap().get_table_name())
            .with_publisher(self.get_name())
            .with_subject(self.publications.first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.publications.first().unwrap())
            .build()?;
        let _ = outbox.insert(out_m.get_name().to_string(), out_m);
        Ok(outbox)
    }
}

/// Compute the relative similarity score between two embeddings
#[allow(dead_code)]
pub struct CandleOpStream {
    /// The messages containing the lhs and rhs
    /// which we cannot determine until we intialize the config
    messages: OutgoingMessageMap,
    /// Parameters for tensor operations
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    baseline_metrics: BaselineMetrics,
    /// Parameters for tensor operations
    config: Option<CandleOpsConfig>,
    /// The polled record batches from the input
    lhs_inbox: Vec<RecordBatch>,
    /// The polled record batches from the input
    rhs_inbox: Vec<RecordBatch>,
    /// The prepared record batches for the output
    outbox: Vec<RecordBatch>,
}

impl CandleOpStream {
    pub fn new(
        messages: OutgoingMessageMap,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        baseline_metrics: BaselineMetrics,
    ) -> Result<Self> {
        Ok(Self {
            messages,
            config_stream,
            baseline_metrics,
            runtime_env,
            config: None,
            lhs_inbox: Vec::new(),
            rhs_inbox: Vec::new(),
            outbox: Vec::new(),
        })
    }

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
                let service = CandleOpsService::new(device);
                let _ = self
                    .runtime_env
                    .try_lock()
                    .unwrap()
                    .tensor_service
                    .replace(Arc::new(RwLock::new(service)));
            }
        } else {
            return Err(anyhow!(
                "The config for Ops processor needs to be initialized before trying to initialize the tensor service."
            ));
        }
        Ok(())
    }
}

impl Stream for CandleOpStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // 1. Accumulate all of the LHS queries which we assume to be less than the document chunks
        // 2. Poll each document chunk one by one which we assume to already be sized to fit in memory
        // 3. Compute the similarity score and send the stream

        // Initialize the metrics
        let metrics = self.baseline_metrics.clone();
        let _timer = metrics.elapsed_compute().timer();

        // Intialize the config
        event!(Level::DEBUG, "Initializing OpsProcessor config.");
        if self.config.is_none() {
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let values = Fields::from_iter(vec![Field::new("values", DataType::Utf8, false)]);
            if batches.first().unwrap().schema().fields().contains(&values) {
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
                config_values["arguments"]["which"] = config_values["name"].clone();
                let config: CandleOpsConfig =
                    serde_json::from_value(config_values.get("arguments").unwrap().clone())?;
                self.config.replace(config);
            } else {
                let config_table = ArrowTableBuilder::new()
                    .with_name("config")
                    .with_record_batches(batches)?
                    .build()?;
                let config: CandleOpsConfig = serde_json::from_value(serde_json::Value::Object(
                    config_table.to_json_object()?.first().unwrap().to_owned(),
                ))?;
                self.config.replace(config);
            }
        }

        // Collect the LHS queries
        event!(Level::DEBUG, "Collecting OpsProcessor LHS.");
        if self.lhs_inbox.is_empty() {
            let lhs_name = self.config.as_ref().unwrap().lhs_name.clone();
            let lhs = match self.messages.get_mut(lhs_name.as_str()) {
                Some(lhs) => {
                    // Poll the input streams and collect/transform the batches
                    let mut batches = Vec::new();
                    while let Some(Ok(batch)) = ready!(lhs.get_message_mut().poll_next_unpin(cx)) {
                        batches.push(batch);
                    }
                    batches
                }
                None => {
                    // Extract the input from the config
                    match self.config.as_ref().unwrap().lhs_args.as_ref() {
                        Some(qs) => {
                            let table = ArrowTableBuilder::new()
                                .with_json(qs.as_bytes(), 512)?
                                .with_name("")
                                .build()?;
                            table.get_record_batches_own()
                        }
                        None => return Poll::Ready(None),
                    }
                }
            };
            // Check that the schema is correct
            if let Err(e) = self.config.as_ref().unwrap().which.check_schema_lhs_input(
                self.config.as_ref().unwrap().lhs_pk.as_str(),
                self.config.as_ref().unwrap().lhs_fk.as_str(),
                self.config.as_ref().unwrap().lhs_values.as_str(),
                lhs.first().unwrap().schema(),
            ) {
                panic!("Error: {e:?}")
            }
            self.lhs_inbox = lhs;
        };

        // Collect the RHS document chunks
        event!(Level::DEBUG, "Collecting OpsProcessor RHS.");
        if self.rhs_inbox.is_empty() && self.config.as_ref().unwrap().rhs_name.is_some() {
            let rhs_name = self
                .config
                .as_ref()
                .unwrap()
                .rhs_name
                .as_ref()
                .unwrap()
                .clone();
            let rhs = match self.messages.get_mut(rhs_name.as_str()) {
                Some(rhs) => {
                    // Poll the input streams and collect/transform the batches
                    match ready!(rhs.get_message_mut().poll_next_unpin(cx)) {
                        Some(Ok(batch)) => vec![batch],
                        _ => return Poll::Ready(None),
                    }
                }
                None => {
                    // Extract the input from the config
                    match self.config.as_ref().unwrap().rhs_args.as_ref() {
                        Some(qs) => {
                            let table = ArrowTableBuilder::new()
                                .with_json(qs.as_bytes(), 512)?
                                .with_name("")
                                .build()?;
                            table.get_record_batches_own()
                        }
                        None => return Poll::Ready(None),
                    }
                }
            };
            // Check that the schema is correct
            if let Err(e) = self.config.as_ref().unwrap().which.check_schema_rhs_input(
                self.config
                    .as_ref()
                    .unwrap()
                    .rhs_pk
                    .as_ref()
                    .unwrap()
                    .as_str(),
                self.config
                    .as_ref()
                    .unwrap()
                    .rhs_fk
                    .as_ref()
                    .unwrap()
                    .as_str(),
                self.config
                    .as_ref()
                    .unwrap()
                    .rhs_values
                    .as_ref()
                    .unwrap()
                    .as_str(),
                rhs.first().unwrap().schema(),
            ) {
                panic!("Error: {e:?}")
            }
            self.rhs_inbox = rhs;
        }

        // Compute the relative similary scores
        event!(
            Level::DEBUG,
            "Executing Ops {}.",
            self.config.as_ref().unwrap().which.get_name()
        );
        self.init_tensor_service()?;
        let batch = match self.config.as_ref().unwrap().which {
            WhichCandleOps::RelativeSimilarityScore => relative_similarity_scores(
                &self.lhs_inbox,
                &self.rhs_inbox,
                &self.config.as_ref().unwrap().lhs_pk,
                &self.config.as_ref().unwrap().lhs_values,
                self.config
                    .as_ref()
                    .unwrap()
                    .rhs_pk
                    .as_ref()
                    .unwrap()
                    .as_str(),
                self.config
                    .as_ref()
                    .unwrap()
                    .rhs_values
                    .as_ref()
                    .unwrap()
                    .as_str(),
                self.runtime_env
                    .try_lock()
                    .unwrap()
                    .tensor_service
                    .as_ref()
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_device(),
            )?,
            WhichCandleOps::SortScoresAndIndices => sort_scores_and_indices(
                &self.lhs_inbox,
                false,
                self.runtime_env
                    .try_lock()
                    .unwrap()
                    .tensor_service
                    .as_ref()
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_device(),
            )?,
            WhichCandleOps::HumanInTheLoops => self.lhs_inbox.first().unwrap().to_owned(),
            WhichCandleOps::ChunkDocuments => chunk_documents(
                &self.lhs_inbox,
                &self.config.as_ref().unwrap().lhs_pk,
                &self.config.as_ref().unwrap().lhs_values,
                512, //DM: need to add to op_kwargs
                64,  //DM: need to add to op_kwargs
                self.runtime_env
                    .try_lock()
                    .unwrap()
                    .tensor_service
                    .as_ref()
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_device(),
            )?,
            WhichCandleOps::JoinInner => join_inner(
                &self.lhs_inbox,
                &self.config.as_ref().unwrap().lhs_fk,
                &self.rhs_inbox,
                self.config
                    .as_ref()
                    .unwrap()
                    .rhs_fk
                    .as_ref()
                    .unwrap()
                    .as_str(),
                self.runtime_env
                    .try_lock()
                    .unwrap()
                    .tensor_service
                    .as_ref()
                    .unwrap()
                    .try_read()
                    .unwrap()
                    .get_device(),
            )?,
        };
        // DM: need to update this with the other config options for streaming
        self.rhs_inbox.clear();
        if self.config.as_ref().unwrap().rhs_name.is_none() {
            self.config
                .as_mut()
                .unwrap()
                .rhs_name
                .replace("".to_string());
        }

        // record the poll
        // println!("Record the poll...");
        let poll = Poll::Ready(Some(Ok(batch)));
        metrics.record_poll(poll)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, None)
    }
}

impl RecordBatchStream for CandleOpStream {
    fn schema(&self) -> SchemaRef {
        match self.config.as_ref() {
            Some(config) => config
                .which
                .get_schema_output(
                    config.lhs_pk.as_str(),
                    config.lhs_fk.as_str(),
                    config.lhs_values.as_str(),
                    config.rhs_pk.as_ref().map_or("", |v| v),
                    config.rhs_fk.as_ref().map_or("", |v| v),
                    config.rhs_values.as_ref().map_or("", |v| v),
                    None,
                    None,
                )
                .unwrap(),
            None => Arc::new(Schema::empty()),
        }
    }
}

pub mod test_candle_ops_processor {
    use std::sync::Arc;

    use anyhow::Result;
    use arrow::{
        array::{ArrayData, ArrayRef, FixedSizeListArray, RecordBatch, StringArray},
        buffer::Buffer,
        datatypes::{DataType, Field},
    };

    pub fn make_embeddings_record_batch(
        id_str: &str,
        ids: Vec<&str>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<RecordBatch> {
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
        let embedding: ArrayRef = Arc::new(FixedSizeListArray::from(list_data));

        // Make the batch
        let ids_ar: ArrayRef = Arc::new(StringArray::from(ids));
        let batch = RecordBatch::try_from_iter(vec![(id_str, ids_ar), ("embeddings", embedding)])?;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use crate::candle_ops::ops_which::WhichCandleOps;
    use arrow::array::Float32Array;
    use futures::TryStreamExt;
    use phymes_core::table::{arrow_table::ArrowTable, arrow_table_publish::ArrowTablePublish};

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
        let lhs_batch = test_candle_ops_processor::make_embeddings_record_batch(
            "lhs_pk",
            lhs_ids_vec,
            lhs_embeddings_vec,
        )?;
        let lhs_table = ArrowTable::get_builder()
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
        let rhs_batch = test_candle_ops_processor::make_embeddings_record_batch(
            "rhs_pk",
            rhs_ids_vec,
            rhs_embeddings_vec,
        )?;
        let rhs_table = ArrowTable::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch])?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&ArrowTablePublish::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&ArrowTablePublish::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the config
        let config = CandleOpsConfig {
            lhs_name: "lhs_name".to_string(),
            rhs_name: Some("rhs_name".to_string()),
            lhs_pk: "lhs_pk".to_string(),
            lhs_fk: "lhs_fk".to_string(),
            lhs_values: "embeddings".to_string(),
            rhs_pk: Some("rhs_pk".to_string()),
            rhs_fk: Some("rhs_fk".to_string()),
            rhs_values: Some("embeddings".to_string()),
            which: WhichCandleOps::RelativeSimilarityScore,
            ..Default::default()
        };
        let config_table = ArrowTable::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config)?, 1)?
            .build()?;

        // Make the metrics
        let metrics = ArrowTaskMetricsSet::new();
        let baseline_metrics = BaselineMetrics::new(&metrics.clone(), "candle_ops_processor");

        // Make the runtime environment
        let device = device(config.cpu)?;
        let service = CandleOpsService::new(device);
        let runtime_env = RuntimeEnv {
            token_service: None,
            tensor_service: Some(Arc::new(RwLock::new(service))),
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        };
        let runtime_env = Arc::new(Mutex::new(runtime_env));

        // Make the stream and run
        let ops_stream = CandleOpStream::new(
            messages,
            config_table.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            baseline_metrics,
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
        let baseline_metrics = BaselineMetrics::new(&metrics.clone(), "candle_ops_processor");

        // Make the config
        let config_args = CandleOpsConfig {
            which: WhichCandleOps::HumanInTheLoops,
            lhs_args: Some("{\"role\": \"assistant\", \"content\": \"RESPONSE\"}".to_string()),
            rhs_args: None,
            ..Default::default()
        };
        let config_args_table = ArrowTable::get_builder()
            .with_name("candle_embed_processor")
            .with_json(&serde_json::to_vec(&config_args)?, 1)?
            .build()?;

        // Make the stream and run
        let ops_stream = CandleOpStream::new(
            HashMap::<String, ArrowOutgoingMessage>::new(),
            config_args_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            baseline_metrics,
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

        // Case 3: LHS and RHS messages from multiple stream batch
        println!("Case 3: LHS and RHS messages from multiple stream batch");
        let lhs_ids_vec_1 = vec!["1"];
        let lhs_embeddings_vec_1: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.]];
        let lhs_batch_1 = test_candle_ops_processor::make_embeddings_record_batch(
            "lhs_pk",
            lhs_ids_vec_1,
            lhs_embeddings_vec_1,
        )?;
        let lhs_ids_vec_2 = vec!["2", "3"];
        let lhs_embeddings_vec_2: Vec<Vec<f32>> = vec![vec![0., 1., 0., 1.], vec![0., 0., 0., 1.]];
        let lhs_batch_2 = test_candle_ops_processor::make_embeddings_record_batch(
            "lhs_pk",
            lhs_ids_vec_2,
            lhs_embeddings_vec_2,
        )?;
        let lhs_table = ArrowTable::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch_1, lhs_batch_2])?
            .build()?;
        let rhs_ids_vec_1 = vec!["1", "2"];
        let rhs_embeddings_vec_1: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.], vec![1., 1., 1., 1.]];
        let rhs_batch_1 = test_candle_ops_processor::make_embeddings_record_batch(
            "rhs_pk",
            rhs_ids_vec_1,
            rhs_embeddings_vec_1,
        )?;
        let rhs_ids_vec_2 = vec!["3", "4"];
        let rhs_embeddings_vec_2: Vec<Vec<f32>> = vec![vec![1., 1., 1., 1.], vec![1., 1., 1., 1.]];
        let rhs_batch_2 = test_candle_ops_processor::make_embeddings_record_batch(
            "rhs_pk",
            rhs_ids_vec_2,
            rhs_embeddings_vec_2,
        )?;
        let rhs_table = ArrowTable::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch_1, rhs_batch_2])?
            .build()?;

        // Make the input message
        let mut messages = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = messages.insert(
            lhs_table.get_name().to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name(lhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&ArrowTablePublish::None)
                .with_message(lhs_table.clone().to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            rhs_table.get_name().to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name(rhs_table.get_name())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&ArrowTablePublish::None)
                .with_message(rhs_table.clone().to_record_batch_stream())
                .build()?,
        );

        // Make the config and metrics
        let baseline_metrics = BaselineMetrics::new(&metrics.clone(), "candle_ops_processor");

        // Make the stream and run
        let ops_stream = CandleOpStream::new(
            messages,
            config_table.clone().to_record_batch_stream(),
            Arc::clone(&runtime_env),
            baseline_metrics,
        )?;
        let result = ops_stream.try_collect::<Vec<_>>().await?;

        // Expected values
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

        Ok(())
    }

    #[tokio::test]
    async fn test_candle_ops_processor() -> Result<()> {
        // LHS and RHS messages
        let mut messages = HashMap::<String, ArrowOutgoingMessage>::new();
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch = test_candle_ops_processor::make_embeddings_record_batch(
            "lhs_pk",
            lhs_ids_vec,
            lhs_embeddings_vec,
        )?;
        let lhs_table = ArrowTable::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;
        let _ = messages.insert(
            "lhs_name".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("lhs_name")
                .with_publisher("")
                .with_subject("")
                .with_update(&ArrowTablePublish::None)
                .with_message(lhs_table.to_record_batch_stream())
                .build()?,
        );
        let rhs_ids_vec = vec!["1", "2", "3", "4"];
        let rhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
            vec![1., 1., 1., 1.],
        ];
        let rhs_batch = test_candle_ops_processor::make_embeddings_record_batch(
            "rhs_pk",
            rhs_ids_vec,
            rhs_embeddings_vec,
        )?;
        let rhs_table = ArrowTable::get_builder()
            .with_name("rhs_name")
            .with_record_batches(vec![rhs_batch])?
            .build()?;
        let _ = messages.insert(
            "rhs_name".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("rhs_name")
                .with_publisher("")
                .with_subject("")
                .with_update(&ArrowTablePublish::None)
                .with_message(rhs_table.to_record_batch_stream())
                .build()?,
        );

        // Make the config
        let config = CandleOpsConfig {
            lhs_name: "lhs_name".to_string(),
            rhs_name: Some("rhs_name".to_string()),
            lhs_pk: "lhs_pk".to_string(),
            lhs_fk: "lhs_fk".to_string(),
            lhs_values: "embeddings".to_string(),
            rhs_pk: Some("rhs_pk".to_string()),
            rhs_fk: Some("rhs_fk".to_string()),
            rhs_values: Some("embeddings".to_string()),
            which: WhichCandleOps::RelativeSimilarityScore,
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = ArrowTableBuilder::new()
            .with_name("candle_ops_processor")
            .with_json(&config_json, 1)?
            .build()?;
        let _ = messages.insert(
            "candle_ops_processor".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("candle_ops_processor")
                .with_publisher("")
                .with_subject("")
                .with_update(&ArrowTablePublish::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let metrics = ArrowTaskMetricsSet::new();

        // Make the runtime environment
        let device = device(config.cpu)?;
        let service = CandleOpsService::new(device);
        let runtime_env = RuntimeEnv {
            token_service: None,
            tensor_service: Some(Arc::new(RwLock::new(service))),
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        };
        let runtime_env = Arc::new(Mutex::new(runtime_env));

        // Make the stream and run
        let ops_processor = CandleOpProcessor::new_with_pub_sub_for(
            "candle_ops_processor",
            &[ArrowTablePublish::Replace {
                table_name: "results".to_string(),
            }],
            &[],
            &[],
        );
        let mut ops_stream = ops_processor.process(messages, metrics, runtime_env)?;
        let result = ops_stream
            .remove("results")
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
