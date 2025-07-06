use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics, HashMap},
    session::{
        common_traits::{BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap},
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
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

use anyhow::{Result, anyhow};
use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use tracing::{Level, event};

use super::summary_config::CandleOpsSummaryConfig;

/// Processor that takes the results of an OpsProcessor
///   and creates a summary of the result for chat inference
///
/// # Notes
///
/// - The default role is `tool`
#[derive(Default, Debug)]
pub struct OpsSummaryProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    forward: Vec<String>,
}

impl OpsSummaryProcessor {
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

impl MappableTrait for OpsSummaryProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ArrowProcessorTrait for OpsSummaryProcessor {
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

        // Extract out the messages to be summarized
        // DM: we need to assume that there is only one message left to summarize
        assert_eq!(message.len(), 1);
        let messages = match message.into_iter().next() {
            Some((_k, v)) => v.get_message_own(),
            None => return Err(anyhow!("Messages not provided for {}.", self.get_name())),
        };

        // Make the outbox and send
        let out = Box::pin(OpsSummaryStream::new(
            messages,
            config,
            Arc::clone(&runtime_env),
            BaselineMetrics::new(&metrics, self.get_name()),
        )?);
        let mut outbox = HashMap::<String, ArrowOutgoingMessage>::new();
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

#[allow(dead_code)]
pub struct OpsSummaryStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    message_stream: SendableRecordBatchStream,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    baseline_metrics: BaselineMetrics,
    /// Parameters for chat inference
    config: Option<CandleOpsSummaryConfig>,
}

impl OpsSummaryStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        baseline_metrics: BaselineMetrics,
    ) -> Result<Self> {
        // Output schema
        let field_names = ["role", "content"];
        let fields_vec = field_names
            .iter()
            .map(|f| Field::new(*f, DataType::Utf8, false))
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields_vec));

        Ok(Self {
            schema,
            message_stream,
            config_stream,
            runtime_env,
            baseline_metrics,
            config: None,
        })
    }

    fn init_config(&mut self, config_table: ArrowTable) -> Result<()> {
        if self.config.is_none() {
            let config: CandleOpsSummaryConfig =
                serde_json::from_value(serde_json::Value::Object(
                    config_table.to_json_object()?.first().unwrap().to_owned(),
                ))?;
            self.config.replace(config);
        }
        Ok(())
    }
}

impl Stream for OpsSummaryStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.config.is_some() {
            // The config is set to None before first iteration,
            // and then set to Some after the first iteration
            // breaking the loop
            Poll::Ready(None)
        } else {
            // Initialize the metrics
            let metrics = self.baseline_metrics.clone();
            let _timer = metrics.elapsed_compute().timer();

            // Initialize the config
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let config_table = ArrowTable::get_builder()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;

            // Collect the messages
            let mut batches = Vec::new();
            match self.config.as_ref().unwrap().num_batches {
                Some(num_batches) => {
                    for _iter in 0..num_batches {
                        while let Some(Ok(batch)) = ready!(self.message_stream.poll_next_unpin(cx))
                        {
                            batches.push(batch);
                        }
                    }
                }
                None => {
                    while let Some(Ok(batch)) = ready!(self.message_stream.poll_next_unpin(cx)) {
                        batches.push(batch);
                    }
                }
            }

            // Limit the columns
            let batches_col = match self.config.as_ref().unwrap().col_names.as_ref() {
                Some(col_names) => {
                    // Parse the JSON list of column names
                    let col_names_vec: Vec<String> = serde_json::from_str(col_names)?;

                    // Remove all columns that are not specified
                    batches
                        .into_iter()
                        .map(|batch| {
                            let columns_to_remove = batch
                                .schema()
                                .fields()
                                .iter()
                                .filter(|field| !col_names_vec.contains(field.name()))
                                .map(|field| field.name().to_string())
                                .collect::<Vec<_>>();
                            let schema = batch.schema();
                            let new_fields = schema
                                .fields()
                                .iter()
                                .filter(|field| !columns_to_remove.contains(field.name()))
                                .cloned()
                                .collect::<Vec<_>>();

                            let new_schema = Arc::new(Schema::new(new_fields));

                            let new_columns = batch
                                .columns()
                                .iter()
                                .zip(schema.fields())
                                .filter(|(_, field)| !columns_to_remove.contains(field.name()))
                                .map(|(column, _)| Arc::clone(column))
                                .collect::<Vec<_>>();

                            RecordBatch::try_new(new_schema, new_columns).unwrap()
                        })
                        .collect::<Vec<_>>()
                }
                None => batches,
            };

            // Concatenate into a single record batch
            let mut batch_json = ArrowTable::get_builder()
                .with_name("")
                .with_record_batches(batches_col)?
                .build()?
                .concat_record_batches()?
                .to_json_object()?;

            // Limit the number of rows
            let mut batch_limit = Vec::new();
            match self.config.as_ref().unwrap().num_rows {
                Some(num_rows) => {
                    if batch_json.len() > num_rows {
                        for index in 0..num_rows {
                            batch_limit.push(batch_json.remove(index));
                        }
                    } else {
                        batch_limit = batch_json;
                    }
                }
                None => batch_limit = batch_json,
            }

            // And then a single string
            let content = serde_json::to_string(&batch_limit)?;

            // Wrap into a record batch
            // DM: Change when upgrading to Qwen 3
            // let role: ArrayRef = Arc::new(StringArray::from(vec!["function"]));
            let role: ArrayRef = Arc::new(StringArray::from(vec!["tool"]));
            let content: ArrayRef = Arc::new(StringArray::from(vec![content]));
            let batch = RecordBatch::try_from_iter(vec![("role", role), ("content", content)])?;

            // record the poll
            let poll = Poll::Ready(Some(Ok(batch)));
            metrics.record_poll(poll)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }
}

impl RecordBatchStream for OpsSummaryStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use phymes_core::table::{
        arrow_table::ArrowTableBuilder, arrow_table_publish::ArrowTablePublish,
    };

    use crate::candle_ops::ops_processor::test_candle_ops_processor::make_embeddings_record_batch;

    use super::*;

    #[tokio::test]
    async fn test_summary_processor() -> Result<()> {
        // Create the input
        let mut messages = HashMap::<String, ArrowOutgoingMessage>::new();
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch = make_embeddings_record_batch("lhs_pk", lhs_ids_vec, lhs_embeddings_vec)?;
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

        // Make the config
        let config = CandleOpsSummaryConfig {
            num_rows: Some(2),
            num_batches: Some(1),
            col_names: Some("[\"embeddings\",\"lhs_pk\"]".to_string()),
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = ArrowTableBuilder::new()
            .with_name("sumary_processor")
            .with_json(&config_json, 1)?
            .build()?;
        let _ = messages.insert(
            "sumary_processor".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("sumary_processor")
                .with_publisher("")
                .with_subject("")
                .with_update(&ArrowTablePublish::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let metrics = ArrowTaskMetricsSet::new();

        let runtime_env = Arc::new(Mutex::new(RuntimeEnv {
            token_service: None,
            tensor_service: None,
            name: "service".to_string(),
            memory_limit: None,
            time_limit: None,
        }));

        // Create the processor and run
        let processor = OpsSummaryProcessor::new_with_pub_sub_for(
            "sumary_processor",
            &[ArrowTablePublish::Extend {
                table_name: "messages".to_string(),
            }],
            &[],
            &[],
        );
        let mut stream = processor.process(messages, metrics.clone(), runtime_env)?;

        // Wrap the results in a table
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            stream.remove("messages").unwrap().get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 1);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 10);
        // DM: change after upgrading to Qwen 3 series
        // assert_eq!(partitions.get_column_as_str_vec("role"), ["function"]);
        assert_eq!(partitions.get_column_as_str_vec("role"), ["tool"]);
        assert_eq!(
            partitions.get_column_as_str_vec("content"),
            [
                "[{\"embeddings\":[1.0,1.0,1.0,1.0],\"lhs_pk\":\"1\"},{\"embeddings\":[0.0,0.0,0.0,1.0],\"lhs_pk\":\"3\"}]"
            ]
        );

        Ok(())
    }
}
