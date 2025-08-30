use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use phymes_core::{
    metrics::{ArrowTaskMetricsSet, BaselineMetrics, HashMap},
    schemas::available_subjects::{
        create_blob_batch, create_messages_record_batch, create_timestamp_micros, AvailableSubjects
    },
    session::{
        common_traits::{
            BuildableTrait, BuilderTrait, MappableTrait, OutgoingMessageMap, StateMap,
        },
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table::{ArrowTable, ArrowTableBuilderTrait, ArrowTableTrait},
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
    array::RecordBatch, csv, datatypes::{Schema, SchemaRef}
};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use tracing::{Level, event, instrument};

use crate::candle_data::summary_config::DataSummaryFormat;

use super::summary_config::DataSummaryConfig;

/// Processor that takes the results of an OpsProcessor
///   and creates a summary of the result for chat inference
///   or creates an attachment for the user to download
///
/// # Notes
///
/// - The default role is `tool`
#[derive(Debug)]
pub struct DataSummaryProcessor {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    subscribe: Box<dyn SubscribeTrait>,
}

impl MappableTrait for DataSummaryProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PubSubTrait for DataSummaryProcessor {
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

impl ArrowProcessorTrait for DataSummaryProcessor {
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
            publications: vec![ArrowTablePublish::None],
            subscriptions: vec![ArrowTableSubscribe::None],
            subscribe: AllTableNamesSubscribe::new_box(),
        })
    }

    fn get_subscribe(&self) -> &dyn SubscribeTrait {
        self.subscribe.as_ref()
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

        // Extract out the config
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Extract out the messages to be summarized
        let mut subscriptions = Vec::new();
        for subs in self.subscriptions.iter() {
            if subs.get_table_name() != self.get_name() {
                match message.remove(subs.get_table_name()) {
                    Some(m) => {
                        subscriptions.push(m);
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
        if subscriptions.len() > 1 {
            return Err(anyhow!("More than one subscription was found."));
        } else if subscriptions.is_empty() {
            return Err(anyhow!("No subscriptions were found."));
        }

        // Make the outbox and send
        let out = Box::pin(DataSummaryStream::new(
            subscriptions.swap_remove(0).get_message_own(),
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
pub struct DataSummaryStream {
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
    config: Option<DataSummaryConfig>,
}

impl DataSummaryStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        baseline_metrics: BaselineMetrics,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::Messages.to_schema(),
            message_stream,
            config_stream,
            runtime_env,
            baseline_metrics,
            config: None,
        })
    }

    fn init_config(&mut self, config_table: ArrowTable) -> Result<()> {
        if self.config.is_none() {
            let config: DataSummaryConfig = serde_json::from_value(serde_json::Value::Object(
                config_table.to_json_object()?.first().unwrap().to_owned(),
            ))?;
            self.config.replace(config);
        }
        Ok(())
    }
}

impl Stream for DataSummaryStream {
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
                    event!(Level::DEBUG, "col_names_vec: {:?}", col_names_vec);
                    event!(Level::DEBUG, "batches: {:?}", batches);

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
                            event!(Level::DEBUG, "New schema: {:?}, new columns: {:?}", new_schema, new_columns);

                            RecordBatch::try_new(new_schema, new_columns).unwrap()
                        })
                        .collect::<Vec<_>>()
                }
                None => batches,
            };

            // Concatenate into a single record batch
            let schema = batches_col.first().unwrap().schema();
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

            // Convert to the desired format
            match self.config.as_ref().unwrap().format {
                DataSummaryFormat::Message => {
                    // Wrap into a record batch
                    let content = serde_json::to_string(&batch_limit)?;
                    let batch = create_messages_record_batch(
                        vec!["tool".to_string()], // DM: Change when upgrading to Qwen 3 "function"
                        vec![content.to_string()],
                        vec![create_timestamp_micros()],
                    )?;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    metrics.record_poll(poll)

                }
                DataSummaryFormat::Csv(csv_format) => {
                    // Convert to Values representation
                    let mut values = Vec::new();
                    for row in batch_limit.iter() {
                        let v = serde_json::to_value(row)?;
                        values.push(v);
                    }
                    let table = ArrowTable::get_builder()
                        .with_name("attachment")
                        .with_schema(schema)
                        .with_json_values(&values)?
                        .build()?;

                    // Convert to CSV and wrap into a blob batch
                    let bytes = table.to_csv(csv_format.delimiter, csv_format.header)?;
                    let batch = create_blob_batch(vec!["attachment".to_string()], vec![".csv".to_string()], vec![bytes], vec!["".to_string()])?;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    return metrics.record_poll(poll);
                }
                DataSummaryFormat::Json(_json_format) => {
                    // Convert to Values representation
                    let mut values = Vec::new();
                    for row in batch_limit.iter() {
                        let v = serde_json::to_value(row)?;
                        values.push(v);
                    }
                    let table = ArrowTable::get_builder()
                        .with_name("attachment")
                        .with_schema(schema)
                        .with_json_values(&values)?
                        .build()?;

                    // Convert to CSV and wrap into a blob batch
                    let bytes = table.to_json()?;
                    let batch = create_blob_batch(vec!["attachment".to_string()], vec![".json".to_string()], vec![bytes], vec!["".to_string()])?;

                    // record the poll
                    let poll = Poll::Ready(Some(Ok(batch)));
                    return metrics.record_poll(poll);
                }
                DataSummaryFormat::Pdf => {
                    todo!("Implement PDF output");
                }
                DataSummaryFormat::Bytes => {
                    todo!("Implement Bytes output");
                }
                DataSummaryFormat::IPC => {
                    todo!("Implement Arrow IPC output");
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (1, Some(1))
    }
}

impl RecordBatchStream for DataSummaryStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, StringArray};
    use phymes_core::table::{
        arrow_table::ArrowTableBuilder, arrow_table_publish::ArrowTablePublish,
    };

    use crate::candle_data::data_processor::test_candle_ops_processor::make_embeddings_record_batch_str_f32;

    use super::*;

    #[tokio::test]
    async fn test_summary_processor_message_format() -> Result<()> {
        // Create the input
        let lhs_ids_vec = vec!["1", "2", "3"];
        let lhs_embeddings_vec: Vec<Vec<f32>> = vec![
            vec![1., 1., 1., 1.],
            vec![0., 1., 0., 1.],
            vec![0., 0., 0., 1.],
        ];
        let lhs_batch =
            make_embeddings_record_batch_str_f32("lhs_pk", lhs_ids_vec, lhs_embeddings_vec)?;
        let lhs_table = ArrowTable::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;

        // Make the config
        let config = DataSummaryConfig {
            num_rows: Some(2),
            num_batches: Some(1),
            col_names: Some("[\"embedding\",\"lhs_pk\"]".to_string()),
            format: DataSummaryFormat::Message,
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = ArrowTableBuilder::new()
            .with_name("summary_processor")
            .with_json(&config_json, 1)?
            .build()?;

        // Make the input messages
        let mut messages = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = messages.insert(
            "lhs_name".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("lhs_name")
                .with_publisher("")
                .with_subject("lhs_name")
                .with_update(&ArrowTablePublish::None)
                .with_message(lhs_table.to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            "summary_processor".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("summary_processor")
                .with_publisher("")
                .with_subject("summary_processor")
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
        let processor = DataSummaryProcessor::new_arc_with_pub_sub(
            "summary_processor",
            &[ArrowTablePublish::Extend {
                table_name: "messages".to_string(),
            }],
            &[ArrowTableSubscribe::AlwaysFullTable {
                table_name: "lhs_name".to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        );
        let mut stream = processor.process(messages, metrics.clone(), runtime_env.clone())?;

        // Wrap the results in a table
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            stream.remove("messages").unwrap().get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;

        // Check the results
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 1);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 10);
        // DM: change after upgrading to Qwen 3 series
        // assert_eq!(partitions.get_column_as_vec_str("role"), ["function"]);
        assert_eq!(partitions.get_column_as_vec_str("role"), ["tool"]);
        assert_eq!(
            partitions.get_column_as_vec_str("content"),
            [
                "[{\"embedding\":[1.0,1.0,1.0,1.0],\"lhs_pk\":\"1\"},{\"embedding\":[0.0,0.0,0.0,1.0],\"lhs_pk\":\"3\"}]"
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_summary_processor_blob_formats() -> Result<()> {
        // Create the input
        let lhs_ids_vec = vec!["1", "2", "3"];
        let ids_ar: ArrayRef = Arc::new(StringArray::from(lhs_ids_vec));
        let lhs_batch = RecordBatch::try_from_iter(vec![("lhs_pk", ids_ar)])?;
        let lhs_table = ArrowTable::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;

        // Make the config
        let config = DataSummaryConfig {
            num_rows: Some(2),
            num_batches: Some(1),
            col_names: Some("[\"lhs_pk\"]".to_string()),
            format: DataSummaryFormat::Csv,
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = ArrowTableBuilder::new()
            .with_name("summary_processor")
            .with_json(&config_json, 1)?
            .build()?;

        // Make the input messages
        let mut messages = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = messages.insert(
            "lhs_name".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("lhs_name")
                .with_publisher("")
                .with_subject("lhs_name")
                .with_update(&ArrowTablePublish::None)
                .with_message(lhs_table.to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            "summary_processor".to_string(),
            ArrowOutgoingMessage::get_builder()
                .with_name("summary_processor")
                .with_publisher("")
                .with_subject("summary_processor")
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
        let processor = DataSummaryProcessor::new_arc_with_pub_sub(
            "summary_processor",
            &[ArrowTablePublish::Extend {
                table_name: "messages".to_string(),
            }],
            &[ArrowTableSubscribe::AlwaysFullTable {
                table_name: "lhs_name".to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        );
        let mut stream = processor.process(messages, metrics.clone(), runtime_env.clone())?;

        // Wrap the results in a table
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            stream.remove("messages").unwrap().get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;

        // Check the results
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 1);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 10);
        assert_eq!(partitions.get_column_as_vec_str("filename"), ["attachment"]);
        assert_eq!(partitions.get_column_as_vec_str("extension"), [".csv"]);
        assert_eq!(partitions.get_column_as_vec_str("metadata"), [""]);
        let contents_vec = partitions.get_column_as_vec_nested_primitive::<u8>("bytes")?;
        let mut contents_str = Vec::new();
        for contents in contents_vec.into_iter() {
            contents_str.push(String::from_utf8(contents)?);
        }
        let contents_join = contents_str.join("");
        assert_eq!(contents_join, "lhs_pk\n1\n3\n");

        Ok(())
    }
}
