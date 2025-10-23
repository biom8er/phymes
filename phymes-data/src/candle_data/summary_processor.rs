use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
};

use bytes::Bytes;
use phymes_core::{
    AllTableNamesSubscribe, AvailableSubjects, AvailableSubjectsTrait, BuildableTrait,
    BuilderTrait, CsvFormat, DataFormat, MappableTrait, MessageBuilderTrait, MessageTrait,
    ProcessorTrait, PubSubTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap,
    SubscribeTrait, Table, TableBuilderTrait, TablePublish, TableSubscribe, TableTrait,
    create_blob_batch, create_chat_record_batch,
};

use anyhow::{Result, anyhow};
use arrow::{
    array::RecordBatch,
    datatypes::{Schema, SchemaRef},
};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, TraceBuilderTrait,
    create_timestamp_micros,
};
use tracing::{Level, event, instrument};

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
    publications: Vec<TablePublish>,
    subscriptions: Vec<TableSubscribe>,
    subscribe: Box<dyn SubscribeTrait>,
}

impl MappableTrait for DataSummaryProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PubSubTrait for DataSummaryProcessor {
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

impl ProcessorTrait for DataSummaryProcessor {
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
            publications: vec![TablePublish::None],
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
        let config = match message.remove(self.get_name()) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Extract out the messages to be summarized
        let mut subscriptions = Vec::new();
        let mut table_names = Vec::new();
        for subs in self.subscriptions.iter() {
            if subs.get_table_name() != self.get_name() {
                match message.remove(subs.get_table_name()) {
                    Some(m) => {
                        subscriptions.push(m);
                        table_names.push(subs.get_table_name())
                    }
                    None => {
                        event!(
                            Level::WARN,
                            "Subscription {} not provided for {}.",
                            subs.get_table_name(),
                            self.get_name()
                        );
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
        let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
        let out = Box::pin(DataSummaryStream::new(
            subscriptions.swap_remove(0).get_message_own(),
            table_names.swap_remove(0).to_string(),
            config,
            Arc::clone(&runtime_env),
            stream_diagnostic_builder,
        )?);
        let mut outbox = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let out_m = SendableRecordBatchStreamMessage::get_builder()
            .with_name(self.publications.first().unwrap().get_table_name())
            .with_publisher(self.get_name())
            .with_subject(self.publications.first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.publications.first().unwrap())
            .build()?;
        let _ = outbox.insert(out_m.get_name().to_string(), out_m);

        // Trace the outbox
        if let Some(trace) = trace {
            trace.0.exit(&outbox.values().collect::<Vec<_>>());
        }
        Ok(outbox)
    }
}

#[allow(dead_code)]
pub struct DataSummaryStream {
    /// Output schema (role and content)
    schema: SchemaRef,
    /// The input message to process
    message_stream: SendableRecordBatchStream,
    /// The table name of the subscription
    table_name: String,
    /// Parameters for chat inference
    config_stream: SendableRecordBatchStream,
    /// The Candle model assets needed for inference
    runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for chat inference
    config: Option<DataSummaryConfig>,
}

impl DataSummaryStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        table_name: String,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Result<Self> {
        Ok(Self {
            schema: AvailableSubjects::Messages.to_schema(),
            message_stream,
            table_name,
            config_stream,
            runtime_env,
            diagnostic_builder,
            config: None,
        })
    }

    fn init_config(&mut self, config_table: Table) -> Result<()> {
        if self.config.is_none() {
            let config: DataSummaryConfig = serde_json::from_value(serde_json::Value::Object(
                config_table.to_json_object()?.first().unwrap().to_owned(),
            ))?;
            self.config.replace(config);
        }
        Ok(())
    }
}

/// Helper function to convert a table into the desired output format
pub fn table_and_data_format_to_record_batch(
    table: &Table,
    format: &DataFormat,
) -> Result<RecordBatch> {
    match format {
        DataFormat::None => {
            // Wrap into a record batch
            let content = serde_json::to_string(&table.to_json_object()?)?;
            create_chat_record_batch(
                vec!["tool".to_string()], // DM: Change when upgrading to Qwen 3 "function"
                vec![content],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Csv(csv_format) => {
            // Convert to CSV and wrap into a blob batch
            let bytes = table.to_csv(csv_format.delimiter, csv_format.header)?;
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::CsvDefault => {
            // Convert to CSV and wrap into a blob batch
            let csv_format = CsvFormat {
                ..Default::default()
            };
            let bytes = table.to_csv(csv_format.delimiter, csv_format.header)?;
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Bytes => {
            // Convert to bytes directly
            let bytes = table.to_bytes()?;
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes.to_vec()],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Json(_) | DataFormat::JsonDefault => {
            // Convert to JSON
            let bytes = table.to_json()?;
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Html | DataFormat::Txt => {
            // Extract out the values column and concatenate into a single String to form the document
            let values = table.get_column_as_vec_str("values").join("");
            let bytes = Bytes::from(values);
            create_blob_batch(
                vec![table.get_name().to_string()],
                vec![format.to_extension().to_string()],
                vec![bytes.to_vec()],
                vec!["assistant".to_string()],
                vec![create_timestamp_micros()],
            )
        }
        DataFormat::Pdf | DataFormat::Ipc => Err(anyhow!("{format} format is not yet supported.")),
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

            // Initialize the config
            let mut batches = Vec::new();
            while let Some(Ok(batch)) = ready!(self.config_stream.poll_next_unpin(cx)) {
                batches.push(batch);
            }
            let config_table = Table::get_builder()
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
                    // Remove all columns that are not specified
                    batches
                        .into_iter()
                        .map(|batch| {
                            let columns_to_remove = batch
                                .schema()
                                .fields()
                                .iter()
                                .filter(|field| !col_names.contains(field.name()))
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
                            event!(
                                Level::DEBUG,
                                "New schema: {:?}, new columns: {:?}",
                                new_schema,
                                new_columns
                            );

                            RecordBatch::try_new(new_schema, new_columns).unwrap()
                        })
                        .collect::<Vec<_>>()
                }
                None => batches,
            };

            // Concatenate into a single record batch
            let schema = batches_col.first().unwrap().schema();
            let mut batch_json = Table::get_builder()
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

            // Wrap into a table
            let values = batch_limit
                .into_iter()
                .map(|m| serde_json::to_value(m).unwrap())
                .collect::<Vec<_>>();
            let table = Table::get_builder()
                .with_name(&self.table_name)
                .with_schema(schema)
                .with_json_values(&values)?
                .build()?;

            // Convert to the desired format
            let batch = table_and_data_format_to_record_batch(
                &table,
                &self.config.as_ref().unwrap().format,
            )?;

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

impl RecordBatchStream for DataSummaryStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, StringArray};
    use phymes_core::{MessageTrait, TableBuilder, TablePublish};
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};

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
        let lhs_table = Table::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;

        // Make the config
        let config = DataSummaryConfig {
            num_rows: Some(2),
            num_batches: Some(1),
            col_names: Some(vec!["embedding".to_string(), "lhs_pk".to_string()]),
            format: DataFormat::None,
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = TableBuilder::new()
            .with_name("summary_processor")
            .with_json(&config_json, 1)?
            .build()?;

        // Make the input messages
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            "lhs_name".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("lhs_name")
                .with_publisher("")
                .with_subject("lhs_name")
                .with_update(&TablePublish::None)
                .with_message(lhs_table.to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            "summary_processor".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("summary_processor")
                .with_publisher("")
                .with_subject("summary_processor")
                .with_update(&TablePublish::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

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
            &[TablePublish::Extend {
                table_name: "messages".to_string(),
            }],
            &[TableSubscribe::AlwaysFullTable {
                table_name: "lhs_name".to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        );
        let mut stream =
            processor.process(messages, Some(&diagnostic_builder), runtime_env.clone())?;

        // Wrap the results in a table
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            stream.remove("messages").unwrap().get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;

        // Check the results
        assert_eq!(partitions.count_rows(), 1);
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
        let lhs_table = Table::get_builder()
            .with_name("lhs_name")
            .with_record_batches(vec![lhs_batch])?
            .build()?;

        // Make the config
        let config = DataSummaryConfig {
            num_rows: Some(2),
            num_batches: Some(1),
            col_names: Some(vec!["lhs_pk".to_string()]),
            format: DataFormat::Csv(CsvFormat {
                ..Default::default()
            }),
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = TableBuilder::new()
            .with_name("summary_processor")
            .with_json(&config_json, 1)?
            .build()?;

        // Make the input messages
        let mut messages = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = messages.insert(
            "lhs_name".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("lhs_name")
                .with_publisher("")
                .with_subject("lhs_name")
                .with_update(&TablePublish::None)
                .with_message(lhs_table.to_record_batch_stream())
                .build()?,
        );
        let _ = messages.insert(
            "summary_processor".to_string(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name("summary_processor")
                .with_publisher("")
                .with_subject("summary_processor")
                .with_update(&TablePublish::None)
                .with_message(config_table.to_record_batch_stream())
                .build()?,
        );

        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

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
            &[TablePublish::Extend {
                table_name: "messages".to_string(),
            }],
            &[TableSubscribe::AlwaysFullTable {
                table_name: "lhs_name".to_string(),
            }],
            AllTableNamesSubscribe::new_box(),
        );
        let mut stream =
            processor.process(messages, Some(&diagnostic_builder), runtime_env.clone())?;

        // Wrap the results in a table
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            stream.remove("messages").unwrap().get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;

        // Check the results
        assert_eq!(partitions.count_rows(), 1);
        assert_eq!(partitions.get_column_as_vec_str("filename"), ["lhs_name"]);
        assert_eq!(partitions.get_column_as_vec_str("extension"), ["csv"]);
        assert_eq!(partitions.get_column_as_vec_str("metadata"), ["assistant"]);
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
