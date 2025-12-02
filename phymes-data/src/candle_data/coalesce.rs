use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};
use std::usize;

use anyhow::{Result, anyhow};
use arrow::array::builder::StringViewBuilder;
use arrow::array::cast::AsArray;
use arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use futures::stream::{Stream, StreamExt};
use parking_lot::Mutex;
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait, MessageBuilderTrait, MessageTrait, ProcessorTrait,
    PublishAndSubscribeTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream,
    SendableRecordBatchStreamMessage, SendableRecordBatchStreamMessageMap, StateMap, Table,
    TableBuilderTrait, TablePublication, TableSubscribePolicyTrait, TableSubscription,
    remove_message_by_subject,
};
use phymes_diagnostics::{
    DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait, TraceBuilderTrait,
};
use tracing::{Level, event};

use crate::{DataConfigTrait, DataSummaryConfig};

/// Processor that implements the [RecordBatch] coalesce operator to combine smaller [RecordBatch]es into larger [RecordBatch]es of a specified size
#[derive(Debug)]
pub struct CoalesceProcessor {
    name: String,
    r#type: String,
    publications: Vec<TablePublication>,
    subscriptions: Vec<TableSubscription>,
    subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
}

impl MappableTrait for CoalesceProcessor {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PublishAndSubscribeTrait for CoalesceProcessor {
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

impl ProcessorTrait for CoalesceProcessor {
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

    fn process(
        &self,
        mut message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
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
        let config = match remove_message_by_subject(self.get_name(), &mut message) {
            Some(s) => s.get_message_own(),
            None => return Err(anyhow!("Config not provided for {}.", self.get_name())),
        };

        // Extract out the message to be summarized
        let mut subscriptions = Vec::new();
        let mut table_names = Vec::new();
        for subs in self.subscriptions.iter() {
            if subs.get_table_name() != self.get_name() {
                match remove_message_by_subject(subs.get_table_name(), &mut message) {
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
            return Err(anyhow!("More than one subscription was found for Limit."));
        } else if subscriptions.is_empty() {
            return Err(anyhow!("No subscriptions were found for Limit."));
        }

        // Make the outbox and send
        let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());
        let out = Box::pin(CoalesceStream::new(
            subscriptions.swap_remove(0).get_message_own(),
            config,
            Arc::clone(&runtime_env),
            stream_diagnostic_builder,
        ));
        let out_m = SendableRecordBatchStreamMessage::get_builder()
            .with_publisher(self.get_name())
            .with_subject(self.publications.first().unwrap().get_table_name())
            .with_message(out)
            .with_update(self.publications.first().unwrap())
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

/// Concatenate multiple [`RecordBatch`]es
///
/// `BatchCoalescer` concatenates multiple small [`RecordBatch`]es into larger ones for
/// more efficient processing by subsequent operations.
///
/// # Background
///
/// Generally speaking, larger [`RecordBatch`]es are more efficient to process
/// than smaller record batches (until the CPU cache is exceeded) because there
/// is fixed processing overhead per batch. We try to operate on
/// batches of `target_batch_size` rows to amortize this overhead
///
/// ```text
/// ┌────────────────────┐
/// │    RecordBatch     │
/// │   num_rows = 23    │
/// └────────────────────┘                 ┌────────────────────┐
///                                        │                    │
/// ┌────────────────────┐     Coalesce    │                    │
/// │                    │      Batches    │                    │
/// │    RecordBatch     │                 │                    │
/// │   num_rows = 50    │  ─ ─ ─ ─ ─ ─ ▶  │                    │
/// │                    │                 │    RecordBatch     │
/// │                    │                 │   num_rows = 106   │
/// └────────────────────┘                 │                    │
///                                        │                    │
/// ┌────────────────────┐                 │                    │
/// │                    │                 │                    │
/// │    RecordBatch     │                 │                    │
/// │   num_rows = 33    │                 └────────────────────┘
/// │                    │
/// └────────────────────┘
/// ```
///
/// # Notes:
///
/// 1. Output rows are produced in the same order as the input rows
///
/// 2. The output is a sequence of batches, with all but the last being at least
///    `fetch` rows.
pub struct CoalesceStream {
    /// The input to read from. This is set to None once the limit is
    /// reached to enable early termination
    message_stream: SendableRecordBatchStream,
    /// Copy of the input schema
    schema: SchemaRef,
    /// Parameters for coalesce
    config_stream: SendableRecordBatchStream,
    /// Runtime parameters
    _runtime_env: Arc<Mutex<RuntimeEnv>>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for coalesce
    config: Option<DataSummaryConfig>,
    /// Buffered batches
    buffer: Vec<RecordBatch>,
    /// Buffered row count
    buffered_rows: usize,
    /// Row size of the concatenated batch, `None` means fetch all rows without a limit
    fetch: Option<usize>,
    /// Overflow batch after the limit has been reached
    overflow: Option<RecordBatch>,
    /// Switch to finished polling
    is_finished: bool,
}

impl CoalesceStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Self {
        let schema = message_stream.schema();
        Self {
            message_stream,
            schema,
            config_stream,
            _runtime_env: runtime_env,
            diagnostic_builder,
            config: None,
            buffer: Vec::new(),
            buffered_rows: 0,
            fetch: None,
            overflow: None,
            is_finished: false,
        }
    }

    /// Initialize the config and update the values for skip and fetch
    fn init_config(&mut self, config_table: Table) -> Result<()> {
        if self.config.is_none() {
            let config = DataSummaryConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        self.fetch = self.config.as_ref().unwrap().fetch;
        Ok(())
    }

    /// Push next batch, and returns [`CoalescerState`] indicating the current
    /// state of the buffer.
    fn push_batch(&mut self, batch: RecordBatch) -> CoalescerState {
        // let batch = gc_string_view_batch(&batch);
        if self.limit_reached(batch) {
            CoalescerState::LimitReached
        } else {
            CoalescerState::Continue
        }
    }

    /// Checks if the buffer will reach the specified limit after getting
    /// `batch`.
    ///
    /// If fetch would be exceeded, slices the received batch, updates the
    /// buffer with it, and returns `true`.
    ///
    /// Otherwise: does nothing and returns `false`.
    fn limit_reached(&mut self, batch: RecordBatch) -> bool {
        if let Some(fetch) = self.fetch
            && self.buffered_rows + batch.num_rows() >= fetch
        {
            // Limit is reached
            let remaining_rows = fetch - self.buffered_rows;
            debug_assert!(remaining_rows > 0);

            // Fill the buffer up to fetch
            let batch_buf = batch.slice(0, remaining_rows);
            self.buffered_rows += batch_buf.num_rows();
            let overflow_rows = batch.num_rows() - batch_buf.num_rows();

            // Track the overflow
            if overflow_rows > 0 {
                let batch_over = batch.slice(remaining_rows, overflow_rows);
                self.overflow.replace(batch_over);
            }

            self.buffer.push(batch_buf);
            true
        } else {
            // Limit has not been reached
            self.buffered_rows += batch.num_rows();
            self.buffer.push(batch);
            false
        }
    }

    /// Concatenates and returns all buffered batches, and clears the buffer.
    fn finish_batch(&mut self) -> Result<RecordBatch> {
        let batch = if self.buffer.len() == 1 {
            self.buffer.pop().unwrap()
        } else {
            concat_batches(&self.schema, &self.buffer)?
        };
        if let Some(overflow) = self.overflow.take() {
            self.buffered_rows = overflow.num_rows();
            self.buffer = vec![overflow];
        } else {
            self.buffered_rows = 0;
            self.buffer.clear();
        }
        Ok(batch)
    }
}

impl Stream for CoalesceStream {
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
            let config_table = Table::get_builder()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;
        }

        // Coalesce the batches
        while let Some(Ok(batch)) =
            ready!(self.message_stream.as_mut().poll_next_unpin(cx))
        {
            match self.push_batch(batch) {
                CoalescerState::Continue => {}
                CoalescerState::LimitReached => {
                    let output_batch = self.finish_batch().unwrap();
                    let poll = Poll::Ready(Some(Ok(output_batch)));

                    // Return the poll
                    if let Some(baseline_metrics) = &baseline_metrics {
                        return baseline_metrics.record_poll(poll);
                    } else {
                        return poll;
                    }
                }
            }
        }

        // Clear out the remaining batches from the buffer
        if self.buffer.is_empty() {
            Poll::Ready(None)
        } else {
            let output_batch = self.finish_batch().unwrap();
            let poll = Poll::Ready(Some(Ok(output_batch)));

            // Trigger the end of the stream
            self.is_finished = true;

            // Return the poll
            if let Some(baseline_metrics) = &baseline_metrics {
                baseline_metrics.record_poll(poll)
            } else {
                poll
            }
        }
    }
}

impl RecordBatchStream for CoalesceStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Indicates the state of the [`CoalescerStream`] buffer after the
/// [`BatchCoalescer::push_batch()`] operation.
///
/// The caller should take different actions, depending on the variant returned.
pub enum CoalescerState {
    /// Neither the limit nor the target batch size is reached.
    ///
    /// Action: continue pushing batches.
    Continue,
    /// The limit has been reached.
    ///
    /// Action: call [`BatchCoalescer::finish_batch()`] to get the final
    /// buffered results as a batch and then continue pushing batches.
    LimitReached,
}

/// Heuristically compact `StringViewArray`s to reduce memory usage, if needed
///
/// Decides when to consolidate the StringView into a new buffer to reduce
/// memory usage and improve string locality for better performance.
///
/// This differs from `StringViewArray::gc` because:
/// 1. It may not compact the array depending on a heuristic.
/// 2. It uses a precise block size to reduce the number of buffers to track.
///
/// # Heuristic
///
/// If the average size of each view is larger than 32 bytes, we compact the array.
///
/// `StringViewArray` include pointers to buffer that hold the underlying data.
/// One of the great benefits of `StringViewArray` is that many operations
/// (e.g., `filter`) can be done without copying the underlying data.
///
/// However, after a while (e.g., after `FilterExec` or `HashJoinExec`) the
/// `StringViewArray` may only refer to a small portion of the buffer,
/// significantly increasing memory usage.
fn gc_string_view_batch(batch: &RecordBatch) -> RecordBatch {
    let new_columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| {
            // Try to re-create the `StringViewArray` to prevent holding the underlying buffer too long.
            let Some(s) = c.as_string_view_opt() else {
                return Arc::clone(c);
            };
            let ideal_buffer_size: usize = s
                .views()
                .iter()
                .map(|v| {
                    let len = (*v as u32) as usize;
                    if len > 12 { len } else { 0 }
                })
                .sum();
            let actual_buffer_size = s.get_buffer_memory_size();

            // Re-creating the array copies data and can be time consuming.
            // We only do it if the array is sparse
            if actual_buffer_size > (ideal_buffer_size * 2) {
                // We set the block size to `ideal_buffer_size` so that the new StringViewArray only has one buffer, which accelerate later concat_batches.
                // See https://github.com/apache/arrow-rs/issues/6094 for more details.
                let mut builder = StringViewBuilder::with_capacity(s.len());
                if ideal_buffer_size > 0 {
                    builder = builder.with_fixed_block_size(ideal_buffer_size as u32);
                }

                for v in s.iter() {
                    builder.append_option(v);
                }

                let gc_string = builder.finish();

                debug_assert!(gc_string.data_buffers().len() <= 1); // buffer count can be 0 if the `ideal_buffer_size` is 0

                Arc::new(gc_string)
            } else {
                Arc::clone(c)
            }
        })
        .collect();
    let mut options = RecordBatchOptions::new();
    options = options.with_row_count(Some(batch.num_rows()));
    RecordBatch::try_new_with_options(batch.schema(), new_columns, &options)
        .expect("Failed to re-create the gc'ed record batch")
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;

    use arrow::array::builder::ArrayBuilder;
    use arrow::array::{StringViewArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use phymes_core::{
        AvailableTableSubscribePolicies, RecordBatchStreamAdapter, TableBuilder, TableTrait,
        test_table,
    };
    use phymes_diagnostics::{Diagnostics, SpanBuilder};

    #[tokio::test]
    async fn test_coalesce_processor() -> Result<()> {
        // Make the test batches
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let test_table = test_table::make_test_table("input", 4, 8, 4)?;
        let test_message = SendableRecordBatchStreamMessage::get_builder()
            .with_name("input")
            .with_subject("input")
            .with_publisher("")
            .with_message(test_table.to_record_batch_stream())
            .with_update(&TablePublication::None)
            .build()?;
        let _ = message.insert(test_message.get_name().to_string(), test_message);

        // Make the config
        let config = DataSummaryConfig {
            fetch: Some(6),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = TableBuilder::new()
            .with_name("CoalesceProcessor")
            .with_json(&config_json, 1)?
            .build()?;
        let config_message = SendableRecordBatchStreamMessage::get_builder()
            .with_name(config_table.get_name())
            .with_publisher("")
            .with_subject(config_table.get_name())
            .with_update(&TablePublication::None)
            .with_message(config_table.to_record_batch_stream())
            .build()?;
        let _ = message.insert(config_message.get_name().to_string(), config_message);

        // Make the diagnostics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the Runtime Env
        let runtime_env = Arc::new(Mutex::new(RuntimeEnv {
            name: "service".to_string(),
            ..Default::default()
        }));

        // Coalesce into batches of six
        let processor = CoalesceProcessor::new(
            "CoalesceProcessor",
            "",
            &[TablePublication::Extend {
                table_name: "output".to_string(),
            }],
            &[TableSubscription::AlwaysFullTable {
                table_name: "input".to_string(),
            }],
            AvailableTableSubscribePolicies::AllTableNamesSubscribe.build(),
        );
        let mut stream =
            processor.process(message, Some(&diagnostic_builder), runtime_env.clone())?;

        // Wrap the results in a table
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            stream
                .remove("from_CoalesceProcessor_on_output")
                .unwrap()
                .get_message_own(),
        )
        .await?
        .with_name("")
        .build()?;
        let sizes = partitions
            .get_record_batches()
            .iter()
            .map(|b| b.num_rows())
            .collect::<Vec<_>>();
        assert_eq!(sizes, [6, 6, 4]);
        Ok(())
    }

    #[tokio::test]
    async fn test_coalesce_stream() -> Result<()> {
        // Make the batches
        let batch = uint32_batch(0..8);

        // Make the diagnostics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the Runtime Env
        let runtime_env = Arc::new(Mutex::new(RuntimeEnv {
            name: "service".to_string(),
            ..Default::default()
        }));

        // --- Coalesce without intermediate overflow ---
        // Make the config
        let config = DataSummaryConfig {
            fetch: Some(24),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = TableBuilder::new()
            .with_name("CoalesceStream")
            .with_json(&config_json, 1)?
            .build()?;

        // Coalesce batches
        let stream = futures::stream::iter(std::iter::repeat_n(batch.clone(), 10).map(Ok));
        let input = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&batch.schema()),
            stream,
        ));
        let coalesce_stream = CoalesceStream::new(
            input,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        );

        let results = Box::pin(coalesce_stream).try_collect::<Vec<_>>().await?;
        let num_rows = results
            .into_iter()
            .map(|b| b.num_rows())
            .collect::<Vec<_>>();
        assert_eq!(num_rows, [24, 24, 24, 8]);

        // --- Coalesce without overflow ---
        // Make the config
        let config = DataSummaryConfig {
            fetch: Some(100),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = TableBuilder::new()
            .with_name("CoalesceStream")
            .with_json(&config_json, 1)?
            .build()?;

        // Coalesce batches
        let stream = futures::stream::iter(std::iter::repeat_n(batch.clone(), 10).map(Ok));
        let input = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&batch.schema()),
            stream,
        ));
        let coalesce_stream = CoalesceStream::new(
            input,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        );

        let results = Box::pin(coalesce_stream).try_collect::<Vec<_>>().await?;
        let num_rows = results
            .into_iter()
            .map(|b| b.num_rows())
            .collect::<Vec<_>>();
        assert_eq!(num_rows, [80]);

        // --- Coalesce with intermediate overflow ---
        // Make the config
        let config = DataSummaryConfig {
            fetch: Some(10),
            ..Default::default()
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = TableBuilder::new()
            .with_name("CoalesceStream")
            .with_json(&config_json, 1)?
            .build()?;

        // Coalesce batches
        let stream = futures::stream::iter(std::iter::repeat_n(batch.clone(), 10).map(Ok));
        let input = Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&batch.schema()),
            stream,
        ));
        let coalesce_stream = CoalesceStream::new(
            input,
            config_table.to_record_batch_stream(),
            Arc::clone(&runtime_env),
            Some(diagnostic_builder.clone()),
        );

        let results = Box::pin(coalesce_stream).try_collect::<Vec<_>>().await?;
        let num_rows = results
            .into_iter()
            .map(|b| b.num_rows())
            .collect::<Vec<_>>();
        assert_eq!(num_rows, [10, 10, 10, 10, 10, 10, 10, 10]);

        Ok(())
    }

    /// Return a batch of  UInt32 with the specified range
    fn uint32_batch(range: Range<u32>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("c0", DataType::UInt32, false)]));

        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(UInt32Array::from_iter_values(range))],
        )
        .unwrap()
    }

    #[test]
    fn test_gc_string_view_batch_small_no_compact() {
        // view with only short strings (no buffers) --> no need to compact
        let array = StringViewTest {
            rows: 1000,
            strings: vec![Some("a"), Some("b"), Some("c")],
        }
        .build();

        let gc_array = do_gc(array.clone());
        compare_string_array_values(&array, &gc_array);
        assert_eq!(array.data_buffers().len(), 0);
        assert_eq!(array.data_buffers().len(), gc_array.data_buffers().len()); // no compaction
    }

    #[test]
    fn test_gc_string_view_test_batch_empty() {
        let schema = Schema::empty();
        let batch = RecordBatch::new_empty(schema.into());
        let output_batch = gc_string_view_batch(&batch);
        assert_eq!(batch.num_columns(), output_batch.num_columns());
        assert_eq!(batch.num_rows(), output_batch.num_rows());
    }

    #[test]
    fn test_gc_string_view_batch_large_no_compact() {
        // view with large strings (has buffers) but full --> no need to compact
        let array = StringViewTest {
            rows: 1000,
            strings: vec![Some("This string is longer than 12 bytes")],
        }
        .build();

        let gc_array = do_gc(array.clone());
        compare_string_array_values(&array, &gc_array);
        assert_eq!(array.data_buffers().len(), 5);
        assert_eq!(array.data_buffers().len(), gc_array.data_buffers().len()); // no compaction
    }

    #[test]
    fn test_gc_string_view_batch_large_slice_compact() {
        // view with large strings (has buffers) and only partially used  --> no need to compact
        let array = StringViewTest {
            rows: 1000,
            strings: vec![Some("this string is longer than 12 bytes")],
        }
        .build();

        // slice only 11 rows, so most of the buffer is not used
        let array = array.slice(11, 22);

        let gc_array = do_gc(array.clone());
        compare_string_array_values(&array, &gc_array);
        assert_eq!(array.data_buffers().len(), 5);
        assert_eq!(gc_array.data_buffers().len(), 1); // compacted into a single buffer
    }

    /// Compares the values of two string view arrays
    fn compare_string_array_values(arr1: &StringViewArray, arr2: &StringViewArray) {
        assert_eq!(arr1.len(), arr2.len());
        for (s1, s2) in arr1.iter().zip(arr2.iter()) {
            assert_eq!(s1, s2);
        }
    }

    /// runs garbage collection on string view array
    /// and ensures the number of rows are the same
    fn do_gc(array: StringViewArray) -> StringViewArray {
        let batch = RecordBatch::try_from_iter(vec![("a", Arc::new(array) as ArrayRef)]).unwrap();
        let gc_batch = gc_string_view_batch(&batch);
        assert_eq!(batch.num_rows(), gc_batch.num_rows());
        assert_eq!(batch.schema(), gc_batch.schema());
        gc_batch
            .column(0)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .clone()
    }

    /// Describes parameters for creating a `StringViewArray`
    struct StringViewTest {
        /// The number of rows in the array
        rows: usize,
        /// The strings to use in the array (repeated over and over
        strings: Vec<Option<&'static str>>,
    }

    impl StringViewTest {
        /// Create a `StringViewArray` with the parameters specified in this struct
        fn build(self) -> StringViewArray {
            let mut builder = StringViewBuilder::with_capacity(100).with_fixed_block_size(8192);
            loop {
                for &v in self.strings.iter() {
                    builder.append_option(v);
                    if builder.len() >= self.rows {
                        return builder.finish();
                    }
                }
            }
        }
    }
}
