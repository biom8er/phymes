use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use anyhow::{Result, anyhow};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use futures::stream::{Stream, StreamExt};
use phymes_core::{
    BuildableTrait, BuilderTrait, MappableTrait,
    RecordBatchStream, RuntimeEnv, SendableRecordBatchStream, Subject, SubjectBuilderTrait,
};
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, MetricBuilderTrait};
use phymes_message::{
    MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
    SendableRecordBatchStreamMessageBuilder, SendableRecordBatchStreamMessageBuilderMap,
    SendableRecordBatchStreamMessageMap, remove_message_by_subject,
};
use phymes_processor::ProcessorTrait;

use crate::{DataConfigTrait, LimitConfig};

/// A Limit stream skips `skip` rows, and then fetch up to `fetch` rows.
pub struct LimitStream {
    /// The remaining number of rows to skip
    skip: Option<usize>,
    /// The remaining number of rows to produce
    fetch: Option<usize>,
    /// The input to read from. This is set to None once the limit is
    /// reached to enable early termination
    message_stream: Option<SendableRecordBatchStream>,
    /// Copy of the input schema
    schema: SchemaRef,
    /// Parameters for limit
    config_stream: SendableRecordBatchStream,
    /// Runtime parameters
    _runtime_env: Arc<RuntimeEnv>,
    /// Runtime metrics recording
    diagnostic_builder: Option<DiagnosticBuilder>,
    /// Parameters for limit
    config: Option<LimitConfig>,
}

impl LimitStream {
    pub fn new(
        message_stream: SendableRecordBatchStream,
        config_stream: SendableRecordBatchStream,
        runtime_env: Arc<RuntimeEnv>,
        diagnostic_builder: Option<DiagnosticBuilder>,
    ) -> Self {
        let schema = message_stream.schema();
        Self {
            skip: None,
            fetch: Some(usize::MAX),
            message_stream: Some(message_stream),
            schema,
            config_stream,
            _runtime_env: runtime_env,
            diagnostic_builder,
            config: None,
        }
    }

    /// Initialize the config and update the values for skip and fetch
    fn init_config(&mut self, config_table: Subject) -> Result<()> {
        if self.config.is_none() {
            let config = LimitConfig::from_table(&config_table)?;
            self.config.replace(config);
        }
        self.skip.replace(
            self.config
                .as_ref()
                .unwrap()
                .skip
                .as_ref()
                .unwrap()
                .to_owned(),
        );
        self.fetch
            .replace(self.config.as_ref().unwrap().fetch.to_owned());
        Ok(())
    }

    fn poll_and_skip(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        let message_stream = self.message_stream.as_mut().unwrap();
        loop {
            let poll = message_stream.poll_next_unpin(cx);
            let poll = poll.map_ok(|batch| {
                if &batch.num_rows() <= self.skip.as_ref().unwrap() {
                    let skip = self.skip.take().unwrap() - batch.num_rows();
                    self.skip.replace(skip);
                    RecordBatch::new_empty(message_stream.schema())
                } else {
                    let new_batch = batch.slice(
                        self.skip.as_ref().unwrap().to_owned(),
                        batch.num_rows() - self.skip.as_ref().unwrap().to_owned(),
                    );
                    self.skip.replace(0);
                    new_batch
                }
            });

            match &poll {
                Poll::Ready(Some(Ok(batch))) => {
                    if batch.num_rows() > 0 {
                        break poll;
                    } else {
                        // Continue to poll input stream
                    }
                }
                Poll::Ready(Some(Err(_e))) => break poll,
                Poll::Ready(None) => break poll,
                Poll::Pending => break poll,
            }
        }
    }

    /// Fetches from the batch
    fn stream_limit(&mut self, batch: RecordBatch) -> Option<RecordBatch> {
        // records time on drop
        if self.fetch.as_ref().unwrap() == &0 {
            self.message_stream = None; // Clear input so it can be dropped early
            None
        } else if &batch.num_rows() < self.fetch.as_ref().unwrap() {
            let fetch = self.fetch.take().unwrap() - batch.num_rows();
            self.fetch.replace(fetch);
            Some(batch)
        } else if &batch.num_rows() >= self.fetch.as_ref().unwrap() {
            let batch_rows = self.fetch.take().unwrap();
            self.fetch.replace(0);
            self.message_stream = None; // Clear input so it can be dropped early

            // It is guaranteed that batch_rows is <= batch.num_rows
            Some(batch.slice(0, batch_rows))
        } else {
            unreachable!()
        }
    }
}

impl Stream for LimitStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
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
            let config_table = Subject::get_builder()
                .with_name("config")
                .with_record_batches(batches)?
                .build()?;
            self.init_config(config_table)?;
        }

        // Fetch rows
        let fetch_started = self.skip.as_ref().unwrap() == &0;
        match &mut self.message_stream {
            Some(input) => {
                let poll = if fetch_started {
                    input.poll_next_unpin(cx)
                } else {
                    self.poll_and_skip(cx)
                };

                // Record the poll
                let poll = poll.map(|x| match x {
                    Some(Ok(batch)) => Ok(self.stream_limit(batch)).transpose(),
                    other => other,
                });
                if let Some(baseline_metrics) = &baseline_metrics {
                    baseline_metrics.record_poll(poll)
                } else {
                    poll
                }
            }
            // Input has been cleared
            None => Poll::Ready(None),
        }
    }
}

impl RecordBatchStream for LimitStream {
    /// Get the schema
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use crate::LimitConfig;

    use super::*;
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatchOptions;
    use futures::{Stream, TryStreamExt};
    use phymes_core::{SubjectBuilder, SubjectTrait, test_subject};
    use phymes_diagnostics::{Diagnostics, SpanBuilder};
    use phymes_event::Publication;

    /// Return a RecordBatch with a single Int32 array with values (0..sz) in a field named "i"
    fn make_partition(sz: i32) -> RecordBatch {
        let seq_start = 0;
        let seq_end = sz;
        let values = (seq_start..seq_end).collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, true)]));
        let arr = Arc::new(Int32Array::from(values));
        let arr = arr as ArrayRef;

        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    /// Return a RecordBatch with a single array with row_count sz
    fn make_batch_no_column(sz: usize) -> RecordBatch {
        let schema = Arc::new(Schema::empty());

        let options = RecordBatchOptions::new().with_row_count(Option::from(sz));
        RecordBatch::try_new_with_options(schema, vec![], &options).unwrap()
    }

    /// Index into the data that has been returned so far
    #[derive(Debug, Default, Clone)]
    pub struct BatchIndex {
        inner: Arc<std::sync::Mutex<usize>>,
    }

    impl BatchIndex {
        /// Return the current index
        pub fn value(&self) -> usize {
            let inner = self.inner.lock().unwrap();
            *inner
        }

        // increment the current index by one
        pub fn incr(&self) {
            let mut inner = self.inner.lock().unwrap();
            *inner += 1;
        }
    }

    /// Iterator over batches
    #[derive(Debug, Default)]
    pub struct TestStream {
        /// Vector of record batches
        data: Vec<RecordBatch>,
        /// Index into the data that has been returned so far
        index: BatchIndex,
    }

    impl TestStream {
        /// Create an iterator for a vector of record batches. Assumes at
        /// least one entry in data (for the schema)
        pub fn new(data: Vec<RecordBatch>) -> Self {
            Self {
                data,
                ..Default::default()
            }
        }

        /// Return a handle to the index counter for this stream
        pub fn index(&self) -> BatchIndex {
            self.index.clone()
        }
    }

    impl Stream for TestStream {
        type Item = Result<RecordBatch>;

        fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let next_batch = self.index.value();

            Poll::Ready(if next_batch < self.data.len() {
                let next_batch = self.index.value();
                self.index.incr();
                Some(Ok(self.data[next_batch].clone()))
            } else {
                None
            })
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.data.len(), Some(self.data.len()))
        }
    }

    impl RecordBatchStream for TestStream {
        /// Get the schema
        fn schema(&self) -> SchemaRef {
            self.data[0].schema()
        }
    }

    #[tokio::test]
    async fn limit_early_shutdown() -> Result<()> {
        let batches = vec![
            make_partition(5),
            make_partition(10),
            make_partition(15),
            make_partition(20),
            make_partition(25),
        ];
        let input = TestStream::new(batches);

        let index = input.index();
        assert_eq!(index.value(), 0);

        // Make the config
        let config = LimitConfig {
            skip: Some(0),
            fetch: 6,
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name("LimitStream")
            .with_json(&config_json, 1)?
            .build()?;

        // Make the diagnostics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the Runtime Env
        let runtime_env = Arc::new(RuntimeEnv {
            name: "service".to_string(),
            ..Default::default()
        });

        // Limit of six needs to consume the entire first record batch
        // (5 rows) and 1 row from the second (1 row)
        let limit_stream = LimitStream::new(
            Box::pin(input),
            config_table.to_record_batch_stream(),
            runtime_env,
            Some(diagnostic_builder),
        );
        assert_eq!(index.value(), 0);

        let results = Box::pin(limit_stream).try_collect::<Vec<_>>().await?;
        let num_rows: usize = results.into_iter().map(|b| b.num_rows()).sum();
        // Only 6 rows should have been produced
        assert_eq!(num_rows, 6);

        // Only the first two batches should be consumed
        assert_eq!(index.value(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn limit_equals_batch_size() -> Result<()> {
        let batches = vec![make_partition(6), make_partition(6), make_partition(6)];
        let input = TestStream::new(batches);

        let index = input.index();
        assert_eq!(index.value(), 0);

        // Make the config
        let config = LimitConfig {
            skip: Some(0),
            fetch: 6,
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name("LimitStream")
            .with_json(&config_json, 1)?
            .build()?;

        // Make the diagnostics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the Runtime Env
        let runtime_env = Arc::new(RuntimeEnv {
            name: "service".to_string(),
            ..Default::default()
        });

        // Limit of six needs to consume the entire first record batch
        // (6 rows) and stop immediately
        let limit_stream = LimitStream::new(
            Box::pin(input),
            config_table.to_record_batch_stream(),
            runtime_env,
            Some(diagnostic_builder),
        );
        assert_eq!(index.value(), 0);

        let results = Box::pin(limit_stream).try_collect::<Vec<_>>().await?;
        let num_rows: usize = results.into_iter().map(|b| b.num_rows()).sum();
        // Only 6 rows should have been produced
        assert_eq!(num_rows, 6);

        // Only the first batch should be consumed
        assert_eq!(index.value(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn limit_no_column() -> Result<()> {
        let batches = vec![
            make_batch_no_column(6),
            make_batch_no_column(6),
            make_batch_no_column(6),
        ];
        let input = TestStream::new(batches);

        let index = input.index();
        assert_eq!(index.value(), 0);

        // Make the config
        let config = LimitConfig {
            skip: Some(0),
            fetch: 6,
        };
        let config_json = serde_json::to_vec(&config)?;
        let config_table = SubjectBuilder::new()
            .with_name("LimitStream")
            .with_json(&config_json, 1)?
            .build()?;

        // Make the diagnostics
        let span = SpanBuilder::default().with_span("test").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);

        // Make the Runtime Env
        let runtime_env = Arc::new(RuntimeEnv {
            name: "service".to_string(),
            ..Default::default()
        });

        // Limit of six needs to consume the entire first record batch
        // (6 rows) and stop immediately
        let limit_stream = LimitStream::new(
            Box::pin(input),
            config_table.to_record_batch_stream(),
            runtime_env,
            Some(diagnostic_builder),
        );
        assert_eq!(index.value(), 0);

        let results = Box::pin(limit_stream).try_collect::<Vec<_>>().await?;
        let num_rows: usize = results.into_iter().map(|b| b.num_rows()).sum();
        // Only 6 rows should have been produced
        assert_eq!(num_rows, 6);

        // Only the first batch should be consumed
        assert_eq!(index.value(), 1);

        Ok(())
    }
}
