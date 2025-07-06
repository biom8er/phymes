//! Stream wrappers for physical operators

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::{Result, anyhow};

use crate::metrics::BaselineMetrics;
use crate::table::stream::{RecordBatchStream, SendableRecordBatchStream};
use crate::task::test_exec::SendableRecordBatchExecTrait;

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use futures::stream::BoxStream;
use futures::{Future, Stream, StreamExt};
use pin_project_lite::pin_project;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinSet;
use tracing::{Level, event, instrument};

/// Creates a stream from a collection of producing tasks, routing panics to the stream.
///
/// Note that this is similar to  [`ReceiverStream` from tokio-stream], with the differences being:
///
/// 1. Methods to bound and "detach"  tasks (`spawn()` and `spawn_blocking()`).
///
/// 2. Propagates panics, whereas the `tokio` version doesn't propagate panics to the receiver.
///
/// 3. Automatically cancels any outstanding tasks when the receiver stream is dropped.
///
/// [`ReceiverStream` from tokio-stream]: https://docs.rs/tokio-stream/latest/tokio_stream/wrappers/struct.ReceiverStream.html
#[derive(Debug)]
pub(crate) struct ReceiverStreamBuilder<O> {
    tx: Sender<Result<O>>,
    rx: Receiver<Result<O>>,
    join_set: JoinSet<Result<()>>,
}

impl<O: Send + 'static + std::fmt::Debug> ReceiverStreamBuilder<O> {
    /// Create new channels with the specified buffer size
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(capacity);

        Self {
            tx,
            rx,
            join_set: JoinSet::new(),
        }
    }

    /// Get a handle for sending data to the output
    pub fn tx(&self) -> Sender<Result<O>> {
        self.tx.clone()
    }

    /// Spawn task that will be aborted if this builder (or the stream
    /// built from it) are dropped
    pub fn spawn<F>(&mut self, task: F)
    where
        F: Future<Output = Result<()>>,
        F: Send + 'static,
    {
        self.join_set.spawn(task);
    }

    /// Spawn a blocking task that will be aborted if this builder (or the stream
    /// built from it) are dropped.
    ///
    /// This is often used to spawn tasks that write to the sender
    /// retrieved from `Self::tx`.
    pub fn spawn_blocking<F>(&mut self, f: F)
    where
        F: FnOnce() -> Result<()>,
        F: Send + 'static,
    {
        self.join_set.spawn_blocking(f);
    }

    /// Create a stream of all data written to `tx`
    #[instrument(level = "trace")]
    pub fn build(self) -> BoxStream<'static, Result<O>> {
        let Self {
            tx,
            rx,
            mut join_set,
        } = self;

        // Doesn't need tx
        drop(tx);

        // future that checks the result of the join set, and propagates panic if seen
        let check = async move {
            while let Some(result) = join_set.join_next().await {
                match result {
                    Ok(task_result) => {
                        match task_result {
                            // Nothing to report
                            Ok(_) => continue,
                            // This means a blocking task error
                            Err(error) => return Some(Err(error)),
                        }
                    }
                    // This means a tokio task error, likely a panic
                    Err(e) => {
                        if e.is_panic() {
                            // resume on the main thread
                            std::panic::resume_unwind(e.into_panic());
                        } else {
                            // This should only occur if the task is
                            // cancelled, which would only occur if
                            // the JoinSet were aborted, which in turn
                            // would imply that the receiver has been
                            // dropped and this code is not running
                            return Some(Err(anyhow!("Non Panic Task error: {e}")));
                        }
                    }
                }
            }
            None
        };

        let check_stream = futures::stream::once(check)
            // unwrap Option / only return the error
            .filter_map(|item| async move { item });

        // Convert the receiver into a stream
        let rx_stream = futures::stream::unfold(rx, |mut rx| async move {
            let next_item = rx.recv().await;
            next_item.map(|next_item| (next_item, rx))
        });

        // Merge the streams together so whichever is ready first
        // produces the batch
        futures::stream::select(rx_stream, check_stream).boxed()
    }
}

/// Builder for `RecordBatchReceiverStream` that propagates errors
/// and panic's correctly.
///
/// [`RecordBatchReceiverStreamBuilder`] is used to spawn one or more tasks
/// that produce [`RecordBatch`]es and send them to a single
/// `Receiver` which can improve parallelism.
///
/// This also handles propagating panic`s and canceling the tasks.
///
/// # Example
///
/// The following example spawns 2 tasks that will write [`RecordBatch`]es to
/// the `tx` end of the builder, after building the stream, we can receive
/// those batches with calling `.next()`
///
/// ```
/// # use std::sync::Arc;
/// # use arrow::datatypes::{Schema, Field, DataType};
/// # use arrow::array::RecordBatch;
/// # use phymes_core::table::stream::SendableRecordBatchStream;
/// # use phymes_core::table::stream_adapter::RecordBatchReceiverStreamBuilder;
/// # use futures::stream::StreamExt;
/// # use tokio::runtime::Builder;
/// # let rt = Builder::new_current_thread().build().unwrap();
/// #
/// # rt.block_on(async {
/// let schema = Arc::new(Schema::new(vec![Field::new("foo", DataType::Int8, false)]));
/// let mut builder = RecordBatchReceiverStreamBuilder::new(Arc::clone(&schema), 10);
///
/// // task 1
/// let tx_1 = builder.tx();
/// let schema_1 = Arc::clone(&schema);
/// builder.spawn(async move {
///     // Your task needs to send batches to the tx
///     tx_1.send(Ok(RecordBatch::new_empty(schema_1))).await.unwrap();
///
///     Ok(())
/// });
///
/// // task 2
/// let tx_2 = builder.tx();
/// let schema_2 = Arc::clone(&schema);
/// builder.spawn(async move {
///     // Your task needs to send batches to the tx
///     tx_2.send(Ok(RecordBatch::new_empty(schema_2))).await.unwrap();
///
///     Ok(())
/// });
///
/// let mut stream = builder.build();
/// while let Some(res_batch) = stream.next().await {
///     // `res_batch` can either from task 1 or 2
///
///     // do something with `res_batch`
/// }
/// # });
/// ```
#[derive(Debug)]
pub struct RecordBatchReceiverStreamBuilder {
    schema: SchemaRef,
    inner: ReceiverStreamBuilder<RecordBatch>,
}

impl RecordBatchReceiverStreamBuilder {
    /// Create new channels with the specified buffer size
    pub fn new(schema: SchemaRef, capacity: usize) -> Self {
        Self {
            schema,
            inner: ReceiverStreamBuilder::new(capacity),
        }
    }

    /// Get a handle for sending [`RecordBatch`] to the output
    pub fn tx(&self) -> Sender<Result<RecordBatch>> {
        self.inner.tx()
    }

    /// Spawn task that will be aborted if this builder (or the stream
    /// built from it) are dropped
    ///
    /// This is often used to spawn tasks that write to the sender
    /// retrieved from [`Self::tx`], for examples, see the document
    /// of this type.
    pub fn spawn<F>(&mut self, task: F)
    where
        F: Future<Output = Result<()>>,
        F: Send + 'static,
    {
        self.inner.spawn(task)
    }

    /// Spawn a blocking task that will be aborted if this builder (or the stream
    /// built from it) are dropped
    ///
    /// This is often used to spawn tasks that write to the sender
    /// retrieved from [`Self::tx`], for examples, see the document
    /// of this type.
    pub fn spawn_blocking<F>(&mut self, f: F)
    where
        F: FnOnce() -> Result<()>,
        F: Send + 'static,
    {
        self.inner.spawn_blocking(f)
    }

    /// Runs the `partition` of the `input` ExecutionPlan on the
    /// tokio thread pool and writes its outputs to this stream
    ///
    /// If the input partition produces an error, the error will be
    /// sent to the output stream and no further results are sent.
    #[instrument(level = "trace")]
    pub fn run_input(&mut self, input: Arc<dyn SendableRecordBatchExecTrait>) {
        let output = self.tx();

        self.inner.spawn(async move {
            let mut stream = match input.run() {
                Err(e) => {
                    // If send fails, the plan being torn down, there
                    // is no place to send the error and no reason to continue.
                    output.send(Err(e)).await.ok();

                    event!(
                        Level::DEBUG,
                        "Stopping execution: error executing input: {}",
                        input.get_name()
                    );
                    return Ok(());
                }
                Ok(stream) => stream,
            };

            // Transfer batches from inner stream to the output tx
            // immediately.
            while let Some(item) = stream.next().await {
                let is_err = item.is_err();

                // If send fails, plan being torn down, there is no
                // place to send the error and no reason to continue.
                if output.send(item).await.is_err() {
                    event!(
                        Level::DEBUG,
                        "Stopping execution: output is gone, plan cancelling: {}",
                        input.get_name()
                    );
                    return Ok(());
                }

                // Stop after the first error is encountered (Don't
                // drive all streams to completion)
                if is_err {
                    event!(
                        Level::DEBUG,
                        "Stopping execution: plan returned error: {}",
                        input.get_name()
                    );
                    return Ok(());
                }
            }

            Ok(())
        });
    }

    /// Create a stream of all [`RecordBatch`] written to `tx`
    pub fn build(self) -> SendableRecordBatchStream {
        Box::pin(RecordBatchStreamAdapter::new(
            self.schema,
            self.inner.build(),
        ))
    }
}

#[doc(hidden)]
pub struct RecordBatchReceiverStream {}

impl RecordBatchReceiverStream {
    /// Create a builder with an internal buffer of capacity batches.
    pub fn builder(schema: SchemaRef, capacity: usize) -> RecordBatchReceiverStreamBuilder {
        RecordBatchReceiverStreamBuilder::new(schema, capacity)
    }
}

pin_project! {
    /// Combines a [`Stream`] with a [`SchemaRef`] implementing
    /// [`SendableRecordBatchStream`] for the combination
    ///
    /// See [`Self::new`] for an example
    pub struct RecordBatchStreamAdapter<S> {
        schema: SchemaRef,

        #[pin]
        stream: S,
    }
}

impl<S> RecordBatchStreamAdapter<S> {
    /// Creates a new [`RecordBatchStreamAdapter`] from the provided schema and stream.
    ///
    /// Note to create a [`SendableRecordBatchStream`] you pin the result
    ///
    /// # Example
    /// ```
    /// # use arrow::datatypes::Schema;
    /// # use arrow::datatypes::Field;
    /// # use arrow::datatypes::DataType;
    /// # use arrow::array::Int32Array;
    /// # use arrow::array::Float64Array;
    /// # use arrow::array::ArrayRef;
    /// # use arrow::record_batch::RecordBatch;
    /// # use std::sync::Arc;
    /// # use phymes_core::table::stream::SendableRecordBatchStream;
    /// # use phymes_core::table::stream_adapter::RecordBatchStreamAdapter;
    /// // Create stream of Result<RecordBatch>
    /// let int_arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    /// let float_arr: ArrayRef = Arc::new(Float64Array::from(vec![Some(4.0), None, Some(5.0)]));
    /// let batch = RecordBatch::try_from_iter(vec![("a", int_arr), ("b", float_arr)]
    /// ).expect("created batch");
    /// let schema = batch.schema();
    /// let stream = futures::stream::iter(vec![Ok(batch)]);
    /// // Convert the stream to a SendableRecordBatchStream
    /// let adapter = RecordBatchStreamAdapter::new(schema, stream);
    /// // Now you can use the adapter as a SendableRecordBatchStream
    /// let batch_stream: SendableRecordBatchStream = Box::pin(adapter);
    /// // ...
    /// ```
    pub fn new(schema: SchemaRef, stream: S) -> Self {
        Self { schema, stream }
    }
}

impl<S> std::fmt::Debug for RecordBatchStreamAdapter<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordBatchStreamAdapter")
            .field("schema", &self.schema)
            .finish()
    }
}

impl<S> Stream for RecordBatchStreamAdapter<S>
where
    S: Stream<Item = Result<RecordBatch>>,
{
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.project().stream.poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.stream.size_hint()
    }
}

impl<S> RecordBatchStream for RecordBatchStreamAdapter<S>
where
    S: Stream<Item = Result<RecordBatch>>,
{
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// `EmptyRecordBatchStream` can be used to create a [`RecordBatchStream`]
/// that will produce no results
pub struct EmptyRecordBatchStream {
    /// Schema wrapped by Arc
    schema: SchemaRef,
}

impl EmptyRecordBatchStream {
    /// Create an empty RecordBatchStream
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }
}

impl RecordBatchStream for EmptyRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Stream for EmptyRecordBatchStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }
}

/// Stream wrapper that records `BaselineMetrics` for a particular
/// `[SendableRecordBatchStream]` (likely a partition)
pub struct ObservedStream {
    inner: SendableRecordBatchStream,
    baseline_metrics: BaselineMetrics,
}

impl ObservedStream {
    pub fn new(inner: SendableRecordBatchStream, baseline_metrics: BaselineMetrics) -> Self {
        Self {
            inner,
            baseline_metrics,
        }
    }
}

impl RecordBatchStream for ObservedStream {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl Stream for ObservedStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll = self.inner.poll_next_unpin(cx);
        self.baseline_metrics.record_poll(poll)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    use crate::task::test_exec::BlockingExec;
    use crate::task::test_exec::{MockExec, PanicExecWrapper};

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    use crate::task::test_exec::assert_strong_count_converges_to_zero;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, true)]))
    }

    #[tokio::test]
    #[should_panic(expected = "PanickingStream did panic")]
    async fn record_batch_receiver_stream_propagates_panics() {
        let schema = schema();

        let num_partitions = 3;
        let input = PanicExecWrapper::new(Arc::clone(&schema), num_partitions)
            .with_partition_panic(0, 0)
            .with_partition_panic(1, 0)
            .with_partition_panic(2, 0);
        consume(input, 3).await
    }

    #[tokio::test]
    #[should_panic(expected = "PanickingStream did panic: 1")]
    async fn record_batch_receiver_stream_propagates_panics_early_shutdown() {
        let schema = schema();

        // Make 2 partitions, second partition panics before the first
        let num_partitions = 2;
        let input = PanicExecWrapper::new(Arc::clone(&schema), num_partitions)
            .with_partition_panic(0, 10)
            .with_partition_panic(1, 3); // partition 1 should panic first (after 3 )

        // Ensure that the panic results in an early shutdown (that
        // everything stops after the first panic).

        // Since the stream reads every other batch: (0,1,0,1,0,panic)
        // so should not exceed 5 batches prior to the panic
        let max_batches = 5;
        consume(input, max_batches).await
    }

    #[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
    #[tokio::test]
    async fn record_batch_receiver_stream_drop_cancel() {
        let schema = schema();

        // Make an input that never proceeds
        let input = BlockingExec::new(Arc::clone(&schema), 1);
        let refs = input.refs();

        // Configure a RecordBatchReceiverStream to consume the input
        let mut builder = RecordBatchReceiverStream::builder(schema, 2);
        builder.run_input(Arc::new(input));
        let stream = builder.build();

        // Input should still be present
        assert!(std::sync::Weak::strong_count(&refs) > 0);

        // Drop the stream, ensure the refs go to zero
        drop(stream);
        assert_strong_count_converges_to_zero(refs).await;
    }

    #[tokio::test]
    /// Ensure that if an error is received in one stream, the
    /// `RecordBatchReceiverStream` stops early and does not drive
    /// other streams to completion.
    async fn record_batch_receiver_stream_error_does_not_drive_completion() {
        let schema = schema();

        // make an input that will error twice
        let error_stream = MockExec::new(
            vec![Err(anyhow!("Test1")), Err(anyhow!("Test2"))],
            Arc::clone(&schema),
        )
        .with_use_task(false);

        let mut builder = RecordBatchReceiverStream::builder(schema, 2);
        builder.run_input(Arc::new(error_stream));
        let mut stream = builder.build();

        // Get the first result, which should be an error
        let first_batch = stream.next().await.unwrap();
        assert!(first_batch.is_err());
        let _first_err = first_batch.unwrap_err();
        //assert_eq!(first_err.strip_backtrace(), "Execution error: Test1");

        // There should be no more batches produced (should not get the second error)
        assert!(stream.next().await.is_none());
    }

    /// Consumes all the input's partitions into a
    /// RecordBatchReceiverStream and runs it to completion
    ///
    /// panic's if more than max_batches is seen,
    async fn consume(input: PanicExecWrapper, max_batches: usize) {
        let input = Arc::new(input);
        let num_partitions = input.get_partition_count();

        // Configure a RecordBatchReceiverStream to consume all the input partitions
        let mut builder = RecordBatchReceiverStream::builder(input.get_schema(), num_partitions);
        for partition in 0..num_partitions {
            builder.run_input(Arc::clone(&input.get_partition(partition))
                as Arc<dyn SendableRecordBatchExecTrait>);
        }
        let mut stream = builder.build();

        // Drain the stream until it is complete, panic'ing on error
        let mut num_batches = 0;
        while let Some(next) = stream.next().await {
            next.unwrap();
            num_batches += 1;
            assert!(
                num_batches < max_batches,
                "Got the limit of {num_batches} batches before seeing panic"
            );
        }
    }
}
