use anyhow::{Result, anyhow};
use phymes_core::{MappableTrait, RecordBatchStream, RuntimeEnv, SendableRecordBatchStream};
use phymes_diagnostics::DiagnosticBuilder;
use phymes_message::{
    SendableRecordBatchStreamMessageBuilderMap, SendableRecordBatchStreamMessageMap,
};
use std::fmt::Debug;
use std::sync::Arc;
use tracing::{Level, event};

/// Trait that performs the actual processing
///   and designed to allow for chaining multiple processors into streaming computational trees
pub trait ProcessorTrait: MappableTrait + Send + Sync + Debug {
    /// New processor
    fn new(name: &str, r#type: &str) -> Self
    where
        Self: Sized;

    /// The type used to identify the processor after dynamic dispatching
    /// often just an alias for `get_static_name`
    fn get_type(&self) -> &str;

    /// Trace information `(line!(), file!().to_string())`
    fn line_and_file(&self) -> (u32, String);

    /// Run the `Process`, returning a [`Stream`] of [`RecordBatch`]es.
    ///
    /// [`RecordBatch`]: arrow::record_batch::RecordBatch
    ///
    /// # Process execution
    ///
    /// The `process` method itself is not `async` but it returns an `async`
    /// [`futures::stream::Stream`]. This `Stream` should incrementally compute
    /// the output, `RecordBatch` by `RecordBatch` (in a streaming fashion).
    /// Most `Processor`s should not do any work before the first
    /// `RecordBatch` is requested from the stream.
    ///
    /// [`RecordBatchStreamAdapter`] can be used to convert an `async`
    /// [`Stream`] into a [`SendableRecordBatchStream`].
    ///
    /// Using `async` `Streams` allows for network I/O during execution and
    /// takes advantage of Rust's built in support for `async` continuations and
    /// crate ecosystem.
    ///
    /// [`Stream`]: futures::stream::Stream
    /// [`StreamExt`]: futures::stream::StreamExt
    /// [`TryStreamExt`]: futures::stream::TryStreamExt
    /// [`RecordBatchStreamAdapter`]: crate::RecordBatchStreamAdapter
    ///
    /// # Error handling
    ///
    /// Any error that occurs during execution is sent as an `Err` in the output stream.
    ///
    /// `Task` implementations cancel additional work immediately once an error occurs.
    /// The rationale is that if the overall task will return an error, any additional work such as continued
    /// polling of inputs will be wasted as it will be thrown away.
    ///
    /// # Cancellation / Aborting Execution
    ///
    /// The [`Stream`] that is returned must ensure that any allocated resources
    /// are freed when the stream itself is dropped. This is particularly
    /// important for [`spawn`]ed tasks or threads. Unless care is taken to
    /// "abort" such tasks, they may continue to consume resources even after
    /// the plan is dropped, generating intermediate results that are never
    /// used.
    /// See `join_message_streams` in `SessionStreamStep` for a safe usage of [`spawn`]
    ///
    /// For more details see [`JoinSet`] and [`RecordBatchReceiverStreamBuilder`]
    /// for structures to help ensure all background tasks are cancelled.
    ///
    /// [`spawn`]: tokio::task::spawn
    /// [`JoinSet`]: tokio::task::JoinSet
    /// [`RecordBatchReceiverStreamBuilder`]: crate::RecordBatchReceiverStreamBuilder
    ///
    /// # Messages handling
    ///
    /// Each object that implements a [`ProcessorTrait`] is responsible for the logic used to extract, transform, and publish messages.
    ///
    /// Each processor should subscribe to a message of the same name (i.e., a `config`) that provides the parameters of the processor execution.
    /// DM: todo!() better description of how to implement custom processors
    ///
    /// # Implementation Examples
    /// DM: todo!() update according to the new interface
    ///
    /// While `async` `Stream`s have a non trivial learning curve, the
    /// [`futures`] crate provides [`StreamExt`] and [`TryStreamExt`]
    /// which help simplify many common operations.
    ///
    /// Here are some common patterns:
    ///
    /// ## Return Precomputed `RecordBatch`
    ///
    /// We can return a precomputed `RecordBatch` as a `Stream`:
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow::array::RecordBatch;
    /// # use arrow::datatypes::SchemaRef;
    /// # use anyhow::Result;
    /// # use phymes_core::SendableRecordBatchStream;
    /// # use phymes_core::RecordBatchStreamAdapter;
    ///
    /// struct MyProcessor {
    ///     batch: RecordBatch,
    /// }
    ///
    /// impl MyProcessor {
    ///     fn process(
    ///         &self) -> Result<SendableRecordBatchStream> {
    ///         // use functions from futures crate convert the batch into a stream
    ///         let fut = futures::future::ready(Ok(self.batch.clone()));
    ///         let stream = futures::stream::once(fut);
    ///         Ok(Box::pin(RecordBatchStreamAdapter::new(self.batch.schema(), stream)))
    ///     }
    /// }
    /// ```
    ///
    /// ## Lazily (async) Compute `RecordBatch`
    ///
    /// We can also lazily compute a `RecordBatch` when the returned `Stream` is polled
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow::array::RecordBatch;
    /// # use arrow::datatypes::SchemaRef;
    /// # use anyhow::Result;
    /// # use phymes_core::SendableRecordBatchStream;
    /// # use phymes_core::RecordBatchStreamAdapter;
    ///
    /// struct MyProcessor {
    ///     schema: SchemaRef,
    /// }
    ///
    /// /// Returns a single batch when the returned stream is polled
    /// async fn get_batch() -> Result<RecordBatch> {
    ///     todo!()
    /// }
    ///
    /// impl MyProcessor {
    ///     fn process(
    ///         &self) -> Result<SendableRecordBatchStream> {
    ///         let fut = get_batch();
    ///         let stream = futures::stream::once(fut);
    ///         Ok(Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream)))
    ///     }
    /// }
    /// ```
    ///
    /// ## Lazily (async) create a Stream
    ///
    /// If you need to create the return `Stream` using an `async` function,
    /// you can do so by flattening the result:
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use arrow::array::RecordBatch;
    /// # use arrow::datatypes::SchemaRef;
    /// # use futures::TryStreamExt;
    /// # use anyhow::Result;
    /// # use phymes_core::SendableRecordBatchStream;
    /// # use phymes_core::RecordBatchStreamAdapter;
    ///
    /// struct MyProcessor {
    ///     schema: SchemaRef,
    /// }
    ///
    /// /// async function that returns a stream
    /// async fn get_batch_stream() -> Result<SendableRecordBatchStream> {
    ///     todo!()
    /// }
    ///
    /// impl MyProcessor {
    ///     fn process(
    ///         &self) -> Result<SendableRecordBatchStream> {
    ///         // A future that yields a stream
    ///         let fut = get_batch_stream();
    ///         // Use TryStreamExt::try_flatten to flatten the stream of streams
    ///         let stream = futures::stream::once(fut).try_flatten();
    ///         Ok(Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream)))
    ///     }
    /// }
    /// ```
    ///
    fn process(
        &self,
        message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        runtime_env: Arc<RuntimeEnv>,
    ) -> Result<SendableRecordBatchStreamMessageBuilderMap>;
}

/// Mock objects and functions for processor testing
pub mod test_processor {
    use super::*;

    use arrow::{array::RecordBatch, compute::concat_batches, datatypes::SchemaRef};
    use futures::{Stream, StreamExt};
    use phymes_core::{BuildableTrait, BuilderTrait, test_subject::make_test_record_batch};
    use phymes_diagnostics::{DiagnosticBuilderTrait, HashMap, MetricBuilderTrait};
    use phymes_message::{
        MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
        SendableRecordBatchStreamMessageBuilder,
    };
    use std::{
        pin::Pin,
        sync::Arc,
        task::{Context, Poll, ready},
    };

    /// Mock processor that adds an additional record batch
    #[derive(Debug)]
    pub struct ProcessorMock {
        name: String,
        r#type: String,
    }

    impl MappableTrait for ProcessorMock {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl ProcessorTrait for ProcessorMock {
        fn new(name: &str, r#type: &str) -> Self {
            Self {
                name: name.to_string(),
                r#type: r#type.to_string(),
            }
        }

        fn get_type(&self) -> &str {
            &self.r#type
        }

        fn line_and_file(&self) -> (u32, String) {
            (line!(), file!().to_string())
        }

        fn process(
            &self,
            message: SendableRecordBatchStreamMessageMap,
            diagnostic_builder: Option<&DiagnosticBuilder>,
            _runtime_env: Arc<RuntimeEnv>,
        ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
            event!(Level::INFO, "Starting processor {}", self.get_name());

            // Add another record batch to the input
            let mut builder_map = HashMap::<String, SendableRecordBatchStreamMessageBuilder>::new();
            for (s_name, s) in message.into_iter() {
                if s.get_subject() == self.get_name() {
                    continue;
                }
                let out = Box::pin(ProcessorMockStream {
                    schema: s.get_message().schema(),
                    input: s.get_message_own(),
                    diagnostic_builder: diagnostic_builder.cloned(),
                });
                let out_m = SendableRecordBatchStreamMessage::get_builder()
                    .with_name(s_name.as_str())
                    .with_message(out);
                let _ = builder_map.insert(s_name, out_m);
            }

            Ok(builder_map)
        }
    }

    struct ProcessorMockStream {
        /// Output schema after the projection
        schema: SchemaRef,
        /// The input task to process.
        input: SendableRecordBatchStream,
        /// Runtime metrics recording
        diagnostic_builder: Option<DiagnosticBuilder>,
    }

    fn add_test_table_row(
        batch: RecordBatch,
        // could also be other arguments required for processing
    ) -> Result<RecordBatch> {
        let new_data = make_test_record_batch(1, 8)?;
        if new_data.schema().eq(&batch.schema()) {
            let concatenated = concat_batches(&batch.schema(), &vec![batch, new_data])?;
            Ok(concatenated)
        } else {
            Ok(batch)
        }
    }

    impl Stream for ProcessorMockStream {
        type Item = Result<RecordBatch>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let poll;
            let baseline_metrics = if let Some(diagnostic_builder) = &self.diagnostic_builder {
                Some(
                    diagnostic_builder
                        .clone()
                        .to_child("ProcessorMockStream")?
                        .baseline_metrics(line!(), file!(), "poll_next"),
                )
            } else {
                None
            };
            #[allow(clippy::never_loop)]
            loop {
                match ready!(self.input.poll_next_unpin(cx)) {
                    Some(Ok(batch)) => {
                        let timer = baseline_metrics
                            .as_ref()
                            .map(|baseline_metrics| baseline_metrics.elapsed_compute().timer());
                        let processed_batch = add_test_table_row(batch)?;
                        if let Some(timer) = timer {
                            timer.done();
                        }
                        poll = Poll::Ready(Some(Ok(processed_batch)));
                        break;
                    }
                    value => {
                        poll = Poll::Ready(value);
                        break;
                    }
                }
            }
            if let Some(baseline_metrics) = baseline_metrics {
                baseline_metrics.record_poll(poll)
            } else {
                poll
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            // Same number of record batches
            self.input.size_hint()
        }
    }

    impl RecordBatchStream for ProcessorMockStream {
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
    }

    /// Error processor that emits an error
    #[derive(Debug)]
    pub struct ProcessorError {
        name: String,
        r#type: String,
    }

    impl MappableTrait for ProcessorError {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl ProcessorTrait for ProcessorError {
        fn new(name: &str, r#type: &str) -> Self {
            Self {
                name: name.to_string(),
                r#type: r#type.to_string(),
            }
        }

        fn get_type(&self) -> &str {
            &self.r#type
        }

        fn line_and_file(&self) -> (u32, String) {
            (line!(), file!().to_string())
        }

        fn process(
            &self,
            _message: SendableRecordBatchStreamMessageMap,
            _diagnostic_builder: Option<&DiagnosticBuilder>,
            _runtime_env: Arc<RuntimeEnv>,
        ) -> Result<SendableRecordBatchStreamMessageBuilderMap> {
            Err(anyhow!("This is an error!"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use phymes_core::{
        BuildableTrait, BuilderTrait, RuntimeEnv, SubjectBuilder, SubjectBuilderTrait,
        SubjectTrait, test_subject::make_test_subject,
    };
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, HashMap, SpanBuilder};
    use phymes_event::Publication;
    use phymes_message::{MessageBuilderTrait, SendableRecordBatchStreamMessage};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_processor() -> Result<()> {
        let span = SpanBuilder::default().with_span("test_processor").build()?;
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics).with_span(&span);
        let runtime_env = RuntimeEnv::default();
        let name = "process_1".to_string();
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            name.clone(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(name.clone().as_str())
                .with_publisher("s1")
                .with_subject("test_table")
                .with_update(&Publication::Extend {
                    subject_name: "test_table".to_string(),
                })
                .with_message(make_test_subject("test_table", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );
        let processor_1 = test_processor::ProcessorMock::new(
            "processor_1",
            test_processor::ProcessorMock::get_static_name(),
        );
        let mut stream =
            processor_1.process(message, Some(&diagnostic_builder), Arc::new(runtime_env))?;
        let partitions = SubjectBuilder::new_from_sendable_record_batch_stream(
            stream.remove(&name).unwrap().message.take().unwrap(),
        )
        .await?
        .with_name("test_message_table")
        .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 15);
        Ok(())
    }
}
