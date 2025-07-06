use crate::{
    metrics::ArrowTaskMetricsSet,
    session::{
        common_traits::{MappableTrait, OutgoingMessageMap},
        runtime_env::RuntimeEnv,
    },
    table::{
        arrow_table_publish::ArrowTablePublish,
        arrow_table_subscribe::ArrowTableSubscribe,
        stream::{RecordBatchStream, SendableRecordBatchStream},
    },
};
use anyhow::Result;
use parking_lot::Mutex;
use std::fmt::Debug;
use std::sync::Arc;
use tracing::{Level, event};

/// For inner task objects that perform the actual processing
/// and designed to allow for chaining multiple processors
/// into streaming computational tree
pub trait ArrowProcessorTrait: MappableTrait + Send + Sync + Debug {
    /// New processor
    ///
    /// # Notes
    ///
    /// The builder pattern is bypassed in favor
    /// of a simple initializer with options for members
    /// who are not always required depending upon the users implementation
    ///
    /// # Examples
    /// ## 1. Chaining processing steps
    ///
    /// Initialize with `input` that is called with the `message`
    /// Process with `metrics` to record each processor
    /// Process with `message` and define a processor that operates
    /// over individual `RecordBatch`es as they are polled
    ///
    /// ## 2. Streaming response
    ///
    /// Process with `message` and define a processor that returns
    /// a stream of `RecordBatches` via a receiver wrapped into a future
    /// Process with `metrics` to record the processor
    ///
    /// ## 3. Remote RPC call
    /// Process with `message` and make an RPC call
    /// that returns a stream or batch of `RecordBatch`es
    fn new_arc(name: &str) -> Arc<dyn ArrowProcessorTrait>
    where
        Self: Sized;

    /// Get publications
    fn get_publications(&self) -> &[ArrowTablePublish];

    /// Get subscriptions
    fn get_subscriptions(&self) -> &[ArrowTableSubscribe];

    /// Get forwarded subscriptions
    fn get_forward_subscriptions(&self) -> &[String];

    /// Begin execution of `task`, returning a [`Stream`] of
    /// [`RecordBatch`]es.
    ///
    /// [`RecordBatch`]: arrow::record_batch::RecordBatch
    ///
    /// # Notes
    ///
    /// The `process` method itself is not `async` but it returns an `async`
    /// [`futures::stream::Stream`]. This `Stream` should incrementally compute
    /// the output, `RecordBatch` by `RecordBatch` (in a streaming fashion).
    /// Most `ArrowTask`s should not do any work before the first
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
    /// [`RecordBatchStreamAdapter`]: crate::table::stream_adapter::RecordBatchStreamAdapter
    ///
    /// # Error handling
    ///
    /// Any error that occurs during execution is sent as an `Err` in the output
    /// stream.
    ///
    /// `ArrowTask` implementations in DataFusion cancel additional work
    /// immediately once an error occurs. The rationale is that if the overall
    /// query will return an error, any additional work such as continued
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
    /// See `join_message_streams` in [`SessionStreamStep`] for a safe usage of [`spawn`]
    ///
    /// For more details see [`JoinSet`] and [`RecordBatchReceiverStreamBuilder`]
    /// for structures to help ensure all background tasks are cancelled.
    ///
    /// [`spawn`]: tokio::task::spawn
    /// [`JoinSet`]: tokio::task::JoinSet
    /// [`SessionStreamStep`]: crate::session::session_context::SessionStreamStep
    /// [`RecordBatchReceiverStreamBuilder`]: crate::table::stream_adapter::RecordBatchReceiverStreamBuilder
    ///
    /// # Implementation Examples
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
    /// # use phymes_core::table::stream::SendableRecordBatchStream;
    /// # use phymes_core::table::stream_adapter::RecordBatchStreamAdapter;
    /// # use phymes_core::session::common_traits::StateMap;
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
    /// # use phymes_core::table::stream::SendableRecordBatchStream;
    /// # use phymes_core::table::stream_adapter::RecordBatchStreamAdapter;
    /// # use phymes_core::session::common_traits::StateMap;
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
    /// # use phymes_core::table::stream::SendableRecordBatchStream;
    /// # use phymes_core::table::stream_adapter::RecordBatchStreamAdapter;
    /// # use phymes_core::session::common_traits::StateMap;
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
        message: OutgoingMessageMap,
        metrics: ArrowTaskMetricsSet,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<OutgoingMessageMap>;
}

/// Processor that returns the input
/// with optional conversion to another format
/// e.g., Bytes for web app streaming
#[derive(Default, Debug)]
pub struct ArrowProcessorEcho {
    name: String,
    publications: Vec<ArrowTablePublish>,
    subscriptions: Vec<ArrowTableSubscribe>,
    forward: Vec<String>,
}

impl ArrowProcessorEcho {
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

impl MappableTrait for ArrowProcessorEcho {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl ArrowProcessorTrait for ArrowProcessorEcho {
    fn new_arc(name: &str) -> Arc<dyn ArrowProcessorTrait>
    where
        Self: Sized,
    {
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
        message: OutgoingMessageMap,
        _metrics: ArrowTaskMetricsSet,
        _runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<OutgoingMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());
        Ok(message)
    }
}

/// Mock objects and functions for processor testing
pub mod test_processor {
    use super::*;
    use crate::{
        metrics::BaselineMetrics,
        session::common_traits::{BuildableTrait, BuilderTrait},
        table::arrow_table::test_table::make_test_record_batch,
        task::arrow_message::{
            ArrowMessageBuilderTrait, ArrowMessageTrait, ArrowOutgoingMessage,
            ArrowOutgoingMessageBuilderTrait, ArrowOutgoingMessageTrait,
        },
    };

    use arrow::{array::RecordBatch, compute::concat_batches, datatypes::SchemaRef};
    use futures::{Stream, StreamExt};
    use hashbrown::HashMap;
    use std::{
        pin::Pin,
        sync::Arc,
        task::{Context, Poll, ready},
    };

    /// Mock processor that adds an additional record batch
    #[derive(Default, Debug)]
    pub struct ArrowProcessorMock {
        name: String,
        publications: Vec<ArrowTablePublish>,
        subscriptions: Vec<ArrowTableSubscribe>,
        forward: Vec<String>,
    }

    impl ArrowProcessorMock {
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

    impl MappableTrait for ArrowProcessorMock {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl ArrowProcessorTrait for ArrowProcessorMock {
        fn new_arc(name: &str) -> Arc<dyn ArrowProcessorTrait>
        where
            Self: Sized,
        {
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
            message: OutgoingMessageMap,
            metrics: ArrowTaskMetricsSet,
            _runtime_env: Arc<Mutex<RuntimeEnv>>,
        ) -> Result<OutgoingMessageMap> {
            event!(Level::INFO, "Starting processor {}", self.get_name());

            // Add another record batch to the input
            let mut outbox = HashMap::<String, ArrowOutgoingMessage>::new();
            for (s_name, s) in message.into_iter() {
                let name = s_name.clone();
                let source = s.get_publisher().to_string();
                let subject = s.get_subject().to_string();
                let update = s.get_update().clone();
                let out = Box::pin(ArrowProcessorMockStream {
                    schema: s.get_message().schema(),
                    input: s.get_message_own(),
                    baseline_metrics: BaselineMetrics::new(&metrics, self.get_name()),
                });
                let out_m = ArrowOutgoingMessage::get_builder()
                    .with_name(name.as_str())
                    .with_publisher(source.as_str())
                    .with_subject(subject.as_str())
                    .with_update(&update)
                    .with_message(out)
                    .build()?;
                let _ = outbox.insert(s_name, out_m);
            }
            Ok(outbox)
        }
    }

    struct ArrowProcessorMockStream {
        /// Output schema after the projection
        schema: SchemaRef,
        /// The input task to process.
        input: SendableRecordBatchStream,
        /// Runtime metrics recording
        baseline_metrics: BaselineMetrics,
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

    impl Stream for ArrowProcessorMockStream {
        type Item = Result<RecordBatch>;

        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let poll;
            #[allow(clippy::never_loop)]
            loop {
                match ready!(self.input.poll_next_unpin(cx)) {
                    Some(Ok(batch)) => {
                        let timer = self.baseline_metrics.elapsed_compute().timer();
                        let processed_batch = add_test_table_row(batch)?;
                        timer.done();
                        poll = Poll::Ready(Some(Ok(processed_batch)));
                        break;
                    }
                    value => {
                        poll = Poll::Ready(value);
                        break;
                    }
                }
            }
            self.baseline_metrics.record_poll(poll)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            // Same number of record batches
            self.input.size_hint()
        }
    }

    impl RecordBatchStream for ArrowProcessorMockStream {
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::test_processor::ArrowProcessorMock;
    use crate::{
        metrics::{ArrowTaskMetricsSet, HashMap},
        session::{
            common_traits::{BuildableTrait, BuilderTrait},
            runtime_env::RuntimeEnv,
        },
        table::{
            arrow_table::{
                ArrowTableBuilder, ArrowTableBuilderTrait, ArrowTableTrait,
                test_table::make_test_table,
            },
            arrow_table_publish::ArrowTablePublish,
        },
        task::{
            arrow_message::{
                ArrowMessageBuilderTrait, ArrowOutgoingMessage, ArrowOutgoingMessageBuilderTrait,
                ArrowOutgoingMessageTrait,
            },
            arrow_processor::ArrowProcessorTrait,
        },
    };
    use anyhow::Result;
    use parking_lot::lock_api::Mutex;

    #[tokio::test]
    async fn test_processor() -> Result<()> {
        let metrics = ArrowTaskMetricsSet::new();
        let runtime_env = RuntimeEnv::default();
        let name = "process_1".to_string();
        let mut message = HashMap::<String, ArrowOutgoingMessage>::new();
        let _ = message.insert(
            name.clone(),
            ArrowOutgoingMessage::get_builder()
                .with_name(name.clone().as_str())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&ArrowTablePublish::Extend {
                    table_name: "test_table".to_string(),
                })
                .with_message(make_test_table("test_table", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );
        let processor_1 = ArrowProcessorMock::new_arc("processor_1");
        let mut stream =
            processor_1.process(message, metrics.clone(), Arc::new(Mutex::new(runtime_env)))?;
        let partitions = ArrowTableBuilder::new_from_sendable_record_batch_stream(
            stream.remove(&name).unwrap().get_message_own(),
        )
        .await?
        .with_name("test_message_table")
        .build()?;
        let n_rows: usize = partitions.count_rows();
        dbg!(&metrics);
        assert_eq!(n_rows, 15);
        assert_eq!(metrics.clone_inner().output_rows().unwrap(), 15);
        assert!(metrics.clone_inner().elapsed_compute().unwrap() > 100);
        Ok(())
    }
}
