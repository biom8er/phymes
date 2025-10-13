use crate::{
    session::{
        common_traits::{MappableTrait, SendableRecordBatchStreamMessageMap},
        runtime_env::RuntimeEnv,
    },
    table::{
        stream::{RecordBatchStream, SendableRecordBatchStream}, table_publish::TablePublish, table_subscribe::{AllTableNamesSubscribe, SubscribeTrait, TableSubscribe}
    },
    task::publish_subscribe::PubSubTrait,
};
use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use phymes_diagnostics::{DiagnosticBuilder, HashMap};
use std::fmt::Debug;
use std::sync::Arc;
use tracing::{Level, event};

/// For inner task objects that perform the actual processing
/// and designed to allow for chaining multiple processors
/// into streaming computational tree
pub trait ProcessorTrait: MappableTrait + PubSubTrait + Send + Sync + Debug {
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
    fn new_arc_with_pub_sub(
        name: &str,
        publications: &[TablePublish],
        subscriptions: &[TableSubscribe],
        subscribe: Box<dyn SubscribeTrait>,
    ) -> Arc<dyn ProcessorTrait>
    where
        Self: Sized;

    /// Default new implementation
    fn new_arc(name: &str) -> Arc<dyn ProcessorTrait>
    where
        Self: Sized;

    /// Get the subscription policy
    fn get_subscribe(&self) -> &dyn SubscribeTrait;

    /// Alias for `get_static_name`
    fn get_type(&self) -> &str;

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
        message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: &DiagnosticBuilder,
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<SendableRecordBatchStreamMessageMap>;
}

/// Processor that returns the input
/// with optional conversion to another format
/// e.g., Bytes for web app streaming
#[derive(Debug)]
pub struct ProcessorEcho {
    name: String,
    publications: Vec<TablePublish>,
    subscriptions: Vec<TableSubscribe>,
    subscribe: Box<dyn SubscribeTrait>,
}

impl MappableTrait for ProcessorEcho {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PubSubTrait for ProcessorEcho {
    fn get_publications(&self) -> Vec<&TablePublish> {
        self.publications.iter().collect::<Vec<_>>()
    }

    fn get_subscriptions(&self) -> Vec<&TableSubscribe> {
        self.subscriptions.iter().collect::<Vec<_>>()
    }
    fn check_subscriptions(
        &self,
        updates: &HashMap<String, bool>,
        state: &crate::session::common_traits::StateMap,
    ) -> bool {
        self.subscribe
            .check_subscriptions(&self.subscriptions, updates, state)
    }
}

impl ProcessorTrait for ProcessorEcho {
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

    fn process(
        &self,
        message: SendableRecordBatchStreamMessageMap,
        _diagnostic_builder: &DiagnosticBuilder,
        _runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<SendableRecordBatchStreamMessageMap> {
        event!(Level::INFO, "Starting processor {}", self.get_name());
        Ok(message)
    }
}

/// A lightweight builder for structures implementing the [ArrowProcessorTrait]
///
/// # Notes
/// * A full `ArrowProcessorBuilderTrait` will be provided in the future
///   once the API stabilizes
#[derive(Default)]
pub struct ProcessorBuilder {
    pub publications: Option<Vec<TablePublish>>,
    pub subscriptions: Option<Vec<TableSubscribe>>,
    pub subscribe: Option<Box<dyn SubscribeTrait>>,
    pub processor_name: Option<String>,
    pub processor_type: Option<String>,
}

type ProcessorInput = (
    String,
    Vec<TablePublish>,
    Vec<TableSubscribe>,
    Box<dyn SubscribeTrait>,
);

impl ProcessorBuilder {
    pub fn take(mut self) -> Result<ProcessorInput> {
        if self.processor_name.as_ref().is_none() {
            return Err(anyhow!("Missing processor name"));
        } else if self.publications.as_ref().is_none() {
            return Err(anyhow!(
                "Missing publications for processor {}",
                self.processor_name.as_ref().unwrap()
            ));
        } else if self.subscriptions.as_ref().is_none() {
            return Err(anyhow!(
                "Missing subscriptions for processor {}",
                self.processor_name.as_ref().unwrap()
            ));
        } else if self.subscribe.as_ref().is_none() {
            return Err(anyhow!(
                "Missing subscribe for processor {}",
                self.processor_name.as_ref().unwrap()
            ));
        }

        Ok((
            self.processor_name.take().unwrap(),
            self.publications.take().unwrap(),
            self.subscriptions.take().unwrap(),
            self.subscribe.take().unwrap(),
        ))
    }
}

/// Mock objects and functions for processor testing
pub mod test_processor {
    use super::*;
    use crate::{
        session::common_traits::{BuildableTrait, BuilderTrait, StateMap},
        table::table_trait::test_table::make_test_record_batch,
        task::message::{
            MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage,
        },
    };

    use arrow::{array::RecordBatch, compute::concat_batches, datatypes::SchemaRef};
    use futures::{Stream, StreamExt};
    use hashbrown::HashMap;
    use phymes_diagnostics::{DiagnosticBuilderTrait, MetricBuilderTrait};
    use std::{
        pin::Pin,
        sync::Arc,
        task::{Context, Poll, ready},
    };

    /// Mock processor that adds an additional record batch
    #[derive(Debug)]
    pub struct ProcessorMock {
        name: String,
        publications: Vec<TablePublish>,
        subscriptions: Vec<TableSubscribe>,
        subscribe: Box<dyn SubscribeTrait>,
    }

    impl MappableTrait for ProcessorMock {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl PubSubTrait for ProcessorMock {
        fn get_publications(&self) -> Vec<&TablePublish> {
            self.publications.iter().collect::<Vec<_>>()
        }

        fn get_subscriptions(&self) -> Vec<&TableSubscribe> {
            self.subscriptions.iter().collect::<Vec<_>>()
        }
        fn check_subscriptions(
            &self,
            updates: &HashMap<String, bool>,
            state: &StateMap,
        ) -> bool {
            self.subscribe
                .check_subscriptions(&self.subscriptions, updates, state)
        }
    }

    impl ProcessorTrait for ProcessorMock {
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

        fn process(
            &self,
            message: SendableRecordBatchStreamMessageMap,
            diagnostic_builder: &DiagnosticBuilder,
            _runtime_env: Arc<Mutex<RuntimeEnv>>,
        ) -> Result<SendableRecordBatchStreamMessageMap> {
            event!(Level::INFO, "Starting processor {}", self.get_name());

            // Add another record batch to the input
            let mut outbox = HashMap::<String, SendableRecordBatchStreamMessage>::new();
            for (s_name, s) in message.into_iter() {
                let name = s_name.clone();
                let source = s.get_publisher().to_string();
                let subject = s.get_subject().to_string();
                let update = s.get_update().clone();
                let out = Box::pin(ProcessorMockStream {
                    schema: s.get_message().schema(),
                    input: s.get_message_own(),
                    diagnostic_builder: diagnostic_builder.clone().to_child(self.get_name())?,
                });
                let out_m = SendableRecordBatchStreamMessage::get_builder()
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

    struct ProcessorMockStream {
        /// Output schema after the projection
        schema: SchemaRef,
        /// The input task to process.
        input: SendableRecordBatchStream,
        /// Runtime metrics recording
        diagnostic_builder: DiagnosticBuilder,
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
            let baseline_metrics = self.diagnostic_builder.clone().to_child("ProcessorMockStream")?.baseline_metrics("poll_next");
            #[allow(clippy::never_loop)]
            loop {
                match ready!(self.input.poll_next_unpin(cx)) {
                    Some(Ok(batch)) => {
                        let timer = baseline_metrics.elapsed_compute().timer();
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
            baseline_metrics.record_poll(poll)
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
        publications: Vec<TablePublish>,
        subscriptions: Vec<TableSubscribe>,
        subscribe: Box<dyn SubscribeTrait>,
    }

    impl MappableTrait for ProcessorError {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl PubSubTrait for ProcessorError {
        fn get_publications(&self) -> Vec<&TablePublish> {
            self.publications.iter().collect::<Vec<_>>()
        }

        fn get_subscriptions(&self) -> Vec<&TableSubscribe> {
            self.subscriptions.iter().collect::<Vec<_>>()
        }
        fn check_subscriptions(
            &self,
            updates: &HashMap<String, bool>,
            state: &StateMap,
        ) -> bool {
            self.subscribe
                .check_subscriptions(&self.subscriptions, updates, state)
        }
    }

    impl ProcessorTrait for ProcessorError {
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

        fn process(
            &self,
            _message: SendableRecordBatchStreamMessageMap,
            _diagnostic_builder: &DiagnosticBuilder,
            _runtime_env: Arc<Mutex<RuntimeEnv>>,
        ) -> Result<SendableRecordBatchStreamMessageMap> {
            event!(Level::INFO, "Starting processor {}", self.get_name());
            Err(anyhow!("This is an error!"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::{
        session::{
            common_traits::{BuildableTrait, BuilderTrait},
            runtime_env::RuntimeEnv,
        },
        table::{
            table_publish::TablePublish, table_trait::{
                test_table::make_test_table, TableBuilder, TableBuilderTrait, TableTrait
            }
        },
        task::message::{MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage}
    };
    use anyhow::Result;
    use parking_lot::lock_api::Mutex;
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics};

    #[tokio::test]
    async fn test_processor() -> Result<()> {
        let diagnostics = Diagnostics::new();
        let diagnostic_builder = DiagnosticBuilder::new(&diagnostics);
        let runtime_env = RuntimeEnv::default();
        let name = "process_1".to_string();
        let mut message = HashMap::<String, SendableRecordBatchStreamMessage>::new();
        let _ = message.insert(
            name.clone(),
            SendableRecordBatchStreamMessage::get_builder()
                .with_name(name.clone().as_str())
                .with_publisher("s1")
                .with_subject("d1")
                .with_update(&TablePublish::Extend {
                    table_name: "test_table".to_string(),
                })
                .with_message(make_test_table("test_table", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );
        let processor_1 = test_processor::ProcessorMock::new_arc("processor_1");
        let mut stream =
            processor_1.process(message, &diagnostic_builder, Arc::new(Mutex::new(runtime_env)))?;
        let partitions = TableBuilder::new_from_sendable_record_batch_stream(
            stream.remove(&name).unwrap().get_message_own(),
        )
        .await?
        .with_name("test_message_table")
        .build()?;
        let n_rows: usize = partitions.count_rows();
        assert_eq!(n_rows, 15);
        Ok(())
    }
}
