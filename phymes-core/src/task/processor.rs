use crate::{
    AvailableTableSubscribePolicies, session::{MappableTrait, RuntimeEnv, SendableRecordBatchStreamMessageMap, StateMap}, table::{
        RecordBatchStream, SendableRecordBatchStream, TablePublication, TableSubscribePolicyTrait, TableSubscription
    }, task::PublishAndSubscribeTrait
};
use anyhow::{Result, anyhow};
use parking_lot::Mutex;
use phymes_diagnostics::{DiagnosticBuilder, DiagnosticBuilderTrait, HashMap, TraceBuilderTrait};
use std::fmt::Debug;
use std::sync::Arc;
use tracing::{Level, event};

/// Trait that performs the actual processing
/// 
/// # Notes
/// - designed to allow for chaining multiple processors into streaming computational trees
pub trait ProcessorTrait: MappableTrait + PublishAndSubscribeTrait + Send + Sync + Debug {
    /// New processor
    fn new_arc_with_pub_sub(
        name: &str,
        publications: &[TablePublication],
        subscriptions: &[TableSubscription],
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    ) -> Arc<dyn ProcessorTrait>
    where
        Self: Sized;

    /// Default new implementation
    fn new_arc(name: &str) -> Arc<dyn ProcessorTrait>
    where
        Self: Sized;

    /// Get the subscription policy
    fn get_subscribe_policy(&self) -> &dyn TableSubscribePolicyTrait;

    /// Alias for `get_static_name`
    fn get_type(&self) -> &str;

    /// Begin execution of `Process`, returning a [`Stream`] of [`RecordBatch`]es.
    ///
    /// [`RecordBatch`]: arrow::record_batch::RecordBatch
    ///
    /// # Notes
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
    /// [`RecordBatchStreamAdapter`]: crate::table::RecordBatchStreamAdapter
    ///
    /// # Error handling
    ///
    /// Any error that occurs during execution is sent as an `Err` in the output stream.
    ///
    /// `Task` implementations cancel additional work immediately once an error occurs. 
    /// The rationale is that if the overall query will return an error, any additional work such as continued
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
    /// [`SessionStreamStep`]: crate::session::SessionStreamStep
    /// [`RecordBatchReceiverStreamBuilder`]: crate::table::RecordBatchReceiverStreamBuilder
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
    /// # use phymes_core::SendableRecordBatchStream;
    /// # use phymes_core::RecordBatchStreamAdapter;
    /// # use phymes_core::StateMap;
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
    /// # use phymes_core::StateMap;
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
    /// # use phymes_core::StateMap;
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
        runtime_env: Arc<Mutex<RuntimeEnv>>,
    ) -> Result<SendableRecordBatchStreamMessageMap>;
}

/// Processor that returns the input
#[derive(Debug)]
pub struct ProcessorEcho {
    name: String,
    publications: Vec<TablePublication>,
    subscriptions: Vec<TableSubscription>,
    subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
}

impl MappableTrait for ProcessorEcho {
    fn get_name(&self) -> &str {
        &self.name
    }
}

impl PublishAndSubscribeTrait for ProcessorEcho {
    fn get_publications(&self) -> Vec<&TablePublication> {
        self.publications.iter().collect::<Vec<_>>()
    }

    fn get_subscriptions(&self) -> Vec<&TableSubscription> {
        self.subscriptions.iter().collect::<Vec<_>>()
    }
    fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
        self.subscribe_policy
            .check_subscriptions(&self.subscriptions, updates, state)
    }
}

impl ProcessorTrait for ProcessorEcho {
    fn new_arc_with_pub_sub(
        name: &str,
        publications: &[TablePublication],
        subscriptions: &[TableSubscription],
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    ) -> Arc<dyn ProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: publications.to_owned(),
            subscriptions: subscriptions.to_owned(),
            subscribe_policy,
        })
    }

    fn new_arc(name: &str) -> Arc<dyn ProcessorTrait> {
        Arc::new(Self {
            name: name.to_string(),
            publications: vec![TablePublication::None],
            subscriptions: vec![TableSubscription::None],
            subscribe_policy: AvailableTableSubscribePolicies::default().build(),
        })
    }

    fn get_subscribe_policy(&self) -> &dyn TableSubscribePolicyTrait {
        self.subscribe_policy.as_ref()
    }

    fn get_type(&self) -> &str {
        Self::get_static_name()
    }

    fn process(
        &self,
        message: SendableRecordBatchStreamMessageMap,
        diagnostic_builder: Option<&DiagnosticBuilder>,
        _runtime_env: Arc<Mutex<RuntimeEnv>>,
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

        // Trace the outbox
        if let Some(trace) = trace {
            trace.0.exit(&message.values().collect::<Vec<_>>());
        }

        Ok(message)
    }
}

/// A lightweight builder for structures implementing the [ProcessorTrait]
///
/// # Notes
/// * A full `ProcessorBuilderTrait` will be provided in the future
///   once the API stabilizes
#[derive(Default)]
pub struct ProcessorBuilder {
    pub publications: Option<Vec<TablePublication>>,
    pub subscriptions: Option<Vec<TableSubscription>>,
    pub subscribe_policy: Option<Box<dyn TableSubscribePolicyTrait>>,
    pub processor_name: Option<String>,
    pub processor_type: Option<String>,
}

type ProcessorInput = (
    String,
    Vec<TablePublication>,
    Vec<TableSubscription>,
    Box<dyn TableSubscribePolicyTrait>,
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
        } else if self.subscribe_policy.as_ref().is_none() {
            return Err(anyhow!(
                "Missing subscribe for processor {}",
                self.processor_name.as_ref().unwrap()
            ));
        }

        Ok((
            self.processor_name.take().unwrap(),
            self.publications.take().unwrap(),
            self.subscriptions.take().unwrap(),
            self.subscribe_policy.take().unwrap(),
        ))
    }
}

/// Mock objects and functions for processor testing
pub mod test_processor {
    use super::*;
    use crate::{
        session::{BuildableTrait, BuilderTrait, StateMap},
        table::test_table::make_test_record_batch,
        task::{MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage},
    };

    use arrow::{array::RecordBatch, compute::concat_batches, datatypes::SchemaRef};
    use futures::{Stream, StreamExt};
    use hashbrown::HashMap;
    use phymes_diagnostics::{DiagnosticBuilderTrait, MetricBuilderTrait, TraceBuilderTrait};
    use std::{
        pin::Pin,
        sync::Arc,
        task::{Context, Poll, ready},
    };

    /// Mock processor that adds an additional record batch
    #[derive(Debug)]
    pub struct ProcessorMock {
        name: String,
        publications: Vec<TablePublication>,
        subscriptions: Vec<TableSubscription>,
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    }

    impl MappableTrait for ProcessorMock {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl PublishAndSubscribeTrait for ProcessorMock {
        fn get_publications(&self) -> Vec<&TablePublication> {
            self.publications.iter().collect::<Vec<_>>()
        }

        fn get_subscriptions(&self) -> Vec<&TableSubscription> {
            self.subscriptions.iter().collect::<Vec<_>>()
        }
        fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
            self.subscribe_policy
                .check_subscriptions(&self.subscriptions, updates, state)
        }
    }

    impl ProcessorTrait for ProcessorMock {
        fn new_arc_with_pub_sub(
            name: &str,
            publications: &[TablePublication],
            subscriptions: &[TableSubscription],
            subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
        ) -> Arc<dyn ProcessorTrait> {
            Arc::new(Self {
                name: name.to_string(),
                publications: publications.to_owned(),
                subscriptions: subscriptions.to_owned(),
                subscribe_policy,
            })
        }

        fn new_arc(name: &str) -> Arc<dyn ProcessorTrait> {
            Arc::new(Self {
                name: name.to_string(),
                publications: vec![TablePublication::None],
                subscriptions: vec![TableSubscription::None],
                subscribe_policy: AvailableTableSubscribePolicies::default().build(),
            })
        }

        fn get_subscribe_policy(&self) -> &dyn TableSubscribePolicyTrait {
            self.subscribe_policy.as_ref()
        }

        fn get_type(&self) -> &str {
            Self::get_static_name()
        }

        fn process(
            &self,
            message: SendableRecordBatchStreamMessageMap,
            diagnostic_builder: Option<&DiagnosticBuilder>,
            _runtime_env: Arc<Mutex<RuntimeEnv>>,
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

            // Add another record batch to the input
            let mut outbox = HashMap::<String, SendableRecordBatchStreamMessage>::new();
            for (s_name, s) in message.into_iter() {
                let stream_diagnostic_builder = trace.as_ref().map(|trace| trace.1.clone());

                let name = s_name.clone();
                let source = s.get_publisher().to_string();
                let subject = s.get_subject().to_string();
                let update = s.get_update().clone();
                let out = Box::pin(ProcessorMockStream {
                    schema: s.get_message().schema(),
                    input: s.get_message_own(),
                    diagnostic_builder: stream_diagnostic_builder,
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

            // Trace the outbox
            if let Some(trace) = trace {
                trace.0.exit(&outbox.values().collect::<Vec<_>>());
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
        publications: Vec<TablePublication>,
        subscriptions: Vec<TableSubscription>,
        subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
    }

    impl MappableTrait for ProcessorError {
        fn get_name(&self) -> &str {
            &self.name
        }
    }

    impl PublishAndSubscribeTrait for ProcessorError {
        fn get_publications(&self) -> Vec<&TablePublication> {
            self.publications.iter().collect::<Vec<_>>()
        }

        fn get_subscriptions(&self) -> Vec<&TableSubscription> {
            self.subscriptions.iter().collect::<Vec<_>>()
        }
        fn check_subscriptions(&self, updates: &HashMap<String, bool>, state: &StateMap) -> bool {
            self.subscribe_policy
                .check_subscriptions(&self.subscriptions, updates, state)
        }
    }

    impl ProcessorTrait for ProcessorError {
        fn new_arc_with_pub_sub(
            name: &str,
            publications: &[TablePublication],
            subscriptions: &[TableSubscription],
            subscribe_policy: Box<dyn TableSubscribePolicyTrait>,
        ) -> Arc<dyn ProcessorTrait> {
            Arc::new(Self {
                name: name.to_string(),
                publications: publications.to_owned(),
                subscriptions: subscriptions.to_owned(),
                subscribe_policy,
            })
        }

        fn new_arc(name: &str) -> Arc<dyn ProcessorTrait> {
            Arc::new(Self {
                name: name.to_string(),
                publications: vec![TablePublication::None],
                subscriptions: vec![TableSubscription::None],
                subscribe_policy: AvailableTableSubscribePolicies::default().build(),
            })
        }

        fn get_subscribe_policy(&self) -> &dyn TableSubscribePolicyTrait {
            self.subscribe_policy.as_ref()
        }

        fn get_type(&self) -> &str {
            Self::get_static_name()
        }

        fn process(
            &self,
            _message: SendableRecordBatchStreamMessageMap,
            _diagnostic_builder: Option<&DiagnosticBuilder>,
            _runtime_env: Arc<Mutex<RuntimeEnv>>,
        ) -> Result<SendableRecordBatchStreamMessageMap> {
            Err(anyhow!("This is an error!"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::{
        session::{BuildableTrait, BuilderTrait, RuntimeEnv},
        table::{
            TableBuilder, TableBuilderTrait, TablePublication, TableTrait, test_table::make_test_table,
        },
        task::{MessageBuilderTrait, MessageTrait, SendableRecordBatchStreamMessage},
    };
    use anyhow::Result;
    use parking_lot::lock_api::Mutex;
    use phymes_diagnostics::{DiagnosticBuilderTrait, Diagnostics, SpanBuilder};

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
                .with_update(&TablePublication::Extend {
                    table_name: "test_table".to_string(),
                })
                .with_message(make_test_table("test_table", 4, 8, 3)?.to_record_batch_stream())
                .build()?,
        );
        let processor_1 = test_processor::ProcessorMock::new_arc("processor_1");
        let mut stream = processor_1.process(
            message,
            Some(&diagnostic_builder),
            Arc::new(Mutex::new(runtime_env)),
        )?;
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
