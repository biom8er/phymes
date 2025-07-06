// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// https://github.com/apache/datafusion/blob/fda500ab8a5b4232a6619ad9875f99cf97b95c5c/datafusion/physical-plan/src/test/exec.rs

//! Simple iterator over batches for use in testing

use anyhow::{Result, anyhow};
use futures::{Stream, TryStreamExt};
use std::{
    fmt::Debug,
    pin::Pin,
    sync::{Arc, Weak},
    task::{Context, Poll},
};
use tokio::task::JoinSet;

// Required for documentation
#[allow(unused_imports)]
use super::arrow_task::ArrowTask;

use crate::table::{
    stream::{RecordBatchStream, SendableRecordBatchStream},
    stream_adapter::{EmptyRecordBatchStream, RecordBatchReceiverStream, RecordBatchStreamAdapter},
};

use arrow::{
    array::{ArrayRef, Int32Array},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};

/// For objects that run computation and send/recieve
/// streaming `RecordBatch`es as messages
pub trait SendableRecordBatchExecTrait: Debug + Send + Sync {
    fn get_static_name() -> &'static str
    where
        Self: Sized,
    {
        let full_name = std::any::type_name::<Self>();
        let maybe_start_idx = full_name.rfind(':');
        match maybe_start_idx {
            Some(start_idx) => &full_name[start_idx + 1..],
            None => "UNKNOWN",
        }
    }
    /// Get the name of the task
    fn get_name(&self) -> &str;
    /// Run the computation without mutating self
    fn run(&self) -> Result<SendableRecordBatchStream>;
}

/// Collect the results of running all tasks in memory
/// Each record batch represents the results of a single stream
pub async fn collect_partitions_runs(
    partitions: Vec<Arc<dyn SendableRecordBatchExecTrait>>,
) -> Result<Vec<Vec<RecordBatch>>> {
    let streams = collect_partitions_runs_helper(partitions)?;

    let mut join_set = JoinSet::new();
    // run_task the plan and collect the results into batches.
    streams.into_iter().enumerate().for_each(|(idx, stream)| {
        join_set.spawn(async move {
            let result: Result<Vec<RecordBatch>> = stream.try_collect().await;
            (idx, result)
        });
    });

    let mut batches = vec![];
    // Note that currently this doesn't identify the thread that panicked
    //
    // TODO: Replace with [join_next_with_id](https://docs.rs/tokio/latest/tokio/task/struct.JoinSet.html#method.join_next_with_id
    // once it is stable
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok((idx, res)) => batches.push((idx, res?)),
            Err(e) => {
                if e.is_panic() {
                    std::panic::resume_unwind(e.into_panic());
                } else {
                    unreachable!();
                }
            }
        }
    }

    batches.sort_by_key(|(idx, _)| *idx);
    let batches = batches.into_iter().map(|(_, batch)| batch).collect();

    Ok(batches)
}

/// Run the [`ArrowTask`] and return a vec with one stream per output
/// partition
///
/// # Aborting Execution
///
/// Dropping the stream will abort the execution of the query, and free up
/// any allocated resources
fn collect_partitions_runs_helper(
    partitions: Vec<Arc<dyn SendableRecordBatchExecTrait>>,
) -> Result<Vec<SendableRecordBatchStream>> {
    let num_partitions = partitions.len();
    let mut streams = Vec::with_capacity(num_partitions);
    for partition in partitions.iter().take(num_partitions) {
        streams.push(partition.run()?);
    }
    Ok(streams)
}

/// Collect the results of running all partitions
/// Each record batch represents the results of a single partitions
pub async fn collect_task_runs(
    partitions: Vec<Arc<dyn SendableRecordBatchExecTrait>>,
    schema: SchemaRef,
) -> Result<Vec<RecordBatch>> {
    let stream = collect_task_runs_helper(partitions, schema)?;
    collect_stream_helper(stream).await
}

/// Create a vector of record batches from a stream
/// See <https://github.com/apache/datafusion/blob/main/datafusion/physical-plan/src/common.rs#L43C1-L46C2>
pub async fn collect_stream_helper(stream: SendableRecordBatchStream) -> Result<Vec<RecordBatch>> {
    stream.try_collect::<Vec<_>>().await
}

/// Run the [`ArrowTask`] and return a single stream of `RecordBatch`es.
///
/// # Aborting Execution
///
/// Dropping the stream will abort the execution of the query, and free up
/// any allocated resources
pub fn collect_task_runs_helper(
    partitions: Vec<Arc<dyn SendableRecordBatchExecTrait>>,
    schema: SchemaRef,
) -> Result<SendableRecordBatchStream> {
    match partitions.len() {
        0 => Ok(Box::pin(EmptyRecordBatchStream::new(Arc::new(
            Schema::empty(),
        )))),
        1 => partitions[0].run(),
        2.. => {
            // merge into a single partition
            let num_partitions = partitions.len();

            // use a stream that allows each sender to put in at
            // least one result in an attempt to maximize
            // parallelism.
            // TODO: check that all partitions have the same schema!
            let mut builder = RecordBatchReceiverStream::builder(schema, num_partitions);

            // spawn independent partitions whose resulting streams (of batches)
            // are sent to the channel for consumption.
            for partition in partitions.iter().take(num_partitions) {
                builder.run_input(Arc::clone(partition));
            }
            Ok(builder.build())
        }
    }
}

/// Return a RecordBatch with a single Int32 array with values (0..sz) in a field named "i"
pub fn make_partition(sz: i32) -> RecordBatch {
    let seq_start = 0;
    let seq_end = sz;
    let values = (seq_start..seq_end).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, true)]));
    let arr = Arc::new(Int32Array::from(values));
    let arr = arr as ArrayRef;

    RecordBatch::try_new(schema, vec![arr]).unwrap()
}

/// Runs the provided execution plan and returns a vector of the number of
/// rows in each partition
pub async fn collect_num_rows(exec: Vec<Arc<dyn SendableRecordBatchExecTrait>>) -> Vec<usize> {
    let partition_batches = collect_partitions_runs(exec).await.unwrap();
    partition_batches
        .into_iter()
        .map(|batches| batches.iter().map(|b| b.num_rows()).sum::<usize>())
        .collect()
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

/// A Mock ExecutionPlan that can be used for writing tests of other
/// ExecutionPlans
#[derive(Debug)]
pub struct MockExec {
    /// the results to send back
    data: Vec<Result<RecordBatch>>,
    schema: SchemaRef,
    /// if true (the default), sends data using a separate task to ensure the
    /// batches are not available without this stream yielding first
    use_task: bool,
}

impl MockExec {
    /// Create a new `MockExec` with a single partition that returns
    /// the specified `Results`s.
    ///
    /// By default, the batches are not produced immediately (the
    /// caller has to actually yield and another task must run) to
    /// ensure any poll loops are correct. This behavior can be
    /// changed with `with_use_task`
    pub fn new(data: Vec<Result<RecordBatch>>, schema: SchemaRef) -> Self {
        Self {
            data,
            schema,
            use_task: true,
        }
    }

    /// If `use_task` is true (the default) then the batches are sent
    /// back using a separate task to ensure the underlying stream is
    /// not immediately ready
    pub fn with_use_task(mut self, use_task: bool) -> Self {
        self.use_task = use_task;
        self
    }

    pub fn get_task_count(&self) -> usize {
        self.data.len()
    }

    pub fn get_schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl SendableRecordBatchExecTrait for MockExec {
    fn get_name(&self) -> &str {
        MockExec::get_static_name()
    }
    fn run(&self) -> Result<SendableRecordBatchStream> {
        // Result doesn't implement clone, so do it ourself
        let data: Vec<_> = self
            .data
            .iter()
            .map(|r| match r {
                Ok(batch) => Ok(batch.clone()),
                Err(e) => Err(anyhow!("Execution error: {}", e)),
            })
            .collect();

        if self.use_task {
            let mut builder = RecordBatchReceiverStream::builder(self.get_schema(), 2);
            // send data in order but in a separate task (to ensure
            // the batches are not available without the stream
            // yielding).
            let tx = builder.tx();
            builder.spawn(async move {
                for batch in data {
                    println!("Sending batch via delayed stream");
                    if let Err(e) = tx.send(batch).await {
                        println!("ERROR batch via delayed stream: {e}");
                    }
                }

                Ok(())
            });
            // returned stream simply reads off the rx stream
            Ok(builder.build())
        } else {
            // make an input that will error
            let stream = futures::stream::iter(data);
            Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.get_schema(),
                stream,
            )))
        }
    }
}

/// Execution plan that emits streams that block forever.
///
/// This is useful to test shutdown / cancellation behavior of certain execution plans.
#[derive(Debug)]
pub struct BlockingExec {
    /// Schema that is mocked by this plan.
    schema: SchemaRef,

    /// Ref-counting helper to check if the plan and the produced stream are still in memory.
    refs: Arc<()>,
}

impl BlockingExec {
    /// Create new [`BlockingExec`] with a give schema and number of partitions.
    pub fn new(schema: SchemaRef, _n_partitions: usize) -> Self {
        Self {
            schema,
            refs: Default::default(),
        }
    }

    /// Weak pointer that can be used for ref-counting this execution plan and its streams.
    ///
    /// Use [`Weak::strong_count`] to determine if the plan itself and its streams are dropped (should be 0 in that
    /// case). Note that tokio might take some time to cancel spawned tasks, so you need to wrap this check into a retry
    /// loop. Use `assert_strong_count_converges_to_zero` to archive this.
    pub fn refs(&self) -> Weak<()> {
        Arc::downgrade(&self.refs)
    }

    pub fn get_task_count(&self) -> usize {
        0
    }
}

impl SendableRecordBatchExecTrait for BlockingExec {
    fn get_name(&self) -> &str {
        BlockingExec::get_static_name()
    }
    fn run(&self) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(BlockingStream {
            schema: Arc::clone(&self.schema),
            _refs: Arc::clone(&self.refs),
        }))
    }
}

/// A [`RecordBatchStream`] that is pending forever.
#[derive(Debug)]
pub struct BlockingStream {
    /// Schema mocked by this stream.
    schema: SchemaRef,

    /// Ref-counting helper to check if the stream are still in memory.
    _refs: Arc<()>,
}

impl Stream for BlockingStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl RecordBatchStream for BlockingStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(all(not(target_family = "wasm"), not(feature = "wasip2")))]
/// Asserts that the strong count of the given [`Weak`] pointer converges to zero.
///
/// This might take a while but has a timeout.
pub async fn assert_strong_count_converges_to_zero<T>(refs: Weak<T>) {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            if dbg!(Weak::strong_count(&refs)) == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

/// Task that emits streams that panics.
///
/// This is useful to test panic handling of certain tasks.
#[derive(Debug)]
struct PanicTask {
    /// Mock schema
    schema: SchemaRef,
    /// Partition number
    partition: usize,
    /// Batches to emit until panic
    until_panic: usize,
}

impl PanicTask {
    pub fn new(schema: SchemaRef, partition: usize, until_panic: usize) -> Self {
        Self {
            schema,
            partition,
            until_panic,
        }
    }
}

impl SendableRecordBatchExecTrait for PanicTask {
    fn get_name(&self) -> &str {
        PanicTask::get_static_name()
    }

    fn run(&self) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(PanicStream {
            partition: self.partition,
            batches_until_panic: self.until_panic,
            schema: Arc::clone(&self.schema),
            ready: false,
        }))
    }
}

/// Task wrapper to coordinate the running of multiple tasks
///
/// The task struct is stateless, so each task must have
/// its own struct
#[derive(Debug)]
pub struct PanicExecWrapper {
    /// Schema that is mocked by this plan.
    schema: SchemaRef,

    /// Number of output partitions. Each partition will produce this
    /// many empty output record batches prior to panicking
    batches_until_panics: Vec<Arc<dyn SendableRecordBatchExecTrait>>,
}

impl PanicExecWrapper {
    /// Create new [`PanicExecWrapper`] with a give schema and number of
    /// partitions, which will each panic immediately.
    pub fn new(schema: SchemaRef, n_partitions: usize) -> Self {
        let batches_until_panics = Vec::with_capacity(n_partitions);
        Self {
            schema,
            batches_until_panics,
        }
    }

    /// Set the number of batches prior to panic for a partition
    pub fn with_partition_panic(mut self, partition: usize, count: usize) -> Self {
        let panic_task = Arc::new(PanicTask::new(Arc::clone(&self.schema), partition, count));
        assert_eq!(self.batches_until_panics.len(), partition);
        self.batches_until_panics.push(panic_task);
        self
    }

    pub fn get_partition_count(&self) -> usize {
        self.batches_until_panics.len()
    }

    pub fn get_schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    pub fn get_partition(&self, partition: usize) -> Arc<dyn SendableRecordBatchExecTrait> {
        Arc::clone(self.batches_until_panics.get(partition).unwrap())
    }
}

/// A [`RecordBatchStream`] that yields every other batch and panics
/// after `batches_until_panic` batches have been produced.
///
/// Useful for testing the behavior of streams on panic
#[derive(Debug)]
struct PanicStream {
    /// Which partition was this
    partition: usize,
    /// How may batches will be produced until panic
    batches_until_panic: usize,
    /// Schema mocked by this stream.
    schema: SchemaRef,
    /// Should we return ready ?
    ready: bool,
}

impl Stream for PanicStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.batches_until_panic > 0 {
            if self.ready {
                self.batches_until_panic -= 1;
                self.ready = false;
                let batch = RecordBatch::new_empty(Arc::clone(&self.schema));
                return Poll::Ready(Some(Ok(batch)));
            } else {
                self.ready = true;
                // get called again
                cx.waker().wake_by_ref();
                return Poll::Pending;
            }
        }
        panic!("PanickingStream did panic: {}", self.partition)
    }
}

impl RecordBatchStream for PanicStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
