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

//! Defines the LIMIT plan

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use anyhow::Result;

use crate::table::stream::{RecordBatchStream, SendableRecordBatchStream};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

use futures::stream::{Stream, StreamExt};

/// A Limit stream skips `skip` rows, and then fetch up to `fetch` rows.
pub struct LimitStream {
    /// The remaining number of rows to skip
    skip: usize,
    /// The remaining number of rows to produce
    fetch: usize,
    /// The input to read from. This is set to None once the limit is
    /// reached to enable early termination
    input: Option<SendableRecordBatchStream>,
    /// Copy of the input schema
    schema: SchemaRef,
}

impl LimitStream {
    pub fn new(input: SendableRecordBatchStream, skip: usize, fetch: Option<usize>) -> Self {
        let schema = input.schema();
        Self {
            skip,
            fetch: fetch.unwrap_or(usize::MAX),
            input: Some(input),
            schema,
        }
    }

    fn poll_and_skip(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        let input = self.input.as_mut().unwrap();
        loop {
            let poll = input.poll_next_unpin(cx);
            let poll = poll.map_ok(|batch| {
                if batch.num_rows() <= self.skip {
                    self.skip -= batch.num_rows();
                    RecordBatch::new_empty(input.schema())
                } else {
                    let new_batch = batch.slice(self.skip, batch.num_rows() - self.skip);
                    self.skip = 0;
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
        if self.fetch == 0 {
            self.input = None; // Clear input so it can be dropped early
            None
        } else if batch.num_rows() < self.fetch {
            //
            self.fetch -= batch.num_rows();
            Some(batch)
        } else if batch.num_rows() >= self.fetch {
            let batch_rows = self.fetch;
            self.fetch = 0;
            self.input = None; // Clear input so it can be dropped early

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
        let fetch_started = self.skip == 0;
        match &mut self.input {
            Some(input) => {
                let poll = if fetch_started {
                    input.poll_next_unpin(cx)
                } else {
                    self.poll_and_skip(cx)
                };

                poll.map(|x| match x {
                    Some(Ok(batch)) => Ok(self.stream_limit(batch)).transpose(),
                    other => other,
                })
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
    use super::*;
    use crate::task::test_exec::collect_stream_helper;
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatchOptions;

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

        // Limit of six needs to consume the entire first record batch
        // (5 rows) and 1 row from the second (1 row)
        let limit_stream = LimitStream::new(Box::pin(input), 0, Some(6));
        assert_eq!(index.value(), 0);

        let results = collect_stream_helper(Box::pin(limit_stream)).await.unwrap();
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

        // Limit of six needs to consume the entire first record batch
        // (6 rows) and stop immediately
        let limit_stream = LimitStream::new(Box::pin(input), 0, Some(6));
        assert_eq!(index.value(), 0);

        let results = collect_stream_helper(Box::pin(limit_stream)).await.unwrap();
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

        // Limit of six needs to consume the entire first record batch
        // (6 rows) and stop immediately
        let limit_stream = LimitStream::new(Box::pin(input), 0, Some(6));
        assert_eq!(index.value(), 0);

        let results = collect_stream_helper(Box::pin(limit_stream)).await.unwrap();
        let num_rows: usize = results.into_iter().map(|b| b.num_rows()).sum();
        // Only 6 rows should have been produced
        assert_eq!(num_rows, 6);

        // Only the first batch should be consumed
        assert_eq!(index.value(), 1);

        Ok(())
    }
}
