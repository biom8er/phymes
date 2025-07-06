// See https://docs.rs/datafusion-execution/43.0.0/src/datafusion_execution/stream.rs.html

use anyhow::Result;
use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use futures::Stream;
use std::pin::Pin;

/// Trait for types that stream [RecordBatch]
///
/// See [`SendableRecordBatchStream`] for more details.
pub trait RecordBatchStream: Stream<Item = Result<RecordBatch>> {
    /// Returns the schema of this `RecordBatchStream`.
    ///
    /// Implementation of this trait should guarantee that all `RecordBatch`'s returned by this
    /// stream should have the same schema as returned from this method.
    fn schema(&self) -> SchemaRef;
}

/// Trait for a [`Stream`] of [`RecordBatch`]es that can be passed between threads
///
/// This trait is used to retrieve the results of DataFusion execution plan nodes.
///
/// The trait is a specialized Rust Async [`Stream`] that also knows the schema
/// of the data it will return (even if the stream has no data). Every
/// `RecordBatch` returned by the stream should have the same schema as returned
/// by [`schema`](`RecordBatchStream::schema`).
///
/// # Error Handling
///
/// Once a stream returns an error, it should not be polled again (the caller
/// should stop calling `next`) and handle the error.
///
/// However, returning `Ready(None)` (end of stream) is likely the safest
/// behavior after an error. Like [`Stream`]s, `RecordBatchStream`s should not
/// be polled after end of stream or returning an error. However, also like
/// [`Stream`]s there is no mechanism to prevent callers polling  so returning
/// `Ready(None)` is recommended.
pub type SendableRecordBatchStream = Pin<Box<dyn RecordBatchStream + Send>>;

pub trait IPCRecordBatchStream: Stream<Item = Result<Vec<u8>>> {
    /// Returns the schema of this `IPCRecordBatchStream`.
    ///
    /// Implementation of this trait should guarantee that all `RecordBatch`'s returned by this
    /// stream should have the same schema as returned from this method.
    fn schema(&self) -> SchemaRef;
}

/// Outgoing IPC message type
pub type SendableIPCRecordBatchStream = Pin<Box<dyn IPCRecordBatchStream + Send>>;
