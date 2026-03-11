use anyhow::Result;
use arrow::{array::RecordBatch, csv::reader::Format, datatypes::{DataType, Field, Schema, SchemaRef}, ipc::reader::StreamReader, json::{Reader, ReaderBuilder, reader::infer_json_schema}};
use bytes::Bytes;
use futures::Stream;
use object_store::{GetResult, ObjectStore, ObjectStoreExt, path::Path};
use std::{
    io::{BufRead, Cursor, Read, Seek},
    pin::Pin,
    sync::Arc,
};

/// Get the results from object storage
pub fn storage_reader_get_result<'a>(store: &'a Arc<dyn ObjectStore>, path: &'a Path,
) -> Pin<Box<dyn Future<Output = Result<GetResult, object_store::Error>> + Send + 'a>> {
    Box::pin(store.get(path))
}

/// Stream the results from object storage after polling `get_result`
pub fn storage_reader_stream_result(result: GetResult,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, object_store::Error>> + Send>> {
    Box::pin(result.into_stream())
}

/// Trait for reading from object storage
pub trait StorageReaderTrait {
    type SR;

    /// Build the [StorageReader] after polling `get_result` and `stream_result`
    fn new(reader: Self::SR) -> Self
    where
        Self: Sized;
}

pub trait StorageStreamReaderTrait<R> {
    fn poll_next_batch(&mut self) -> Result<Option<RecordBatch>>;
}
