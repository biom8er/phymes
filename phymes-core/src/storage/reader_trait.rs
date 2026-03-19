use anyhow::Result;
use arrow::array::RecordBatch;
use futures::Stream;
use object_store::{GetResult, ObjectMeta, ObjectStore, ObjectStoreExt, path::Path};
use std::{
    pin::Pin,
    sync::Arc,
};

/// List the partitions in the object storage
pub fn storage_reader_list<'a>(store: &'a Arc<dyn ObjectStore>, path: Option<&'a Path>) -> Pin<Box<dyn Stream<Item = Result<ObjectMeta, object_store::Error>> + Send>> {
    store.list(path)
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
