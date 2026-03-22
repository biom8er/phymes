use anyhow::Result;
use arrow::array::RecordBatch;

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
