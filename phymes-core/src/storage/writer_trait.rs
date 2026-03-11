use std::{collections::VecDeque, pin::Pin, sync::Arc};

use anyhow::Result;
use arrow::array::RecordBatch;
use bytes::Bytes;
use object_store::{MultipartUpload, ObjectStore, ObjectStoreExt, PutResult, path::Path};
use parking_lot::Mutex;

use crate::storage::chunked_writer::{ChunkedWriter, OnChunk};

/// Mutipart storage writing
pub fn storage_writer_multipart<'a>(store: &'a Arc<dyn ObjectStore>, path: &'a Path,
) -> Pin<
    Box<dyn Future<Output = Result<Box<dyn MultipartUpload>, object_store::Error>> + Send + 'a>,
> {
    Box::pin(store.put_multipart(path))
}

/// Trait for writing to object storage
pub trait StorageWriterTrait {
    type SW;

    fn pending_mut(&mut self) -> &mut Arc<Mutex<VecDeque<Vec<u8>>>>;
    fn mp_mut(&mut self) -> &mut Box<dyn MultipartUpload>;

    /// ChunkWriter
    fn chunk_writer(chunk_size: usize) -> (Arc<Mutex<VecDeque<Vec<u8>>>>, ChunkedWriter<OnChunk>) {
        let pending = Arc::new(Mutex::new(VecDeque::new()));
        let on_chunk = OnChunk::new(&pending);
        let chunk_writer = ChunkedWriter::new(chunk_size, on_chunk);
        (pending, chunk_writer)
    }

    /// New Writer
    fn new(writer: Self::SW, mp: Box<dyn MultipartUpload>, pending: Arc<Mutex<VecDeque<Vec<u8>>>>) -> Self
    where
        Self: Sized;

    /// Poll the next chunk
    fn poll_chunk(
        &mut self,
    ) -> Option<Pin<Box<dyn Future<Output = Result<(), object_store::Error>> + Send>>> {
        let chunk = if let Some(chunk) = self.pending_mut().lock().pop_front() {
            Some(chunk)
        } else {
            None
        };
        if let Some(chunk) = chunk {
            Some(self.mp_mut().put_part(Bytes::from(chunk).into()))
        } else {
            None
        }
    }

    /// Poll the next chunk
    fn finish_chunks(
        &mut self,
    ) -> Pin<Box<dyn Future<Output = Result<PutResult, object_store::Error>> + Send + '_>> {
        self.mp_mut().complete()
    }
}

pub trait StorageStreamWriterTrait<W> {

    /// Add the batch to the writer
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()>;

    /// Finish adding batches to the writer
    fn finish_batch(&mut self) -> Result<()>;
}