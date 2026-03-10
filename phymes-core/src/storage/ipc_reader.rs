use arrow::{array::RecordBatch, ipc::reader::StreamReader};
use bytes::Bytes;
use futures::Stream;
use object_store::{GetResult, ObjectStore, ObjectStoreExt, path::Path};
use std::{io::{Cursor, Read}, pin::Pin, sync::Arc};
use anyhow::Result;

/// Trait for reading from object storage
pub trait StorageReader<'a> {
    /// Get the results from object storage
    fn get_result(store: &'a Arc<dyn ObjectStore>, path: &'a Path) -> Pin<Box<dyn Future<Output = Result<GetResult, object_store::Error>> + Send + 'a>> {
        Box::pin(store.get(path))
    }
    /// Stream the results from object storage after polling `get_result`
    fn stream_result(result: GetResult) -> Pin<Box<dyn Stream<Item = Result<Bytes, object_store::Error>> + Send>> {
        Box::pin(result.into_stream())
    }
    /// Build the [StorageReader] after polling `get_result` and `stream_result`
    fn from_bytes(bytes: &[u8], projection: Option<Vec<usize>>) -> Result<Self> where Self: Sized;
}

/// Read IPC from storage
pub struct IpcReader<R: Read> {
    reader: StreamReader<R>,
}

impl IpcReader<Cursor<Vec<u8>>> {
    pub async fn from_object(
        store: Arc<dyn ObjectStore>,
        path: Path,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let bytes = store.get(&path).await?.bytes().await?;
        let cursor = Cursor::new(bytes.to_vec());
        let reader = StreamReader::try_new(cursor, projection)?;
        Ok(Self { reader })
    }
}

impl<'a> StorageReader<'a> for IpcReader<Cursor<Vec<u8>>> {
    fn from_bytes(bytes: &[u8], projection: Option<Vec<usize>>) -> Result<Self> where Self: Sized {
        let cursor = Cursor::new(bytes.to_vec());
        let reader = StreamReader::try_new(cursor, projection)?;
        Ok(Self { reader })
    }
}

impl<R: Read> IpcReader<R> {
    pub fn poll_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}
