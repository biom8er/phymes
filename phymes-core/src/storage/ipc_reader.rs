use arrow::{array::RecordBatch, ipc::reader::StreamReader};
use object_store::{ObjectStore, ObjectStoreExt, path::Path};
use std::{io::{Read, Cursor}, sync::Arc};
use anyhow::Result;

pub struct IpcReader<R: Read> {
    reader: StreamReader<R>,
}

impl IpcReader<Cursor<Vec<u8>>> {
    /// Construct from an object in object_store (native or WASM memory backend).
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

impl<R: Read> IpcReader<R> {
    /// Poll the next decoded RecordBatch.
    pub fn poll_next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }
}
