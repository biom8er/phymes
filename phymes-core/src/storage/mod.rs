mod backend;
mod chunked_writer;
mod ipc_reader;
mod ipc_writer;
mod multiparts;

pub use ipc_reader::IpcReader;
pub use ipc_writer::IpcWriter;
pub use multiparts::upload_multipart;
pub use backend::{StorageBackendConfig, make_store};

#[cfg(test)]
mod tests {
    use crate::storage::ipc_reader::StorageReader;

    use super::*;
    use arrow::{array::{Int64Array, RecordBatch}, datatypes::{DataType, Field, Schema}};
    use futures::TryStreamExt;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use std::sync::Arc;

    #[tokio::test(flavor = "current_thread")]
    async fn test_storage_ipc_roundtrip() -> anyhow::Result<()> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("roundtrip.arrow.ipc");

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..10))],
        )?;

        // --- Write ---
        let mut writer = IpcWriter::new(schema.clone(), 64 * 1024)?;
        writer.write_batch(&batch)?;

        upload_multipart(store.clone(), path.clone(), &mut writer).await?;

        // --- Read ---
        let mut reader = IpcReader::from_object(store.clone(), path.clone(), None).await?;
        let read = reader.poll_next_batch()?.unwrap();

        assert_eq!(read.num_rows(), batch.num_rows());
        assert_eq!(read.schema(), batch.schema());

        Ok(())
    }
    
    #[tokio::test(flavor = "current_thread")]
    async fn test_storage_ipc_multi_batch_stress() -> anyhow::Result<()> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("stress.arrow.ipc");

        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
        ]));

        // Generate many batches
        let mut batches = Vec::new();
        for i in 0..500 {
            let size = (i * 17) % 10_000;
            let arr = Int64Array::from_iter_values(0..size as i64);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?;
            batches.push(batch);
        }

        // --- Write ---
        let mut writer = IpcWriter::new(schema.clone(), 64 * 1024)?;
        for batch in &batches {
            writer.write_batch(batch)?;
        }

        upload_multipart(store.clone(), path.clone(), &mut writer).await?;

        // --- Read without streaming ---
        let mut reader = IpcReader::from_object(store.clone(), path.clone(), None).await?;

        let mut read_batches = Vec::new();
        while let Some(batch) = reader.poll_next_batch()? {
            read_batches.push(batch);
        }

        assert_eq!(read_batches.len(), batches.len());

        for (expected, actual) in batches.iter().zip(read_batches.iter()) {
            assert_eq!(expected.num_rows(), actual.num_rows());
            assert_eq!(expected.schema(), actual.schema());
        }

        // --- Read with streaming ---
        let result = IpcReader::get_result(&store, &path).await?;
        let mut read_batches = Vec::new();
        let mut stream = IpcReader::stream_result(result);
        while let Some(bytes) = stream.try_next().await? {
            let mut reader = IpcReader::from_bytes(&bytes, None)?;
            while let Some(batch) = reader.poll_next_batch()? {
                read_batches.push(batch);
            }
        };

        assert_eq!(read_batches.len(), batches.len());

        for (expected, actual) in batches.iter().zip(read_batches.iter()) {
            assert_eq!(expected.num_rows(), actual.num_rows());
            assert_eq!(expected.schema(), actual.schema());
        }

        Ok(())
    }
}