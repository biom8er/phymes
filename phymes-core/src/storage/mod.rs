mod backend;
mod chunked_writer;
mod reader_trait;
mod readers;
mod storage_reader;
mod storage_writer;
mod writer_trait;
mod writers;

pub use backend::{ObjectStorageBackend, make_store};
pub use chunked_writer::{BatchWriter, ChunkedWriter, OnChunk, OnChunkTrait};
pub use reader_trait::{StorageReaderTrait, StorageStreamReaderTrait, storage_reader_get_result, storage_reader_stream_result};
pub use readers::{IpcReader, JsonReader, CsvReader};
pub use storage_reader::ObjectStorageReader;
pub use storage_writer::ObjectStorageWriter;
pub use writer_trait::{StorageWriterTrait, StorageWriterMultipartTrait, StorageStreamWriterTrait, storage_writer_multipart};
pub use writers::{IpcWriter, JsonWriter, CsvWriter, IpcWriterMultipart, JsonWriterMultipart, CsvWriterMultipart};

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Int64Array, RecordBatch},
        datatypes::{DataType, Field, Schema},
    };
    use futures::TryStreamExt;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use std::sync::Arc;

    #[tokio::test(flavor = "current_thread")]
    async fn test_storage_to_from_storage_ipc_multipart() -> anyhow::Result<()> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("stress.arrow.ipc");

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        // Generate many batches
        let mut batches = Vec::new();
        for i in 0..500 {
            let size = (i * 17) % 10_000;
            let arr = Int64Array::from_iter_values(0..size as i64);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?;
            batches.push(batch);
        }

        // --- Write ---
        let mp = storage_writer_multipart(&store, &path).await?;

        let mut writer = IpcWriterMultipart::new_with_config(mp, schema, 64 * 1024)?;
        for batch in &batches {
            writer.write_batch(batch)?;
        }
        writer.finish_batch()?;

        while let Some(chunk) = writer.poll_chunk() {
            chunk.await?;
        }
        writer.finish_chunks().await?;

        // --- Read ---
        let result = storage_reader_get_result(&store, &path).await?;
        let mut stream = storage_reader_stream_result(result);
        let mut read_batches = Vec::new();
        while let Some(bytes) = stream.try_next().await? {
            let mut reader = IpcReader::new_with_bytes(&bytes, None)?;
            while let Some(batch) = reader.poll_next_batch()? {
                read_batches.push(batch);
            }
        }

        assert_eq!(read_batches.len(), batches.len());

        for (expected, actual) in batches.iter().zip(read_batches.iter()) {
            assert_eq!(expected.num_rows(), actual.num_rows());
            assert_eq!(expected.schema(), actual.schema());
        }

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_storage_to_from_storage_json_multipart() -> anyhow::Result<()> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("stress.arrow.json");

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        // Generate many batches
        let mut batches = Vec::new();
        for i in 0..500 {
            let size = (i * 17) % 10_000;
            let arr = Int64Array::from_iter_values(0..size as i64);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?;
            batches.push(batch);
        }

        // --- Write ---
        let mp = storage_writer_multipart(&store, &path).await?;

        let mut writer = JsonWriterMultipart::new_with_config(mp, 64 * 1024)?;
        for batch in &batches {
            writer.write_batch(batch)?;
        }
        writer.finish_batch()?;

        while let Some(chunk) = writer.poll_chunk() {
            chunk.await?;
        }
        writer.finish_chunks().await?;

        // --- Read ---
        let result = storage_reader_get_result(&store, &path).await?;
        let mut stream = storage_reader_stream_result(result);
        let mut read_batches = Vec::new();
        while let Some(bytes) = stream.try_next().await? {
            let mut reader = JsonReader::new_with_bytes(&bytes, 512, Some(schema.clone()))?;
            while let Some(batch) = reader.poll_next_batch()? {
                read_batches.push(batch);
            }
        }

        assert_ne!(read_batches.len(), batches.len());
        let n_rows_read = read_batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        let n_rows = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        assert_eq!(n_rows_read, n_rows);
        assert_eq!(read_batches.first().unwrap().schema(), batches.first().unwrap().schema());

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_storage_to_from_storage_csv_multipart() -> anyhow::Result<()> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("stress.arrow.csv");

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        // Generate many batches
        let mut batches = Vec::new();
        for i in 0..500 {
            let size = (i * 17) % 10_000;
            let arr = Int64Array::from_iter_values(0..size as i64);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?;
            batches.push(batch);
        }

        // --- Write ---
        let mp = storage_writer_multipart(&store, &path).await?;

        let mut writer = CsvWriterMultipart::new_with_config(mp, 64 * 1024, false, b';')?;
        for batch in &batches {
            writer.write_batch(batch)?;
        }
        writer.finish_batch()?;

        while let Some(chunk) = writer.poll_chunk() {
            chunk.await?;
        }
        writer.finish_chunks().await?;

        // --- Read ---
        let result = storage_reader_get_result(&store, &path).await?;
        let mut stream = storage_reader_stream_result(result);
        let mut read_batches = Vec::new();
        while let Some(bytes) = stream.try_next().await? {
            let mut reader = CsvReader::new_with_bytes(&bytes, false, b';', 512, Some(schema.clone()))?;
            while let Some(batch) = reader.poll_next_batch()? {
                read_batches.push(batch);
            }
        }

        assert_ne!(read_batches.len(), batches.len());
        let n_rows_read = read_batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        let n_rows = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        assert_eq!(n_rows_read, n_rows);
        assert_eq!(read_batches.first().unwrap().schema(), batches.first().unwrap().schema());

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_storage_to_from_storage_ipc_singlepart() -> anyhow::Result<()> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("arrow.ipc");

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        // Generate batches
        let mut batches = Vec::new();
        for _ in 0..2 {
            let size = 10;
            let arr = Int64Array::from_iter_values(0..size as i64);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?;
            batches.push(batch);
        }

        // --- Write ---
        let mut writer = IpcWriter::new_with_config(schema)?;
        for batch in &batches {
            writer.write_batch(batch)?;
        }
        writer.finish_batch()?;
        writer.put(&store, &path).await?;

        // --- Read ---
        let result = storage_reader_get_result(&store, &path).await?;
        let mut stream = storage_reader_stream_result(result);
        let mut read_batches = Vec::new();
        while let Some(bytes) = stream.try_next().await? {
            let mut reader = IpcReader::new_with_bytes(&bytes, None)?;
            while let Some(batch) = reader.poll_next_batch()? {
                read_batches.push(batch);
            }
        }

        assert_eq!(read_batches.len(), batches.len());

        for (expected, actual) in batches.iter().zip(read_batches.iter()) {
            assert_eq!(expected.num_rows(), actual.num_rows());
            assert_eq!(expected.schema(), actual.schema());
        }

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_storage_to_from_storage_json_singlepart() -> anyhow::Result<()> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("arrow.ipc");

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        // Generate batches
        let mut batches = Vec::new();
        for _ in 0..2 {
            let size = 10;
            let arr = Int64Array::from_iter_values(0..size as i64);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?;
            batches.push(batch);
        }

        // --- Write ---
        let mut writer = JsonWriter::new_with_config()?;
        for batch in &batches {
            writer.write_batch(batch)?;
        }
        writer.finish_batch()?;

        writer.put(&store, &path).await?;

        // --- Read ---
        let result = storage_reader_get_result(&store, &path).await?;
        let mut stream = storage_reader_stream_result(result);
        let mut read_batches = Vec::new();
        while let Some(bytes) = stream.try_next().await? {
            let mut reader = IpcReader::new_with_bytes(&bytes, None)?;
            while let Some(batch) = reader.poll_next_batch()? {
                read_batches.push(batch);
            }
        }

        assert_eq!(read_batches.len(), batches.len());

        for (expected, actual) in batches.iter().zip(read_batches.iter()) {
            assert_eq!(expected.num_rows(), actual.num_rows());
            assert_eq!(expected.schema(), actual.schema());
        }

        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_storage_to_from_storage_csv_singlepart() -> anyhow::Result<()> {
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("arrow.ipc");

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));

        // Generate batches
        let mut batches = Vec::new();
        for _ in 0..2 {
            let size = 10;
            let arr = Int64Array::from_iter_values(0..size as i64);
            let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])?;
            batches.push(batch);
        }

        // --- Write ---
        let mut writer = CsvWriter::new_with_config(false, b';')?;
        for batch in &batches {
            writer.write_batch(batch)?;
        }
        writer.finish_batch()?;

        writer.put(&store, &path).await?;

        // --- Read ---
        let result = storage_reader_get_result(&store, &path).await?;
        let mut stream = storage_reader_stream_result(result);
        let mut read_batches = Vec::new();
        while let Some(bytes) = stream.try_next().await? {
            let mut reader = IpcReader::new_with_bytes(&bytes, None)?;
            while let Some(batch) = reader.poll_next_batch()? {
                read_batches.push(batch);
            }
        }

        assert_eq!(read_batches.len(), batches.len());

        for (expected, actual) in batches.iter().zip(read_batches.iter()) {
            assert_eq!(expected.num_rows(), actual.num_rows());
            assert_eq!(expected.schema(), actual.schema());
        }

        Ok(())
    }
}
