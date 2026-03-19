use std::{collections::VecDeque, fmt::Debug, io::Write, sync::Arc};

use anyhow::Result;
use arrow::{csv::{Writer, WriterBuilder}, {array::RecordBatch, json::LineDelimitedWriter}, datatypes::SchemaRef, ipc::writer::StreamWriter};
use object_store::MultipartUpload;
use parking_lot::Mutex;

use crate::{BatchWriter, StorageStreamWriterTrait, StorageWriterMultipartTrait, StorageWriterTrait, storage::chunked_writer::{ChunkedWriter, OnChunk}};

/// Write IPC to storage
pub struct IpcWriterMultipart<W> {
    writer: StreamWriter<W>,
    pending: Arc<Mutex<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<W> Debug for IpcWriterMultipart<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcWriter").field("writer", &"arrow::ipc::writer::StreamWriter").field("pending", &self.pending).field("mp", &self.mp).finish()
    }
}

impl IpcWriterMultipart<ChunkedWriter<OnChunk>> {
    /// New IPC Writer with schema and chunk_size
    pub fn new_with_config(mp: Box<dyn MultipartUpload>, schema: SchemaRef, chunk_size: usize) -> Result<Self> {
        let (pending, cw) = Self::chunk_writer(chunk_size);
        let writer = StreamWriter::try_new(cw, &schema)?;
        Ok(Self::new(writer, mp, pending))
    }
}

impl<W: Write> StorageStreamWriterTrait<W> for IpcWriterMultipart<W> {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch)?;
        Ok(())
    }

    fn finish_batch(&mut self) -> Result<()> {
        self.writer.finish()?;
        self.writer.get_mut().flush()?;
        Ok(())
    }
}

impl StorageWriterMultipartTrait for IpcWriterMultipart<ChunkedWriter<OnChunk>> {
    type SW = StreamWriter<ChunkedWriter<OnChunk>>;

    fn new(writer: Self::SW, mp: Box<dyn MultipartUpload>, pending: Arc<Mutex<VecDeque<Vec<u8>>>>) -> Self {
        Self { writer, pending, mp }
    }

    fn mp_mut(&mut self) -> &mut Box<dyn MultipartUpload> {
        &mut self.mp
    }
    
    fn pending_mut(&mut self) -> &mut Arc<Mutex<VecDeque<Vec<u8>>>> {
        &mut self.pending
    }
}

/// Write JSON to storage
#[derive(Debug)]
pub struct JsonWriterMultipart<W: Write> {
    writer: LineDelimitedWriter<W>,
    pending: Arc<Mutex<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<W: Write> StorageStreamWriterTrait<W> for JsonWriterMultipart<W> {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch)?;
        Ok(())
    }

    fn finish_batch(&mut self) -> Result<()> {
        self.writer.finish()?;
        self.writer.get_mut().flush()?;
        Ok(())
    }
}

impl JsonWriterMultipart<ChunkedWriter<OnChunk>> {
    /// New JSON Writer with schema and chunk_size
    pub fn new_with_config(mp: Box<dyn MultipartUpload>, chunk_size: usize) -> Result<Self> {
        let (pending, cw) = Self::chunk_writer(chunk_size);
        let writer = LineDelimitedWriter::new(cw);
        Ok(Self::new(writer, mp, pending))
    }
}

impl StorageWriterMultipartTrait for JsonWriterMultipart<ChunkedWriter<OnChunk>> {
    type SW = LineDelimitedWriter<ChunkedWriter<OnChunk>>;
    fn new(writer: Self::SW, mp: Box<dyn MultipartUpload>, pending: Arc<Mutex<VecDeque<Vec<u8>>>>) -> Self {
        Self { writer, pending, mp }
    }
    
    fn pending_mut(&mut self) -> &mut Arc<Mutex<VecDeque<Vec<u8>>>> {
        &mut self.pending
    }

    fn mp_mut(&mut self) -> &mut Box<dyn MultipartUpload> {
        &mut self.mp
    }
}

/// Write CSV to storage
#[derive(Debug)]
pub struct CsvWriterMultipart<W: Write> {
    writer: Writer<W>,
    pending: Arc<Mutex<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<W: Write> StorageStreamWriterTrait<W> for CsvWriterMultipart<W> {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch)?;
        Ok(())
    }

    fn finish_batch(&mut self) -> Result<()> {
        Ok(())
    }
}

impl CsvWriterMultipart<ChunkedWriter<OnChunk>> {
    /// New CSV Writer with schema and chunk_size
    pub fn new_with_config(mp: Box<dyn MultipartUpload>, chunk_size: usize, header: bool, delimiter: u8) -> Result<Self> {
        let (pending, cw) = Self::chunk_writer(chunk_size);
        let builder = WriterBuilder::new()
            .with_header(header)
            .with_delimiter(delimiter)
            .with_quote(b'\'')
            .with_null("NULL".to_string())
            .with_time_format("%r".to_string());
        let writer = builder.build(cw);
        Ok(Self::new(writer, mp, pending))
    }
}

impl StorageWriterMultipartTrait for CsvWriterMultipart<ChunkedWriter<OnChunk>> {
    type SW = Writer<ChunkedWriter<OnChunk>>;
    fn new(writer: Self::SW, mp: Box<dyn MultipartUpload>, pending: Arc<Mutex<VecDeque<Vec<u8>>>>) -> Self {
        Self { writer, pending, mp }
    }
    
    fn pending_mut(&mut self) -> &mut Arc<Mutex<VecDeque<Vec<u8>>>> {
        &mut self.pending
    }

    fn mp_mut(&mut self) -> &mut Box<dyn MultipartUpload> {
        &mut self.mp
    }
}

/// Write IPC to storage
pub struct IpcWriter<W> {
    writer: StreamWriter<W>,
    pending: Vec<u8>,
}

impl<W> Debug for IpcWriter<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcWriter").field("writer", &"arrow::ipc::writer::StreamWriter").field("pending", &self.pending).finish()
    }
}

impl IpcWriter<BatchWriter<OnChunk>> {
    pub fn new_with_config(schema: SchemaRef) -> Result<Self> {
        let (pending, cw) = Self::batch_writer();
        let writer = StreamWriter::try_new(cw, &schema)?;
        Ok(Self::new(writer, pending))
    }
}

impl<W: Write> StorageStreamWriterTrait<W> for IpcWriter<W> {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch)?;
        Ok(())
    }

    fn finish_batch(&mut self) -> Result<()> {
        self.writer.finish()?;
        self.writer.get_mut().flush()?;
        Ok(())
    }
}

impl StorageWriterTrait for IpcWriter<BatchWriter<OnChunk>> {
    type SW = StreamWriter<BatchWriter<OnChunk>>;

    fn new(writer: Self::SW, pending: Arc<Mutex<VecDeque<Vec<u8>>>>) -> Self {
        Self { writer, pending }
    }
    
    fn pending_mut(&mut self) -> &mut Arc<Mutex<VecDeque<Vec<u8>>>> {
        &mut self.pending
    }
}