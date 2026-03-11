use std::{cell::{RefCell, RefMut}, collections::VecDeque, fmt::Debug, io::Write, pin::Pin, rc::Rc, sync::Arc};

use anyhow::Result;
use arrow::{csv::{Writer, WriterBuilder}, {array::RecordBatch, json::LineDelimitedWriter}, datatypes::SchemaRef, ipc::writer::StreamWriter};
use bytes::Bytes;
use object_store::{MultipartUpload, ObjectStore, ObjectStoreExt, PutResult, path::Path};
use parking_lot::Mutex;

use crate::{StorageStreamWriterTrait, StorageWriterTrait, storage::chunked_writer::{ChunkedWriter, OnChunk}};

/// Write IPC to storage
pub struct IpcWriter<W> {
    writer: StreamWriter<W>,
    pending: Arc<Mutex<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<W> Debug for IpcWriter<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcWriter").field("writer", &"arrow::ipc::writer::StreamWriter").field("pending", &self.pending).field("mp", &self.mp).finish()
    }
}

impl IpcWriter<ChunkedWriter<OnChunk>> {
    /// New IPC Writer with schema and chunk_size
    pub fn new_with_config(mp: Box<dyn MultipartUpload>, schema: SchemaRef, chunk_size: usize) -> Result<Self> {
        let (pending, cw) = Self::chunk_writer(chunk_size);
        let writer = StreamWriter::try_new(cw, &schema)?;
        Ok(Self::new(writer, mp, pending))
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

impl StorageWriterTrait for IpcWriter<ChunkedWriter<OnChunk>> {
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
pub struct JsonWriter<W: Write> {
    writer: LineDelimitedWriter<W>,
    pending: Arc<Mutex<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<W: Write> StorageStreamWriterTrait<W> for JsonWriter<W> {
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

impl JsonWriter<ChunkedWriter<OnChunk>> {
    /// New JSON Writer with schema and chunk_size
    pub fn new_with_config(mp: Box<dyn MultipartUpload>, chunk_size: usize) -> Result<Self> {
        let (pending, cw) = Self::chunk_writer(chunk_size);
        let writer = LineDelimitedWriter::new(cw);
        Ok(Self::new(writer, mp, pending))
    }
}

impl StorageWriterTrait for JsonWriter<ChunkedWriter<OnChunk>> {
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
pub struct CsvWriter<W: Write> {
    writer: Writer<W>,
    pending: Arc<Mutex<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<W: Write> StorageStreamWriterTrait<W> for CsvWriter<W> {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch)?;
        Ok(())
    }

    fn finish_batch(&mut self) -> Result<()> {
        Ok(())
    }
}

impl CsvWriter<ChunkedWriter<OnChunk>> {
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

impl StorageWriterTrait for CsvWriter<ChunkedWriter<OnChunk>> {
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