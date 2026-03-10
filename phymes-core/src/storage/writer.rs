use std::{cell::{RefCell, RefMut}, collections::VecDeque, io::Write, pin::Pin, rc::Rc, sync::Arc};

use anyhow::Result;
use arrow::{csv::{Writer, WriterBuilder}, {array::RecordBatch, json::LineDelimitedWriter}, datatypes::SchemaRef, ipc::writer::StreamWriter};
use bytes::Bytes;
use object_store::{MultipartUpload, ObjectStore, ObjectStoreExt, PutResult, path::Path};

use crate::storage::chunked_writer::ChunkedWriter;

/// Mutipart storage writing
pub fn storage_writer_multipart<'a>(store: &'a Arc<dyn ObjectStore>, path: &'a Path,
) -> Pin<
    Box<dyn Future<Output = Result<Box<dyn MultipartUpload>, object_store::Error>> + Send + 'a>,
> {
    Box::pin(store.put_multipart(path))
}

/// Trait for writing to object storage
pub trait StorageWriter {
    type SW;

    fn pending_borrow_mut(&mut self) -> RefMut<'_, VecDeque<Vec<u8>>>;
    fn mp_mut(&mut self) -> &mut Box<dyn MultipartUpload>;

    /// ChunkWriter
    fn chunk_writer(chunk_size: usize) -> (Rc<RefCell<VecDeque<Vec<u8>>>>, ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>) {
        let pending = Rc::new(RefCell::new(VecDeque::new()));
        let pending_for_closure = Rc::clone(&pending);
        let on_chunk: Box<dyn FnMut(Vec<u8>)> = Box::new(move |chunk| {
            pending_for_closure.borrow_mut().push_back(chunk);
        });
        let chunk_writer = ChunkedWriter::new(chunk_size, on_chunk);
        (pending, chunk_writer)
    }

    /// New Writer
    fn new(writer: Self::SW, mp: Box<dyn MultipartUpload>, pending: Rc<RefCell<VecDeque<Vec<u8>>>>) -> Self
    where
        Self: Sized;

    /// Poll the next chunk
    fn poll_chunk(
        &mut self,
    ) -> Option<Pin<Box<dyn Future<Output = Result<(), object_store::Error>> + Send>>> {
        let chunk = if let Some(chunk) = self.pending_borrow_mut().pop_front() {
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

pub trait StorageStreamWriter<W> {

    /// Add the batch to the writer
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()>;

    /// Finish adding batches to the writer
    fn finish_batch(&mut self) -> Result<()>;
}

/// Write IPC to storage
pub struct IpcWriter<W> {
    writer: StreamWriter<W>,
    pending: Rc<RefCell<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl IpcWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>> {
    /// New IPC Writer with schema and chunk_size
    pub fn new_with_config(mp: Box<dyn MultipartUpload>, schema: SchemaRef, chunk_size: usize) -> Result<Self> {
        let (pending, cw) = Self::chunk_writer(chunk_size);
        let writer = StreamWriter::try_new(cw, &schema)?;
        Ok(Self::new(writer, mp, pending))
    }
}

impl<W: Write> StorageStreamWriter<W> for IpcWriter<W> {
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

impl StorageWriter for IpcWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>> {
    type SW = StreamWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>>;

    fn new(writer: Self::SW, mp: Box<dyn MultipartUpload>, pending: Rc<RefCell<VecDeque<Vec<u8>>>>) -> Self {
        Self { writer, pending, mp }
    }

    fn pending_borrow_mut(&mut self) -> RefMut<'_, VecDeque<Vec<u8>>> {
        self.pending.borrow_mut()
    }

    fn mp_mut(&mut self) -> &mut Box<dyn MultipartUpload> {
        &mut self.mp
    }
}

/// Write JSON to storage
pub struct JsonWriter<W: Write> {
    writer: LineDelimitedWriter<W>,
    pending: Rc<RefCell<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<W: Write> StorageStreamWriter<W> for JsonWriter<W> {
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

impl JsonWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>> {
    /// New JSON Writer with schema and chunk_size
    pub fn new_with_config(mp: Box<dyn MultipartUpload>, chunk_size: usize) -> Result<Self> {
        let (pending, cw) = Self::chunk_writer(chunk_size);
        let writer = LineDelimitedWriter::new(cw);
        Ok(Self::new(writer, mp, pending))
    }
}

impl StorageWriter for JsonWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>> {
    type SW = LineDelimitedWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>>;
    fn new(writer: Self::SW, mp: Box<dyn MultipartUpload>, pending: Rc<RefCell<VecDeque<Vec<u8>>>>) -> Self {
        Self { writer, pending, mp }
    }

    fn pending_borrow_mut(&mut self) -> RefMut<'_, VecDeque<Vec<u8>>> {
        self.pending.borrow_mut()
    }

    fn mp_mut(&mut self) -> &mut Box<dyn MultipartUpload> {
        &mut self.mp
    }
}

/// Write CSV to storage
pub struct CsvWriter<W: Write> {
    writer: Writer<W>,
    pending: Rc<RefCell<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<W: Write> StorageStreamWriter<W> for CsvWriter<W> {
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch)?;
        Ok(())
    }

    fn finish_batch(&mut self) -> Result<()> {
        Ok(())
    }
}

impl CsvWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>> {
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

impl StorageWriter for CsvWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>> {
    type SW = Writer<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>>;
    fn new(writer: Self::SW, mp: Box<dyn MultipartUpload>, pending: Rc<RefCell<VecDeque<Vec<u8>>>>) -> Self {
        Self { writer, pending, mp }
    }

    fn pending_borrow_mut(&mut self) -> RefMut<'_, VecDeque<Vec<u8>>> {
        self.pending.borrow_mut()
    }

    fn mp_mut(&mut self) -> &mut Box<dyn MultipartUpload> {
        &mut self.mp
    }
}