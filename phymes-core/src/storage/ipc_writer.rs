use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::Write;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::ipc::writer::StreamWriter;
use bytes::Bytes;
use object_store::{MultipartUpload, ObjectStore, ObjectStoreExt, PutResult, UploadPart, path::Path};

use crate::storage::chunked_writer::ChunkedWriter;

pub trait StorageWriter<'a> {
    /// Mutipart
    fn multipart(store: &'a Arc<dyn ObjectStore>, path: &'a Path) -> Pin<Box<dyn Future<Output = Result<Box<dyn MultipartUpload>, object_store::Error>> + Send + 'a>> {
        Box::pin(store.put_multipart(path))
    }

    /// New Writer
    fn new(mp: Box<dyn MultipartUpload>, schema: SchemaRef, chunk_size: usize) -> Result<Self> where Self: Sized;

    /// Add the batch to the writer
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()>;

    /// Finish adding batches to the writer
    fn finish_batch(&mut self) -> Result<()>;

    /// Poll the next chunk
    fn poll_chunk(&mut self) -> Pin<Box<dyn Future<Output = Result<(), object_store::Error>> + Send>>;

    /// Poll the next chunk
    fn finish_chunks(&mut self) -> Pin<Box<dyn Future<Output = Result<PutResult, object_store::Error>> + Send>>;
}

pub struct IpcWriter {
    ipc: StreamWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>>,
    pending: Rc<RefCell<VecDeque<Vec<u8>>>>,
    mp: Box<dyn MultipartUpload>,
}

impl<'a> StorageWriter<'a> for IpcWriter {
    fn new(mp: Box<dyn MultipartUpload>, schema: SchemaRef, chunk_size: usize) -> Result<Self> {
        let pending = Rc::new(RefCell::new(VecDeque::new()));
        let pending_for_closure = Rc::clone(&pending);

        let on_chunk: Box<dyn FnMut(Vec<u8>)> = Box::new(move |chunk| {
            pending_for_closure.borrow_mut().push_back(chunk);
        });

        let cw = ChunkedWriter::new(chunk_size, on_chunk);
        let ipc = StreamWriter::try_new(cw, &schema)?;

        Ok(Self { ipc, pending, mp })
    }
    
    fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.ipc.write(batch)?;
        Ok(())
    }
    
    fn finish_batch(&mut self) -> Result<()> {
        self.ipc.finish()?;
        self.ipc.get_mut().flush()?;
        Ok(())
    }
    
    fn poll_chunk(&mut self) -> Pin<Box<dyn Future<Output = Result<(), object_store::Error>> + Send>> {
        if let Some(chunk) = self.pending.borrow_mut().pop_front() {
            self.mp.put_part(Bytes::from(chunk).into())
        } else {
            None
        }
    }

    fn finish_chunks(&mut self) -> Pin<Box<dyn Future<Output = Result<PutResult, object_store::Error>> + Send>> {
        self.mp.complete()
    }
}

pub struct IpcWriterOld {
    ipc: StreamWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>>,
    pending: Rc<RefCell<VecDeque<Vec<u8>>>>,
}

impl IpcWriterOld {
    pub fn new(schema: SchemaRef, chunk_size: usize) -> Result<Self> {
        let pending = Rc::new(RefCell::new(VecDeque::new()));
        let pending_for_closure = Rc::clone(&pending);

        let on_chunk: Box<dyn FnMut(Vec<u8>)> = Box::new(move |chunk| {
            pending_for_closure.borrow_mut().push_back(chunk);
        });

        let cw = ChunkedWriter::new(chunk_size, on_chunk);
        let ipc = StreamWriter::try_new(cw, &schema)?;

        Ok(Self { ipc, pending })
    }

    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.ipc.write(batch)?;
        Ok(())
    }

    pub fn poll_chunk(&mut self) -> Option<Vec<u8>> {
        self.pending.borrow_mut().pop_front()
    }

    pub fn finish(&mut self) -> Result<()> {
        self.ipc.finish()?;
        self.ipc.get_mut().flush()?;
        Ok(())
    }
}
