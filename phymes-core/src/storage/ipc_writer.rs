use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::Write;
use std::rc::Rc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::ipc::writer::StreamWriter;

use crate::storage::chunked_writer::ChunkedWriter;

pub struct IpcWriter {
    ipc: StreamWriter<ChunkedWriter<Box<dyn FnMut(Vec<u8>)>>>,
    pending: Rc<RefCell<VecDeque<Vec<u8>>>>,
}

impl IpcWriter {
    pub fn new(schema: SchemaRef, chunk_size: usize) -> anyhow::Result<Self> {
        let pending = Rc::new(RefCell::new(VecDeque::new()));
        let pending_for_closure = Rc::clone(&pending);

        let on_chunk: Box<dyn FnMut(Vec<u8>)> = Box::new(move |chunk| {
            pending_for_closure.borrow_mut().push_back(chunk);
        });

        let cw = ChunkedWriter::new(chunk_size, on_chunk);
        let ipc = StreamWriter::try_new(cw, &schema)?;

        Ok(Self { ipc, pending })
    }

    pub fn write_batch(&mut self, batch: &RecordBatch) -> anyhow::Result<()> {
        self.ipc.write(batch)?;
        Ok(())
    }

    pub fn poll_chunk(&mut self) -> Option<Vec<u8>> {
        self.pending.borrow_mut().pop_front()
    }

    pub fn finish(&mut self) -> anyhow::Result<()> {
        self.ipc.finish()?;
        self.ipc.get_mut().flush()?;
        Ok(())
    }
}
