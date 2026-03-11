use std::{collections::VecDeque, fmt::Debug, io::{Result as IoResult, Write}, sync::Arc};
use parking_lot::Mutex;

pub trait OnChunkTrait {
    fn on_chunk(&self, chunk: Vec<u8>);
}

#[derive(Debug)]
pub struct OnChunk {
    pending: Arc<Mutex<VecDeque<Vec<u8>>>>,
}

impl OnChunk {
    pub fn new(pending: &Arc<Mutex<VecDeque<Vec<u8>>>>) -> Self {
        OnChunk { pending: Arc::clone(pending) }
    }
}

impl OnChunkTrait for OnChunk {
    fn on_chunk(&self, chunk: Vec<u8>) {
        self.pending.lock().push_back(chunk);
    }
}

pub struct ChunkedWriter<F>
    where F: OnChunkTrait
{
    buf: Vec<u8>,
    chunk_size: usize,
    on_chunk: F,
}

impl<F: OnChunkTrait> ChunkedWriter<F> {
    pub fn new(chunk_size: usize, on_chunk: F) -> Self {
        Self {
            buf: Vec::with_capacity(chunk_size),
            chunk_size,
            on_chunk,
        }
    }

    fn flush_inner(&mut self) {
        if !self.buf.is_empty() {
            let new_buf = Vec::with_capacity(self.chunk_size);
            let old = std::mem::replace(&mut self.buf, new_buf);
            self.on_chunk.on_chunk(old);
        }
    }
}

impl<F> Debug for ChunkedWriter<F>
where
    F: OnChunkTrait,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkedWriter").field("buf", &self.buf).field("chunk_size", &self.chunk_size).field("on_chunk", &"FnMut(Vec<u8>)").finish()
    }
}

impl<F> Write for ChunkedWriter<F>
where
    F: OnChunkTrait,
{
    fn write(&mut self, mut data: &[u8]) -> IoResult<usize> {
        let total = data.len();
        while !data.is_empty() {
            let remaining = self.chunk_size - self.buf.len();
            let to_take = remaining.min(data.len());
            self.buf.extend_from_slice(&data[..to_take]);
            data = &data[to_take..];

            if self.buf.len() == self.chunk_size {
                self.flush_inner();
            }
        }
        Ok(total)
    }

    fn flush(&mut self) -> IoResult<()> {
        self.flush_inner();
        Ok(())
    }
}
