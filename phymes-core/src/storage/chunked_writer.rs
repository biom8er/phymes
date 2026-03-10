use std::io::{Result as IoResult, Write};

pub struct ChunkedWriter<F>
where
    F: FnMut(Vec<u8>),
{
    buf: Vec<u8>,
    chunk_size: usize,
    on_chunk: F,
}

impl<F> ChunkedWriter<F>
where
    F: FnMut(Vec<u8>),
{
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
            (self.on_chunk)(old);
        }
    }
}

impl<F> Write for ChunkedWriter<F>
where
    F: FnMut(Vec<u8>),
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
