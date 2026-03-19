use std::fmt::Display;

use crate::{CsvWriterMultipart, IpcWriterMultipart, JsonWriterMultipart, storage::chunked_writer::{ChunkedWriter, OnChunk}};

/// Available object storage writers
/// 
/// # Todo
/// - Add in `to_bytes`, `to_struct`, `to_values`, etc.
#[derive(Debug)]
pub enum ObjectStorageWriter {
    Ipc(IpcWriterMultipart<ChunkedWriter<OnChunk>>),
    Json(JsonWriterMultipart<ChunkedWriter<OnChunk>>),
    Csv(CsvWriterMultipart<ChunkedWriter<OnChunk>>),
}
impl Display for ObjectStorageWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ipc(_) => write!(f, "Ipc"),
            Self::Json(_) => write!(f, "Json"),
            Self::Csv(_) => write!(f, "Csv"),
        }
    }
}