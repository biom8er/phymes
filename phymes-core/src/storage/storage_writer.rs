use std::fmt::Display;

use crate::{BatchWriter, CsvWriter, CsvWriterMultipart, IpcWriter, IpcWriterMultipart, JsonWriter, JsonWriterMultipart, storage::chunked_writer::{ChunkedWriter, OnChunk}};

/// Available object storage writers
/// 
/// # Todo
/// - Add in `to_bytes`, `to_struct`, `to_values`, etc.
#[derive(Debug)]
pub enum ObjectStorageWriter {
    IpcMultipart(IpcWriterMultipart<ChunkedWriter<OnChunk>>),
    JsonMultipart(JsonWriterMultipart<ChunkedWriter<OnChunk>>),
    CsvMultipart(CsvWriterMultipart<ChunkedWriter<OnChunk>>),
    Ipc(IpcWriter<BatchWriter<OnChunk>>),
    Json(JsonWriter<BatchWriter<OnChunk>>),
    Csv(CsvWriter<BatchWriter<OnChunk>>),
}
impl Display for ObjectStorageWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IpcMultipart(_) => write!(f, "IpcMultipart"),
            Self::JsonMultipart(_) => write!(f, "JsonMultipart"),
            Self::CsvMultipart(_) => write!(f, "CsvMultipart"),
            Self::Ipc(_) => write!(f, "Ipc"),
            Self::Json(_) => write!(f, "Json"),
            Self::Csv(_) => write!(f, "Csv"),
        }
    }
}