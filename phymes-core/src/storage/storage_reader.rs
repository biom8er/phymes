use std::{fmt::Display, io::Cursor};

use crate::{CsvReader, IpcReader, JsonReader};

/// Available object storage readers
///
/// # Todo
/// - Add in `fom_bytes`, `fom_struct`, `fom_values`, etc.
#[derive(Debug)]
pub enum ObjectStorageReader {
    Ipc(IpcReader<Cursor<Vec<u8>>>),
    Json(JsonReader<Cursor<Vec<u8>>>),
    Csv(CsvReader<Cursor<Vec<u8>>>),
}
impl Display for ObjectStorageReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ipc(_) => write!(f, "Ipc"),
            Self::Json(_) => write!(f, "Json"),
            Self::Csv(_) => write!(f, "Csv"),
        }
    }
}
