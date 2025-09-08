use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Clone)]
pub struct CsvFormat {
    pub delimiter: u8,
    pub header: bool,
    pub batch_size: usize,
}

impl Default for CsvFormat {
    fn default() -> Self {
        CsvFormat {
            delimiter: b',',
            header: true,
            batch_size: 1024,
        }
    }
}

#[derive(Parser, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Clone)]
pub struct JsonFormat {
    pub batch_size: usize,
}

impl Default for JsonFormat {
    fn default() -> Self {
        JsonFormat { batch_size: 1024 }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, ValueEnum, Deserialize, Default)]
pub enum DataFormat {
    /// Comma Seperated Values string
    #[value(name = "CsvDefault")]
    CsvDefault,
    #[clap(skip)]
    #[value(name = "Csv")]
    Csv(CsvFormat),
    /// Json attachment
    #[value(name = "JsonDefault")]
    JsonDefault,
    #[clap(skip)]
    #[value(name = "Json")]
    Json(JsonFormat),
    /// Pdf attachment
    #[value(name = "Pdf")]
    Pdf,    
    /// JSON Object written to Bytes
    #[value(name = "Bytes")]
    Bytes,
    /// Arrow IPC stream
    #[value(name = "Ipc")]
    Ipc,
    /// The raw [RecordBatch]
    #[default]
    #[value(name = "None")]
    None,
}

impl DataFormat {
    /// Convert from a filename extension
    pub fn from_extension(extension: &str) -> Result<Self> {
        let format = match extension {
            "csv" => DataFormat::CsvDefault,
            "json" => DataFormat::JsonDefault,
            "pdf" => DataFormat::Pdf,
            "bytes" => DataFormat::Bytes,
            "ipc" => DataFormat::Ipc,
            _ => return Err(anyhow!("File extension {extension} was not recognized. Supported extensions are .csv, .json, .pdf, .bytes, and .ipc")),
        };
        Ok(format)
    }
    
    /// The file extension for the format
    pub fn to_extension(&self) -> &str {
        match self {
            Self::Csv(_) | Self::CsvDefault => "csv",
            Self::Json(_) | Self::JsonDefault => "json",
            Self::Bytes => "bytes",
            Self::Ipc => "ipc",
            Self::Pdf => "pdf",
            Self::None => "",
        }
    }
}