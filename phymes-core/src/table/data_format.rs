use std::fmt::Display;

use anyhow::{Result, anyhow};
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
    ///
    /// [RecordBatch]: arrow::record_batch::RecordBatch
    #[default]
    #[value(name = "None")]
    None,
    /// HTML format
    #[value(name = "Html")]
    Html,
    /// HTML format
    #[value(name = "Txt")]
    Txt,
    /// XML format
    #[value(name = "Xml")]
    Xml,
    /// OWL format
    #[value(name = "Owl")]
    Owl,
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
            "html" => DataFormat::Html,
            "txt" => DataFormat::Txt,
            "Xml" => DataFormat::Xml,
            "Owl" => DataFormat::Owl,
            _ => {
                return Err(anyhow!(
                    "File extension {extension} was not recognized. Supported extensions are .csv, .json, .pdf, .bytes, .ipc, .txt, .xml,, .owl, and .html"
                ));
            }
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
            Self::Html => "html",
            Self::Txt => "txt",
            Self::Xml => "Xml",
            Self::Owl => "Owl",
            Self::None => "",
        }
    }
}

impl Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csv(_) => write!(f, "Csv"),
            Self::CsvDefault => write!(f, "CsvDefault"),
            Self::Json(_) => write!(f, "Json"),
            Self::JsonDefault => write!(f, "JsonDefault"),
            Self::Bytes => write!(f, "Bytes"),
            Self::Ipc => write!(f, "Ipc"),
            Self::Pdf => write!(f, "Pdf"),
            Self::Html => write!(f, "Html"),
            Self::Txt => write!(f, "Txt"),
            Self::Xml => write!(f, "Xml"),
            Self::Owl => write!(f, "Owl"),
            Self::None => write!(f, "None"),
        }
    }
}
