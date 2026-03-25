use std::fmt::Display;

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Clone, Hash)]
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

#[derive(Parser, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Clone, Hash)]
pub struct JsonFormat {
    pub batch_size: usize,
}

impl Default for JsonFormat {
    fn default() -> Self {
        JsonFormat { batch_size: 1024 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, ValueEnum, Deserialize, Default, Hash)]
pub enum DataFormat {
    /// Comma Seperated Values string
    #[value(name = "CsvDefault")]
    CsvDefault,
    #[clap(skip)]
    #[value(name = "Csv")]
    Csv(CsvFormat),
    /// Json attachment with common defaults
    #[value(name = "JsonDefault")]
    JsonDefault,
    /// Json attachment
    #[clap(skip)]
    #[value(name = "Json")]
    Json(JsonFormat),
    /// Json attachment with a specified schema
    ///   and `JsonSchemaTrait` to parse to [RecordBatch]es
    ///
    /// [RecordBatch]: arrow::record_batch::RecordBatch
    #[value(name = "JsonSchema")]
    JsonSchema,
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
    /// Convert from a filename prefix
    pub fn from_prefix(prefix: &str) -> Result<Self> {
        let format = match prefix {
            "csv" => DataFormat::CsvDefault,
            "json" => DataFormat::JsonDefault,
            "pdf" => DataFormat::Pdf,
            "bytes" => DataFormat::Bytes,
            "ipc" => DataFormat::Ipc,
            "html" => DataFormat::Html,
            "txt" => DataFormat::Txt,
            "xml" => DataFormat::Xml,
            "owl" => DataFormat::Owl,
            _ => {
                return Err(anyhow!(
                    "File prefix {prefix} was not recognized. Supported prefixes are .csv, .json, .pdf, .bytes, .ipc, .txt, .xml,, .owl, and .html"
                ));
            }
        };
        Ok(format)
    }

    /// The file prefix for the format
    pub fn to_prefix(&self) -> &str {
        match self {
            Self::Csv(_) | Self::CsvDefault => "csv",
            Self::Json(_) | Self::JsonDefault | Self::JsonSchema => "json",
            Self::Bytes => "bytes",
            Self::Ipc => "ipc",
            Self::Pdf => "pdf",
            Self::Html => "html",
            Self::Txt => "txt",
            Self::Xml => "xml",
            Self::Owl => "owl",
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
            Self::JsonSchema => write!(f, "JsonSchema"),
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
