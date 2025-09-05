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
pub enum DataSummaryFormat {
    /// Comma Seperated Values string
    #[value(name = "CsvDefault")]
    CsvDefault,
    #[clap(skip)]
    #[value(name = "Csv")]
    Csv(CsvFormat),
    /// Json Object
    #[value(name = "JsonObject")]
    JsonObject,
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

impl DataSummaryFormat {
    /// Convert from a filename extension
    pub fn from_extension(extension: &str) -> Result<Self> {
        let format = match extension {
            ".csv" => DataSummaryFormat::CsvDefault,
            ".json" => DataSummaryFormat::JsonDefault,
            ".pdf" => DataSummaryFormat::Pdf,
            ".bytes" => DataSummaryFormat::Bytes,
            ".ipc" => DataSummaryFormat::Ipc,
            _ => return Err(anyhow!("File extension {extension} was not recognized. Supported extensions are .csv, .json, .pdf, .bytes, and .ipc")),
        };
        Ok(format)
    }
    
    /// The file extension for the format
    pub fn to_extension(&self) -> &str {
        match self {
            Self::Csv(_) | Self::CsvDefault => ".csv",
            Self::Json(_) | Self::JsonObject | Self::JsonDefault => ".json",
            Self::Bytes => ".bytes",
            Self::Ipc => ".ipc",
            Self::Pdf => ".pdf",
            Self::None => "",
        }
    }
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct DataSummaryConfig {
    /// The column names
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub col_names: Option<Vec<String>>,

    /// The number of rows
    #[arg(long, default_value = "10")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_rows: Option<usize>,

    /// The number of batches
    #[arg(long, default_value = "1")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_batches: Option<usize>,

    /// The output format
    #[arg(long)]
    pub format: DataSummaryFormat,
}