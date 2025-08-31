use std::str::FromStr;

use anyhow::Result;
use clap::Parser;
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

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DataSummaryFormat {
    /// ToolMessage format
    Message,
    /// Comma Seperated Values string    
    Csv(CsvFormat),
    /// Json attachment
    Json(JsonFormat),
    /// Pdf attachment
    Pdf,    
    /// JSON Object written to Bytes
    Bytes,
    /// Arrow IPC
    IPC,
    #[default]
    None,
}

impl FromStr for DataSummaryFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match serde_json::from_str::<DataSummaryFormat>(s) {
            Ok(format) => Ok(format),
            Err(err) => Err(anyhow::anyhow!("{err:?}. Invalid DataSummaryFormat string, expected one of Message, Csv, Json, Pdf, Bytes, IPC, or None.")),
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