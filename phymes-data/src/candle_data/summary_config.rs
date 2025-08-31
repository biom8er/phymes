use std::{fmt::Display, str::FromStr};

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
    Ipc,
    #[default]
    None,
}

impl FromStr for DataSummaryFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = s.split("-").collect::<Vec<&str>>();
        let value = match *parts.first().unwrap() {
            "Message" => DataSummaryFormat::Message,
            "Pdf" => DataSummaryFormat::Pdf,
            "Bytes" => DataSummaryFormat::Bytes,
            "Ipc" => DataSummaryFormat::Ipc,
            "None" => DataSummaryFormat::None,
            "Csv" => {
                if parts.len() != 4 {
                    return Err(anyhow::anyhow!("Invalid DataSummaryFormat::Csv string {}, expected Csv-<delimiter>-<header>-<batch_size>.", parts.join("-")));
                }
                DataSummaryFormat::Csv( CsvFormat { delimiter: parts.get(1).unwrap().parse()?, header: parts.get(2).unwrap().parse()?, batch_size: parts.get(3).unwrap().parse()? })
            },
            "Json" => {
                if parts.len() != 2 {
                    return Err(anyhow::anyhow!("Invalid DataSummaryFormat::Json string {}, expected Json-<batch_size>.", parts.join("-")));
                }
                DataSummaryFormat::Json( JsonFormat { batch_size: parts.get(1).unwrap().parse()? })
            },
            _ => return Err(anyhow::anyhow!("Invalid DataSummaryFormat string {}, expected Message, Pdf, Bytes, Ipc, None, Csv-<delimiter>-<header>-<batch_size> or Json-<batch_size>.", parts.join("-"))),
        };
        // match serde_json::from_str::<DataSummaryFormat>(s) {
        //     Ok(format) => Ok(format),
        //     Err(err) => Err(anyhow::anyhow!("{err:?}. Invalid DataSummaryFormat string, expected one of Message, Csv, Json, Pdf, Bytes, IPC, or None.")),
        // }
        Ok(value)
    }
}

impl Display for DataSummaryFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Message => write!(f, "Message"),
            Self::Csv(format) => write!(f, "Csv-{}-{}-{}", format.delimiter, format.header, format.batch_size),
            Self::Json(format) => write!(f, "Json-{}", format.batch_size),
            Self::Pdf => write!(f, "Pdf"),
            Self::Bytes => write!(f, "Bytes"),
            Self::Ipc => write!(f, "IPC"),
            Self::None => write!(f, "None"),
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