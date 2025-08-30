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

impl FromStr for CsvFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let items = s.split("-").collect::<Vec<&str>>();
        if items.len() !=3 {
            return Err(anyhow::anyhow!("Invalid CsvFormat string, expected format: <delimiter>-<header>-<batch_size>."));
        }

        Ok(CsvFormat { delimiter: items[0].as_bytes()[0], header: items[1].parse::<bool>()?, batch_size: items[2].parse::<usize>()? } )
    }
}

#[derive(Parser, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Clone)]
pub struct JsonFormat {
    pub batch_size: usize,
}

impl FromStr for JsonFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(JsonFormat { batch_size: s.parse::<usize>()? } )
    }
}

impl Default for JsonFormat {
    fn default() -> Self {
        JsonFormat { batch_size: 1024 }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DataSummaryFormat {
    /// ToolMessage format
    #[default]
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
}

impl FromStr for DataSummaryFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let items = s.split("-").collect::<Vec<&str>>();
        match items.first() {
            Some(&"Message") => Ok(DataSummaryFormat::Message),
            Some(&"Csv") => {
                if items.len() != 4 {
                    return Err(anyhow::anyhow!("Invalid CsvFormat string, expected format: Csv-<delimiter>-<header>-<batch_size>."));
                }
                let csv_format = CsvFormat { delimiter: items[1].as_bytes()[0], header: items[2].parse::<bool>()?, batch_size: items[3].parse::<usize>()? };
                Ok(DataSummaryFormat::Csv(csv_format))
            },
            Some(&"Json") => {
                if items.len() != 2 {
                    return Err(anyhow::anyhow!("Invalid JsonFormat string, expected format: Json-<batch_size>."));
                }
                let json_format = JsonFormat { batch_size: items[1].parse::<usize>()? };
                Ok(DataSummaryFormat::Json(json_format))
            },
            Some(&"Pdf") => Ok(DataSummaryFormat::Pdf),
            Some(&"Bytes") => Ok(DataSummaryFormat::Bytes),
            Some(&"IPC") => Ok(DataSummaryFormat::IPC),
            _ => Err(anyhow::anyhow!("Invalid DataSummaryFormat string, expected one of Message, Csv-<delimiter>-<header>-<batch_size>, Json-<batch_size>, Pdf, Bytes, IPC.")),
        }
    }
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct DataSummaryConfig {
    /// The column names
    #[arg(long)]
    pub col_names: Option<Vec<String>>,

    /// The number of rows
    #[arg(long, default_value = "10")]
    pub num_rows: Option<usize>,

    /// The number of batches
    #[arg(long, default_value = "1")]
    pub num_batches: Option<usize>,

    /// The output format
    #[arg(long)]
    pub format: DataSummaryFormat,
}

impl DataSummaryConfig {
    #[allow(dead_code)]
    fn new_from_json(input: &str) -> Result<Self> {
        let self_data: DataSummaryConfig = serde_json::from_str(input)?;
        Ok(self_data)
    }
}
