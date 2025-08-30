use anyhow::Result;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};


#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum DataSummaryFormat {
    /// ToolMessage format
    #[default]
    Message,
    /// Attachment formats
    Csv,
    Json,
    Pdf,
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct DataSummaryConfig {
    /// The column names
    #[arg(long, default_value = "[\"lhs_pk\", \"lhs_fk\"]")]
    pub col_names: Option<String>,

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
