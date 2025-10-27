use clap::Parser;
use phymes_core::DataFormat;
use serde::{Deserialize, Serialize};

use crate::DataConfigTrait;

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
    pub format: DataFormat,
}

impl DataConfigTrait for DataSummaryConfig {
    fn to_example(name: &str) -> Self {
        match name {
            "Function" => Self {
                num_rows: Some(10),
                num_batches: Some(1),
                format: DataFormat::None,
                ..Default::default()
            },
            _ => Self::default(),
        }
    }
}
