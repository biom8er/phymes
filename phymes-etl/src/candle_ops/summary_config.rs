use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct CandleOpsSummaryConfig {
    /// The column names
    #[arg(long, default_value = "[\"lhs_pk\", \"lhs_fk\"]")]
    pub col_names: Option<String>,

    /// The number of rows
    #[arg(long, default_value = "10")]
    pub num_rows: Option<usize>,

    /// The number of batches
    #[arg(long, default_value = "1")]
    pub num_batches: Option<usize>,
}

impl CandleOpsSummaryConfig {
    #[allow(dead_code)]
    fn new_from_json(input: &str) -> Result<Self> {
        let self_data: CandleOpsSummaryConfig = serde_json::from_str(input)?;
        Ok(self_data)
    }
}
