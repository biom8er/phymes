use anyhow::{Result, anyhow};
use clap::Parser;
use phymes_core::{DataFormat, MappableTrait, Table, TableTrait};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

use crate::DataConfigTrait;

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct DataSummaryConfig {
    /// The number of rows to skip when limiting the row count
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip: Option<usize>,

    /// The number of rows to fetch when limiting the row count
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fetch: Option<usize>,

    /// The output format
    #[arg(long)]
    pub summary_format: DataFormat,
}

impl DataConfigTrait for DataSummaryConfig {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Self::default())
    }
    fn from_table(table: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // Check for the required fields
        let column_names = table
            .get_schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<HashSet<_>>();
        if !column_names.contains("summary_format") {
            return Err(anyhow!(
                "Table {} is missing required Field for `summary_format` in DataSummaryConfig.",
                table.get_name()
            ));
        }

        // Try to build the config
        match table.to_struct::<DataSummaryConfig>() {
            Ok(config_vec) => match config_vec.first() {
                Some(config) => Ok(config.to_owned()),
                None => Err(anyhow!(
                    "No config data found for DataSummaryConfig with subject {}",
                    table.get_name()
                )),
            },
            Err(err) => Err(anyhow!(
                "DataSummaryConfig could not be built for subject {}. {err}",
                table.get_name()
            )),
        }
    }
}
