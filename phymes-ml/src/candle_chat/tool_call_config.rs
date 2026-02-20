use anyhow::{Result, anyhow};
use clap::Parser;
use phymes_core::{MappableTrait, Table, TableTrait};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

/// Configuration for [ToolCallProcessor]
/// 
/// [ToolCallProcessor]: phymes_ml::ToolCallProcessor
#[derive(Parser, Debug, Serialize, Deserialize, Default, Clone)]
#[command(author, version, about, long_about = None)]
pub struct ToolCallConfig {
    /// The name of the subject containing the `ViewTasksSubscribePublishAggregated` subject
    #[arg(long)]
    pub all_subscribe_publish: String,

    /// The name of the subjects containing the processor configs
    #[arg(long)]
    pub subject_names: Vec<String>,

    /// The subscription keys in the configs to search for
    #[arg(long)]
    pub subscription_table_names: Vec<String>,

    /// The default subscription to use 
    #[arg(long)]
    pub subscription_name: Option<String>,

    /// The publication keys in the configs to search for
    #[arg(long)]
    pub publication_table_names: Option<Vec<String>>,

    /// The default publication to use 
    #[arg(long)]
    pub publication_name: Option<String>,
}

impl DataConfigTrait for ToolCallConfig {
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
        if !(column_names.contains("all_subscribe_publish")
            && column_names.contains("subject_names")
            && column_names.contains("subscription_table_names"))
        {
            return Err(anyhow!(
                "Table {} is missing required Field for `all_subscribe_publish`, `subject_names`, `subscription_table_names` in ToolCallConfig.",
                table.get_name()
            ));
        }

        // Try to build the config
        match table.to_struct::<ToolCallConfig>() {
            Ok(config_vec) => match config_vec.first() {
                Some(config) => Ok(config.to_owned()),
                None => Err(anyhow!(
                    "No config data found for ToolCallConfig with subject {}",
                    table.get_name()
                )),
            },
            Err(err) => Err(anyhow!(
                "ToolCallConfig could not be built for subject {}. {err}",
                table.get_name()
            )),
        }
    }
}
