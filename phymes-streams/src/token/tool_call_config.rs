use anyhow::{Result, anyhow};
use clap::Parser;
use phymes_subject::{MappableTrait, Subject, SubjectTrait};
use phymes_data::DataConfigTrait;
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

/// Configuration for [ToolCallProcessor]
///
/// [ToolCallProcessor]: crate::ToolCallProcessor
#[derive(Parser, Debug, Serialize, Deserialize, Default, Clone)]
#[command(author, version, about, long_about = None)]
pub struct ToolCallConfig {
    /// The name of the subject containing the `ViewTasksSubscribePublishAggregated` subject
    #[arg(long)]
    #[serde(alias = "lhs_name")]
    pub subject_name: String,

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
    fn from_subject(subject: &Subject) -> Result<Self>
    where
        Self: Sized,
    {
        if let Some(bytes) = Self::from_subject_as_bytes(subject) {
            // Try to build the config
            match serde_json::from_slice::<ToolCallConfig>(&bytes) {
                Ok(config) => {
                    config.check_required_members(subject.get_name())?;
                    Ok(config)
                },
                Err(err) => Err(anyhow!(
                    "`{}` could not be built for subject `{}`. {err}",
                    Self::get_static_name(), 
                    subject.get_name()
                )),
            }
        } else {
            // Check for the required fields
            let required_fields = &["subject_name", "subject_names", "subscription_table_names"];
            let column_names = subject
                .get_schema()
                .fields()
                .iter()
                .map(|f| f.name().to_string())
                .collect::<HashSet<_>>();
            Self::check_required_fields(subject.get_name(), &column_names, required_fields)?;            

            // Try to build the config
            match subject.to_struct::<ToolCallConfig>() {
                Ok(mut config_vec) => match config_vec.pop() {
                    Some(config) => Ok(config),
                    None => Err(anyhow!(
                        "No config data found for `{}` with subject {}",
                        Self::get_static_name(),
                        subject.get_name()
                    )),
                },
                Err(err) => Err(anyhow!(
                    "`{}` could not be built for subject `{}`. {err}",
                    Self::get_static_name(), 
                    subject.get_name()
                )),
            }
        }
    }
    
    fn check_required_members(&self, _subject_name: &str) -> Result<()> {
        Ok(())
    }
}

impl MappableTrait for ToolCallConfig {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}
