use anyhow::{Result, anyhow};
use clap::Parser;
use phymes_data::DataConfigTrait;
use phymes_diagnostics::HashSet;
use phymes_subject::{MappableTrait, Subject, SubjectTrait};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct LimitConfig {
    /// The number of rows to skip when limiting the row count
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip: Option<usize>,

    /// The number of rows to fetch when limiting the row count
    #[arg(long)]
    pub fetch: usize,
}

impl DataConfigTrait for LimitConfig {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Self::default())
    }
    fn from_subject(subject: &Subject) -> Result<Self>
    where
        Self: Sized,
    {
        if let Some(bytes) = Self::from_subject_as_bytes(subject) {
            // Try to build the config
            match serde_json::from_slice::<LimitConfig>(&bytes) {
                Ok(config) => {
                    config.check_required_members(subject.get_name())?;
                    Ok(config)
                }
                Err(err) => Err(anyhow!(
                    "`{}` could not be built for subject `{}`. {err}",
                    Self::get_static_name(),
                    subject.get_name()
                )),
            }
        } else {
            // Check for the required fields
            let required_fields = &["fetch"];
            let column_names = subject
                .get_schema()
                .fields()
                .iter()
                .map(|f| f.name().to_string())
                .collect::<HashSet<_>>();
            Self::check_required_fields(subject.get_name(), &column_names, required_fields)?;

            // Try to build the config
            match subject.to_struct::<LimitConfig>() {
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

impl MappableTrait for LimitConfig {
    fn get_name(&self) -> &str {
        Self::get_static_name()
    }
}
