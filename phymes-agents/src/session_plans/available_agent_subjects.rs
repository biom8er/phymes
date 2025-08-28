use anyhow::Result;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// The available session plans
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, Default)]
pub enum AvailableAgentSubjects {
    #[default]
    UserMessages,
    AssistantMessages,
    ToolMessages,
    Tools,
    Pdfs,
    Audio,
    Videos,
    Images,
    Scripts,
    Documents,
    Queries,
    /// Any table derived from CSV or JSON
    TabularData,
    /// Any other table adhering to one of the `AvailableSubjects` in phymes-core
    OtherAvailableSubjects,

}

impl MappableTrait for AvailableProcessors {
    fn get_name(&self) -> &str {
        match self {
        }
    }
}