use std::fmt::Display;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Subject partitioning into folders
#[derive(
    Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize, Default, ValueEnum,
)]
pub enum SubjectFolderPartition {
    /// `RecordBatch`es are written to a folder with the last changed superstep
    #[default]
    #[value(name = "Superstep")]
    Superstep,
    /// `RecordBatch`es are written to a folder with the last changed timestamp
    #[value(name = "Timestamp")]
    Timestamp,
    /// `RecordBatch`es are written to a folder with the last changed Date
    #[value(name = "Date")]
    Date,
    /// No additional folder partitioning
    #[value(name = "None")]
    None,
}
impl Display for SubjectFolderPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Superstep => write!(f, "Superstep"),
            Self::Timestamp => write!(f, "Timestamp"),
            Self::Date => write!(f, "Date"),
            Self::None => write!(f, "None"),
        }
    }
}

/// Subject partitioning into files
#[derive(
    Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize, Default, ValueEnum,
)]
pub enum SubjectFilePartition {
    /// `RecordBatch`es partitioned according to the number of rows
    #[value(name = "NumRows")]
    NumRows,
    /// `RecordBatch`es partitioned according to the number of byte size
    #[value(name = "ChunkSize")]
    ChunkSize,
    /// No additional file partitioning
    #[default]
    #[value(name = "None")]
    None,
}
impl Display for SubjectFilePartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumRows => write!(f, "NumRows"),
            Self::ChunkSize => write!(f, "ChunkSize"),
            Self::None => write!(f, "None"),
        }
    }
}
