use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Subject partitioning into folders
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize, Default, ValueEnum)]
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

/// Subject partitioning into files
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize, Default, ValueEnum)]
pub enum SubjectFilePartition {
    /// `RecordBatch`es partitioned according to the number of rows
    #[default]
    #[value(name = "NumRows")]
    NumRows,
    /// `RecordBatch`es partitioned according to the number of byte size
    #[value(name = "ChunkSize")]
    ChunkSize,
    /// No additional file partitioning
    #[value(name = "None")]
    None,
}