use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Subject partitioning
/// 
/// # Notes
/// * Each partitioning variant is converted into a series of Processor steps that implement the actual partitioning
/// * Superstep 
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, Serialize, Deserialize, Default, ValueEnum)]
pub enum SubjectPartition {
    /// Last changed superstep
    #[default]
    #[value(name = "Superstep")]
    Superstep,
    #[value(name = "NumRows")]
    NumRows,
    #[value(name = "ChunkSize")]
    ChunkSize,
    #[value(name = "SuperstepNumRows")]
    SuperstepNumRows,
    #[value(name = "SuperstepChunkSize")]
    SuperstepChunkSize,
    #[value(name = "Timestamp")]
    Timestamp,
    #[value(name = "Date")]
    Date,
}