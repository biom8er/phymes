use anyhow::Result;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

use super::ops_which::WhichCandleOps;

#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum CandleOpsStreamManager {
    /// Accumulate the LHS record batches before
    /// streaming operations for each RHS record batch
    #[value(name = "accumulate-lhs-stream-rhs")]
    AccumulateLHSStreamRHS,
    /// Accumulate the LHS and RHS record batches before
    /// operating over the accumulated record batches
    #[value(name = "accumulate-lhs-accumulate-rhs")]
    AccumulateLHSAccumulateRHS,
    /// Stream LHS and RHS record batches
    #[value(name = "stream-lhs-stream-rhs")]
    StreamLHSStreamRHS,
    /// Stream LHS and RHS record batches but
    /// accumulating the RHS results
    #[value(name = "stream-lhs-accumulate-rhs")]
    StreamLHSAccumulateRHS,
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct CandleOpsConfig {
    /// Run on CPU rather than GPU even if a GPU is available.
    #[arg(long)]
    pub cpu: bool,

    /// The left hand side table name
    #[arg(long, default_value = "lhs_name")]
    pub lhs_name: String,

    /// The right hand side table name
    #[arg(long, default_value = "rhs_name")]
    pub rhs_name: Option<String>,

    /// The left hand side primary key column identifier
    #[arg(long, default_value = "lhs_pk")]
    pub lhs_pk: String,

    /// The right hand side primary key column identifier
    #[arg(long, default_value = "rhs_pk")]
    pub rhs_pk: Option<String>,

    /// The left hand side primary key column identifier
    #[arg(long, default_value = "lhs_fk")]
    pub lhs_fk: String,

    /// The right hand side primary key column identifier
    #[arg(long, default_value = "rhs_fk")]
    pub rhs_fk: Option<String>,

    /// The left hand side values column identifier
    #[arg(long, default_value = "lhs_values")]
    pub lhs_values: String,

    /// The right hand side values column identifier
    #[arg(long, default_value = "rhs_values")]
    pub rhs_values: Option<String>,

    /// The left hand side arguments to the operator
    /// JSONized vector of record batches
    #[arg(long)]
    pub lhs_args: Option<String>,

    /// The right hand side arguments to the operator
    /// JSONized vector of record batches
    #[arg(long)]
    pub rhs_args: Option<String>,

    /// Operator keyword arguments in JSON format
    /// that can be deserialized on the fly
    #[arg(long)]
    pub op_kwargs: Option<String>,

    /// The streaming strategy to use
    #[arg(long, default_value = "stream-lhs-stream-rhs")]
    pub stream: CandleOpsStreamManager,

    /// The operator to invoke
    #[arg(long, default_value = "relative-similarity-score")]
    pub which: WhichCandleOps,
}

impl Default for CandleOpsConfig {
    fn default() -> Self {
        Self {
            cpu: false,
            lhs_name: "lhs_name".to_string(),
            rhs_name: None,
            lhs_pk: "lhs_pk".to_string(),
            rhs_pk: None,
            lhs_fk: "lhs_fk".to_string(),
            rhs_fk: None,
            lhs_values: "lhs_values".to_string(),
            rhs_values: None,
            lhs_args: None,
            rhs_args: None,
            op_kwargs: None,
            stream: CandleOpsStreamManager::StreamLHSStreamRHS,
            which: WhichCandleOps::RelativeSimilarityScore,
        }
    }
}

impl CandleOpsConfig {
    #[allow(dead_code)]
    fn new_from_json(input: &str) -> Result<Self> {
        let self_data: CandleOpsConfig = serde_json::from_str(input)?;
        Ok(self_data)
    }
}
