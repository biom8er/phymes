use anyhow::Result;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::candle_operators::which_operator::WhichCandleOperator;

#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataStreamManager {
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

impl DataStreamManager {
    pub fn get_name(&self) -> &str {
        match self {
            Self::AccumulateLHSStreamRHS => "accumulate-lhs-stream-rhs",
            Self::AccumulateLHSAccumulateRHS => "accumulate-lhs-accumulate-rhs",
            Self::StreamLHSStreamRHS => "stream-lhs-stream-rhs",
            Self::StreamLHSAccumulateRHS => "stream-lhs-accumulate-rhs",
        }
    }
}

/// Data Aggregation (Reduction) operators
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataAggregatorOperator {
    Max,
    Min,
    Sum,
    Mean,
    Var,
    Count,
    Concat,
}

impl DataAggregatorOperator {
    pub fn get_name(&self) -> &str {
        match self {
            Self::Max => "Max",
            Self::Min => "Min",
            Self::Sum => "Sum",
            Self::Mean => "Mean",
            Self::Var => "Var",
            Self::Count => "Count",
            Self::Concat => "Concat",
        }
    }
}

/// Data Comparison operators
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataComparatorOperator {
    /// Primitive
    Equals,
    NotEquals,
    LessThanOrEqualTo,
    GreaterThanOrEqualTo,
    LessThan,
    GreaterThan,
    /// Nested and non-primitive
    Contains,
    EndsWith,
    CaseInsensitiveLike,
    Like,
    CaseInsensitiveNotLike,
    NotLike,
    InList,
    InListUtf8,
    RegExpIsMatch,
    StartsWith,
}

impl DataComparatorOperator {
    pub fn get_name(&self) -> &str {
        match self {
            Self::Equals => "Equals",
            Self::NotEquals => "NotEquals",
            Self::LessThanOrEqualTo => "LessThanOrEqualTo",
            Self::GreaterThanOrEqualTo => "GreaterThanOrEqualTo",
            Self::LessThan => "LessThan",
            Self::GreaterThan => "GreaterThan",
            Self::Contains => "Contains",
            Self::EndsWith => "EndsWith",
            Self::CaseInsensitiveLike => "CaseInsensitiveLike",
            Self::Like => "Like",
            Self::CaseInsensitiveNotLike => "CaseInsensitiveNotLike",
            Self::NotLike => "NotLike",
            Self::InList => "InList",
            Self::InListUtf8  => "InListUtf8",
            Self::RegExpIsMatch => "RegExpIsMatch",
            Self::StartsWith => "StartsWith",
        }
    }
}

/// Data Comparison predicates to evaluate parenthetic groups
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataComparatorPredicate {
    All,
    Any
}

impl DataComparatorPredicate {
    pub fn get_name(&self) -> &str {
        match self {
            Self::All => "All",
            Self::Any => "Any",
        }
    }
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct DataConfig {
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
    #[arg(long, default_value = "accumulate-lhs-accumulate-rhs")]
    pub stream: DataStreamManager,

    /// The operator to invoke
    #[arg(long, default_value = "relative-similarity-score")]
    pub which: WhichCandleOperator,
}

impl Default for DataConfig {
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
            stream: DataStreamManager::AccumulateLHSAccumulateRHS,
            which: WhichCandleOperator::RelativeSimilarityScore,
        }
    }
}

impl DataConfig {
    #[allow(dead_code)]
    fn new_from_json(input: &str) -> Result<Self> {
        let self_data: DataConfig = serde_json::from_str(input)?;
        Ok(self_data)
    }
}
