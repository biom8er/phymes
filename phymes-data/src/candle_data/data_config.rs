use std::fmt::Display;

use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::candle_operators::available_candle_operators::AvailableCandleOperators;

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

impl Display for DataStreamManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AccumulateLHSStreamRHS => write!(f, "AccumulateLHSStreamRHS"),
            Self::AccumulateLHSAccumulateRHS => write!(f, "AccumulateLHSAccumulateRHS"),
            Self::StreamLHSStreamRHS => write!(f, "StreamLHSStreamRHS"),
            Self::StreamLHSAccumulateRHS => write!(f, "StreamLHSAccumulateRHS"),
        }
    }
}

/// Data Aggregation (Reduction) operators
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataAggregatorOperator {
    #[value(name = "Max")]
    Max,
    #[value(name = "Min")]
    Min,
    #[value(name = "Sum")]
    Sum,
    #[value(name = "Mean")]
    Mean,
    #[value(name = "Var")]
    Var,
    #[value(name = "Count")]
    Count,
    #[value(name = "Concat")]
    Concat,
}

impl Display for DataAggregatorOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Max => write!(f, "Max"),
            Self::Min => write!(f, "Min"),
            Self::Sum => write!(f, "Sum"),
            Self::Mean => write!(f, "Mean"),
            Self::Var => write!(f, "Var"),
            Self::Count => write!(f, "Count"),
            Self::Concat => write!(f, "Concat"),
        }
    }
}

/// Data Comparison operators
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataComparatorOperator {
    /// Primitive
    #[value(name = "Equals")]
    Equals,
    #[value(name = "NotEquals")]
    NotEquals,
    #[value(name = "LessThanOrEqualTo")]
    LessThanOrEqualTo,
    #[value(name = "GreaterThanOrEqualTo")]
    GreaterThanOrEqualTo,
    #[value(name = "LessThan")]
    LessThan,
    #[value(name = "GreaterThan")]
    GreaterThan,
    /// Nested and non-primitive
    #[value(name = "Contains")]
    Contains,
    #[value(name = "EndsWith")]
    EndsWith,
    #[value(name = "CaseInsensitiveLike")]
    CaseInsensitiveLike,
    #[value(name = "Like")]
    Like,
    #[value(name = "CaseInsensitiveNotLike")]
    CaseInsensitiveNotLike,
    #[value(name = "NotLike")]
    NotLike,
    #[value(name = "InList")]
    InList,
    #[value(name = "InListUtf8")]
    InListUtf8,
    #[value(name = "RegExpIsMatch")]
    RegExpIsMatch,
    #[value(name = "StartsWith")]
    StartsWith,
}

impl Display for DataComparatorOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Equals => write!(f, "Equals"),
            Self::NotEquals => write!(f, "NotEquals"),
            Self::LessThanOrEqualTo => write!(f, "LessThanOrEqualTo"),
            Self::GreaterThanOrEqualTo => write!(f, "GreaterThanOrEqualTo"),
            Self::LessThan => write!(f, "LessThan"),
            Self::GreaterThan => write!(f, "GreaterThan"),
            Self::Contains => write!(f, "Contains"),
            Self::EndsWith => write!(f, "EndsWith"),
            Self::CaseInsensitiveLike => write!(f, "CaseInsensitiveLike"),
            Self::Like => write!(f, "Like"),
            Self::CaseInsensitiveNotLike => write!(f, "CaseInsensitiveNotLike"),
            Self::NotLike => write!(f, "NotLike"),
            Self::InList => write!(f, "InList"),
            Self::InListUtf8 => write!(f, "InListUtf8"),
            Self::RegExpIsMatch => write!(f, "RegExpIsMatch"),
            Self::StartsWith => write!(f, "StartsWith"),
        }
    }
}

/// Data Comparison predicates to evaluate parenthetic groups
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataComparatorPredicate {
    #[value(name = "All")]
    All,
    #[value(name = "Any")]
    Any,
}

impl Display for DataComparatorPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All => write!(f, "All"),
            Self::Any => write!(f, "Any"),
        }
    }
}

/// Data cast operators with work in conjunction with DataCastAs to change the column name,
///   DataCastDataType to change the data type, and DataCastTemplate to apply a template
/// 
/// # Notes
/// 
/// Casting uses the [arrow_cast] crate to convert between [DatayType]s
/// 
/// 1. Check if conversion is possible https://arrow.apache.org/rust/arrow_cast/cast/fn.can_cast_types.html
/// 2. Convert between types https://arrow.apache.org/rust/arrow_cast/cast/fn.cast_with_options.html
/// 3. Encode/Decode Base64 https://arrow.apache.org/rust/arrow_cast/base64/index.html with BASE64_URL_SAFE_NO_PAD engine
/// 
/// Casting also allows for applying a [String] template using
/// 
/// 1. `format!` from the standard library
/// 2. Jinja2 template
/// 
/// [arrow_cast]: https://arrow.apache.org/rust/arrow_cast/index.html
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataCastOperator {
    #[value(name = "Base64Encode")]
    Base64Encode,
    #[value(name = "Base64Decode")]
    Base64Decode,
    #[value(name = "Cast")]
    Cast,
    #[value(name = "ApplyJinja2Template")]
    ApplyJinja2Template,
    #[value(name = "ApplyStdTemplate")]
    ApplyStdTemplate,
}

impl Display for DataCastOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base64Encode => write!(f, "Base64Encode"),
            Self::Base64Decode => write!(f, "Base64Decode"),
            Self::Cast => write!(f, "Cast"),
            Self::ApplyJinja2Template => write!(f, "ApplyJinja2Template"),
            Self::ApplyStdTemplate => write!(f, "ApplyStdTemplate"),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_name: Option<String>,

    /// The left hand side primary key column identifier
    #[arg(long, default_value = "lhs_pk")]
    pub lhs_pk: String,

    /// The right hand side primary key column identifier
    #[arg(long, default_value = "rhs_pk")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_pk: Option<String>,

    /// The left hand side primary key column identifier
    #[arg(long, default_value = "lhs_fk")]
    pub lhs_fk: String,

    /// The right hand side primary key column identifier
    #[arg(long, default_value = "rhs_fk")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_fk: Option<String>,

    /// The left hand side values column identifier
    #[arg(long, default_value = "lhs_values")]
    pub lhs_values: String,

    /// The right hand side values column identifier
    #[arg(long, default_value = "rhs_values")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_values: Option<String>,

    /// The left hand side arguments to the operator
    /// JSONized vector of record batches
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_args: Option<String>,

    /// The right hand side arguments to the operator
    /// JSONized vector of record batches
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_args: Option<String>,

    /// Operator keyword arguments in JSON format
    /// that can be deserialized on the fly
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_kwargs: Option<String>,

    /// The streaming strategy to use
    #[arg(long, default_value = "accumulate-lhs-accumulate-rhs")]
    pub stream: DataStreamManager,

    /// The operator to invoke
    #[arg(long, default_value = "relative-similarity-score")]
    pub operator: AvailableCandleOperators,
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
            operator: AvailableCandleOperators::RelativeSimilarityScore,
        }
    }
}