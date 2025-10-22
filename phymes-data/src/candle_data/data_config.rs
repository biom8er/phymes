use std::fmt::Display;

use clap::{Parser, ValueEnum};
use phymes_core::DataFormat;
use serde::{Deserialize, Serialize};

use crate::candle_operators::AvailableCandleOperators;

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
    // #[value(name = "Set")]
    // Set,
    // #[value(name = "First")]
    // First,
    // #[value(name = "Last")]
    // Last,
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
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataComparatorPredicate {
    #[default]
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

/// Data distance operators between two equal length vectors
#[derive(Default, Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataDistanceOperator {
    #[default]
    #[value(name = "NormalizedDotProduct")]
    NormalizedDotProduct,
    #[value(name = "NormalizedDifference")]
    NormalizedDifference,
}

impl Display for DataDistanceOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NormalizedDotProduct => write!(f, "NormalizedDotProduct"),
            Self::NormalizedDifference => write!(f, "NormalizedDifference"),
        }
    }
}

/// Data cast operators work in conjunction with DataCastAs to change the column name,
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
/// Casting allows for applying a [String] template for formatting
/// 
/// Casting allows for converting between DateTime strings and numeric Timestamps
/// 
/// [arrow_cast]: https://arrow.apache.org/rust/arrow_cast/index.html
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum)]
pub enum DataCastOperator {
    // #[value(name = "Base64Encode")]
    // Base64Encode,
    // #[value(name = "Base64Decode")]
    // Base64Decode,
    #[value(name = "Cast")]
    Cast,
    #[value(name = "None")]
    None,
}

impl Display for DataCastOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Self::Base64Encode => write!(f, "Base64Encode"),
            // Self::Base64Decode => write!(f, "Base64Decode"),
            Self::Cast => write!(f, "Cast"),
            Self::None => write!(f, "None"),
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
    pub lhs_values: Vec<String>,

    /// The right hand side values column identifier
    #[arg(long, default_value = "rhs_values")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_values: Option<Vec<String>>,

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

    /// Minijinja [String] template
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_template: Option<String>,

    /// The name of the resulting document after applying the minijinja template
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_name: Option<String>,

    /// The expression for the table within the minijinja template
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub table_expression: Option<String>,

    /// A serialized JSON [Value] representing the input for the template beyond the table_expression
    ///   where the table_expression will be inserted into to complete the input for the template
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_input: Option<String>,

    /// The length of the document chunks
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,

    /// The length of overlap between document chunks
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_overlap: Option<usize>,

    /// The data format to extract
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<DataFormat>,
    
    /// Vec of Strings for the comparator columns
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmp_columns: Option<Vec<String>>,

    /// Vec of [DataComparatorOperator]s specifying the comparator operator to apply to each cmp_column
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmp_operators: Option<Vec<DataComparatorOperator>>,

    /// Data Comparison predicates to evaluate parenthetic groups
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cmp_predicate: Option<DataComparatorPredicate>,

    /// Vec of Strings for the aggregation columns
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agg_columns: Option<Vec<String>>,

    /// Vec of [DataAggregatorOperator]s specifying the aggregator operator to apply to each agg_column
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agg_operators: Option<Vec<DataAggregatorOperator>>,    

    /// Vec of [String]s for the columns to rename to
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub as_columns: Option<Vec<String>>,

    /// Vec of of [DataCastOperator]s specifying the cast operator to apply to each lhs_values
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast_operators: Option<Vec<DataCastOperator>>,

    /// Vec of [DataType]s cast to [String]s specifying the data type to cast each lhs_values to
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast_datatypes: Option<Vec<String>>,

    /// Vec of Slice of [String]s specifying the template to use when casting each lhs_value to a [String] representation
    ///   where the template is a simple minijinja template with a single expression for the column
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast_templates: Option<Vec<String>>,

    /// true for ascending and false for descending
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asc: Option<bool>,

    /// Vec of [String]s for the pivot table columns
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pvt_columns: Option<Vec<String>>,

    /// Vec of [String]s for default values when missing values are encountered
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_values: Option<Vec<String>>,

    /// Data distance operator to apply between vectors
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dist_operator: Option<DataDistanceOperator>,
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
            lhs_values: vec!["lhs_values".to_string()],
            rhs_values: None,
            lhs_args: None,
            rhs_args: None,
            op_kwargs: None,
            stream: DataStreamManager::AccumulateLHSAccumulateRHS,
            operator: AvailableCandleOperators::VectorDistance,
            doc_template: None,
            doc_name: None,
            table_expression: None,
            doc_input: None,
            chunk_size: None,
            chunk_overlap: None,
            format: None,
            cmp_columns: None,
            cmp_operators: None,
            cmp_predicate: None,
            agg_columns: None,
            agg_operators: None,
            as_columns: None,
            cast_operators: None,
            cast_datatypes: None,
            cast_templates: None,
            asc: None,
            pvt_columns: None,
            default_values: None,
            dist_operator: None,
        }
    }
}