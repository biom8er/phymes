use std::fmt::Display;

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use phymes_core::{AvailableSubjects, DataFormat, MappableTrait, Table, TableTrait};
use phymes_diagnostics::HashSet;
use serde::{Deserialize, Serialize};

use crate::{AvailableJinja2Templates, candle_operators::AvailableCandleOperators};

#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataStreamManager {
    /// Accumulate the record batches before streaming operations for each record batch
    #[default]
    #[value(name = "Accumulate")]
    Accumulate,
    /// Stream the record batches
    #[value(name = "Stream")]
    Stream,
}

impl Display for DataStreamManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Accumulate => write!(f, "Accumulate"),
            Self::Stream => write!(f, "Stream"),
        }
    }
}

/// Data Aggregation (Reduction) operators
///
/// # Notes
/// - `Max`, `Min`, `Sum`, `Mean`, and `Var` can only be applied to non-nested primitive [DataType]s
/// - `Count` can be applied to all [DataType]s and generates a UInt32Array
/// - `Concat` can only be applied to Utf8 [DataType] to generate a new Utf8 [DataType] by joining the [String]s together
/// - `List` and `Set` can be applied to all primitive [DataType]s except floats.
///   Non-nested primitive and non-primitive [DataType]s will generate a nested primitive or non-primitive Array.
///   Nested primitive or non-primitive [DataType]s will maintain the nested primitive or non-primitive Array through extension of the list or set.
/// - `First` and `Last` can be applied to all [DataType]s.
///
/// [DataType]: arrow::datatypes::DataType
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
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
    #[default]
    #[value(name = "Count")]
    Count,
    #[value(name = "Concat")]
    Concat,
    #[value(name = "ConcatSemicolonSeperator")]
    ConcatSemicolonSeperator,
    #[value(name = "List")]
    List,
    #[value(name = "Set")]
    Set,
    #[value(name = "First")]
    First,
    #[value(name = "Last")]
    Last,
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
            Self::ConcatSemicolonSeperator => write!(f, "ConcatSemicolonSeperator"),
            Self::List => write!(f, "List"),
            Self::Set => write!(f, "Set"),
            Self::First => write!(f, "First"),
            Self::Last => write!(f, "Last"),
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
/// Casting uses the [arrow_cast] crate to convert between [DataType]s
///
/// [DataType]: arrow::datatypes::DataType
///
/// 1. Check if conversion is possible <https://arrow.apache.org/rust/arrow_cast/cast/fn.can_cast_types.html>
/// 2. Convert between types <https://arrow.apache.org/rust/arrow_cast/cast/fn.cast_with_options.html>
/// 3. Encode/Decode Base64 <https://arrow.apache.org/rust/arrow_cast/base64/index.html> with BASE64_URL_SAFE_NO_PAD engine
/// 4. Convert a List-UInt8 (Bytes) to Utf8
///
/// Casting allows for applying a [String] template for formatting
///
/// Casting allows for converting between DateTime strings and numeric Timestamps
///
/// [arrow_cast]: <https://arrow.apache.org/rust/arrow_cast/index.html>
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataCastOperator {
    // #[value(name = "Base64Encode")]
    // Base64Encode,
    // #[value(name = "Base64Decode")]
    // Base64Decode,
    #[value(name = "Cast")]
    Cast,
    #[value(name = "BytesToString")]
    BytesToString,
    #[value(name = "Hash")]
    Hash,
    #[default]
    #[value(name = "None")]
    None,
}

impl Display for DataCastOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Self::Base64Encode => write!(f, "Base64Encode"),
            // Self::Base64Decode => write!(f, "Base64Decode"),
            Self::Cast => write!(f, "Cast"),
            Self::BytesToString => write!(f, "BytesToString"),
            Self::Hash => write!(f, "Hash"),
            Self::None => write!(f, "None"),
        }
    }
}

/// Data Column operators between one (unary) or two (binary) columns
///
/// # Notes on binary operators
/// - `Max`, `Min`, `Add`, `Sub`, `Mult`, `Div`, and `Var` can only be applied to non-nested primitive [DataType]s
/// - `And`, `Or`, `XOr`, `LeftShift`, and `RightShift` can only be applied to non-nested primitive [DataType]s
/// - `Concat` can only be applied to Utf8 [DataType] to generate a new Utf8 [DataType] by joining the [String]s together
/// - `List` and `Set` can be applied to all primitive [DataType]s except floats.
///   Non-nested primitive and non-primitive [DataType]s will generate a nested primitive or non-primitive Array.
///   Nested primitive or non-primitive [DataType]s will maintain the nested primitive or non-primitive Array through extension of the list or set.
///
/// # Notes on unary operators
/// - `Not`can only be applied to non-nested primitive [DataType]s
/// - `Len` can be applied to all [DataType]s and generates a UInt32Array
///
/// # Notes on intialization operators
/// - `Zeros` and `Ones` will create a new column filled with primitive zero's or one's
/// - `String` will create a new column filled with an empty Utf8
/// - `Value` will create a new column filled with a specified primitive or non-primitive value
///  
///
/// [DataType]: arrow::datatypes::DataType
#[derive(Debug, Serialize, Deserialize, Clone, ValueEnum, Default)]
pub enum DataColumnOperator {
    #[value(name = "And")]
    And,
    #[value(name = "AndNot")]
    AndNot,
    #[value(name = "Or")]
    Or,
    #[value(name = "XOr")]
    XOr,
    #[value(name = "Not")]
    Not,
    #[value(name = "LeftShift")]
    LeftShift,
    #[value(name = "RightShift")]
    RightShift,
    #[value(name = "Max")]
    Max,
    #[value(name = "Min")]
    Min,
    #[value(name = "Add")]
    Add,
    #[value(name = "Sub")]
    Sub,
    #[value(name = "Mult")]
    Mult,
    #[value(name = "Div")]
    Div,
    #[value(name = "Rem")]
    Rem,
    #[value(name = "List")]
    List,
    #[value(name = "Set")]
    Set,
    #[value(name = "Concat")]
    Concat,
    #[value(name = "Len")]
    Len,
    #[default]
    #[value(name = "None")]
    None,
    #[value(name = "Zeros")]
    Zeros,
    #[value(name = "Ones")]
    Ones,
    // #[value(name = "Rand")]
    // Rand,
    #[value(name = "String")]
    String,
    #[value(name = "Value")]
    Value,
    #[value(name = "BroadcastMax")]
    BroadcastMax,
    #[value(name = "BroadcastMin")]
    BroadcastMin,
    #[value(name = "BroadcastMean")]
    BroadcastMean,
    #[value(name = "BroadcastVar")]
    BroadcastVar,
    #[value(name = "BroadcastCount")]
    BroadcastCount,
    #[value(name = "BroadcastList")]
    BroadcastList,
    #[value(name = "BroadcastSet")]
    BroadcastSet,
    #[value(name = "CumSum")]
    CumSum,
}

impl DataColumnOperator {
    /// Can the operator be applied to two columns
    pub fn is_binary(&self) -> bool {
        match self {
            Self::And
            | Self::AndNot
            | Self::Or
            | Self::XOr
            | Self::Not
            | Self::LeftShift
            | Self::RightShift
            | Self::Max
            | Self::Min
            | Self::Add
            | Self::Sub
            | Self::Mult
            | Self::Div
            | Self::Rem
            | Self::List
            | Self::Set
            | Self::Concat
            | Self::Len
            | Self::None => true,
            Self::Zeros
            | Self::Ones
            | Self::String
            | Self::Value
            | Self::BroadcastMax
            | Self::BroadcastMin
            | Self::BroadcastMean
            | Self::BroadcastVar
            | Self::BroadcastCount
            | Self::BroadcastList
            | Self::BroadcastSet
            | Self::CumSum => false,
        }
    }

    /// Can the operator be applied to one columns
    pub fn is_unary(&self) -> bool {
        match self {
            Self::And
            | Self::AndNot
            | Self::Or
            | Self::XOr
            | Self::LeftShift
            | Self::RightShift
            | Self::Max
            | Self::Min
            | Self::Add
            | Self::Sub
            | Self::Mult
            | Self::Div
            | Self::Rem
            | Self::List
            | Self::Set
            | Self::Concat
            | Self::Zeros
            | Self::Ones
            | Self::String
            | Self::Value => false,
            Self::BroadcastMax
            | Self::BroadcastMin
            | Self::BroadcastMean
            | Self::BroadcastVar
            | Self::BroadcastCount
            | Self::BroadcastList
            | Self::BroadcastSet
            | Self::CumSum
            | Self::Not
            | Self::Len
            | Self::None => true,
        }
    }

    /// Can the operator initialize a new column
    pub fn is_init(&self) -> bool {
        match self {
            Self::And
            | Self::AndNot
            | Self::Or
            | Self::XOr
            | Self::Not
            | Self::LeftShift
            | Self::RightShift
            | Self::Max
            | Self::Min
            | Self::Add
            | Self::Sub
            | Self::Mult
            | Self::Div
            | Self::Rem
            | Self::List
            | Self::Set
            | Self::Concat
            | Self::Len
            | Self::BroadcastMax
            | Self::BroadcastMin
            | Self::BroadcastMean
            | Self::BroadcastVar
            | Self::BroadcastCount
            | Self::BroadcastList
            | Self::BroadcastSet
            | Self::CumSum
            | Self::None => false,
            Self::Zeros | Self::Ones | Self::String | Self::Value => true,
        }
    }
}

impl Display for DataColumnOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::And => write!(f, "And"),
            Self::AndNot => write!(f, "AndNot"),
            Self::Or => write!(f, "Or"),
            Self::XOr => write!(f, "XOr"),
            Self::Not => write!(f, "Not"),
            Self::LeftShift => write!(f, "LeftShift"),
            Self::RightShift => write!(f, "RightShift"),
            Self::Max => write!(f, "Max"),
            Self::Min => write!(f, "Min"),
            Self::Add => write!(f, "Add"),
            Self::Sub => write!(f, "Sub"),
            Self::Mult => write!(f, "Mult"),
            Self::Div => write!(f, "Div"),
            Self::Rem => write!(f, "Rem"),
            Self::List => write!(f, "List"),
            Self::Set => write!(f, "Set"),
            Self::Concat => write!(f, "Concat"),
            Self::Len => write!(f, "Len"),
            Self::None => write!(f, "None"),
            Self::Zeros => write!(f, "Zeros"),
            Self::Ones => write!(f, "Ones"),
            Self::String => write!(f, "String"),
            Self::Value => write!(f, "Value"),
            Self::BroadcastMax => write!(f, "BroadcastMax"),
            Self::BroadcastMin => write!(f, "BroadcastMin"),
            Self::BroadcastMean => write!(f, "BroadcastMean"),
            Self::BroadcastVar => write!(f, "BroadcastVar"),
            Self::BroadcastCount => write!(f, "BroadcastCount"),
            Self::BroadcastList => write!(f, "BroadcastList"),
            Self::BroadcastSet => write!(f, "BroadcastSet"),
            Self::CumSum => write!(f, "CumSum"),
        }
    }
}

/// Traits for all configs
pub trait DataConfigTrait {
    /// Create an example and serialize to JSON
    ///
    /// # Notes
    /// - example implementation: serde_json::to_vec(&DataConfig)
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error>
    where
        Self: Serialize;

    /// Build the config from a [Table]
    fn from_table(table: &Table) -> Result<Self>
    where
        Self: Sized;
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone, Default)]
#[command(author, version, about, long_about = None)]
#[serde(default)]
pub struct DataConfig {
    /// Run on CPU rather than GPU even if a GPU is available.
    #[arg(long)]
    pub cpu: bool,

    /// The left hand side table name
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_name: Option<String>,

    /// The right hand side table name
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_name: Option<String>,

    /// The left hand side primary key column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_pk: Option<String>,

    /// The right hand side primary key column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_pk: Option<String>,

    /// The left hand side primary key column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_fk: Option<String>,

    /// The right hand side primary key column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_fk: Option<String>,

    /// The left hand side values column identifier
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lhs_values: Option<Vec<String>>,

    /// The right hand side values column identifier
    #[arg(long)]
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

    /// The streaming strategy to use for the LHS
    #[arg(long)]
    pub lhs_stream: DataStreamManager,

    /// The streaming strategy to use for the RHS if different than the LHS
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rhs_stream: Option<DataStreamManager>,

    /// The operator to invoke
    #[arg(long)]
    pub operator: AvailableCandleOperators,

    /// Minijinja [String] template
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_template: Option<AvailableJinja2Templates>,

    /// The name of the resulting document after applying the minijinja template
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_name: Option<String>,

    /// A serialized JSON [Value] representing the input for the template beyond the table_expression
    ///   where the table_expression will be inserted into to complete the input for the template
    ///
    /// [Value]: serde_json::Value
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

    /// The data schema to extract with
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<AvailableSubjects>,

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

    /// Vec of [String]s for the columns to reorder and include in the schema
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reorder_columns: Option<Vec<String>>,

    /// Vec of of [DataColumnOperator]s specifying the column transformation operator to apply to each lhs_values and optionally rhs_values
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column_operators: Option<Vec<DataColumnOperator>>,

    /// Vec of of [DataCastOperator]s specifying the cast operator to apply to each lhs_values
    #[arg(long)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cast_operators: Option<Vec<DataCastOperator>>,

    /// Vec of [DataType]s cast to [String]s specifying the data type to cast each lhs_values to
    ///
    /// [DataType]: arrow::datatypes::DataType
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

impl DataConfigTrait for DataConfig {
    fn to_example_json(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Self::default())
    }
    fn from_table(table: &Table) -> Result<Self>
    where
        Self: Sized,
    {
        // Check for the required fields
        let column_names = table
            .get_schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<HashSet<_>>();
        if !(column_names.contains("operator")
            && column_names.contains("cpu")
            && column_names.contains("lhs_stream"))
        {
            return Err(anyhow!(
                "Table {} is missing required Field for `operator`, `cpu`, or `lhs_stream` in DataConfig.",
                table.get_name()
            ));
        }

        // Try to build the config
        match table.to_struct::<DataConfig>() {
            Ok(config_vec) => match config_vec.first() {
                Some(config) => Ok(config.to_owned()),
                None => Err(anyhow!(
                    "No config data found for DataConfig with subject {}",
                    table.get_name()
                )),
            },
            Err(err) => Err(anyhow!(
                "DataConfig could not be built for subject {}. {err}",
                table.get_name()
            )),
        }
    }
}
